# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from typing import Any

import imageio_ffmpeg
from google import genai
from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.cloud import storage
from google.genai import types

from .utils.face_lock import (
    apply_face_lock_to_video,
    resolve_user_photo_bytes_and_uri,
)
from .utils.utils import load_prompt_from_file

# Set logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configuration constants
MODEL = "gemini-2.5-flash"
VIDEO_MODEL = "veo-3.1-generate-001"
VIDEO_MODEL_LOCATION = "us-central1"
DESCRIPTION = (
    "Agent responsible for generating video clips with exact face preservation "
    "and stitching them into a final short movie."
)
ASPECT_RATIO = "16:9"
AUTHORIZED_URI = "https://storage.mtls.cloud.google.com/"


def _load_image_bytes(
    path_or_uri: str, project_id: str, bucket_name: str
) -> bytes | None:
    """Helper to load image bytes from GCS, local file, or URL."""
    if not path_or_uri:
        return None
    try:
        normalized = path_or_uri.strip()
        if normalized.startswith(AUTHORIZED_URI):
            normalized = normalized.replace(AUTHORIZED_URI, "gs://")
        elif normalized.startswith("https://storage.googleapis.com/"):
            normalized = normalized.replace("https://storage.googleapis.com/", "gs://")

        if normalized.startswith("gs://"):
            storage_client = storage.Client(project=project_id)
            blob = storage.Blob.from_string(normalized, client=storage_client)
            return blob.download_as_bytes()
        elif os.path.exists(normalized):
            with open(normalized, "rb") as f:
                return f.read()
        elif normalized.startswith("http://") or normalized.startswith("https://"):
            ctx = urllib.request.ssl._create_unverified_context()
            req = urllib.request.Request(
                normalized, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, context=ctx) as resp:
                return resp.read()
    except Exception as e:
        logger.warning(f"Failed to load image bytes from '{path_or_uri}': {e}")
    return None


# Video generation tool
def video_generate(
    prompt: str,
    scene_number: int,
    image_link: str = "",
    tool_context: ToolContext = None,  # type: ignore[assignment]
) -> list[str]:
    """Generates a video clip from a prompt and storyboard image with exact face locking.

    Args:
        prompt (str): Prompt describing the video scene to generate.
        scene_number (int): Scene number for the video.
        image_link (str): Optional GCS or HTTPS link to the storyboard image for
          Image-to-Video generation.
        tool_context (ToolContext): ToolContext needed by the tool.

    Returns:
        list[str]: Authorized link to the generated video stored in GCS.
    """
    try:
        session_id = tool_context._invocation_context.session.id
        bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        gcs_path = f"gs://{bucket_name}/{session_id}"

        # Extract dialogue from screenplay if available
        dialogue = ""
        screenplay = (
            tool_context._invocation_context.session.state.get(
                "screenplay", ""
            )
            or ""
        )
        dialogue += "\n".join(
            re.findall(r"^\s{2,}.+$", screenplay, re.MULTILINE)
        )

        if dialogue:
            prompt += f"\n\nAudio:\n{dialogue}"

        # Resolve storyboard image input for Image-to-Video (I2V)
        image_input = None
        if image_link:
            gs_image_uri = image_link.strip()
            if gs_image_uri.startswith(AUTHORIZED_URI):
                gs_image_uri = gs_image_uri.replace(AUTHORIZED_URI, "gs://")
            elif gs_image_uri.startswith("https://storage.googleapis.com/"):
                gs_image_uri = gs_image_uri.replace(
                    "https://storage.googleapis.com/", "gs://"
                )

            if gs_image_uri.startswith("gs://"):
                image_input = types.Image(gcs_uri=gs_image_uri, mime_type="image/png")
                logger.info(f"Using Image-to-Video with storyboard: {gs_image_uri}")

        # Actual video generation
        logger.info(
            f"Generating video for prompt '{prompt}' and image '{image_link}'"
        )

        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=VIDEO_MODEL_LOCATION,
        )

        operation = client.models.generate_videos(
            model=VIDEO_MODEL,
            prompt=prompt,
            image=image_input,
            config=types.GenerateVideosConfig(
                aspect_ratio=ASPECT_RATIO,
                output_gcs_uri=f"{gcs_path}/scene_{scene_number}",
                number_of_videos=1,
                duration_seconds=8,
                person_generation="allow_adult",
            ),
        )

        while not operation.done:
            time.sleep(15)
            operation = client.operations.get(operation)
            logger.info(f"Video generation operation: {operation}")

        if operation.response:
            logger.info(
                f"Generated {len(operation.result.generated_videos)} video(s) for prompt: {prompt}"
            )
            raw_uris = [
                video.video.uri
                for video in operation.result.generated_videos
            ]

            # Check if user photo is available for post-processing face-lock pass
            user_photo_bytes, _ = resolve_user_photo_bytes_and_uri(tool_context)

            if user_photo_bytes:
                storage_client = storage.Client(project=project_id)
                for uri in raw_uris:
                    try:
                        logger.info(f"Refining video with Exact Face-Lock: {uri}")
                        blob = storage.Blob.from_string(uri, client=storage_client)
                        temp_raw = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                        temp_locked = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
                        blob.download_to_filename(temp_raw)

                        success = apply_face_lock_to_video(
                            temp_raw, user_photo_bytes, temp_locked
                        )
                        if (
                            success
                            and os.path.exists(temp_locked)
                            and os.path.getsize(temp_locked) > 1000
                        ):
                            blob.upload_from_filename(
                                temp_locked, content_type="video/mp4"
                            )
                            logger.info(
                                f"Uploaded face-locked video back to GCS: {uri}"
                            )

                        if os.path.exists(temp_raw):
                            os.remove(temp_raw)
                        if os.path.exists(temp_locked):
                            os.remove(temp_locked)
                    except Exception as fe:
                        logger.warning(f"Face-lock video refinement failed for {uri}: {fe}")

            return [
                uri.replace("gs://", AUTHORIZED_URI)
                for uri in raw_uris
            ]
        else:
            logger.info(f"Generated no (0) video for prompt: {prompt}")
            return []  # Return an empty list if no video
    except Exception as e:
        logger.error(f"Error generating a video for {prompt}: {e}")
        return []


# Video merge tool
def merge_scene_videos(
    video_links: list[str],
    tool_context: ToolContext,
) -> str:
    """Consolidates and stitches all generated scene video clips into a single continuous movie.

    Args:
        video_links (list[str]): List of GCS or HTTPS URLs for each generated
          scene video clip in chronological order.
        tool_context (): ToolContext needed by the tool.

    Returns:
        str: Authorized link to the final merged movie stored in the GCS bucket.
    """
    session_id = tool_context._invocation_context.session.id
    bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")

    if not video_links:
        logger.warning("No video links provided for merging.")
        return ""

    logger.info(
        f"Merging {len(video_links)} scene video clips for session {session_id}"
    )

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    temp_dir = tempfile.mkdtemp(prefix="short_movie_merge_")
    local_clip_paths: list[str] = []

    try:
        # Step 1: Download each video clip from GCS / URL
        for idx, link in enumerate(video_links):
            gs_uri = link.strip()
            if gs_uri.startswith(AUTHORIZED_URI):
                gs_uri = gs_uri.replace(AUTHORIZED_URI, "gs://")
            elif gs_uri.startswith("https://storage.googleapis.com/"):
                gs_uri = gs_uri.replace(
                    "https://storage.googleapis.com/", "gs://"
                )

            local_clip = os.path.join(temp_dir, f"clip_{idx:03d}.mp4")

            if gs_uri.startswith("gs://"):
                blob = storage.Blob.from_string(gs_uri, client=storage_client)
                blob.download_to_filename(local_clip)
                logger.info(f"Downloaded {gs_uri} -> {local_clip}")
            elif os.path.exists(gs_uri):
                # Local path
                subprocess.run(
                    ["cp", gs_uri, local_clip], check=True, capture_output=True
                )
            else:
                logger.warning(
                    f"Could not resolve video link: {link}, skipping."
                )
                continue

            local_clip_paths.append(local_clip)

        if not local_clip_paths:
            logger.error("No valid video files were downloaded for merging.")
            return ""

        # Step 2: Create FFmpeg concat file list
        concat_list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for clip_path in local_clip_paths:
                f.write(f"file '{clip_path}'\n")

        # Step 3: Run FFmpeg to concatenate video files
        merged_output_path = os.path.join(temp_dir, "final_movie.mp4")
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            merged_output_path,
        ]

        logger.info(f"Running FFmpeg merge: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"FFmpeg stdout: {result.stdout}")

        # Step 4: Upload final merged video to GCS
        final_blob_name = f"{session_id}/final_movie.mp4"
        final_blob = bucket.blob(final_blob_name)
        final_blob.upload_from_filename(
            merged_output_path, content_type="video/mp4"
        )

        final_gcs_uri = f"gs://{bucket_name}/{final_blob_name}"
        final_authorized_link = final_gcs_uri.replace("gs://", AUTHORIZED_URI)

        logger.info(f"✅ Final movie uploaded successfully: {final_gcs_uri}")
        return final_authorized_link

    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg merge failed with returncode {e.returncode}")
        logger.error(f"FFmpeg stderr: {e.stderr}")
        return ""
    except Exception as e:
        logger.error(f"Error during video merge: {e}")
        return ""
    finally:
        # Cleanup temporary files
        try:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


# --- Video Agent ---
video_agent = None
try:
    video_agent = Agent(
        model=MODEL,
        name="video_agent",
        description=DESCRIPTION,
        instruction=load_prompt_from_file("video_agent.txt"),
        output_key="video",
        tools=[video_generate, merge_scene_videos],
    )
    logger.info(f"✅ Agent '{video_agent.name}' created using model '{MODEL}'.")
except Exception as e:
    logger.error(
        f"❌ Could not create Video agent. Check API Key ({MODEL}). Error: {e}"
    )
