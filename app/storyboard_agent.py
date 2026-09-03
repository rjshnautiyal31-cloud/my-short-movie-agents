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
import mimetypes
import os
import re
import urllib.request
from typing import Any

from google import genai
from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.cloud import storage
from google.genai import types

from .utils.utils import load_prompt_from_file

# Set logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configuration constants
MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"
DESCRIPTION = (
    "Agent responsible for creating consistent character reference sheets and "
    "scene storyboards based on a screenplay, story, and optional user photos."
)


def _load_image_part(
    path_or_uri: str, project_id: str, bucket_name: str
) -> types.Part | None:
    """Helper to load an image from a local file path, GCS URI, or HTTPS URL as a genai Part."""
    if not path_or_uri:
        return None

    try:
        # Standardize URI
        normalized = path_or_uri.strip()
        authorized_uri = "https://storage.mtls.cloud.google.com/"

        if normalized.startswith(authorized_uri):
            normalized = normalized.replace(authorized_uri, "gs://")
        elif normalized.startswith("https://storage.googleapis.com/"):
            normalized = normalized.replace("https://storage.googleapis.com/", "gs://")

        # Case 1: GCS URI
        if normalized.startswith("gs://"):
            storage_client = storage.Client(project=project_id)
            blob = storage.Blob.from_string(normalized, client=storage_client)
            img_bytes = blob.download_as_bytes()
            mime = blob.content_type or "image/png"
            return types.Part.from_bytes(data=img_bytes, mime_type=mime)

        # Case 2: Local file path
        if os.path.exists(normalized):
            with open(normalized, "rb") as f:
                img_bytes = f.read()
            mime, _ = mimetypes.guess_type(normalized)
            mime = mime or "image/jpeg"
            return types.Part.from_bytes(data=img_bytes, mime_type=mime)

        # Case 3: Public HTTP/HTTPS URL
        if normalized.startswith("http://") or normalized.startswith("https://"):
            req = urllib.request.Request(
                normalized, headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req) as resp:
                img_bytes = resp.read()
            mime = resp.headers.get_content_type() or "image/jpeg"
            return types.Part.from_bytes(data=img_bytes, mime_type=mime)

    except Exception as e:
        logger.warning(f"Could not load reference image from '{path_or_uri}': {e}")

    return None


def create_character_profile(
    character_name: str,
    visual_description: str,
    photo_path_or_url: str = "",
    visual_style: str = "Cinematic 3D Animation, vibrant cinematic lighting, cohesive aesthetic",
    tool_context: ToolContext = None,  # type: ignore[assignment]
) -> str:
    """Create a Master Character Reference Sheet from a description or user photo.

    Ensures the character's facial structure, hair, clothing, and palette remain
    consistent across all scene storyboards and videos without style clashes.

    Args:
        character_name (str): Name of the character (e.g., 'Rajesh', 'Sammy').
        visual_description (str): Detailed visual traits, clothing, colors, and distinct features.
        photo_path_or_url (str): Optional local file path (e.g. 'me.jpg') or URL/GCS link to
          a reference photo (e.g. photo of the user) to base the character on.
        visual_style (str): The unified art style (e.g., 'Cinematic 3D Animation', 'Photorealistic').
        tool_context (ToolContext): ADK runtime tool context.

    Returns:
        str: GCS link to the generated master character reference sheet.
    """
    try:
        session_id = tool_context._invocation_context.session.id
        bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        authorized_uri = "https://storage.mtls.cloud.google.com/"

        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", character_name.lower())
        logger.info(
            f"Creating character reference profile for '{character_name}' (photo: {photo_path_or_url})"
        )

        contents: list[Any] = []
        photo_part = None
        if photo_path_or_url:
            photo_part = _load_image_part(photo_path_or_url, project_id, bucket_name)

        if photo_part:
            contents.append(photo_part)
            contents.append(
                f"[REFERENCE PHOTO ATTACHED FOR CHARACTER '{character_name}']"
            )
            prompt = (
                f"Create a unified master character sheet for character '{character_name}'.\n"
                f"Art Style: {visual_style}.\n"
                f"Instructions:\n"
                f"1. Accurately capture the person's real facial structure, hairstyle, eyes, and distinct likeness from the attached photo.\n"
                f"2. Seamlessly adapt their likeness into the {visual_style} aesthetic so it looks completely organic (not a cutout or collage).\n"
                f"3. Character costume and attributes: {visual_description}.\n"
                f"4. Render full-body and 3/4 front views on a clean studio background with consistent lighting and colors."
            )
        else:
            prompt = (
                f"Create a master character sheet illustration for character '{character_name}'.\n"
                f"Art Style: {visual_style}.\n"
                f"Character Visual Details: {visual_description}.\n"
                f"Render full-body and 3/4 front views on a clean studio background with clear, consistent color palette and features."
            )

        contents.append(prompt)

        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            ),
        )

        image_bytes = None
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    image_bytes = part.inline_data.data
                    break

        if image_bytes:
            storage_client = storage.Client(project=project_id)
            bucket = storage_client.bucket(bucket_name)
            blob_path = f"{session_id}/character_{clean_name}.png"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(image_bytes, content_type="image/png")

            gcs_uri = f"gs://{bucket_name}/{blob_path}"

            # Save in session state for subsequent scene storyboards
            state = tool_context._invocation_context.session.state
            char_refs = state.get("character_refs", {})
            char_refs[clean_name] = gcs_uri
            state["character_refs"] = char_refs

            logger.info(f"Saved master character profile to {gcs_uri}")
            return gcs_uri.replace("gs://", authorized_uri)
        else:
            logger.warning(f"No image generated for character profile {character_name}")
            return f"Failed to generate character profile for {character_name}"

    except Exception as e:
        logger.error(f"Error in create_character_profile: {e}")
        return f"Error: {e}"


def storyboard_generate(
    prompt: str,
    scene_number: int,
    character_reference_link: str = "",
    tool_context: ToolContext = None,  # type: ignore[assignment]
) -> list[str]:
    """Generate a scene storyboard image, conditioned on master character reference sheets.

    Args:
        prompt (str): Text prompt describing the scene action, environment, and characters.
        scene_number (int): Scene number (1, 2, 3...).
        character_reference_link (str): Optional specific character reference image link.
        tool_context (ToolContext): ADK runtime tool context.

    Returns:
        list[str]: Link to the generated storyboard image stored in GCS.
    """
    try:
        session_id = tool_context._invocation_context.session.id
        bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        authorized_uri = "https://storage.mtls.cloud.google.com/"

        logger.info(
            f"Generating storyboard for scene {scene_number} with prompt: {prompt}"
        )

        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

        contents: list[Any] = []

        # Retrieve and explicitly label character references to prevent character blending
        char_ref_dict: dict[str, str] = {}
        if character_reference_link:
            char_ref_dict["character"] = character_reference_link
        else:
            state = tool_context._invocation_context.session.state
            stored_refs = state.get("character_refs", {})
            if isinstance(stored_refs, dict):
                char_ref_dict = stored_refs

        for char_name, uri in char_ref_dict.items():
            part = _load_image_part(uri, project_id, bucket_name)
            if part:
                contents.append(part)
                contents.append(
                    f"[REFERENCE SHEET FOR CHARACTER: '{char_name.upper()}']"
                )

        if char_ref_dict:
            scene_text = (
                f"SCENE {scene_number} STORYBOARD GENERATION:\n"
                f"CRITICAL RULES:\n"
                f"1. For each character appearing in this scene, strictly match their appearance, face, hair, clothing, and colors from their respective attached reference sheet.\n"
                f"2. DO NOT mix up or swap traits between characters.\n"
                f"3. Render the entire scene in full, cohesive cinematic color and lighting matching the movie's style.\n\n"
                f"Scene Action & Environment:\n{prompt}"
            )
        else:
            scene_text = (
                f"SCENE {scene_number} STORYBOARD GENERATION:\n"
                f"Render the scene in full cinematic color, dynamic composition, and consistent lighting.\n\n"
                f"Scene Action & Environment:\n{prompt}"
            )

        contents.append(scene_text)

        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            ),
        )

        image_bytes = None
        if response.candidates and response.candidates[0].content:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    image_bytes = part.inline_data.data
                    break

        if image_bytes:
            storage_client = storage.Client(project=project_id)
            bucket = storage_client.bucket(bucket_name)
            blob_path = f"{session_id}/scene_{scene_number}.png"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(image_bytes, content_type="image/png")

            gcs_uri = f"gs://{bucket_name}/{blob_path}"
            logger.info(f"Generated and saved storyboard to {gcs_uri}")
            return [gcs_uri.replace("gs://", authorized_uri)]
        else:
            logger.info(f"Generated no (0) images for prompt: {prompt}")
            return []
    except Exception as e:
        logger.error(f"Error generating storyboard for {prompt}: {e}")
        return []


# --- Storyboard Agent ---
storyboard_agent = None
try:
    storyboard_agent = Agent(
        model=MODEL,
        name="storyboard_agent",
        description=DESCRIPTION,
        instruction=load_prompt_from_file("storyboard_agent.txt"),
        output_key="storyboard",
        tools=[create_character_profile, storyboard_generate],
    )
    logger.info(
        f"✅ Agent '{storyboard_agent.name}' created using model '{MODEL}'."
    )
except Exception as e:
    logger.error(
        f"❌ Could not create Storyboard agent. Check API Key ({MODEL}). Error: {e}"
    )
