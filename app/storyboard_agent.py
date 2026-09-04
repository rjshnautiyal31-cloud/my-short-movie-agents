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

from .utils.face_lock import (
    _load_image_bytes,
    composite_face_into_image_bytes,
    resolve_user_photo_bytes_and_uri,
)
from .utils.utils import load_prompt_from_file

# Set logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Configuration constants
MODEL = "gemini-2.5-flash"
IMAGE_MODEL = "gemini-2.5-flash-image"
DESCRIPTION = (
    "Agent responsible for creating unified cinematic master character sheets and "
    "consistent scene storyboards with exact face locking from user photos."
)

DEFAULT_CINEMATIC_STYLE = (
    "High-End Cinematic Live-Action Feature Film, 8k resolution, crisp photorealistic details, "
    "realistic human skin and clothing textures, volumetric cinema lighting."
)
STRICT_NEGATIVE_PROMPT = (
    "STRICT AVOID / NEGATIVE CONSTRAINTS: Absolutely NO pencil sketch, NO rough drawing, "
    "NO 2D cartoon, NO black and white line art, NO caricature, NO watermarks, NO storyboard text borders."
)


def _load_image_part(
    path_or_uri: str, project_id: str, bucket_name: str
) -> types.Part | None:
    """Helper to load an image as a genai Part."""
    img_bytes = _load_image_bytes(path_or_uri, project_id, bucket_name)
    if img_bytes:
        mime = "image/png"
        if path_or_uri.lower().endswith(".jpg") or path_or_uri.lower().endswith(".jpeg"):
            mime = "image/jpeg"
        return types.Part.from_bytes(data=img_bytes, mime_type=mime)
    return None


def create_character_profile(
    character_name: str,
    visual_description: str,
    photo_path_or_url: str = "",
    visual_style: str = "",
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
        visual_style (str): The unified art style. Defaults to photorealistic cinematic film.
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

        state = tool_context._invocation_context.session.state
        clean_name = re.sub(r"[^a-zA-Z0-9_]", "_", character_name.lower())

        # Resolve user photo bytes across all modalities
        user_photo_bytes, resolved_uri = resolve_user_photo_bytes_and_uri(
            tool_context, photo_path_or_url
        )

        # Unify visual style
        if not visual_style:
            visual_style = state.get("visual_style", DEFAULT_CINEMATIC_STYLE)
        state["visual_style"] = visual_style

        logger.info(
            f"Creating character reference profile for '{character_name}' (has_photo={user_photo_bytes is not None}, style: {visual_style})"
        )

        contents: list[Any] = []

        if user_photo_bytes:
            # Save raw user photo to GCS for downstream face locking
            storage_client = storage.Client(project=project_id)
            bucket = storage_client.bucket(bucket_name)
            photo_blob = bucket.blob(f"{session_id}/user_photo.png")
            photo_blob.upload_from_string(user_photo_bytes, content_type="image/png")
            user_photo_gcs = f"gs://{bucket_name}/{session_id}/user_photo.png"

            state["user_photo_gcs_uri"] = user_photo_gcs
            state["user_photo_uri"] = resolved_uri or user_photo_gcs

            photo_part = types.Part.from_bytes(data=user_photo_bytes, mime_type="image/png")
            contents.append(photo_part)
            contents.append(
                f"[REFERENCE PHOTO ATTACHED FOR CHARACTER '{character_name}']"
            )
            prompt = (
                f"Create a high-resolution Master Character Reference Profile for character '{character_name}'.\n"
                f"GLOBAL ART STYLE: {visual_style}.\n"
                f"MANDATORY INSTRUCTIONS:\n"
                f"1. Accurately capture the person's exact facial structure, eyes, nose, hairstyle, and likeness from the attached reference photo.\n"
                f"2. Seamlessly render the character in full cinematic photographic realism matching the global film style.\n"
                f"3. Character costume and attributes: {visual_description}.\n"
                f"4. Render full-body and 3/4 front views on a clean studio background with consistent lighting and colors.\n"
                f"{STRICT_NEGATIVE_PROMPT}"
            )
        else:
            prompt = (
                f"Create a high-resolution Master Character Reference Profile for character '{character_name}'.\n"
                f"GLOBAL ART STYLE: {visual_style}.\n"
                f"Character Visual Details: {visual_description}.\n"
                f"Render full-body and 3/4 front views on a clean studio background with consistent lighting and colors.\n"
                f"{STRICT_NEGATIVE_PROMPT}"
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
            # Composite user's exact face onto the master character profile
            if user_photo_bytes:
                image_bytes = composite_face_into_image_bytes(image_bytes, user_photo_bytes)

            storage_client = storage.Client(project=project_id)
            bucket = storage_client.bucket(bucket_name)
            blob_path = f"{session_id}/character_{clean_name}.png"
            blob = bucket.blob(blob_path)
            blob.upload_from_string(image_bytes, content_type="image/png")

            gcs_uri = f"gs://{bucket_name}/{blob_path}"

            # Save in session state for subsequent scene storyboards
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
        state = tool_context._invocation_context.session.state

        visual_style = state.get("visual_style", DEFAULT_CINEMATIC_STYLE)

        logger.info(
            f"Generating storyboard for scene {scene_number} (style: {visual_style}) with prompt: {prompt}"
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
            stored_refs = state.get("character_refs", {})
            if isinstance(stored_refs, dict):
                char_ref_dict = stored_refs

        for char_name, uri in char_ref_dict.items():
            part = _load_image_part(uri, project_id, bucket_name)
            if part:
                contents.append(part)
                contents.append(
                    f"[MASTER CHARACTER REFERENCE SHEET FOR: '{char_name.upper()}']"
                )

        scene_text = (
            f"SCENE {scene_number} CINEMATIC VISUALIZATION:\n"
            f"GLOBAL ART STYLE: {visual_style}\n\n"
            f"MANDATORY RULES:\n"
            f"1. RENDER QUALITY: Render this entire scene in full, vibrant, high-fidelity cinematic realism matching the global art style.\n"
            f"2. CHARACTER CONSISTENCY: For every character present, strictly match the exact facial structure, hair, clothing, and palette from their attached reference sheet.\n"
            f"3. COHESIVE LIGHTING: Match the natural environmental lighting and camera lens of a blockbuster movie.\n"
            f"{STRICT_NEGATIVE_PROMPT}\n\n"
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
            # Check for user photo face compositing to lock exact face on Frame 0
            user_photo_bytes, _ = resolve_user_photo_bytes_and_uri(tool_context)

            if user_photo_bytes:
                logger.info(f"Compositing user exact face onto Scene {scene_number} storyboard frame")
                image_bytes = composite_face_into_image_bytes(image_bytes, user_photo_bytes)

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
