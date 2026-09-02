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
    "Agent responsible for creating storyboards based on a screenplay and story"
)


# Storyboard generate tool
def storyboard_generate(
    prompt: str, scene_number: int, tool_context: ToolContext
) -> list[str]:
    """Generate storyboard image representing the passed prompt.

    Args:
        prompt (str): A text prompt describing the storyboard image that should
          be generated and returned by the tool.
        scene_number (int): Scene number
        tool_context (): ToolContext needed by the tool

    Returns:
        str: Link to the image stored in GCS bucket.
    """
    try:
        session_id = tool_context._invocation_context.session.id
        bucket_name = os.getenv("GOOGLE_CLOUD_BUCKET_NAME")
        project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
        location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
        authorized_uri = "https://storage.mtls.cloud.google.com/"

        logger.info(
            f"Generating image for scene {scene_number} with prompt: {prompt}"
        )

        client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location,
        )

        response = client.models.generate_content(
            model=IMAGE_MODEL,
            contents=prompt,
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
            logger.info(f"Generated and saved image to {gcs_uri}")
            return [gcs_uri.replace("gs://", authorized_uri)]
        else:
            logger.info(f"Generated no (0) images for prompt: {prompt}")
            return []
    except Exception as e:
        logger.error(f"Error generating an image for {prompt}: {e}")
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
        tools=[storyboard_generate],
    )
    logger.info(
        f"✅ Agent '{storyboard_agent.name}' created using model '{MODEL}'."
    )
except Exception as e:
    logger.error(
        f"❌ Could not create Storyboard agent. Check API Key ({MODEL}). Error: {e}"
    )

