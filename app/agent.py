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

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext

from .screenplay_agent import screenplay_agent
from .story_agent import story_agent
from .storyboard_agent import storyboard_agent
from .utils.face_lock import resolve_user_photo_bytes_and_uri
from .utils.utils import load_prompt_from_file
from .video_agent import video_agent

# Set logging
logger = logging.getLogger(__name__)

# Configuration constants
MODEL = "gemini-2.5-flash"
DESCRIPTION = (
    "Orchestrates the creation of a short, animated campfire story based on user input, "
    "utilizing specialized sub-agents for story generation, storyboard creation, and video generation."
)


def extract_and_store_user_photo(callback_context: CallbackContext) -> None:
    """Automatically extracts user uploaded photos from events, inline data, or input and saves to GCS."""
    try:
        raw_bytes, uri = resolve_user_photo_bytes_and_uri(callback_context)
        if raw_bytes:
            logger.info(f"✅ Root Director auto-captured user photo into session -> {uri}")
    except Exception as e:
        logger.warning(f"Error in extract_and_store_user_photo callback: {e}")


# --- Director Agent (root agent) ---

if story_agent and screenplay_agent and storyboard_agent and video_agent:
    root_agent = Agent(
        name="director_agent",
        model=MODEL,
        description=(DESCRIPTION),
        instruction=load_prompt_from_file("director_agent.txt"),
        before_agent_callback=extract_and_store_user_photo,
        sub_agents=[
            story_agent,
            screenplay_agent,
            storyboard_agent,
            video_agent,
        ],
    )
    logger.info(f"✅ Agent '{root_agent.name}' created using model '{MODEL}'.")
else:
    logger.error(
        "❌ Cannot create root agent because one or more sub-agents failed to initialize or a tool is missing."
    )
    if not story_agent:
        logger.error(" - Story Agent is missing.")
    if not screenplay_agent:
        logger.error(" - Screenplay Agent is missing.")
    if not storyboard_agent:
        logger.error(" - Storyboard Agent is missing.")
    if not video_agent:
        logger.error(" - Video Agent is missing.")
