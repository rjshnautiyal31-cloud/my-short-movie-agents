# Short Movie Agents

A collaborative multi-agent AI system built on the **Google Agent Development Kit (ADK)** and **Google Cloud Vertex AI** that guides users end-to-end through creating AI short movies—from initial concept to screenplays, storyboard images, video clips, and final movie consolidation.

---

## Architecture & Multi-Agent Workflow

The system is coordinated by a **Director Agent** that orchestrates specialized sub-agents through a 4-step interactive pipeline:

```
[User] ──► [Director Agent] (Coordinator)
                 │
                 ├─► 1. [Story Agent] (Gemini 2.5 Flash) ──► Campfire Story
                 │
                 ├─► 2. [Screenplay Agent] (Gemini 2.5 Flash) ──► Scene-by-Scene Script
                 │
                 ├─► 3. [Storyboard Agent] (Gemini 2.5 Flash Image) ──► Scene PNGs (GCS)
                 │
                 └─► 4. [Video Agent] (Veo 3.1 + FFmpeg) ──► Scene MP4s & Final Movie (GCS)
```

| Agent | File | Model / Technology | Role & Responsibilities |
|---|---|---|---|
| **Director Agent** | [`app/agent.py`](app/agent.py) | `gemini-2.5-flash` | Main root coordinator. Guides the user step-by-step and manages asset approvals. |
| **Story Agent** | [`app/story_agent.py`](app/story_agent.py) | `gemini-2.5-flash` | Generates short, compelling narrative concepts and campfire stories. |
| **Screenplay Agent** | [`app/screenplay_agent.py`](app/screenplay_agent.py) | `gemini-2.5-flash` | Breaks stories into structured scenes with character descriptions, actions, and dialogue. |
| **Storyboard Agent** | [`app/storyboard_agent.py`](app/storyboard_agent.py) | `gemini-2.5-flash-image` | Generates high-resolution storyboard images for each scene and stores them in Cloud Storage. |
| **Video Agent** | [`app/video_agent.py`](app/video_agent.py) | `veo-3.1-generate-001` + FFmpeg | Generates scene video clips and automatically stitches them into a continuous `final_movie.mp4`. |

---

## Project Structure

```
my-short-movie-agents/
├── app/                        # Multi-agent application package
│   ├── agent.py                # Director root agent definition
│   ├── story_agent.py          # Story generation agent
│   ├── screenplay_agent.py     # Screenplay generation agent
│   ├── storyboard_agent.py     # Storyboard image generator (GenAI SDK)
│   ├── video_agent.py          # Video generation & video merge tool
│   ├── merge_session.py        # CLI utility to stitch past session clips
│   ├── server.py               # FastAPI backend server
│   ├── prompts/                # System instructions & agent prompts
│   └── utils/                  # Helper utilities (GCS, tracing, typing)
├── Makefile                    # Automation shortcuts (run, test, lint)
├── GEMINI.md                   # Google ADK development guide
├── pyproject.toml              # Dependencies and configuration
└── .env-template               # Environment variables template
```

---

## Requirements & Prerequisites

- **Python**: 3.13+
- **[uv](https://docs.astral.sh/uv/)**: Fast Python package manager
- **[Google Cloud SDK](https://cloud.google.com/sdk/docs/install)** (`gcloud` CLI)
- **Google Cloud Project** with Vertex AI enabled and a Cloud Storage bucket

---

## Getting Started

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/rjshnautiyal31-cloud/my-short-movie-agents.git
cd my-short-movie-agents
uv sync --dev
```

### 2. Configure Environment Variables

Create your `.env` file from the template:

```bash
cp .env-template .env
```

Edit `.env` with your Google Cloud details:

```ini
GOOGLE_CLOUD_PROJECT=your-gcp-project-id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_CLOUD_BUCKET_NAME=your-gcs-bucket-name
GOOGLE_GENAI_USE_VERTEXAI=TRUE
```

### 3. Grant Vertex AI Bucket Permissions

Ensure the Vertex AI service agent has access to write generated media into your Cloud Storage bucket:

```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe YOUR_PROJECT_ID --format='value(projectNumber)')

# Create Vertex AI service identity if not already created
gcloud beta services identity create --service=aiplatform.googleapis.com --project=YOUR_PROJECT_ID

# Grant Storage Object Admin role on the bucket
gcloud storage buckets add-iam-policy-binding gs://YOUR_BUCKET_NAME \
    --member="serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-aiplatform.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

---

## Running the Agents

### Option A: Interactive CLI (Terminal)

Launch the conversational Director Agent directly in your terminal:

```bash
uv run adk run app
```

Example prompt:
> *"I want to create a short movie about a brave little squirrel discovering a glowing crystal in the forest."*

---

### Option B: Interactive Web UI (Playground)

Launch the ADK Web Playground:

```bash
make playground
# or: uv run adk web app --port 8501
```

Open your browser at `http://localhost:8501`.

---

### Option C: Consolidating Previous Session Clips

If you already have scene clips from a previous session and want to stitch them into a single movie:

```bash
uv run python -m app.merge_session <SESSION_ID>
```

*(e.g., `uv run python -m app.merge_session a0ea7f1f-b17d-4cb3-b3a8-aab6e8c91e58`)*

The output `final_movie.mp4` will be uploaded directly to `gs://YOUR_BUCKET_NAME/<SESSION_ID>/final_movie.mp4`.

---

## Commands

| Command | Description |
|---|---|
| `uv run adk run app` | Run the Director Agent in terminal mode |
| `make playground` | Launch the ADK web UI on port 8501 |
| `make local-backend` | Launch local FastAPI server with hot-reload |
| `make test` | Run unit and integration tests |
| `make lint` | Run code quality checks (ruff, codespell, mypy) |
| `uv run python -m app.merge_session <ID>` | Stitch all scene clips for a session into `final_movie.mp4` |

---

## License

Apache License 2.0