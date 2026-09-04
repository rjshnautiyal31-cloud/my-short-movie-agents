# Short Movie Agents

A collaborative multi-agent AI system built on the **Google Agent Development Kit (ADK)** and **Google Cloud Vertex AI** that guides users end-to-end through creating AI short movies—from initial concept and custom character photos to screenplays, consistent storyboard illustrations, video clips, and final movie consolidation.

---

## Architecture & Multi-Agent Workflow

The system is coordinated by a **Director Agent** that orchestrates specialized sub-agents through a 4-step interactive, human-in-the-loop pipeline with strict approval gating at each phase:

```
[User / Custom Photo] ──► [Director Agent] (Coordinator & Gatekeeper)
                                 │
                                 ├─► Phase 1: [Story Agent] (Gemini 2.5 Flash)
                                 │            └── Campfire Story & Character Profiles ──► [User Approval]
                                 │
                                 ├─► Phase 2: [Screenplay Agent] (Gemini 2.5 Flash)
                                 │            └── Structured Screenplay ──► [User Approval]
                                 │
                                 ├─► Phase 3: [Storyboard Agent] (Gemini 2.5 Flash Image)
                                 │            └── Master Character Sheets & Storyboards ──► [User Approval]
                                 │
                                 └─► Phase 4: [Video Agent] (Veo 3.1 + FFmpeg)
                                              └── Image-to-Video Animation & Final Movie Consolidation
```

| Agent | File | Model / Technology | Role & Responsibilities |
|---|---|---|---|
| **Director Agent** | [`app/agent.py`](app/agent.py) | `gemini-2.5-flash` | Main root coordinator. Enforces strict step-by-step user approval before advancing phases. |
| **Story Agent** | [`app/story_agent.py`](app/story_agent.py) | `gemini-2.5-flash` | Generates narrative concepts, campfire stories, and detailed visual character descriptions. |
| **Screenplay Agent** | [`app/screenplay_agent.py`](app/screenplay_agent.py) | `gemini-2.5-flash` | Breaks stories into structured scenes with sluglines, action lines, and character dialogue. |
| **Storyboard Agent** | [`app/storyboard_agent.py`](app/storyboard_agent.py) | `gemini-2.5-flash-image` | Generates master character profile sheets (supporting user photo uploads) and consistent scene storyboards. |
| **Video Agent** | [`app/video_agent.py`](app/video_agent.py) | `veo-3.1-generate-001` + FFmpeg | Generates scene video clips via Image-to-Video (I2V) and stitches them into a continuous `final_movie.mp4`. |

---

## Key Features

### 1. Human-in-the-Loop Phase Gating
* The Director Agent strictly presents each asset (Story, Screenplay, Storyboards) and halts execution to ask for your review and approval.
* Revisions can be requested at any stage, refining the assets before proceeding to video generation.

### 2. Strict Character Consistency Across Scenes
* **Master Character Profile Sheets**: Generates front and 3/4 reference sheets that anchor the character's facial structure, hair, clothing, and palette.
* **Image-to-Image Conditioning**: Gemini 2.5 Flash Image uses the master character anchors across all scene storyboards.
* **Image-to-Video (I2V) Animation**: Veo 3.1 directly takes the generated scene storyboard as visual input, preventing character drift across video scenes.

### 3. Exact Face Preservation ("Self-as-Character" Hybrid Pipeline)
* **Direct Photo Composition (Storyboard Stage)**: Seamlessly composites your real, high-resolution face onto the character's body in scene storyboards using landmark detection, Reinhard color/lighting transfer, and Poisson blending.
* **Image-to-Video Animation (Veo 3.1)**: Veo uses the photo-anchored storyboard as Frame 0, animating the character with authentic facial likeness.
* **Post-Processing Video Face-Lock**: Automatically tracks and refines facial landmarks across all video frames with temporal smoothing to guarantee a **100% exact facial identity** across every scene.

---

## Project Structure

```
my-short-movie-agents/
├── app/                        # Multi-agent application package
│   ├── agent.py                # Director root agent definition
│   ├── story_agent.py          # Story generation agent
│   ├── screenplay_agent.py     # Screenplay generation agent
│   ├── storyboard_agent.py     # Storyboard image generator (Master Character Profiles & I2I)
│   ├── video_agent.py          # Video generation (I2V) & video merge tool
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
> *"I want to create a short movie about an adventurous explorer named Rajesh discovering a glowing crystal in the Himalayas. Use my photo at /path/to/my_photo.jpg for the explorer!"*

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