# Convert Videos to Short Clips

A Python application that automatically converts long videos into engaging vertical short clips with smart cropping and auto-generated captions using AI.

## Architecture

This project supports three modes:

### 1. Standalone Mode (Streamlit)
Original simple interface for single-user local processing.

### 2. Async Mode (FastAPI + Celery)
Production-ready asynchronous architecture for multi-user scalability.

### 3. Full Stack (Next.js Frontend + FastAPI Backend)
Modern SaaS-style frontend with the async backend.

```
┌─────────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│    Next.js      │────▶│   FastAPI    │────▶│    Redis    │────▶│   Celery    │
│   (Frontend)    │     │   (API)      │     │  (Broker)   │     │  (Workers)  │
└─────────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                               │                                         │
                               ▼                                         ▼
                        ┌─────────────┐                          ┌─────────────┐
                        │ PostgreSQL  │◀─────────────────────────│  Processing │
                        │  (Storage)  │                          │   Pipeline  │
                        └─────────────┘                          └─────────────┘
```

## Features

- **Multi-Clip Generation**: Generate 1-3 clips from a single video simultaneously
- **Configurable Clip Length**: Choose between 15s, 30s, or 60s target duration
- **AI-Powered Clip Categorization**: GPT-4 identifies three types of clips:
  - **Hook Focus**: Strongest opening grab - stops the scroll
  - **Emotional Peak**: Laughter, surprise, awe, or powerful insights
  - **Viral Moment**: Most shareable and quotable segment
- **GPT-4o Vision Analysis** (optional): Analyzes keyframes for visual hooks and ranks clips by virality potential
- **Smart Vertical Cropping**: Automatically detects and follows subjects (faces or motion) to create 9:16 vertical videos
- **Auto-Captions**: Generates captions using OpenAI Whisper and burns them into the video
- **Caption Styles**: Three configurable styles via JSON templates:
  - **Default**: Solid black background
  - **Minimal**: Transparent, subtle text
  - **Highlight**: Bold text with gold emphasis
- **Crop Profiles**: YAML-based profiles for different tracking behaviors (smooth_follow, snappy)
- **Async Processing**: Background workers with job tracking and progress updates
- **Retry & Recovery**: Automatic retry with configurable attempts for failed tasks
- **REST API**: Full REST API for job management and status polling
- **Modern Frontend**: Next.js 14 with dark mode, drag-and-drop upload, real-time progress

## Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Copy environment file
cp .env.example .env
# Edit .env with your OpenAI API key

# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api
docker-compose logs -f worker-pipeline
docker-compose logs -f worker-clips
```

Services will be available at:
- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Flower (Celery Monitor)**: http://localhost:5555

### Option 2: Local Development

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start PostgreSQL and Redis (or use Docker)
docker-compose up -d postgres redis

# Copy and configure environment
cp .env.example .env
# Edit .env with your settings

# Start the API server
cd backend
uvicorn app.main:app --reload --port 8000

# Start Celery workers (in separate terminals)
celery -A app.tasks.celery_app worker -Q pipeline,default --loglevel=info --concurrency=2
celery -A app.tasks.celery_app worker -Q clips --loglevel=info --concurrency=4

# Start the frontend (in another terminal)
cd frontend
npm install
npm run dev
```

### Option 3: Streamlit (Original)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Frontend

The Next.js frontend provides a modern SaaS-style interface:

### Pages

| Page | Route | Description |
|------|-------|-------------|
| Landing | `/` | Marketing page with features and pricing |
| Login | `/auth/login` | Email/password + social login |
| Register | `/auth/register` | New account creation |
| Dashboard | `/dashboard` | Main upload + job overview |
| Job Details | `/dashboard/jobs/[id]` | Processing progress + clip player |
| History | `/dashboard/history` | All jobs with search and filters |

### Key Components

- **Drag-and-Drop Upload**: File upload with configuration panel
- **Processing Progress**: Real-time step-by-step progress visualization
- **Video Player**: Custom player with controls, volume, fullscreen
- **Clip Gallery**: Side-by-side clip comparison with scores
- **Download Manager**: Individual or bulk clip downloads

### Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:3000`.

## API Endpoints

### Jobs

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/v1/jobs` | Create a new video processing job |
| `GET` | `/api/v1/jobs` | List all jobs (paginated) |
| `GET` | `/api/v1/jobs/{job_id}` | Get job status and progress |
| `POST` | `/api/v1/jobs/{job_id}/cancel` | Cancel a running job |
| `DELETE` | `/api/v1/jobs/{job_id}` | Delete a completed job |

### Clips

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/jobs/{job_id}/clips/{clip_id}/original` | Download original clip |
| `GET` | `/api/v1/jobs/{job_id}/clips/{clip_id}/final` | Download processed clip |

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |

## Job Status Flow

```
pending
  └── extracting_audio (10%)
      └── transcribing (25%)
          └── analyzing (40%)
              └── extracting_clips (55%)
                  └── processing_clips (60%)
                      └── completed (100%)
                          OR
                      └── failed
```

## Processing Pipeline

The async pipeline breaks video processing into independent tasks:

1. **Extract Audio** - Extracts audio track from video file
2. **Transcribe** - Converts audio to text with timestamps using Whisper
3. **Analyze** - GPT-4 identifies optimal clip segments
4. **Extract Clips** - Parses timestamps and creates clip records
5. **Process Clips** (parallel per clip):
   - Extract video segment
   - Crop to vertical 9:16 format
   - Add captions with selected style
6. **Vision Analysis** (optional) - GPT-4o analyzes visual hooks
7. **Finalize** - Updates job status and rankings

## Project Structure

```
├── app.py                          # Streamlit standalone app
├── core/                           # Shared video processing modules
│   ├── clip_analyzer.py
│   ├── clip_generator.py
│   ├── transcription.py
│   ├── video_processor.py
│   └── vision_analysis.py
├── crop_videos.py                  # Smart vertical cropping
├── captions.py                     # Caption generation
├── backend/                        # FastAPI async backend
│   └── app/
│       ├── main.py                 # FastAPI application
│       ├── core/                   # Config & database
│       ├── models/                 # SQLAlchemy models
│       ├── schemas/                # Pydantic schemas
│       ├── services/               # Business logic
│       ├── routers/                # API endpoints
│       └── tasks/                  # Celery tasks
├── frontend/                       # Next.js frontend
│   └── src/
│       ├── app/                    # Pages (App Router)
│       ├── components/             # React components
│       ├── lib/                    # Utilities & API client
│       ├── hooks/                  # Custom React hooks
│       └── types/                  # TypeScript types
├── config/
│   └── crop_profiles/              # YAML crop profiles
├── templates/
│   └── captions/                   # JSON caption templates
├── docker-compose.yml              # Docker services
├── Dockerfile                      # Container build
├── requirements.txt                # Python dependencies
└── .env.example                    # Environment template
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key |
| `DATABASE_URL` | `postgresql+asyncpg://...` | Database connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `UPLOAD_DIR` | `./storage/uploads` | Video upload directory |
| `OUTPUT_DIR` | `./storage/outputs` | Processed clips directory |
| `TEMP_DIR` | `./storage/temp` | Temporary files directory |
| `MAX_FILE_SIZE_MB` | `500` | Maximum upload size |
| `TASK_MAX_RETRIES` | `3` | Max retry attempts per task |
| `TASK_RETRY_DELAY` | `60` | Delay between retries (seconds) |
| `DEBUG` | `false` | Enable debug mode |

### Celery Queues

| Queue | Concurrency | Purpose |
|-------|-------------|---------|
| `pipeline` | 2 | Main processing pipeline tasks |
| `clips` | 4 | Individual clip processing (parallel) |
| `default` | - | Fallback queue |

## Requirements

- Python 3.11+
- Node.js 18+
- PostgreSQL 14+
- Redis 7+
- FFmpeg
- ImageMagick (for captions)
- OpenAI API key

## Dependencies

### Backend
- `fastapi` - REST API framework
- `celery` - Distributed task queue
- `redis` - Message broker
- `sqlalchemy` - Database ORM
- `asyncpg` - PostgreSQL async driver
- `opencv-python` - Video processing
- `moviepy` - Video editing
- `openai` - AI API access

### Frontend
- `next` - React framework
- `tailwindcss` - CSS framework
- `lucide-react` - Icons
- `react-dropzone` - File upload
- `zustand` - State management
- `framer-motion` - Animations
