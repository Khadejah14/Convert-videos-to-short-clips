import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Any

from app.core.database import get_db
from app.core.config import get_settings
from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.auth import MessageResponse
from app.agent.schemas import (
    AgentConfig,
    EditingStyle,
    CaptionStyle,
)

settings = get_settings()
router = APIRouter(prefix="/api/v1/agent", tags=["ai-agent"])


@router.get("/styles")
async def list_editing_styles(current_user: User = Depends(get_current_user)):
    from app.agent.styles import StyleManager
    sm = StyleManager()
    return {"styles": sm.list_styles()}


@router.post("/edit")
async def agent_edit_video(
    file: UploadFile = File(..., description="Video file to edit"),
    style: str = Form(default="viral"),
    caption_style: str = Form(default="highlight"),
    enable_silence_removal: bool = Form(default=True),
    silence_threshold_db: float = Form(default=-35.0),
    enable_zoom_effects: bool = Form(default=True),
    zoom_intensity: float = Form(default=0.5),
    enable_punch_in: bool = Form(default=True),
    punch_in_scale: float = Form(default=1.15),
    enable_keyword_highlight: bool = Form(default=True),
    enable_pacing_optimization: bool = Form(default=True),
    target_energy_curve: str = Form(default="hook_rising"),
    max_effects_per_minute: int = Form(default=6),
    project_id: uuid.UUID | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    try:
        EditingStyle(style)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style. Available: {[s.value for s in EditingStyle]}",
        )

    try:
        CaptionStyle(caption_style)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid caption style. Available: {[s.value for s in CaptionStyle]}",
        )

    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    if not file.filename or "." not in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(allowed_extensions)}",
        )

    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max: {settings.MAX_FILE_SIZE_MB}MB",
        )

    import os
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    job_id = uuid.uuid4()
    video_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(video_path, "wb") as f:
        f.write(file_bytes)

    from app.models.job import Job, JobStatus
    job = Job(
        id=job_id,
        owner_id=current_user.id,
        project_id=project_id,
        status=JobStatus.PENDING,
        original_filename=file.filename,
        video_path=video_path,
        clip_count=1,
        clip_length=0,
        caption_style=caption_style,
        use_vision=False,
    )
    db.add(job)
    await db.commit()

    agent_config = {
        "style": style,
        "caption_style": caption_style,
        "enable_silence_removal": enable_silence_removal,
        "silence_threshold_db": silence_threshold_db,
        "enable_zoom_effects": enable_zoom_effects,
        "zoom_intensity": zoom_intensity,
        "enable_punch_in": enable_punch_in,
        "punch_in_scale": punch_in_scale,
        "enable_keyword_highlight": enable_keyword_highlight,
        "enable_pacing_optimization": enable_pacing_optimization,
        "target_energy_curve": target_energy_curve,
        "max_effects_per_minute": max_effects_per_minute,
    }

    from app.tasks.agent_tasks import run_agent_pipeline
    run_agent_pipeline.delay(str(job_id), agent_config)

    return {
        "job_id": str(job_id),
        "status": "pending",
        "style": style,
        "message": "Agent editing pipeline started",
    }


@router.post("/analyze")
async def agent_analyze_only(
    file: UploadFile = File(..., description="Video file to analyze"),
    style: str = Form(default="viral"),
    target_energy_curve: str = Form(default="hook_rising"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    if not file.filename or "." not in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Unsupported format")

    file_bytes = await file.read()
    import os
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    job_id = uuid.uuid4()
    video_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}_{file.filename}")
    with open(video_path, "wb") as f:
        f.write(file_bytes)

    from app.models.job import Job, JobStatus
    job = Job(
        id=job_id,
        owner_id=current_user.id,
        status=JobStatus.PENDING,
        original_filename=file.filename,
        video_path=video_path,
        clip_count=1,
        clip_length=0,
    )
    db.add(job)
    await db.commit()

    agent_config = {
        "style": style,
        "target_energy_curve": target_energy_curve,
    }

    from app.tasks.agent_tasks import agent_analyze_only
    agent_analyze_only.delay(str(job_id), agent_config)

    return {
        "job_id": str(job_id),
        "status": "pending",
        "message": "Analysis started. Check job status for results.",
    }
