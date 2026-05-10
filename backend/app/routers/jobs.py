import uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import get_settings
from app.core.deps import get_current_user
from app.models.user import User
from app.schemas.job import JobCreate, JobResponse, JobStatusResponse, JobListResponse
from app.services.job_service import JobService

settings = get_settings()
router = APIRouter(prefix="/api/v1/jobs", tags=["jobs"])


@router.post("", response_model=JobResponse, status_code=201)
async def create_job(
    file: UploadFile = File(..., description="Video file to process"),
    clip_count: int = Form(default=3, ge=1, le=3),
    clip_length: int = Form(default=30, ge=15, le=60),
    caption_style: str = Form(default="default"),
    use_vision: bool = Form(default=False),
    project_id: uuid.UUID | None = Form(default=None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
    if not file.filename or "." not in file.filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_ext = "." + file.filename.rsplit(".", 1)[-1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}",
        )

    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB}MB",
        )

    config = JobCreate(
        clip_count=clip_count,
        clip_length=clip_length,
        caption_style=caption_style,
        use_vision=use_vision,
    )

    service = JobService(db)
    job = await service.create_job(
        file_bytes, file.filename, config,
        owner_id=current_user.id, project_id=project_id,
    )

    return job


@router.get("", response_model=JobListResponse)
async def list_jobs(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = JobService(db)
    jobs, total = await service.list_jobs(page, per_page, owner_id=current_user.id)

    return JobListResponse(jobs=jobs, total=total, page=page, per_page=per_page)


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = JobService(db)
    job = await service.get_job(job_id, owner_id=current_user.id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job


@router.post("/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = JobService(db)
    success = await service.cancel_job(job_id, owner_id=current_user.id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job cannot be cancelled (not found or already completed/failed)",
        )

    job = await service.get_job(job_id, owner_id=current_user.id)
    return job


@router.delete("/{job_id}", status_code=204)
async def delete_job(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = JobService(db)
    success = await service.delete_job(job_id, owner_id=current_user.id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job cannot be deleted (not found or still processing)",
        )


@router.get("/{job_id}/clips/{clip_id}/original")
async def download_original_clip(
    job_id: uuid.UUID,
    clip_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = JobService(db)
    file_path = await service.get_clip_file_path(job_id, clip_id, "original", owner_id=current_user.id)

    if not file_path:
        raise HTTPException(status_code=404, detail="Clip file not found")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=f"clip_{clip_id}_original.mp4",
    )


@router.get("/{job_id}/clips/{clip_id}/final")
async def download_final_clip(
    job_id: uuid.UUID,
    clip_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = JobService(db)
    file_path = await service.get_clip_file_path(job_id, clip_id, "final", owner_id=current_user.id)

    if not file_path:
        raise HTTPException(status_code=404, detail="Clip file not found")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=f"clip_{clip_id}_final.mp4",
    )


@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": settings.APP_NAME}
