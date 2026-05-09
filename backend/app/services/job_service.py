import os
import uuid
import shutil
from datetime import datetime
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.models.job import Job, Clip, JobStatus, ClipStatus
from app.schemas.job import JobCreate, JobResponse, JobStatusResponse, ClipResponse
from app.tasks.video_tasks import process_video_pipeline

settings = get_settings()


class JobService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_job(
        self, file_bytes: bytes, filename: str, config: JobCreate
    ) -> JobResponse:
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(settings.OUTPUT_DIR, exist_ok=True)
        os.makedirs(settings.TEMP_DIR, exist_ok=True)

        job_id = uuid.uuid4()
        video_path = os.path.join(settings.UPLOAD_DIR, f"{job_id}_{filename}")

        with open(video_path, "wb") as f:
            f.write(file_bytes)

        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            original_filename=filename,
            video_path=video_path,
            clip_count=config.clip_count,
            clip_length=config.clip_length,
            caption_style=config.caption_style,
            use_vision=config.use_vision,
        )

        self.db.add(job)
        await self.db.commit()
        await self.db.refresh(job)

        process_video_pipeline.delay(str(job_id))

        return JobResponse.model_validate(job)

    async def get_job(self, job_id: uuid.UUID) -> JobStatusResponse | None:
        result = await self.db.execute(
            select(Job)
            .where(Job.id == job_id)
            .options(selectinload(Job.clips))
        )
        job = result.scalar_one_or_none()

        if not job:
            return None

        response = JobStatusResponse.model_validate(job)

        response.clips = []
        for clip in job.clips:
            clip_resp = ClipResponse(
                id=clip.id,
                clip_number=clip.clip_number,
                status=clip.status,
                category=clip.category,
                start_time=clip.start_time,
                end_time=clip.end_time,
                duration=clip.duration,
                text_score=clip.text_score,
                visual_score=clip.visual_score,
                combined_score=clip.combined_score,
                visual_hook=clip.visual_hook,
                rank=clip.rank,
                no_captions=clip.no_captions,
                error_message=clip.error_message,
            )

            if clip.original_path and os.path.exists(clip.original_path):
                clip_resp.original_url = f"/api/v1/jobs/{job_id}/clips/{clip.id}/original"
            if clip.final_path and os.path.exists(clip.final_path):
                clip_resp.final_url = f"/api/v1/jobs/{job_id}/clips/{clip.id}/final"

            response.clips.append(clip_resp)

        return response

    async def list_jobs(
        self, page: int = 1, per_page: int = 20
    ) -> tuple[list[JobResponse], int]:
        count_result = await self.db.execute(select(func.count(Job.id)))
        total = count_result.scalar()

        result = await self.db.execute(
            select(Job)
            .order_by(Job.created_at.desc())
            .offset((page - 1) * per_page)
            .limit(per_page)
        )
        jobs = result.scalars().all()

        return [JobResponse.model_validate(job) for job in jobs], total

    async def cancel_job(self, job_id: uuid.UUID) -> bool:
        result = await self.db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()

        if not job:
            return False

        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False

        if job.celery_task_id:
            from app.tasks.celery_app import celery_app

            try:
                celery_app.control.revoke(job.celery_task_id, terminate=True)
            except Exception:
                pass

        job.status = JobStatus.CANCELLED
        job.current_step = "Cancelled by user"
        job.updated_at = datetime.utcnow()
        await self.db.commit()

        return True

    async def delete_job(self, job_id: uuid.UUID) -> bool:
        result = await self.db.execute(
            select(Job).where(Job.id == job_id).options(selectinload(Job.clips))
        )
        job = result.scalar_one_or_none()

        if not job:
            return False

        if job.status not in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
            return False

        if job.video_path and os.path.exists(job.video_path):
            os.remove(job.video_path)
        if job.audio_path and os.path.exists(job.audio_path):
            os.remove(job.audio_path)

        for clip in job.clips:
            for path_attr in ["original_path", "cropped_path", "final_path"]:
                path = getattr(clip, path_attr, None)
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception:
                        pass

        await self.db.delete(job)
        await self.db.commit()

        return True

    async def get_clip_file_path(
        self, job_id: uuid.UUID, clip_id: uuid.UUID, file_type: str
    ) -> str | None:
        result = await self.db.execute(
            select(Clip).where(Clip.id == clip_id, Clip.job_id == job_id)
        )
        clip = result.scalar_one_or_none()

        if not clip:
            return None

        if file_type == "original":
            path = clip.original_path
        elif file_type == "final":
            path = clip.final_path
        else:
            return None

        if path and os.path.exists(path):
            return path

        return None
