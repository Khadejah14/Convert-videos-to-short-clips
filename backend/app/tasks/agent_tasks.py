import os
import uuid
import traceback
import asyncio
from datetime import datetime

from celery import chain
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.tasks.celery_app import celery_app
from app.core.config import get_settings
from app.models.job import Job, Clip, JobStatus, ClipStatus

settings = get_settings()

sync_db_url = settings.DATABASE_URL.replace("+asyncpg", "")
sync_engine = create_engine(sync_db_url, pool_size=5, max_overflow=10)
SyncSession = sessionmaker(bind=sync_engine)


def get_sync_db() -> Session:
    return SyncSession()


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_agent_pipeline(self, job_id: str, agent_config: dict):
    db = get_sync_db()
    try:
        job = db.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
        if not job:
            return

        job.celery_task_id = self.request.id
        job.status = JobStatus.EXTRACTING_AUDIO
        job.current_step = "Extracting audio..."
        job.progress = 5.0
        db.commit()

        from app.tasks.video_tasks import extract_audio_sync, transcribe_sync
        from app.agent.orchestrator import EditingAgent
        from app.agent.schemas import AgentConfig, EditingStyle, CaptionStyle

        audio_path = extract_audio_sync(db, job)

        job.status = JobStatus.TRANSCRIBING
        job.current_step = "Transcribing with Whisper..."
        job.progress = 15.0
        db.commit()

        transcript_text, segments = transcribe_sync(db, job, audio_path)

        job.status = JobStatus.ANALYZING
        job.current_step = "AI agent analyzing content..."
        job.progress = 30.0
        db.commit()

        config = AgentConfig(
            style=EditingStyle(agent_config.get("style", "viral")),
            caption_style=CaptionStyle(agent_config.get("caption_style", "highlight")),
            enable_silence_removal=agent_config.get("enable_silence_removal", True),
            silence_threshold_db=agent_config.get("silence_threshold_db", -35.0),
            min_silence_duration=agent_config.get("min_silence_duration", 0.3),
            silence_padding_ms=agent_config.get("silence_padding_ms", 100),
            enable_zoom_effects=agent_config.get("enable_zoom_effects", True),
            zoom_intensity=agent_config.get("zoom_intensity", 0.5),
            enable_punch_in=agent_config.get("enable_punch_in", True),
            punch_in_scale=agent_config.get("punch_in_scale", 1.15),
            enable_keyword_highlight=agent_config.get("enable_keyword_highlight", True),
            enable_pacing_optimization=agent_config.get("enable_pacing_optimization", True),
            target_energy_curve=agent_config.get("target_energy_curve", "hook_rising"),
            max_effects_per_minute=agent_config.get("max_effects_per_minute", 6),
            plugins=agent_config.get("plugins", []),
        )

        agent = EditingAgent(
            api_key=settings.OPENAI_API_KEY,
            model=agent_config.get("model", "gpt-4o"),
            plugins_dir=agent_config.get("plugins_dir"),
            custom_styles_dir=agent_config.get("custom_styles_dir"),
        )

        output_dir = os.path.join(settings.OUTPUT_DIR, str(job_id))
        os.makedirs(output_dir, exist_ok=True)

        job.status = JobStatus.PROCESSING_CLIPS
        job.current_step = "Agent editing video..."
        job.progress = 40.0
        db.commit()

        result = _run_async(agent.edit(
            video_path=job.video_path,
            transcript_segments=segments,
            full_transcript=transcript_text,
            config=config,
            output_dir=output_dir,
            job_id=uuid.UUID(job_id),
        ))

        clip = Clip(
            id=uuid.uuid4(),
            job_id=job.id,
            clip_number=1,
            status=ClipStatus.COMPLETED,
            category=f"agent_{config.style.value}",
            start_time=0,
            end_time=result.duration_after,
            duration=result.duration_after,
            final_path=result.output_path,
            text_score=result.plan.pacing_optimizer.compute_retention_score(result.plan) if result.plan.beats else None,
        )
        db.add(clip)

        job.status = JobStatus.COMPLETED
        job.current_step = "Complete"
        job.progress = 100.0
        job.completed_at = datetime.utcnow()
        db.commit()

    except Exception as e:
        job = db.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
        if job:
            job.status = JobStatus.FAILED
            job.error_message = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()[:1000]}"
            job.current_step = "Failed"
            db.commit()
        raise
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=1)
def agent_analyze_only(self, job_id: str, agent_config: dict):
    db = get_sync_db()
    try:
        job = db.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
        if not job:
            return

        from app.agent.orchestrator import EditingAgent
        from app.agent.schemas import AgentConfig, EditingStyle, CaptionStyle

        config = AgentConfig(
            style=EditingStyle(agent_config.get("style", "viral")),
            caption_style=CaptionStyle(agent_config.get("caption_style", "highlight")),
            target_energy_curve=agent_config.get("target_energy_curve", "hook_rising"),
        )

        agent = EditingAgent(api_key=settings.OPENAI_API_KEY)

        segments = []
        if job.transcript_text:
            import re
            for match in re.finditer(
                r"\[([\d.]+)s?\s*-\s*([\d.]+)s?\]\s*(.+)",
                job.transcript_text,
            ):
                segments.append({
                    "start": float(match.group(1)),
                    "end": float(match.group(2)),
                    "text": match.group(3).strip(),
                })

        plan = _run_async(agent.analyze_only(
            transcript_segments=segments,
            full_transcript=job.transcript_text or "",
            config=config,
        ))

        job.gpt_analysis = plan.model_dump_json(indent=2)
        db.commit()

    except Exception as e:
        job = db.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
        if job:
            job.error_message = str(e)
            db.commit()
    finally:
        db.close()
