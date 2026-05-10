import re
import uuid
import logging
import asyncio

from celery import group
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.tasks.celery_app import celery_app
from app.core.config import get_settings
from app.models.job import Job

settings = get_settings()

sync_db_url = settings.DATABASE_URL.replace("+asyncpg", "")
sync_engine = create_engine(sync_db_url, pool_size=5, max_overflow=10)
SyncSession = sessionmaker(bind=sync_engine)

logger = logging.getLogger(__name__)


def _run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _parse_transcript(text: str) -> list[dict]:
    segments = []
    for match in re.finditer(
        r"\[([\d.]+)s?\s*-\s*([\d.]+)s?\]\s*(.+)", text
    ):
        segments.append({
            "start": float(match.group(1)),
            "end": float(match.group(2)),
            "text": match.group(3).strip(),
        })
    return segments


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def generate_embeddings_task(self, job_id: str, user_id: str):
    db = SyncSession()
    try:
        job = db.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
        if not job or not job.transcript_text:
            logger.warning(f"Job {job_id} not found or has no transcript")
            return

        segments = _parse_transcript(job.transcript_text)
        if not segments:
            logger.warning(f"No segments parsed for job {job_id}")
            return

        from app.services.embedding_service import EmbeddingService
        emb_svc = EmbeddingService()

        _run_async(emb_svc.store_job_embeddings(
            db=db,
            job_id=uuid.UUID(job_id),
            segments=segments,
            owner_id=uuid.UUID(user_id),
        ))

        logger.info(f"Embedded {len(segments)} segments for job {job_id}")

    except Exception as e:
        logger.error(f"Embedding task failed for job {job_id}: {e}")
        raise self.retry(exc=e)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=2, default_retry_delay=60)
def build_clusters_task(self, user_id: str, n_clusters: int = 8):
    db = SyncSession()
    try:
        from app.services.embedding_service import EmbeddingService
        emb_svc = EmbeddingService()

        _run_async(emb_svc.compute_cluster_centroids(
            db=db,
            owner_id=uuid.UUID(user_id),
            n_clusters=n_clusters,
        ))

        logger.info(f"Built {n_clusters} clusters for user {user_id}")

    except Exception as e:
        logger.error(f"Cluster build failed for user {user_id}: {e}")
        raise self.retry(exc=e)
    finally:
        db.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=30)
def generate_clip_embeddings_task(self, job_id: str, user_id: str):
    """Generate embeddings for each clip individually (with clip_id linkage)."""
    db = SyncSession()
    try:
        job = db.query(Job).filter(Job.id == uuid.UUID(job_id)).first()
        if not job or not job.transcript_text:
            return

        from app.models.job import Clip
        clips = db.query(Clip).filter(Clip.job_id == job.id).all()

        if not clips:
            generate_embeddings_task.delay(job_id, user_id)
            return

        from app.services.embedding_service import EmbeddingService
        emb_svc = EmbeddingService()

        for clip in clips:
            if not clip.final_path:
                continue

            clip_segments = _parse_transcript_for_clip(
                job.transcript_text, clip.start_time, clip.end_time
            )
            if clip_segments:
                _run_async(emb_svc.store_job_embeddings(
                    db=db,
                    job_id=job.id,
                    segments=clip_segments,
                    owner_id=uuid.UUID(user_id),
                    clip_id=clip.id,
                ))

        logger.info(f"Embedded clips for job {job_id}")

    except Exception as e:
        logger.error(f"Clip embedding task failed: {e}")
        raise self.retry(exc=e)
    finally:
        db.close()


def _parse_transcript_for_clip(
    transcript: str, clip_start: float, clip_end: float
) -> list[dict]:
    all_segments = _parse_transcript(transcript)
    return [
        s for s in all_segments
        if s["start"] >= clip_start and s["end"] <= clip_end
    ]
