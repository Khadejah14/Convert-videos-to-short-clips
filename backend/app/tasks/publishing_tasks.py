import uuid
import logging
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.tasks.celery_app import celery_app
from app.core.database import engine
from app.models.publishing import (
    PublishHistory, PublishStatus,
    PublishDraft, DraftStatus,
    PublishSchedule, ScheduleStatus,
    ConnectedAccount, Platform,
    AnalyticsSnapshot,
)
from app.models.job import Clip

logger = logging.getLogger(__name__)


def _get_sync_session() -> Session:
    from sqlalchemy.orm import sessionmaker
    sync_engine = engine.sync_engine if hasattr(engine, 'sync_engine') else engine
    SessionLocal = sessionmaker(bind=sync_engine)
    return SessionLocal()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def publish_clip_task(self, history_id: str):
    """Publish a clip to a platform."""
    session = _get_sync_session()
    try:
        entry = session.get(PublishHistory, uuid.UUID(history_id))
        if not entry:
            logger.error(f"PublishHistory {history_id} not found")
            return

        draft = session.get(PublishDraft, entry.draft_id)
        if not draft:
            entry.status = PublishStatus.FAILED
            entry.error_message = "Draft not found"
            session.commit()
            return

        account = session.get(ConnectedAccount, entry.account_id)
        if not account:
            entry.status = PublishStatus.FAILED
            entry.error_message = "Connected account not found"
            session.commit()
            return

        clip = session.get(Clip, draft.clip_id)
        if not clip or not clip.final_path:
            entry.status = PublishStatus.FAILED
            entry.error_message = "Clip file not found"
            session.commit()
            return

        import asyncio
        from app.services.platform_adapters import get_adapter

        adapter = get_adapter(entry.platform)

        async def _publish():
            from app.services.publishing_service import PublishingService
            from app.core.database import async_session_factory

            async with async_session_factory() as db:
                svc = PublishingService(db)
                token = await svc.get_valid_token(account)

                tags = []
                if draft.tags:
                    tags = [t.strip() for t in draft.tags.split(",") if t.strip()]

                async def progress_cb(progress: float, step: str):
                    entry.upload_progress = progress
                    entry.status = PublishStatus.UPLOADING
                    session.commit()

                result = await adapter.upload_video(
                    access_token=token,
                    video_path=clip.final_path,
                    title=draft.title,
                    description=draft.description,
                    tags=tags,
                    visibility=draft.visibility,
                    progress_callback=progress_cb,
                )

                entry.platform_post_id = result.get("platform_post_id")
                entry.platform_post_url = result.get("platform_post_url")
                entry.status = PublishStatus.PUBLISHED
                entry.upload_progress = 100.0
                entry.published_at = datetime.utcnow()

                draft.status = DraftStatus.PUBLISHED
                session.commit()

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_publish())
        finally:
            loop.close()

    except Exception as e:
        logger.error(f"Publish failed for {history_id}: {e}")
        try:
            entry = session.get(PublishHistory, uuid.UUID(history_id))
            if entry:
                entry.status = PublishStatus.FAILED
                entry.error_message = str(e)[:1000]
                entry.retry_count += 1
                session.commit()

                draft = session.get(PublishDraft, entry.draft_id)
                if draft:
                    draft.status = DraftStatus.FAILED
                    session.commit()
        except Exception:
            pass

        raise self.retry(exc=e)
    finally:
        session.close()


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def execute_scheduled_publish(self, schedule_id: str):
    """Execute a scheduled publish."""
    session = _get_sync_session()
    try:
        schedule = session.get(PublishSchedule, uuid.UUID(schedule_id))
        if not schedule:
            logger.error(f"Schedule {schedule_id} not found")
            return

        if schedule.status != ScheduleStatus.PENDING:
            logger.info(f"Schedule {schedule_id} is not pending, skipping")
            return

        schedule.status = ScheduleStatus.PROCESSING
        session.commit()

        draft = session.get(PublishDraft, schedule.draft_id)
        if not draft:
            schedule.status = ScheduleStatus.FAILED
            schedule.error_message = "Draft not found"
            session.commit()
            return

        from app.services.publishing_service import PublishingService
        import asyncio
        from app.core.database import async_session_factory

        async def _create_history():
            async with async_session_factory() as db:
                svc = PublishingService(db)
                entry = await svc.create_history_entry(
                    schedule.user_id, draft, draft.account_id
                )
                return str(entry.id)

        loop = asyncio.new_event_loop()
        try:
            history_id = loop.run_until_complete(_create_history())
        finally:
            loop.close()

        publish_clip_task.delay(history_id)

        schedule.status = ScheduleStatus.COMPLETED
        schedule.executed_at = datetime.utcnow()
        session.commit()

    except Exception as e:
        logger.error(f"Scheduled publish failed for {schedule_id}: {e}")
        try:
            schedule = session.get(PublishSchedule, uuid.UUID(schedule_id))
            if schedule:
                schedule.status = ScheduleStatus.FAILED
                schedule.error_message = str(e)[:1000]
                schedule.retry_count += 1
                session.commit()
        except Exception:
            pass
        raise self.retry(exc=e)
    finally:
        session.close()


@celery_app.task(bind=True, max_retries=2)
def retry_publish_task(self, history_id: str):
    """Retry a failed publish."""
    publish_clip_task.delay(history_id)


@celery_app.task
def collect_analytics_task():
    """Periodic task to collect analytics for all published posts."""
    session = _get_sync_session()
    try:
        entries = session.execute(
            select(PublishHistory).where(
                PublishHistory.status == PublishStatus.PUBLISHED,
                PublishHistory.platform_post_id.isnot(None),
            )
        ).scalars().all()

        import asyncio
        from app.core.database import async_session_factory
        from app.services.analytics_service import AnalyticsService

        async def _collect():
            async with async_session_factory() as db:
                svc = AnalyticsService(db)
                for entry in entries:
                    try:
                        await svc.collect_snapshot(entry.id)
                    except Exception as e:
                        logger.warning(f"Analytics collection failed for {entry.id}: {e}")

        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(_collect())
        finally:
            loop.close()

    finally:
        session.close()
