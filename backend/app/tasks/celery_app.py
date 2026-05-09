from celery import Celery
from app.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "video_processor",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=600,
    task_time_limit=900,
    worker_max_tasks_per_child=50,
    task_default_queue="default",
    task_routes={
        "app.tasks.video_tasks.process_video_pipeline": {"queue": "pipeline"},
        "app.tasks.video_tasks.extract_audio_task": {"queue": "pipeline"},
        "app.tasks.video_tasks.transcribe_audio_task": {"queue": "pipeline"},
        "app.tasks.video_tasks.analyze_clips_task": {"queue": "pipeline"},
        "app.tasks.video_tasks.extract_clips_task": {"queue": "pipeline"},
        "app.tasks.video_tasks.process_single_clip_task": {"queue": "clips"},
        "app.tasks.video_tasks.vision_analysis_task": {"queue": "clips"},
    },
)

celery_app.autodiscover_tasks(["app.tasks"])
