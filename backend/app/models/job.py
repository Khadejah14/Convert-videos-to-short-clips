import enum
import uuid
from datetime import datetime
from sqlalchemy import String, Integer, Float, Enum, DateTime, ForeignKey, Text, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    EXTRACTING_AUDIO = "extracting_audio"
    TRANSCRIBING = "transcribing"
    ANALYZING = "analyzing"
    EXTRACTING_CLIPS = "extracting_clips"
    PROCESSING_CLIPS = "processing_clips"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ClipStatus(str, enum.Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    CROPPING = "cropping"
    CAPTIONING = "captioning"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True
    )
    project_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("projects.id", ondelete="SET NULL"), index=True, nullable=True
    )
    status: Mapped[JobStatus] = mapped_column(
        Enum(JobStatus), default=JobStatus.PENDING, index=True
    )
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    current_step: Mapped[str] = mapped_column(String(255), default="pending")

    # Input config
    original_filename: Mapped[str] = mapped_column(String(500))
    video_path: Mapped[str] = mapped_column(String(1000))
    clip_count: Mapped[int] = mapped_column(Integer, default=3)
    clip_length: Mapped[int] = mapped_column(Integer, default=30)
    caption_style: Mapped[str] = mapped_column(String(50), default="default")
    use_vision: Mapped[bool] = mapped_column(Boolean, default=False)

    # Processing results
    audio_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    transcript_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    gpt_analysis: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Retry tracking
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, index=True
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # Relationships
    clips: Mapped[list["Clip"]] = relationship(
        back_populates="job", cascade="all, delete-orphan"
    )
    project: Mapped["Project | None"] = relationship(back_populates="jobs")
    owner: Mapped["User | None"] = relationship(foreign_keys=[owner_id])


class Clip(Base):
    __tablename__ = "clips"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    clip_number: Mapped[int] = mapped_column(Integer)
    status: Mapped[ClipStatus] = mapped_column(
        Enum(ClipStatus), default=ClipStatus.PENDING
    )
    category: Mapped[str] = mapped_column(String(100))

    # Timestamps from analysis
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    duration: Mapped[float] = mapped_column(Float)

    # File paths
    original_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    cropped_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    final_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    # Vision analysis
    text_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    visual_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    combined_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    visual_hook: Mapped[str | None] = mapped_column(Text, nullable=True)
    rank: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Error tracking
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    no_captions: Mapped[bool] = mapped_column(Boolean, default=False)

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    # Relationships
    job: Mapped["Job"] = relationship(back_populates="clips")
