import enum
import uuid
from datetime import datetime
from sqlalchemy import (
    String, Integer, Float, Enum, DateTime, ForeignKey, Text, Boolean, JSON
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from app.core.database import Base


class Platform(str, enum.Enum):
    TIKTOK = "tiktok"
    YOUTUBE = "youtube"
    INSTAGRAM = "instagram"


class AccountStatus(str, enum.Enum):
    ACTIVE = "active"
    EXPIRED = "expired"
    REVOKED = "revoked"
    ERROR = "error"


class DraftStatus(str, enum.Enum):
    DRAFT = "draft"
    READY = "ready"
    PUBLISHING = "publishing"
    PUBLISHED = "published"
    FAILED = "failed"


class ScheduleStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PublishStatus(str, enum.Enum):
    UPLOADING = "uploading"
    PROCESSING = "processing"
    PUBLISHED = "published"
    FAILED = "failed"


class ConnectedAccount(Base):
    __tablename__ = "connected_accounts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    platform: Mapped[Platform] = mapped_column(Enum(Platform), nullable=False)
    platform_user_id: Mapped[str] = mapped_column(String(255), nullable=False)
    platform_username: Mapped[str | None] = mapped_column(String(255), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    avatar_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    access_token: Mapped[str] = mapped_column(Text, nullable=False)
    refresh_token: Mapped[str | None] = mapped_column(Text, nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    scopes: Mapped[str | None] = mapped_column(Text, nullable=True)

    status: Mapped[AccountStatus] = mapped_column(
        Enum(AccountStatus), default=AccountStatus.ACTIVE
    )
    platform_metadata: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    user: Mapped["User"] = relationship(foreign_keys=[user_id])
    drafts: Mapped[list["PublishDraft"]] = relationship(
        back_populates="account", cascade="all, delete-orphan"
    )


class ExportPreset(Base):
    __tablename__ = "export_presets"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    platform: Mapped[Platform] = mapped_column(Enum(Platform), nullable=False)
    is_default: Mapped[bool] = mapped_column(Boolean, default=False)
    is_system: Mapped[bool] = mapped_column(Boolean, default=False)

    resolution: Mapped[str] = mapped_column(String(20), default="1080x1920")
    fps: Mapped[int] = mapped_column(Integer, default=30)
    bitrate: Mapped[str] = mapped_column(String(20), default="8M")
    format: Mapped[str] = mapped_column(String(20), default="mp4")
    codec: Mapped[str] = mapped_column(String(50), default="h264")
    max_duration: Mapped[int] = mapped_column(Integer, default=60)
    caption_style: Mapped[str] = mapped_column(String(50), default="default")
    watermark_enabled: Mapped[bool] = mapped_column(Boolean, default=False)
    watermark_position: Mapped[str] = mapped_column(String(20), default="bottom-right")
    custom_settings: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )


class PublishDraft(Base):
    __tablename__ = "publish_drafts"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    clip_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clips.id", ondelete="CASCADE"), index=True
    )
    account_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("connected_accounts.id", ondelete="SET NULL"), nullable=True
    )
    preset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("export_presets.id", ondelete="SET NULL"), nullable=True
    )

    platform: Mapped[Platform] = mapped_column(Enum(Platform), nullable=False)
    status: Mapped[DraftStatus] = mapped_column(
        Enum(DraftStatus), default=DraftStatus.DRAFT
    )

    title: Mapped[str] = mapped_column(String(500), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    thumbnail_path: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    cover_frame_time: Mapped[float | None] = mapped_column(Float, nullable=True)

    visibility: Mapped[str] = mapped_column(String(20), default="public")
    category: Mapped[str | None] = mapped_column(String(100), nullable=True)
    language: Mapped[str] = mapped_column(String(10), default="en")

    publish_config: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    user: Mapped["User"] = relationship(foreign_keys=[user_id])
    clip: Mapped["Clip"] = relationship(foreign_keys=[clip_id])
    account: Mapped["ConnectedAccount | None"] = relationship(back_populates="drafts")
    schedules: Mapped[list["PublishSchedule"]] = relationship(
        back_populates="draft", cascade="all, delete-orphan"
    )
    history: Mapped[list["PublishHistory"]] = relationship(
        back_populates="draft", cascade="all, delete-orphan"
    )


class PublishSchedule(Base):
    __tablename__ = "publish_schedules"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    draft_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("publish_drafts.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )

    scheduled_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    status: Mapped[ScheduleStatus] = mapped_column(
        Enum(ScheduleStatus), default=ScheduleStatus.PENDING
    )
    celery_task_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    executed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    draft: Mapped["PublishDraft"] = relationship(back_populates="schedules")
    user: Mapped["User"] = relationship(foreign_keys=[user_id])


class PublishHistory(Base):
    __tablename__ = "publish_history"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    draft_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("publish_drafts.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    account_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("connected_accounts.id", ondelete="SET NULL"), nullable=True
    )

    platform: Mapped[Platform] = mapped_column(Enum(Platform), nullable=False, index=True)
    status: Mapped[PublishStatus] = mapped_column(Enum(PublishStatus), default=PublishStatus.UPLOADING)

    platform_post_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    platform_post_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)

    title: Mapped[str] = mapped_column(String(500), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    tags: Mapped[str | None] = mapped_column(Text, nullable=True)
    visibility: Mapped[str] = mapped_column(String(20), default="public")

    upload_progress: Mapped[float] = mapped_column(Float, default=0.0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)

    published_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    draft: Mapped["PublishDraft"] = relationship(back_populates="history")
    user: Mapped["User"] = relationship(foreign_keys=[user_id])
    account: Mapped["ConnectedAccount | None"] = relationship(foreign_keys=[account_id])
    analytics: Mapped[list["AnalyticsSnapshot"]] = relationship(
        back_populates="publish_entry", cascade="all, delete-orphan"
    )


class AnalyticsSnapshot(Base):
    __tablename__ = "analytics_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    publish_history_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("publish_history.id", ondelete="CASCADE"), index=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    platform: Mapped[Platform] = mapped_column(Enum(Platform), nullable=False, index=True)

    views: Mapped[int] = mapped_column(Integer, default=0)
    likes: Mapped[int] = mapped_column(Integer, default=0)
    comments: Mapped[int] = mapped_column(Integer, default=0)
    shares: Mapped[int] = mapped_column(Integer, default=0)
    watch_time_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    average_watch_time: Mapped[float] = mapped_column(Float, default=0.0)
    engagement_rate: Mapped[float] = mapped_column(Float, default=0.0)
    follower_count: Mapped[int] = mapped_column(Integer, default=0)

    raw_metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    snapshot_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    publish_entry: Mapped["PublishHistory"] = relationship(back_populates="analytics")
    user: Mapped["User"] = relationship(foreign_keys=[user_id])
