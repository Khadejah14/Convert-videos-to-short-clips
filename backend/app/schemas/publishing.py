import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.publishing import (
    Platform, AccountStatus, DraftStatus, ScheduleStatus, PublishStatus
)


# --- Connected Accounts ---

class ConnectedAccountResponse(BaseModel):
    id: uuid.UUID
    platform: Platform
    platform_user_id: str
    platform_username: str | None = None
    display_name: str | None = None
    avatar_url: str | None = None
    status: AccountStatus
    scopes: str | None = None
    created_at: datetime
    updated_at: datetime
    last_used_at: datetime | None = None

    class Config:
        from_attributes = True


class ConnectedAccountListResponse(BaseModel):
    accounts: list[ConnectedAccountResponse]
    total: int


class OAuthInitResponse(BaseModel):
    authorization_url: str
    state: str


class OAuthCallbackRequest(BaseModel):
    code: str
    state: str


# --- Export Presets ---

class ExportPresetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    platform: Platform
    resolution: str = Field(default="1080x1920")
    fps: int = Field(default=30, ge=15, le=60)
    bitrate: str = Field(default="8M")
    format: str = Field(default="mp4")
    codec: str = Field(default="h264")
    max_duration: int = Field(default=60, ge=5, le=600)
    caption_style: str = Field(default="default")
    watermark_enabled: bool = False
    watermark_position: str = Field(default="bottom-right")
    custom_settings: dict | None = None


class ExportPresetUpdate(BaseModel):
    name: str | None = None
    resolution: str | None = None
    fps: int | None = None
    bitrate: str | None = None
    format: str | None = None
    codec: str | None = None
    max_duration: int | None = None
    caption_style: str | None = None
    watermark_enabled: bool | None = None
    watermark_position: str | None = None
    custom_settings: dict | None = None


class ExportPresetResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID | None = None
    name: str
    platform: Platform
    is_default: bool
    is_system: bool
    resolution: str
    fps: int
    bitrate: str
    format: str
    codec: str
    max_duration: int
    caption_style: str
    watermark_enabled: bool
    watermark_position: str
    custom_settings: dict | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ExportPresetListResponse(BaseModel):
    presets: list[ExportPresetResponse]
    total: int


# --- Publish Drafts ---

class PublishDraftCreate(BaseModel):
    clip_id: uuid.UUID
    platform: Platform
    account_id: uuid.UUID | None = None
    preset_id: uuid.UUID | None = None
    title: str = Field(default="", max_length=500)
    description: str = Field(default="")
    tags: str | None = None
    visibility: str = Field(default="public")
    category: str | None = None
    language: str = Field(default="en")
    cover_frame_time: float | None = None
    publish_config: dict | None = None


class PublishDraftUpdate(BaseModel):
    account_id: uuid.UUID | None = None
    preset_id: uuid.UUID | None = None
    title: str | None = None
    description: str | None = None
    tags: str | None = None
    visibility: str | None = None
    category: str | None = None
    language: str | None = None
    cover_frame_time: float | None = None
    publish_config: dict | None = None
    status: DraftStatus | None = None


class PublishDraftResponse(BaseModel):
    id: uuid.UUID
    user_id: uuid.UUID
    clip_id: uuid.UUID
    account_id: uuid.UUID | None = None
    preset_id: uuid.UUID | None = None
    platform: Platform
    status: DraftStatus
    title: str
    description: str
    tags: str | None = None
    thumbnail_path: str | None = None
    cover_frame_time: float | None = None
    visibility: str
    category: str | None = None
    language: str
    publish_config: dict | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PublishDraftListResponse(BaseModel):
    drafts: list[PublishDraftResponse]
    total: int
    page: int
    per_page: int


# --- Publish Schedules ---

class PublishScheduleCreate(BaseModel):
    draft_id: uuid.UUID
    scheduled_at: datetime


class PublishScheduleResponse(BaseModel):
    id: uuid.UUID
    draft_id: uuid.UUID
    user_id: uuid.UUID
    scheduled_at: datetime
    status: ScheduleStatus
    retry_count: int
    error_message: str | None = None
    created_at: datetime
    executed_at: datetime | None = None

    class Config:
        from_attributes = True


class PublishScheduleListResponse(BaseModel):
    schedules: list[PublishScheduleResponse]
    total: int


# --- Publish History ---

class PublishHistoryResponse(BaseModel):
    id: uuid.UUID
    draft_id: uuid.UUID
    user_id: uuid.UUID
    account_id: uuid.UUID | None = None
    platform: Platform
    status: PublishStatus
    platform_post_id: str | None = None
    platform_post_url: str | None = None
    title: str
    description: str
    tags: str | None = None
    visibility: str
    upload_progress: float
    error_message: str | None = None
    retry_count: int
    published_at: datetime | None = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class PublishHistoryListResponse(BaseModel):
    history: list[PublishHistoryResponse]
    total: int
    page: int
    per_page: int


# --- Publish Now ---

class PublishNowRequest(BaseModel):
    draft_id: uuid.UUID


# --- Analytics ---

class AnalyticsSnapshotResponse(BaseModel):
    id: uuid.UUID
    publish_history_id: uuid.UUID
    platform: Platform
    views: int
    likes: int
    comments: int
    shares: int
    watch_time_seconds: float
    average_watch_time: float
    engagement_rate: float
    follower_count: int
    raw_metrics: dict | None = None
    snapshot_at: datetime

    class Config:
        from_attributes = True


class AnalyticsSummary(BaseModel):
    total_views: int = 0
    total_likes: int = 0
    total_comments: int = 0
    total_shares: int = 0
    total_watch_time: float = 0.0
    average_engagement_rate: float = 0.0
    posts_count: int = 0
    platform_breakdown: dict[str, int] = {}
