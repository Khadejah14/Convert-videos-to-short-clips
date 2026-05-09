import uuid
from datetime import datetime
from pydantic import BaseModel, Field
from app.models.job import JobStatus, ClipStatus


class JobCreate(BaseModel):
    clip_count: int = Field(default=3, ge=1, le=3, description="Number of clips to generate (1-3)")
    clip_length: int = Field(default=30, ge=15, le=60, description="Target clip length in seconds")
    caption_style: str = Field(default="default", description="Caption style preset")
    use_vision: bool = Field(default=False, description="Enable GPT-4o vision analysis")


class ClipResponse(BaseModel):
    id: uuid.UUID
    clip_number: int
    status: ClipStatus
    category: str
    start_time: float
    end_time: float
    duration: float
    original_url: str | None = None
    final_url: str | None = None
    text_score: float | None = None
    visual_score: float | None = None
    combined_score: float | None = None
    visual_hook: str | None = None
    rank: int | None = None
    no_captions: bool = False
    error_message: str | None = None

    class Config:
        from_attributes = True


class JobResponse(BaseModel):
    id: uuid.UUID
    status: JobStatus
    progress: float
    current_step: str
    original_filename: str
    clip_count: int
    clip_length: int
    caption_style: str
    use_vision: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class JobStatusResponse(JobResponse):
    transcript_text: str | None = None
    gpt_analysis: str | None = None
    error_message: str | None = None
    retry_count: int = 0
    completed_at: datetime | None = None
    clips: list[ClipResponse] = []

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int
    page: int
    per_page: int
