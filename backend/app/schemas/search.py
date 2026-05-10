import uuid
from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    job_id: uuid.UUID | None = None
    project_id: uuid.UUID | None = None
    limit: int = Field(default=20, ge=1, le=100)
    threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    is_hook: bool | None = None
    is_emotional_peak: bool | None = None
    emotion: str | None = None
    min_energy: float | None = Field(default=None, ge=0.0, le=1.0)
    topic_cluster_id: int | None = None


class SimilarRequest(BaseModel):
    segment_id: uuid.UUID
    limit: int = Field(default=10, ge=1, le=50)
    same_job: bool = False


class SegmentResult(BaseModel):
    id: str
    job_id: str
    clip_id: str | None = None
    segment_index: int | None = None
    start_time: float
    end_time: float
    duration: float
    text: str
    emotion: str | None = None
    energy: float | None = None
    is_hook: bool = False
    is_emotional_peak: bool = False
    keywords: dict | list | None = None
    topic: str | None = None
    topic_cluster_id: int | None = None
    similarity: float | None = None
    preview_url: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SegmentResult]
    total: int


class TopicClusterResponse(BaseModel):
    id: int
    label: str
    description: str | None = None
    size: int
    avg_energy: float | None = None
    dominant_emotion: str | None = None
    created_at: str


class ClusterListResponse(BaseModel):
    clusters: list[TopicClusterResponse]
    total: int


class EmbeddingStatsResponse(BaseModel):
    total_segments: int
    total_hooks: int
    total_emotional_peaks: int
    total_jobs_indexed: int
    total_clusters: int


class EmbedJobRequest(BaseModel):
    job_id: uuid.UUID


class EmbedJobResponse(BaseModel):
    job_id: str
    segments_indexed: int
    status: str
