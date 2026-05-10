import uuid
from datetime import datetime
from sqlalchemy import String, Float, DateTime, ForeignKey, Text, Integer, Index, JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID
from pgvector.sqlalchemy import Vector

from app.core.database import Base
from app.core.config import get_settings

settings = get_settings()


class TranscriptEmbedding(Base):
    __tablename__ = "transcript_embeddings"
    __table_args__ = (
        Index("ix_emb_owner", "owner_id"),
        Index("ix_emb_job", "job_id"),
        Index("ix_emb_clip", "clip_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("jobs.id", ondelete="CASCADE"), index=True
    )
    clip_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("clips.id", ondelete="SET NULL"), nullable=True
    )

    # Segment info
    segment_index: Mapped[int] = mapped_column(Integer)
    start_time: Mapped[float] = mapped_column(Float)
    end_time: Mapped[float] = mapped_column(Float)
    text: Mapped[str] = mapped_column(Text)

    # Metadata from analysis
    emotion: Mapped[str | None] = mapped_column(String(50), nullable=True)
    energy: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_hook: Mapped[bool] = mapped_column(default=False)
    is_emotional_peak: Mapped[bool] = mapped_column(default=False)
    keywords: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    topic: Mapped[str | None] = mapped_column(String(200), nullable=True)
    topic_cluster_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # pgvector embedding
    embedding = mapped_column(Vector(settings.EMBEDDING_DIMENSIONS))

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class TopicCluster(Base):
    __tablename__ = "topic_clusters"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )
    label: Mapped[str] = mapped_column(String(200))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    centroid = mapped_column(Vector(settings.EMBEDDING_DIMENSIONS))
    size: Mapped[int] = mapped_column(Integer, default=0)
    avg_energy: Mapped[float | None] = mapped_column(Float, nullable=True)
    dominant_emotion: Mapped[str | None] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
