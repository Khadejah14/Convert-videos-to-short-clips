from __future__ import annotations

import logging
import uuid
from typing import Any

from sqlalchemy import select, text, func, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.embedding import TranscriptEmbedding, TopicCluster
from app.models.job import Job, Clip
from app.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)
settings = get_settings()


class SearchService:
    def __init__(self, db: AsyncSession, embedding_service: EmbeddingService | None = None):
        self.db = db
        self.emb = embedding_service or EmbeddingService()

    async def semantic_search(
        self,
        query: str,
        owner_id: uuid.UUID | None = None,
        job_id: uuid.UUID | None = None,
        project_id: uuid.UUID | None = None,
        limit: int = 20,
        threshold: float = 0.3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query_embedding = await self.emb.embed_single(query)

        stmt = select(
            TranscriptEmbedding.id,
            TranscriptEmbedding.job_id,
            TranscriptEmbedding.clip_id,
            TranscriptEmbedding.segment_index,
            TranscriptEmbedding.start_time,
            TranscriptEmbedding.end_time,
            TranscriptEmbedding.text,
            TranscriptEmbedding.emotion,
            TranscriptEmbedding.energy,
            TranscriptEmbedding.is_hook,
            TranscriptEmbedding.is_emotional_peak,
            TranscriptEmbedding.keywords,
            TranscriptEmbedding.topic,
            TranscriptEmbedding.topic_cluster_id,
            TranscriptEmbedding.embedding.cosine_distance(query_embedding).label("distance"),
        )

        conditions = []
        if owner_id:
            conditions.append(TranscriptEmbedding.owner_id == owner_id)
        if job_id:
            conditions.append(TranscriptEmbedding.job_id == job_id)
        if project_id:
            conditions.append(
                TranscriptEmbedding.job_id.in_(
                    select(Job.id).where(Job.project_id == project_id)
                )
            )

        if filters:
            if filters.get("is_hook"):
                conditions.append(TranscriptEmbedding.is_hook == True)
            if filters.get("is_emotional_peak"):
                conditions.append(TranscriptEmbedding.is_emotional_peak == True)
            if filters.get("emotion"):
                conditions.append(TranscriptEmbedding.emotion == filters["emotion"])
            if filters.get("min_energy"):
                conditions.append(TranscriptEmbedding.energy >= filters["min_energy"])
            if filters.get("topic_cluster_id") is not None:
                conditions.append(
                    TranscriptEmbedding.topic_cluster_id == filters["topic_cluster_id"]
                )

        if conditions:
            stmt = stmt.where(and_(*conditions))

        stmt = stmt.order_by("distance").limit(limit)

        result = await self.db.execute(stmt)
        rows = result.all()

        results = []
        for row in rows:
            similarity = 1.0 - row.distance
            if similarity < threshold:
                continue

            results.append({
                "id": str(row.id),
                "job_id": str(row.job_id),
                "clip_id": str(row.clip_id) if row.clip_id else None,
                "segment_index": row.segment_index,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "duration": row.end_time - row.start_time,
                "text": row.text,
                "emotion": row.emotion,
                "energy": row.energy,
                "is_hook": row.is_hook,
                "is_emotional_peak": row.is_emotional_peak,
                "keywords": row.keywords,
                "topic": row.topic,
                "topic_cluster_id": row.topic_cluster_id,
                "similarity": round(similarity, 4),
                "preview_url": self._build_preview_url(row.job_id, row.clip_id),
            })

        return results

    async def find_similar_segments(
        self,
        segment_id: uuid.UUID,
        owner_id: uuid.UUID | None = None,
        limit: int = 10,
        same_job: bool = False,
    ) -> list[dict[str, Any]]:
        result = await self.db.execute(
            select(TranscriptEmbedding).where(TranscriptEmbedding.id == segment_id)
        )
        source = result.scalar_one_or_none()
        if not source:
            return []

        stmt = (
            select(
                TranscriptEmbedding.id,
                TranscriptEmbedding.job_id,
                TranscriptEmbedding.clip_id,
                TranscriptEmbedding.segment_index,
                TranscriptEmbedding.start_time,
                TranscriptEmbedding.end_time,
                TranscriptEmbedding.text,
                TranscriptEmbedding.emotion,
                TranscriptEmbedding.energy,
                TranscriptEmbedding.is_hook,
                TranscriptEmbedding.is_emotional_peak,
                TranscriptEmbedding.topic,
                TranscriptEmbedding.embedding.cosine_distance(source.embedding).label("distance"),
            )
            .where(TranscriptEmbedding.id != segment_id)
            .order_by("distance")
            .limit(limit + 10)
        )

        if owner_id:
            stmt = stmt.where(TranscriptEmbedding.owner_id == owner_id)
        if same_job:
            stmt = stmt.where(TranscriptEmbedding.job_id == source.job_id)

        rows = (await self.db.execute(stmt)).all()

        results = []
        for row in rows:
            similarity = 1.0 - row.distance
            if similarity < 0.2:
                continue
            if len(results) >= limit:
                break

            results.append({
                "id": str(row.id),
                "job_id": str(row.job_id),
                "clip_id": str(row.clip_id) if row.clip_id else None,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "duration": row.end_time - row.start_time,
                "text": row.text,
                "emotion": row.emotion,
                "energy": row.energy,
                "is_hook": row.is_hook,
                "is_emotional_peak": row.is_emotional_peak,
                "topic": row.topic,
                "similarity": round(similarity, 4),
                "preview_url": self._build_preview_url(row.job_id, row.clip_id),
            })

        return results

    async def get_topic_clusters(
        self,
        owner_id: uuid.UUID | None = None,
    ) -> list[dict[str, Any]]:
        stmt = select(TopicCluster).order_by(TopicCluster.size.desc())
        if owner_id:
            stmt = stmt.where(
                or_(
                    TopicCluster.owner_id == owner_id,
                    TopicCluster.owner_id.is_(None),
                )
            )

        result = await self.db.execute(stmt)
        clusters = result.scalars().all()

        return [
            {
                "id": c.id,
                "label": c.label,
                "description": c.description,
                "size": c.size,
                "avg_energy": round(c.avg_energy, 3) if c.avg_energy else None,
                "dominant_emotion": c.dominant_emotion,
                "created_at": c.created_at.isoformat(),
            }
            for c in clusters
        ]

    async def get_cluster_segments(
        self,
        cluster_id: int,
        owner_id: uuid.UUID | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(TranscriptEmbedding)
            .where(TranscriptEmbedding.topic_cluster_id == cluster_id)
            .order_by(TranscriptEmbedding.energy.desc())
            .limit(limit)
        )
        if owner_id:
            stmt = stmt.where(TranscriptEmbedding.owner_id == owner_id)

        result = await self.db.execute(stmt)
        records = result.scalars().all()

        return [
            {
                "id": str(r.id),
                "job_id": str(r.job_id),
                "start_time": r.start_time,
                "end_time": r.end_time,
                "text": r.text,
                "emotion": r.emotion,
                "energy": r.energy,
                "is_hook": r.is_hook,
                "topic": r.topic,
            }
            for r in records
        ]

    async def get_hooks(
        self,
        owner_id: uuid.UUID | None = None,
        job_id: uuid.UUID | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(TranscriptEmbedding)
            .where(TranscriptEmbedding.is_hook == True)
            .order_by(TranscriptEmbedding.energy.desc())
            .limit(limit)
        )
        if owner_id:
            stmt = stmt.where(TranscriptEmbedding.owner_id == owner_id)
        if job_id:
            stmt = stmt.where(TranscriptEmbedding.job_id == job_id)

        result = await self.db.execute(stmt)
        records = result.scalars().all()

        return [
            {
                "id": str(r.id),
                "job_id": str(r.job_id),
                "start_time": r.start_time,
                "end_time": r.end_time,
                "text": r.text,
                "emotion": r.emotion,
                "energy": r.energy,
                "preview_url": self._build_preview_url(r.job_id, r.clip_id),
            }
            for r in records
        ]

    async def get_emotional_peaks(
        self,
        owner_id: uuid.UUID | None = None,
        job_id: uuid.UUID | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        stmt = (
            select(TranscriptEmbedding)
            .where(TranscriptEmbedding.is_emotional_peak == True)
            .order_by(TranscriptEmbedding.energy.desc())
            .limit(limit)
        )
        if owner_id:
            stmt = stmt.where(TranscriptEmbedding.owner_id == owner_id)
        if job_id:
            stmt = stmt.where(TranscriptEmbedding.job_id == job_id)

        result = await self.db.execute(stmt)
        records = result.scalars().all()

        return [
            {
                "id": str(r.id),
                "job_id": str(r.job_id),
                "start_time": r.start_time,
                "end_time": r.end_time,
                "text": r.text,
                "emotion": r.emotion,
                "energy": r.energy,
                "preview_url": self._build_preview_url(r.job_id, r.clip_id),
            }
            for r in records
        ]

    async def get_embedding_stats(
        self,
        owner_id: uuid.UUID | None = None,
    ) -> dict[str, Any]:
        base = select(func.count(TranscriptEmbedding.id))
        if owner_id:
            base = base.where(TranscriptEmbedding.owner_id == owner_id)
        total = (await self.db.execute(base)).scalar() or 0

        hook_q = select(func.count(TranscriptEmbedding.id)).where(
            TranscriptEmbedding.is_hook == True
        )
        if owner_id:
            hook_q = hook_q.where(TranscriptEmbedding.owner_id == owner_id)
        hooks = (await self.db.execute(hook_q)).scalar() or 0

        peak_q = select(func.count(TranscriptEmbedding.id)).where(
            TranscriptEmbedding.is_emotional_peak == True
        )
        if owner_id:
            peak_q = peak_q.where(TranscriptEmbedding.owner_id == owner_id)
        peaks = (await self.db.execute(peak_q)).scalar() or 0

        jobs_q = select(func.count(func.distinct(TranscriptEmbedding.job_id)))
        if owner_id:
            jobs_q = jobs_q.where(TranscriptEmbedding.owner_id == owner_id)
        jobs = (await self.db.execute(jobs_q)).scalar() or 0

        cluster_q = select(func.count(TopicCluster.id))
        if owner_id:
            cluster_q = cluster_q.where(
                or_(TopicCluster.owner_id == owner_id, TopicCluster.owner_id.is_(None))
            )
        clusters = (await self.db.execute(cluster_q)).scalar() or 0

        return {
            "total_segments": total,
            "total_hooks": hooks,
            "total_emotional_peaks": peaks,
            "total_jobs_indexed": jobs,
            "total_clusters": clusters,
        }

    def _build_preview_url(self, job_id: uuid.UUID, clip_id: uuid.UUID | None) -> str | None:
        if clip_id:
            return f"/api/v1/jobs/{job_id}/clips/{clip_id}/final"
        return None
