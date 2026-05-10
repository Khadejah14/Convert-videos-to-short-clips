from __future__ import annotations

import logging
import uuid
from typing import Any

import openai
import numpy as np
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.embedding import TranscriptEmbedding, TopicCluster
from app.models.job import Job

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    def __init__(self, api_key: str | None = None):
        self.client = openai.AsyncOpenAI(api_key=api_key or settings.OPENAI_API_KEY)
        self.model = settings.EMBEDDING_MODEL
        self.dimensions = settings.EMBEDDING_DIMENSIONS
        self.batch_size = settings.EMBEDDING_BATCH_SIZE

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch = [t[:8000] for t in batch]

            try:
                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                )
                for item in response.data:
                    all_embeddings.append(item.embedding)
            except Exception as e:
                logger.error(f"Embedding batch {i} failed: {e}")
                all_embeddings.extend([[0.0] * self.dimensions] * len(batch))

        return all_embeddings

    async def embed_single(self, text: str) -> list[float]:
        result = await self.embed_texts([text])
        return result[0] if result else [0.0] * self.dimensions

    async def store_job_embeddings(
        self,
        db: AsyncSession,
        job_id: uuid.UUID,
        segments: list[dict[str, Any]],
        owner_id: uuid.UUID | None = None,
        clip_id: uuid.UUID | None = None,
    ) -> list[TranscriptEmbedding]:
        await db.execute(
            delete(TranscriptEmbedding).where(TranscriptEmbedding.job_id == job_id)
        )

        texts = [s["text"] for s in segments]
        embeddings = await self.embed_texts(texts)

        records = []
        for i, (seg, emb) in enumerate(zip(segments, embeddings)):
            record = TranscriptEmbedding(
                owner_id=owner_id,
                job_id=job_id,
                clip_id=clip_id,
                segment_index=i,
                start_time=seg.get("start", 0),
                end_time=seg.get("end", 0),
                text=seg.get("text", ""),
                emotion=seg.get("emotion"),
                energy=seg.get("energy"),
                is_hook=seg.get("is_hook", False),
                is_emotional_peak=seg.get("is_emotional_peak", False),
                keywords=seg.get("keywords"),
                embedding=emb,
            )
            db.add(record)
            records.append(record)

        await db.commit()
        logger.info(f"Stored {len(records)} embeddings for job {job_id}")
        return records

    async def store_segment_embedding(
        self,
        db: AsyncSession,
        job_id: uuid.UUID,
        segment: dict[str, Any],
        owner_id: uuid.UUID | None = None,
        clip_id: uuid.UUID | None = None,
    ) -> TranscriptEmbedding:
        emb = await self.embed_single(segment["text"])

        record = TranscriptEmbedding(
            owner_id=owner_id,
            job_id=job_id,
            clip_id=clip_id,
            segment_index=segment.get("index", 0),
            start_time=segment.get("start", 0),
            end_time=segment.get("end", 0),
            text=segment.get("text", ""),
            emotion=segment.get("emotion"),
            energy=segment.get("energy"),
            is_hook=segment.get("is_hook", False),
            is_emotional_peak=segment.get("is_emotional_peak", False),
            keywords=segment.get("keywords"),
            embedding=emb,
        )
        db.add(record)
        await db.commit()
        return record

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        a_arr = np.array(a)
        b_arr = np.array(b)
        norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
        if norm == 0:
            return 0.0
        return float(np.dot(a_arr, b_arr) / norm)

    async def compute_cluster_centroids(
        self,
        db: AsyncSession,
        owner_id: uuid.UUID | None = None,
        n_clusters: int = 8,
    ) -> list[TopicCluster]:
        query = select(TranscriptEmbedding)
        if owner_id:
            query = query.where(TranscriptEmbedding.owner_id == owner_id)

        result = await db.execute(query)
        records = result.scalars().all()

        if len(records) < n_clusters:
            n_clusters = max(1, len(records))

        embeddings = np.array([r.embedding for r in records])
        labels, centroids = self._kmeans(embeddings, n_clusters)

        await db.execute(
            delete(TopicCluster).where(
                TopicCluster.owner_id == owner_id if owner_id else True
            )
        )

        clusters = []
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            cluster_records = [r for r, m in zip(records, mask) if m]
            if not cluster_records:
                continue

            emotions = [r.emotion for r in cluster_records if r.emotion]
            dominant = max(set(emotions), key=emotions.count) if emotions else None
            avg_e = float(np.mean([r.energy for r in cluster_records if r.energy]))

            label = self._generate_cluster_label(cluster_records)

            cluster = TopicCluster(
                owner_id=owner_id,
                label=label,
                centroid=centroids[cluster_id].tolist(),
                size=len(cluster_records),
                avg_energy=avg_e,
                dominant_emotion=dominant,
            )
            db.add(cluster)
            clusters.append(cluster)

            for rec in cluster_records:
                rec.topic_cluster_id = cluster_id
                rec.topic = label

        await db.commit()
        return clusters

    def _kmeans(
        self, embeddings: np.ndarray, k: int, max_iter: int = 50
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(embeddings)
        if n == 0:
            return np.array([]), np.zeros((k, embeddings.shape[1] if embeddings.ndim == 2 else self.dimensions))

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        rng = np.random.default_rng(42)
        indices = rng.choice(n, size=min(k, n), replace=False)
        centroids = embeddings[indices].copy()

        for _ in range(max_iter):
            dists = np.linalg.norm(embeddings[:, None] - centroids[None, :], axis=2)
            labels = np.argmin(dists, axis=1)

            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                members = embeddings[labels == i]
                if len(members) > 0:
                    new_centroids[i] = members.mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                break
            centroids = new_centroids

        return labels, centroids

    def _generate_cluster_label(self, records: list[TranscriptEmbedding]) -> str:
        all_keywords = []
        for r in records:
            if r.keywords:
                if isinstance(r.keywords, list):
                    all_keywords.extend(r.keywords)
                elif isinstance(r.keywords, dict):
                    all_keywords.extend(r.keywords.values())

        if all_keywords:
            from collections import Counter
            common = Counter(all_keywords).most_common(3)
            return " / ".join(w for w, _ in common)

        texts = [r.text[:30] for r in records[:3]]
        return " | ".join(texts) if texts else "Unknown Topic"
