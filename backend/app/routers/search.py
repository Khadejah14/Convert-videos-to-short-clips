import uuid
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.deps import get_current_user
from app.models.user import User
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.schemas.search import (
    SearchRequest,
    SearchResponse,
    SimilarRequest,
    SegmentResult,
    TopicClusterResponse,
    ClusterListResponse,
    EmbeddingStatsResponse,
    EmbedJobRequest,
    EmbedJobResponse,
)

router = APIRouter(prefix="/api/v1/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search_transcripts(
    body: SearchRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = SearchService(db)
    filters = {
        "is_hook": body.is_hook,
        "is_emotional_peak": body.is_emotional_peak,
        "emotion": body.emotion,
        "min_energy": body.min_energy,
        "topic_cluster_id": body.topic_cluster_id,
    }
    filters = {k: v for k, v in filters.items() if v is not None}

    results = await svc.semantic_search(
        query=body.query,
        owner_id=current_user.id,
        job_id=body.job_id,
        project_id=body.project_id,
        limit=body.limit,
        threshold=body.threshold,
        filters=filters or None,
    )

    return SearchResponse(
        query=body.query,
        results=[SegmentResult(**r) for r in results],
        total=len(results),
    )


@router.post("/similar", response_model=SearchResponse)
async def find_similar(
    body: SimilarRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = SearchService(db)
    results = await svc.find_similar_segments(
        segment_id=body.segment_id,
        owner_id=current_user.id,
        limit=body.limit,
        same_job=body.same_job,
    )

    return SearchResponse(
        query=f"similar to {body.segment_id}",
        results=[SegmentResult(**r) for r in results],
        total=len(results),
    )


@router.get("/hooks", response_model=list[SegmentResult])
async def get_hooks(
    job_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = SearchService(db)
    results = await svc.get_hooks(
        owner_id=current_user.id,
        job_id=job_id,
        limit=limit,
    )
    return [SegmentResult(**r) for r in results]


@router.get("/peaks", response_model=list[SegmentResult])
async def get_emotional_peaks(
    job_id: uuid.UUID | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = SearchService(db)
    results = await svc.get_emotional_peaks(
        owner_id=current_user.id,
        job_id=job_id,
        limit=limit,
    )
    return [SegmentResult(**r) for r in results]


@router.get("/clusters", response_model=ClusterListResponse)
async def list_clusters(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = SearchService(db)
    clusters = await svc.get_topic_clusters(owner_id=current_user.id)
    return ClusterListResponse(
        clusters=[TopicClusterResponse(**c) for c in clusters],
        total=len(clusters),
    )


@router.get("/clusters/{cluster_id}/segments", response_model=list[SegmentResult])
async def get_cluster_segments(
    cluster_id: int,
    limit: int = Query(default=50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = SearchService(db)
    results = await svc.get_cluster_segments(
        cluster_id=cluster_id,
        owner_id=current_user.id,
        limit=limit,
    )
    return [SegmentResult(**r) for r in results]


@router.get("/stats", response_model=EmbeddingStatsResponse)
async def get_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    svc = SearchService(db)
    stats = await svc.get_embedding_stats(owner_id=current_user.id)
    return EmbeddingStatsResponse(**stats)


@router.post("/embed", response_model=EmbedJobResponse)
async def embed_job(
    body: EmbedJobRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    from app.models.job import Job

    result = await db.execute(
        select(Job).where(Job.id == body.job_id, Job.owner_id == current_user.id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.transcript_text:
        raise HTTPException(status_code=400, detail="Job has no transcript yet")

    import re
    segments = []
    for match in re.finditer(
        r"\[([\d.]+)s?\s*-\s*([\d.]+)s?\]\s*(.+)", job.transcript_text
    ):
        segments.append({
            "start": float(match.group(1)),
            "end": float(match.group(2)),
            "text": match.group(3).strip(),
        })

    if not segments:
        raise HTTPException(status_code=400, detail="Could not parse transcript segments")

    emb_svc = EmbeddingService()
    records = await emb_svc.store_job_embeddings(
        db=db,
        job_id=body.job_id,
        segments=segments,
        owner_id=current_user.id,
    )

    return EmbedJobResponse(
        job_id=str(body.job_id),
        segments_indexed=len(records),
        status="completed",
    )


@router.post("/embed/async")
async def embed_job_async(
    body: EmbedJobRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    from sqlalchemy import select
    from app.models.job import Job

    result = await db.execute(
        select(Job).where(Job.id == body.job_id, Job.owner_id == current_user.id)
    )
    job = result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.transcript_text:
        raise HTTPException(status_code=400, detail="Job has no transcript yet")

    from app.tasks.embedding_tasks import generate_embeddings_task
    generate_embeddings_task.delay(str(body.job_id), str(current_user.id))

    return {"job_id": str(body.job_id), "status": "queued"}


@router.post("/clusters/build")
async def build_clusters(
    n_clusters: int = Query(default=8, ge=2, le=20),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    emb_svc = EmbeddingService()
    clusters = await emb_svc.compute_cluster_centroids(
        db=db,
        owner_id=current_user.id,
        n_clusters=n_clusters,
    )

    return {
        "clusters_built": len(clusters),
        "labels": [c.label for c in clusters],
    }
