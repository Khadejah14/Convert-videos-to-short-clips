import uuid
import logging
from datetime import datetime, timedelta

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.publishing import (
    AnalyticsSnapshot, PublishHistory, PublishStatus, Platform
)
from app.schemas.publishing import AnalyticsSnapshotResponse, AnalyticsSummary
from app.services.platform_adapters import get_adapter

logger = logging.getLogger(__name__)


class AnalyticsService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def collect_snapshot(
        self, publish_history_id: uuid.UUID
    ) -> AnalyticsSnapshotResponse | None:
        result = await self.db.execute(
            select(PublishHistory).where(PublishHistory.id == publish_history_id)
        )
        entry = result.scalar_one_or_none()
        if not entry or entry.status != PublishStatus.PUBLISHED:
            return None
        if not entry.platform_post_id or not entry.account_id:
            return None

        account_result = await self.db.execute(
            select(entry.__class__).where(entry.__class__.id == entry.account_id)
        )

        from app.models.publishing import ConnectedAccount
        account_result = await self.db.execute(
            select(ConnectedAccount).where(ConnectedAccount.id == entry.account_id)
        )
        account = account_result.scalar_one_or_none()
        if not account:
            return None

        try:
            adapter = get_adapter(entry.platform)
            from app.services.publishing_service import PublishingService
            svc = PublishingService(self.db)
            token = await svc.get_valid_token(account)

            metrics = await adapter.get_post_analytics(token, entry.platform_post_id)

            snapshot = AnalyticsSnapshot(
                publish_history_id=entry.id,
                user_id=entry.user_id,
                platform=entry.platform,
                views=metrics.get("views", 0),
                likes=metrics.get("likes", 0),
                comments=metrics.get("comments", 0),
                shares=metrics.get("shares", 0),
                watch_time_seconds=metrics.get("watch_time_seconds", 0),
                average_watch_time=metrics.get("average_watch_time", 0),
                engagement_rate=self._calc_engagement(metrics),
                raw_metrics=metrics,
            )
            self.db.add(snapshot)
            await self.db.commit()
            await self.db.refresh(snapshot)
            return AnalyticsSnapshotResponse.model_validate(snapshot)

        except Exception as e:
            logger.warning(f"Failed to collect analytics for {publish_history_id}: {e}")
            return None

    async def get_summary(
        self,
        user_id: uuid.UUID,
        platform: Platform | None = None,
        days: int = 30,
    ) -> AnalyticsSummary:
        since = datetime.utcnow() - timedelta(days=days)

        query = select(
            func.coalesce(func.sum(AnalyticsSnapshot.views), 0).label("total_views"),
            func.coalesce(func.sum(AnalyticsSnapshot.likes), 0).label("total_likes"),
            func.coalesce(func.sum(AnalyticsSnapshot.comments), 0).label("total_comments"),
            func.coalesce(func.sum(AnalyticsSnapshot.shares), 0).label("total_shares"),
            func.coalesce(func.sum(AnalyticsSnapshot.watch_time_seconds), 0).label("total_watch_time"),
            func.coalesce(func.avg(AnalyticsSnapshot.engagement_rate), 0).label("avg_engagement"),
            func.count(AnalyticsSnapshot.id).label("snapshot_count"),
        ).where(
            AnalyticsSnapshot.user_id == user_id,
            AnalyticsSnapshot.snapshot_at >= since,
        )
        if platform:
            query = query.where(AnalyticsSnapshot.platform == platform)

        result = await self.db.execute(query)
        row = result.one()

        platform_query = (
            select(
                PublishHistory.platform,
                func.count(PublishHistory.id),
            )
            .where(
                PublishHistory.user_id == user_id,
                PublishHistory.status == PublishStatus.PUBLISHED,
            )
            .group_by(PublishHistory.platform)
        )
        plat_result = await self.db.execute(platform_query)
        platform_breakdown = {row[0].value: row[1] for row in plat_result.all()}

        return AnalyticsSummary(
            total_views=int(row.total_views),
            total_likes=int(row.total_likes),
            total_comments=int(row.total_comments),
            total_shares=int(row.total_shares),
            total_watch_time=float(row.total_watch_time),
            average_engagement_rate=float(row.avg_engagement),
            posts_count=int(row.snapshot_count),
            platform_breakdown=platform_breakdown,
        )

    async def get_snapshots_for_post(
        self, publish_history_id: uuid.UUID, user_id: uuid.UUID
    ) -> list[AnalyticsSnapshotResponse]:
        result = await self.db.execute(
            select(AnalyticsSnapshot)
            .where(
                AnalyticsSnapshot.publish_history_id == publish_history_id,
                AnalyticsSnapshot.user_id == user_id,
            )
            .order_by(AnalyticsSnapshot.snapshot_at.asc())
        )
        snapshots = result.scalars().all()
        return [AnalyticsSnapshotResponse.model_validate(s) for s in snapshots]

    def _calc_engagement(self, metrics: dict) -> float:
        views = metrics.get("views", 0)
        if views == 0:
            return 0.0
        interactions = (
            metrics.get("likes", 0)
            + metrics.get("comments", 0)
            + metrics.get("shares", 0)
        )
        return round((interactions / views) * 100, 2)
