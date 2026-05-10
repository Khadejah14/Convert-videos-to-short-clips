import uuid
import secrets
import logging
from datetime import datetime
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.core.config import get_settings
from app.core.deps import get_current_user
from app.models.user import User
from app.models.publishing import Platform, ScheduleStatus
from app.schemas.publishing import (
    ConnectedAccountResponse, ConnectedAccountListResponse,
    OAuthInitResponse, OAuthCallbackRequest,
    ExportPresetCreate, ExportPresetUpdate, ExportPresetResponse, ExportPresetListResponse,
    PublishDraftCreate, PublishDraftUpdate, PublishDraftResponse, PublishDraftListResponse,
    PublishScheduleCreate, PublishScheduleResponse, PublishScheduleListResponse,
    PublishHistoryResponse, PublishHistoryListResponse,
    PublishNowRequest,
    AnalyticsSnapshotResponse, AnalyticsSummary,
)
from app.services.publishing_service import PublishingService
from app.services.analytics_service import AnalyticsService
from app.services.platform_adapters import get_adapter, generate_oauth_state

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/v1/publishing", tags=["publishing"])

OAUTH_REDIRECT_BASE = "http://localhost:8000/api/v1/publishing/oauth/callback"


# --- OAuth Account Connection ---

@router.get("/oauth/{platform}/url", response_model=OAuthInitResponse)
async def get_oauth_url(
    platform: Platform,
    current_user: User = Depends(get_current_user),
):
    state = generate_oauth_state(current_user.id, platform)
    adapter = get_adapter(platform)
    redirect_uri = f"{OAUTH_REDIRECT_BASE}/{platform.value}"
    url = adapter.get_oauth_url(state, redirect_uri)
    return OAuthInitResponse(authorization_url=url, state=state)


@router.get("/oauth/callback/{platform}", response_model=ConnectedAccountResponse)
async def oauth_callback(
    platform: Platform,
    code: str = Query(...),
    state: str = Query(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    adapter = get_adapter(platform)
    redirect_uri = f"{OAUTH_REDIRECT_BASE}/{platform.value}"

    try:
        token_data = await adapter.exchange_code(code, redirect_uri)
        user_info = await adapter.get_user_info(token_data["access_token"])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"OAuth exchange failed: {str(e)}")

    service = PublishingService(db)
    account = await service.create_account(
        user_id=current_user.id,
        platform=platform,
        token_data=token_data,
        user_info=user_info,
    )
    return account


@router.get("/accounts", response_model=ConnectedAccountListResponse)
async def list_accounts(
    platform: Platform | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    accounts = await service.list_accounts(current_user.id, platform)
    return ConnectedAccountListResponse(accounts=accounts, total=len(accounts))


@router.delete("/accounts/{account_id}", status_code=204)
async def disconnect_account(
    account_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    success = await service.disconnect_account(account_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Account not found")


# --- Export Presets ---

@router.get("/presets", response_model=ExportPresetListResponse)
async def list_presets(
    platform: Platform | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    presets = await service.list_presets(current_user.id, platform)
    return ExportPresetListResponse(presets=presets, total=len(presets))


@router.post("/presets", response_model=ExportPresetResponse, status_code=201)
async def create_preset(
    data: ExportPresetCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    return await service.create_preset(current_user.id, data)


@router.patch("/presets/{preset_id}", response_model=ExportPresetResponse)
async def update_preset(
    preset_id: uuid.UUID,
    data: ExportPresetUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    result = await service.update_preset(preset_id, current_user.id, data)
    if not result:
        raise HTTPException(status_code=404, detail="Preset not found or is system preset")
    return result


@router.delete("/presets/{preset_id}", status_code=204)
async def delete_preset(
    preset_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    success = await service.delete_preset(preset_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Preset not found or is system preset")


# --- Publish Drafts ---

@router.post("/drafts", response_model=PublishDraftResponse, status_code=201)
async def create_draft(
    data: PublishDraftCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    try:
        return await service.create_draft(current_user.id, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/drafts", response_model=PublishDraftListResponse)
async def list_drafts(
    platform: Platform | None = None,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    drafts, total = await service.list_drafts(
        current_user.id, platform, page, per_page
    )
    return PublishDraftListResponse(
        drafts=drafts, total=total, page=page, per_page=per_page
    )


@router.get("/drafts/{draft_id}", response_model=PublishDraftResponse)
async def get_draft(
    draft_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    draft = await service.get_draft(draft_id, current_user.id)
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")
    return draft


@router.patch("/drafts/{draft_id}", response_model=PublishDraftResponse)
async def update_draft(
    draft_id: uuid.UUID,
    data: PublishDraftUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    result = await service.update_draft(draft_id, current_user.id, data)
    if not result:
        raise HTTPException(status_code=404, detail="Draft not found")
    return result


@router.delete("/drafts/{draft_id}", status_code=204)
async def delete_draft(
    draft_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    success = await service.delete_draft(draft_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Draft not found or currently publishing",
        )


# --- Publish Actions ---

@router.post("/publish", response_model=PublishHistoryResponse)
async def publish_now(
    data: PublishNowRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    try:
        return await service.publish_now(data.draft_id, current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# --- Schedules ---

@router.post("/schedules", response_model=PublishScheduleResponse, status_code=201)
async def create_schedule(
    data: PublishScheduleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    try:
        return await service.create_schedule(current_user.id, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/schedules", response_model=PublishScheduleListResponse)
async def list_schedules(
    status: ScheduleStatus | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    schedules = await service.list_schedules(current_user.id, status)
    return PublishScheduleListResponse(schedules=schedules, total=len(schedules))


@router.post("/schedules/{schedule_id}/cancel")
async def cancel_schedule(
    schedule_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    success = await service.cancel_schedule(schedule_id, current_user.id)
    if not success:
        raise HTTPException(
            status_code=400,
            detail="Schedule not found or not in pending state",
        )
    return {"status": "cancelled"}


# --- Publishing History ---

@router.get("/history", response_model=PublishHistoryListResponse)
async def list_history(
    platform: Platform | None = None,
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    history, total = await service.list_history(
        current_user.id, platform, page, per_page
    )
    return PublishHistoryListResponse(
        history=history, total=total, page=page, per_page=per_page
    )


@router.get("/history/{history_id}", response_model=PublishHistoryResponse)
async def get_history_entry(
    history_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    entry = await service.get_history_entry(history_id, current_user.id)
    if not entry:
        raise HTTPException(status_code=404, detail="History entry not found")
    return entry


@router.post("/history/{history_id}/retry", response_model=PublishHistoryResponse)
async def retry_publish(
    history_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = PublishingService(db)
    result = await service.retry_publish(history_id, current_user.id)
    if not result:
        raise HTTPException(
            status_code=400,
            detail="History entry not found or not in failed state",
        )
    return result


# --- Analytics ---

@router.get("/analytics/summary", response_model=AnalyticsSummary)
async def get_analytics_summary(
    platform: Platform | None = None,
    days: int = Query(default=30, ge=1, le=365),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AnalyticsService(db)
    return await service.get_summary(current_user.id, platform, days)


@router.get("/analytics/posts/{history_id}", response_model=list[AnalyticsSnapshotResponse])
async def get_post_analytics(
    history_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AnalyticsService(db)
    return await service.get_snapshots_for_post(history_id, current_user.id)


@router.post("/analytics/collect/{history_id}", response_model=AnalyticsSnapshotResponse | None)
async def collect_analytics(
    history_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    service = AnalyticsService(db)
    return await service.collect_snapshot(history_id)
