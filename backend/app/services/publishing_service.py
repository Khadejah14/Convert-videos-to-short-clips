import uuid
import logging
from datetime import datetime
from typing import Any

from sqlalchemy import select, func, and_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.models.publishing import (
    ConnectedAccount, Platform, AccountStatus,
    PublishDraft, DraftStatus,
    PublishSchedule, ScheduleStatus,
    PublishHistory, PublishStatus,
    ExportPreset,
)
from app.models.job import Clip
from app.schemas.publishing import (
    ConnectedAccountResponse,
    ExportPresetCreate, ExportPresetUpdate, ExportPresetResponse,
    PublishDraftCreate, PublishDraftUpdate, PublishDraftResponse,
    PublishScheduleCreate, PublishScheduleResponse,
    PublishHistoryResponse,
)
from app.services.platform_adapters import get_adapter, generate_oauth_state

logger = logging.getLogger(__name__)


class PublishingService:
    def __init__(self, db: AsyncSession):
        self.db = db

    # --- Connected Accounts ---

    async def list_accounts(
        self, user_id: uuid.UUID, platform: Platform | None = None
    ) -> list[ConnectedAccountResponse]:
        query = select(ConnectedAccount).where(ConnectedAccount.user_id == user_id)
        if platform:
            query = query.where(ConnectedAccount.platform == platform)
        query = query.order_by(ConnectedAccount.created_at.desc())
        result = await self.db.execute(query)
        accounts = result.scalars().all()
        return [ConnectedAccountResponse.model_validate(a) for a in accounts]

    async def get_account(
        self, account_id: uuid.UUID, user_id: uuid.UUID
    ) -> ConnectedAccount | None:
        result = await self.db.execute(
            select(ConnectedAccount).where(
                ConnectedAccount.id == account_id,
                ConnectedAccount.user_id == user_id,
            )
        )
        return result.scalar_one_or_none()

    async def create_account(
        self,
        user_id: uuid.UUID,
        platform: Platform,
        token_data: dict[str, Any],
        user_info: dict[str, Any],
    ) -> ConnectedAccountResponse:
        existing = await self.db.execute(
            select(ConnectedAccount).where(
                ConnectedAccount.user_id == user_id,
                ConnectedAccount.platform == platform,
                ConnectedAccount.platform_user_id == user_info["platform_user_id"],
            )
        )
        account = existing.scalar_one_or_none()

        expires_at = None
        if "expires_in" in token_data:
            from datetime import timedelta
            expires_at = datetime.utcnow() + timedelta(seconds=int(token_data["expires_in"]))

        if account:
            account.access_token = token_data["access_token"]
            if "refresh_token" in token_data:
                account.refresh_token = token_data["refresh_token"]
            account.token_expires_at = expires_at
            account.platform_username = user_info.get("username")
            account.display_name = user_info.get("display_name")
            account.avatar_url = user_info.get("avatar_url")
            account.status = AccountStatus.ACTIVE
        else:
            account = ConnectedAccount(
                user_id=user_id,
                platform=platform,
                platform_user_id=user_info["platform_user_id"],
                platform_username=user_info.get("username"),
                display_name=user_info.get("display_name"),
                avatar_url=user_info.get("avatar_url"),
                access_token=token_data["access_token"],
                refresh_token=token_data.get("refresh_token"),
                token_expires_at=expires_at,
                status=AccountStatus.ACTIVE,
            )
            self.db.add(account)

        await self.db.commit()
        await self.db.refresh(account)
        return ConnectedAccountResponse.model_validate(account)

    async def disconnect_account(
        self, account_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        account = await self.get_account(account_id, user_id)
        if not account:
            return False
        await self.db.delete(account)
        await self.db.commit()
        return True

    async def get_valid_token(
        self, account: ConnectedAccount
    ) -> str:
        if account.token_expires_at and account.token_expires_at < datetime.utcnow():
            if account.refresh_token:
                adapter = get_adapter(account.platform)
                token_data = await adapter.refresh_access_token(account.refresh_token)
                account.access_token = token_data["access_token"]
                if "expires_in" in token_data:
                    from datetime import timedelta
                    account.token_expires_at = datetime.utcnow() + timedelta(
                        seconds=int(token_data["expires_in"])
                    )
                if "refresh_token" in token_data:
                    account.refresh_token = token_data["refresh_token"]
                account.status = AccountStatus.ACTIVE
                await self.db.commit()
            else:
                account.status = AccountStatus.EXPIRED
                await self.db.commit()
                raise ValueError("Token expired and no refresh token available")
        return account.access_token

    # --- Export Presets ---

    async def list_presets(
        self, user_id: uuid.UUID, platform: Platform | None = None
    ) -> list[ExportPresetResponse]:
        query = select(ExportPreset).where(
            (ExportPreset.user_id == user_id) | (ExportPreset.is_system == True)
        )
        if platform:
            query = query.where(ExportPreset.platform == platform)
        query = query.order_by(ExportPreset.is_system.desc(), ExportPreset.name)
        result = await self.db.execute(query)
        presets = result.scalars().all()
        return [ExportPresetResponse.model_validate(p) for p in presets]

    async def create_preset(
        self, user_id: uuid.UUID, data: ExportPresetCreate
    ) -> ExportPresetResponse:
        preset = ExportPreset(
            user_id=user_id,
            **data.model_dump(),
        )
        self.db.add(preset)
        await self.db.commit()
        await self.db.refresh(preset)
        return ExportPresetResponse.model_validate(preset)

    async def update_preset(
        self, preset_id: uuid.UUID, user_id: uuid.UUID, data: ExportPresetUpdate
    ) -> ExportPresetResponse | None:
        result = await self.db.execute(
            select(ExportPreset).where(
                ExportPreset.id == preset_id,
                ExportPreset.user_id == user_id,
                ExportPreset.is_system == False,
            )
        )
        preset = result.scalar_one_or_none()
        if not preset:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(preset, key, value)

        await self.db.commit()
        await self.db.refresh(preset)
        return ExportPresetResponse.model_validate(preset)

    async def delete_preset(
        self, preset_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        result = await self.db.execute(
            select(ExportPreset).where(
                ExportPreset.id == preset_id,
                ExportPreset.user_id == user_id,
                ExportPreset.is_system == False,
            )
        )
        preset = result.scalar_one_or_none()
        if not preset:
            return False
        await self.db.delete(preset)
        await self.db.commit()
        return True

    # --- Publish Drafts ---

    async def create_draft(
        self, user_id: uuid.UUID, data: PublishDraftCreate
    ) -> PublishDraftResponse:
        clip_result = await self.db.execute(
            select(Clip).where(Clip.id == data.clip_id)
        )
        clip = clip_result.scalar_one_or_none()
        if not clip:
            raise ValueError("Clip not found")

        draft = PublishDraft(
            user_id=user_id,
            clip_id=data.clip_id,
            account_id=data.account_id,
            preset_id=data.preset_id,
            platform=data.platform,
            title=data.title,
            description=data.description,
            tags=data.tags,
            visibility=data.visibility,
            category=data.category,
            language=data.language,
            cover_frame_time=data.cover_frame_time,
            publish_config=data.publish_config,
            status=DraftStatus.DRAFT,
        )
        self.db.add(draft)
        await self.db.commit()
        await self.db.refresh(draft)
        return PublishDraftResponse.model_validate(draft)

    async def update_draft(
        self, draft_id: uuid.UUID, user_id: uuid.UUID, data: PublishDraftUpdate
    ) -> PublishDraftResponse | None:
        result = await self.db.execute(
            select(PublishDraft).where(
                PublishDraft.id == draft_id,
                PublishDraft.user_id == user_id,
            )
        )
        draft = result.scalar_one_or_none()
        if not draft:
            return None

        update_data = data.model_dump(exclude_unset=True)
        for key, value in update_data.items():
            setattr(draft, key, value)

        await self.db.commit()
        await self.db.refresh(draft)
        return PublishDraftResponse.model_validate(draft)

    async def get_draft(
        self, draft_id: uuid.UUID, user_id: uuid.UUID
    ) -> PublishDraftResponse | None:
        result = await self.db.execute(
            select(PublishDraft).where(
                PublishDraft.id == draft_id,
                PublishDraft.user_id == user_id,
            )
        )
        draft = result.scalar_one_or_none()
        if not draft:
            return None
        return PublishDraftResponse.model_validate(draft)

    async def list_drafts(
        self,
        user_id: uuid.UUID,
        platform: Platform | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[PublishDraftResponse], int]:
        base = select(func.count(PublishDraft.id)).where(
            PublishDraft.user_id == user_id
        )
        if platform:
            base = base.where(PublishDraft.platform == platform)
        count_result = await self.db.execute(base)
        total = count_result.scalar()

        query = select(PublishDraft).where(PublishDraft.user_id == user_id)
        if platform:
            query = query.where(PublishDraft.platform == platform)
        query = query.order_by(PublishDraft.updated_at.desc())
        query = query.offset((page - 1) * per_page).limit(per_page)

        result = await self.db.execute(query)
        drafts = result.scalars().all()
        return [PublishDraftResponse.model_validate(d) for d in drafts], total

    async def delete_draft(
        self, draft_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        result = await self.db.execute(
            select(PublishDraft).where(
                PublishDraft.id == draft_id,
                PublishDraft.user_id == user_id,
            )
        )
        draft = result.scalar_one_or_none()
        if not draft:
            return False
        if draft.status == DraftStatus.PUBLISHING:
            return False
        await self.db.delete(draft)
        await self.db.commit()
        return True

    # --- Publish Schedules ---

    async def create_schedule(
        self, user_id: uuid.UUID, data: PublishScheduleCreate
    ) -> PublishScheduleResponse:
        draft_result = await self.db.execute(
            select(PublishDraft).where(
                PublishDraft.id == data.draft_id,
                PublishDraft.user_id == user_id,
            )
        )
        draft = draft_result.scalar_one_or_none()
        if not draft:
            raise ValueError("Draft not found")

        schedule = PublishSchedule(
            draft_id=data.draft_id,
            user_id=user_id,
            scheduled_at=data.scheduled_at,
            status=ScheduleStatus.PENDING,
        )
        self.db.add(schedule)
        draft.status = DraftStatus.READY
        await self.db.commit()
        await self.db.refresh(schedule)

        from app.tasks.publishing_tasks import execute_scheduled_publish
        task = execute_scheduled_publish.apply_async(
            args=[str(schedule.id)],
            eta=data.scheduled_at,
        )
        schedule.celery_task_id = task.id
        await self.db.commit()
        await self.db.refresh(schedule)

        return PublishScheduleResponse.model_validate(schedule)

    async def list_schedules(
        self, user_id: uuid.UUID, status: ScheduleStatus | None = None
    ) -> list[PublishScheduleResponse]:
        query = select(PublishSchedule).where(PublishSchedule.user_id == user_id)
        if status:
            query = query.where(PublishSchedule.status == status)
        query = query.order_by(PublishSchedule.scheduled_at.asc())
        result = await self.db.execute(query)
        schedules = result.scalars().all()
        return [PublishScheduleResponse.model_validate(s) for s in schedules]

    async def cancel_schedule(
        self, schedule_id: uuid.UUID, user_id: uuid.UUID
    ) -> bool:
        result = await self.db.execute(
            select(PublishSchedule).where(
                PublishSchedule.id == schedule_id,
                PublishSchedule.user_id == user_id,
                PublishSchedule.status == ScheduleStatus.PENDING,
            )
        )
        schedule = result.scalar_one_or_none()
        if not schedule:
            return False

        if schedule.celery_task_id:
            from app.tasks.celery_app import celery_app
            try:
                celery_app.control.revoke(schedule.celery_task_id, terminate=True)
            except Exception:
                pass

        schedule.status = ScheduleStatus.CANCELLED
        await self.db.commit()
        return True

    # --- Publish History ---

    async def create_history_entry(
        self,
        user_id: uuid.UUID,
        draft: PublishDraft,
        account_id: uuid.UUID | None = None,
    ) -> PublishHistory:
        entry = PublishHistory(
            draft_id=draft.id,
            user_id=user_id,
            account_id=account_id or draft.account_id,
            platform=draft.platform,
            status=PublishStatus.UPLOADING,
            title=draft.title,
            description=draft.description,
            tags=draft.tags,
            visibility=draft.visibility,
        )
        self.db.add(entry)
        await self.db.commit()
        await self.db.refresh(entry)
        return entry

    async def update_history_progress(
        self,
        history_id: uuid.UUID,
        progress: float,
        status: PublishStatus | None = None,
        platform_post_id: str | None = None,
        platform_post_url: str | None = None,
        error_message: str | None = None,
    ) -> None:
        result = await self.db.execute(
            select(PublishHistory).where(PublishHistory.id == history_id)
        )
        entry = result.scalar_one_or_none()
        if not entry:
            return

        entry.upload_progress = progress
        if status:
            entry.status = status
        if platform_post_id:
            entry.platform_post_id = platform_post_id
        if platform_post_url:
            entry.platform_post_url = platform_post_url
        if error_message:
            entry.error_message = error_message
        if status == PublishStatus.PUBLISHED:
            entry.published_at = datetime.utcnow()
            entry.upload_progress = 100.0

        await self.db.commit()

    async def list_history(
        self,
        user_id: uuid.UUID,
        platform: Platform | None = None,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[PublishHistoryResponse], int]:
        base = select(func.count(PublishHistory.id)).where(
            PublishHistory.user_id == user_id
        )
        if platform:
            base = base.where(PublishHistory.platform == platform)
        count_result = await self.db.execute(base)
        total = count_result.scalar()

        query = select(PublishHistory).where(PublishHistory.user_id == user_id)
        if platform:
            query = query.where(PublishHistory.platform == platform)
        query = query.order_by(PublishHistory.created_at.desc())
        query = query.offset((page - 1) * per_page).limit(per_page)

        result = await self.db.execute(query)
        entries = result.scalars().all()
        return [PublishHistoryResponse.model_validate(e) for e in entries], total

    async def get_history_entry(
        self, history_id: uuid.UUID, user_id: uuid.UUID
    ) -> PublishHistoryResponse | None:
        result = await self.db.execute(
            select(PublishHistory).where(
                PublishHistory.id == history_id,
                PublishHistory.user_id == user_id,
            )
        )
        entry = result.scalar_one_or_none()
        if not entry:
            return None
        return PublishHistoryResponse.model_validate(entry)

    async def retry_publish(
        self, history_id: uuid.UUID, user_id: uuid.UUID
    ) -> PublishHistoryResponse | None:
        result = await self.db.execute(
            select(PublishHistory).where(
                PublishHistory.id == history_id,
                PublishHistory.user_id == user_id,
                PublishHistory.status == PublishStatus.FAILED,
            )
        )
        entry = result.scalar_one_or_none()
        if not entry:
            return None

        entry.status = PublishStatus.UPLOADING
        entry.upload_progress = 0.0
        entry.error_message = None
        entry.retry_count += 1
        await self.db.commit()

        from app.tasks.publishing_tasks import retry_publish_task
        retry_publish_task.delay(str(entry.id))

        await self.db.refresh(entry)
        return PublishHistoryResponse.model_validate(entry)

    # --- Publish Now ---

    async def publish_now(
        self, draft_id: uuid.UUID, user_id: uuid.UUID
    ) -> PublishHistoryResponse:
        draft_result = await self.db.execute(
            select(PublishDraft).where(
                PublishDraft.id == draft_id,
                PublishDraft.user_id == user_id,
            )
        )
        draft = draft_result.scalar_one_or_none()
        if not draft:
            raise ValueError("Draft not found")
        if draft.status == DraftStatus.PUBLISHING:
            raise ValueError("Draft is already being published")
        if not draft.account_id:
            raise ValueError("No connected account selected for this draft")

        account = await self.get_account(draft.account_id, user_id)
        if not account:
            raise ValueError("Connected account not found")

        draft.status = DraftStatus.PUBLISHING
        entry = await self.create_history_entry(user_id, draft)
        await self.db.commit()

        from app.tasks.publishing_tasks import publish_clip_task
        publish_clip_task.delay(str(entry.id))

        return PublishHistoryResponse.model_validate(entry)
