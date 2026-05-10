import abc
import uuid
import hashlib
import secrets
import logging
from datetime import datetime, timedelta
from urllib.parse import urlencode
from typing import Any

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.publishing import (
    ConnectedAccount, Platform, AccountStatus, PublishHistory, PublishStatus
)

logger = logging.getLogger(__name__)
settings = get_settings()


class PlatformAdapter(abc.ABC):
    """Base class for platform publishing adapters."""

    platform: Platform

    @abc.abstractmethod
    def get_oauth_url(self, state: str, redirect_uri: str) -> str:
        ...

    @abc.abstractmethod
    async def exchange_code(
        self, code: str, redirect_uri: str
    ) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    async def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    async def get_user_info(self, access_token: str) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    async def upload_video(
        self,
        access_token: str,
        video_path: str,
        title: str,
        description: str,
        tags: list[str],
        visibility: str,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        ...

    @abc.abstractmethod
    async def get_post_analytics(
        self, access_token: str, post_id: str
    ) -> dict[str, Any]:
        ...


class TikTokAdapter(PlatformAdapter):
    platform = Platform.TIKTOK

    AUTH_BASE = "https://www.tiktok.com/v2/auth/authorize/"
    TOKEN_URL = "https://open.tiktokapis.com/v2/oauth/token/"
    USER_INFO_URL = "https://open.tiktokapis.com/v2/user/info/"
    VIDEO_INIT_URL = "https://open.tiktokapis.com/v2/post/publish/inbox/video/init/"
    VIDEO_STATUS_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
    QUERY_VIDEO_URL = "https://open.tiktokapis.com/v2/post/publish/video/query/"

    def get_oauth_url(self, state: str, redirect_uri: str) -> str:
        params = {
            "client_key": settings.TIKTOK_CLIENT_KEY,
            "scope": "user.info.basic,video.upload,video.publish",
            "response_type": "code",
            "redirect_uri": redirect_uri,
            "state": state,
        }
        return f"{self.AUTH_BASE}?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.TOKEN_URL,
                data={
                    "client_key": settings.TIKTOK_CLIENT_KEY,
                    "client_secret": settings.TIKTOK_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if "data" in data:
                return data["data"]
            raise ValueError(f"TikTok token exchange failed: {data}")

    async def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.TOKEN_URL,
                data={
                    "client_key": settings.TIKTOK_CLIENT_KEY,
                    "client_secret": settings.TIKTOK_CLIENT_SECRET,
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            if "data" in data:
                return data["data"]
            raise ValueError(f"TikTok token refresh failed: {data}")

    async def get_user_info(self, access_token: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                self.USER_INFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                params={"fields": "open_id,union_id,avatar_url,display_name"},
            )
            resp.raise_for_status()
            data = resp.json()
            user_data = data.get("data", {}).get("user", {})
            return {
                "platform_user_id": user_data.get("open_id", ""),
                "display_name": user_data.get("display_name", ""),
                "avatar_url": user_data.get("avatar_url", ""),
                "username": user_data.get("display_name", ""),
            }

    async def upload_video(
        self,
        access_token: str,
        video_path: str,
        title: str,
        description: str,
        tags: list[str],
        visibility: str,
        progress_callback=None,
    ) -> dict[str, Any]:
        import os

        file_size = os.path.getsize(video_path)

        async with httpx.AsyncClient(timeout=300) as client:
            if progress_callback:
                await progress_callback(10, "Initializing upload")

            init_resp = await client.post(
                self.VIDEO_INIT_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={
                    "post_info": {
                        "title": title[:150],
                        "privacy_level": "PUBLIC_TO_EVERYONE" if visibility == "public" else "SELF_ONLY",
                        "description": description[:2200],
                        "disable_duet": False,
                        "disable_comment": False,
                        "disable_stitch": False,
                    },
                    "source_info": {
                        "source": "FILE_UPLOAD",
                        "video_size": file_size,
                    },
                },
            )
            init_resp.raise_for_status()
            init_data = init_resp.json().get("data", {})
            upload_url = init_data.get("upload_url", "")
            publish_id = init_data.get("publish_id", "")

            if not upload_url:
                raise ValueError("No upload URL returned from TikTok")

            if progress_callback:
                await progress_callback(30, "Uploading video")

            with open(video_path, "rb") as f:
                video_bytes = f.read()

            upload_resp = await client.put(
                upload_url,
                content=video_bytes,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
                },
            )
            if upload_resp.status_code not in (200, 201):
                raise ValueError(f"TikTok upload failed: {upload_resp.status_code}")

            if progress_callback:
                await progress_callback(90, "Upload complete, processing")

            return {
                "platform_post_id": publish_id,
                "platform_post_url": f"https://www.tiktok.com/@user/video/{publish_id}",
            }

    async def get_post_analytics(
        self, access_token: str, post_id: str
    ) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.QUERY_VIDEO_URL,
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                },
                json={"filters": {"video_ids": [post_id]}},
            )
            resp.raise_for_status()
            data = resp.json().get("data", {}).get("videos", [{}])
            video = data[0] if data else {}
            stats = video.get("statistics", {})
            return {
                "views": stats.get("view_count", 0),
                "likes": stats.get("like_count", 0),
                "comments": stats.get("comment_count", 0),
                "shares": stats.get("share_count", 0),
            }


class YouTubeAdapter(PlatformAdapter):
    platform = Platform.YOUTUBE

    AUTH_BASE = "https://accounts.google.com/o/oauth2/v2/auth"
    TOKEN_URL = "https://oauth2.googleapis.com/token"
    UPLOAD_URL = "https://www.googleapis.com/upload/youtube/v3/videos"
    CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
    ANALYTICS_URL = "https://youtubeanalytics.googleapis.com/v2/reports"

    def get_oauth_url(self, state: str, redirect_uri: str) -> str:
        params = {
            "client_id": settings.YOUTUBE_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "https://www.googleapis.com/auth/youtube.upload https://www.googleapis.com/auth/youtube.readonly https://www.googleapis.com/auth/yt-analytics.readonly",
            "access_type": "offline",
            "prompt": "consent",
            "state": state,
        }
        return f"{self.AUTH_BASE}?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.TOKEN_URL,
                data={
                    "client_id": settings.YOUTUBE_CLIENT_ID,
                    "client_secret": settings.YOUTUBE_CLIENT_SECRET,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                self.TOKEN_URL,
                data={
                    "client_id": settings.YOUTUBE_CLIENT_ID,
                    "client_secret": settings.YOUTUBE_CLIENT_SECRET,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            resp.raise_for_status()
            return resp.json()

    async def get_user_info(self, access_token: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                self.CHANNELS_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                params={"part": "snippet", "mine": "true"},
            )
            resp.raise_for_status()
            items = resp.json().get("items", [])
            if not items:
                raise ValueError("No YouTube channel found")
            snippet = items[0].get("snippet", {})
            return {
                "platform_user_id": items[0].get("id", ""),
                "display_name": snippet.get("title", ""),
                "avatar_url": snippet.get("thumbnails", {}).get("default", {}).get("url", ""),
                "username": snippet.get("customUrl", snippet.get("title", "")),
            }

    async def upload_video(
        self,
        access_token: str,
        video_path: str,
        title: str,
        description: str,
        tags: list[str],
        visibility: str,
        progress_callback=None,
    ) -> dict[str, Any]:
        import os

        privacy_status = "public" if visibility == "public" else "private"

        metadata = {
            "snippet": {
                "title": title[:100],
                "description": description[:5000],
                "tags": tags[:500],
                "categoryId": "22",
            },
            "status": {
                "privacyStatus": privacy_status,
                "selfDeclaredMadeForKids": False,
            },
        }

        if progress_callback:
            await progress_callback(10, "Initializing upload")

        async with httpx.AsyncClient(timeout=600) as client:
            resp = await client.post(
                f"{self.UPLOAD_URL}?uploadType=resumable&part=snippet,status",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json",
                    "X-Upload-Content-Type": "video/mp4",
                    "X-Upload-Content-Length": str(os.path.getsize(video_path)),
                },
                json=metadata,
            )
            resp.raise_for_status()
            upload_url = resp.headers.get("Location")

            if not upload_url:
                raise ValueError("No upload URL returned from YouTube")

            if progress_callback:
                await progress_callback(30, "Uploading video")

            with open(video_path, "rb") as f:
                video_bytes = f.read()

            upload_resp = await client.put(
                upload_url,
                content=video_bytes,
                headers={
                    "Content-Type": "video/mp4",
                    "Content-Length": str(len(video_bytes)),
                },
            )
            upload_resp.raise_for_status()
            result = upload_resp.json()
            video_id = result.get("id", "")

            if progress_callback:
                await progress_callback(90, "Upload complete")

            return {
                "platform_post_id": video_id,
                "platform_post_url": f"https://youtube.com/shorts/{video_id}",
            }

    async def get_post_analytics(
        self, access_token: str, post_id: str
    ) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                self.ANALYTICS_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "ids": "channel==MINE",
                    "startDate": "2020-01-01",
                    "endDate": datetime.utcnow().strftime("%Y-%m-%d"),
                    "metrics": "views,likes,comments,shares,estimatedMinutesWatched,averageViewDuration",
                    "dimensions": "video",
                    "filters": f"video=={post_id}",
                },
            )
            if resp.status_code != 200:
                return {"views": 0, "likes": 0, "comments": 0, "shares": 0}
            data = resp.json()
            rows = data.get("rows", [[0, 0, 0, 0, 0, 0]])
            row = rows[0] if rows else [0, 0, 0, 0, 0, 0]
            return {
                "views": row[0] if len(row) > 0 else 0,
                "likes": row[1] if len(row) > 1 else 0,
                "comments": row[2] if len(row) > 2 else 0,
                "shares": row[3] if len(row) > 3 else 0,
                "watch_time_seconds": (row[4] * 60) if len(row) > 4 else 0,
                "average_watch_time": row[5] if len(row) > 5 else 0,
            }


class InstagramAdapter(PlatformAdapter):
    platform = Platform.INSTAGRAM

    AUTH_BASE = "https://www.facebook.com/v19.0/dialog/oauth"
    TOKEN_URL = "https://graph.facebook.com/v19.0/oauth/access_token"
    USER_INFO_URL = "https://graph.instagram.com/v19.0/me"
    MEDIA_URL = "https://graph.facebook.com/v19.0"

    def get_oauth_url(self, state: str, redirect_uri: str) -> str:
        params = {
            "client_id": settings.INSTAGRAM_CLIENT_ID,
            "redirect_uri": redirect_uri,
            "scope": "instagram_basic,instagram_content_publish,instagram_manage_insights,pages_show_list",
            "response_type": "code",
            "state": state,
        }
        return f"{self.AUTH_BASE}?{urlencode(params)}"

    async def exchange_code(self, code: str, redirect_uri: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                self.TOKEN_URL,
                params={
                    "client_id": settings.INSTAGRAM_CLIENT_ID,
                    "client_secret": settings.INSTAGRAM_CLIENT_SECRET,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                    "code": code,
                },
            )
            resp.raise_for_status()
            short_token_data = resp.json()
            short_token = short_token_data.get("access_token", "")

            resp2 = await client.get(
                "https://graph.facebook.com/v19.0/oauth/access_token",
                params={
                    "grant_type": "fb_exchange_token",
                    "client_id": settings.INSTAGRAM_CLIENT_ID,
                    "client_secret": settings.INSTAGRAM_CLIENT_SECRET,
                    "fb_exchange_token": short_token,
                },
            )
            resp2.raise_for_status()
            long_token_data = resp2.json()

            return {
                "access_token": long_token_data.get("access_token", short_token),
                "expires_in": long_token_data.get("expires_in", short_token_data.get("expires_in", 3600)),
            }

    async def refresh_access_token(self, refresh_token: str) -> dict[str, Any]:
        return {"access_token": refresh_token, "expires_in": 5184000}

    async def get_user_info(self, access_token: str) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                self.USER_INFO_URL,
                headers={"Authorization": f"Bearer {access_token}"},
                params={"fields": "id,username,name,profile_picture_url"},
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "platform_user_id": data.get("id", ""),
                "display_name": data.get("name", data.get("username", "")),
                "avatar_url": data.get("profile_picture_url", ""),
                "username": data.get("username", ""),
            }

    async def upload_video(
        self,
        access_token: str,
        video_path: str,
        title: str,
        description: str,
        tags: list[str],
        visibility: str,
        progress_callback=None,
    ) -> dict[str, Any]:
        import os

        user_info = await self.get_user_info(access_token)
        ig_user_id = user_info["platform_user_id"]

        if progress_callback:
            await progress_callback(10, "Creating media container")

        video_url = f"file://{os.path.abspath(video_path)}"

        async with httpx.AsyncClient(timeout=600) as client:
            create_resp = await client.post(
                f"{self.MEDIA_URL}/{ig_user_id}/media",
                data={
                    "media_type": "REELS",
                    "video_url": video_url,
                    "caption": f"{title}\n\n{description}",
                    "access_token": access_token,
                },
            )
            create_resp.raise_for_status()
            container_id = create_resp.json().get("id", "")

            if progress_callback:
                await progress_callback(30, "Processing video")

            import asyncio
            for i in range(60):
                status_resp = await client.get(
                    f"{self.MEDIA_URL}/{container_id}",
                    params={
                        "fields": "status_code",
                        "access_token": access_token,
                    },
                )
                status_data = status_resp.json()
                status_code = status_data.get("status_code", "")
                if status_code == "FINISHED":
                    break
                if status_code == "ERROR":
                    raise ValueError("Instagram video processing failed")
                if progress_callback:
                    progress = 30 + int((i / 60) * 50)
                    await progress_callback(progress, "Processing video")
                await asyncio.sleep(5)

            if progress_callback:
                await progress_callback(85, "Publishing")

            publish_resp = await client.post(
                f"{self.MEDIA_URL}/{ig_user_id}/media_publish",
                data={
                    "creation_id": container_id,
                    "access_token": access_token,
                },
            )
            publish_resp.raise_for_status()
            media_id = publish_resp.json().get("id", "")

            if progress_callback:
                await progress_callback(100, "Published")

            return {
                "platform_post_id": media_id,
                "platform_post_url": f"https://www.instagram.com/reel/{media_id}",
            }

    async def get_post_analytics(
        self, access_token: str, post_id: str
    ) -> dict[str, Any]:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.MEDIA_URL}/{post_id}/insights",
                headers={"Authorization": f"Bearer {access_token}"},
                params={
                    "metric": "impressions,reach,likes,comments,shares,saved,video_views",
                },
            )
            if resp.status_code != 200:
                return {"views": 0, "likes": 0, "comments": 0, "shares": 0}
            data = resp.json().get("data", [])
            metrics = {}
            for item in data:
                name = item.get("name", "")
                values = item.get("values", [{}])
                metrics[name] = values[0].get("value", 0) if values else 0

            return {
                "views": metrics.get("video_views", metrics.get("impressions", 0)),
                "likes": metrics.get("likes", 0),
                "comments": metrics.get("comments", 0),
                "shares": metrics.get("shares", 0),
            }


def get_adapter(platform: Platform) -> PlatformAdapter:
    adapters = {
        Platform.TIKTOK: TikTokAdapter,
        Platform.YOUTUBE: YouTubeAdapter,
        Platform.INSTAGRAM: InstagramAdapter,
    }
    adapter_cls = adapters.get(platform)
    if not adapter_cls:
        raise ValueError(f"Unsupported platform: {platform}")
    return adapter_cls()


def generate_oauth_state(user_id: uuid.UUID, platform: Platform) -> str:
    raw = f"{user_id}:{platform.value}:{secrets.token_hex(16)}"
    return hashlib.sha256(raw.encode()).hexdigest()
