from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    APP_NAME: str = "Video Clip Generator"
    DEBUG: bool = False

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/video_clips"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # OpenAI
    OPENAI_API_KEY: str = ""

    # Storage
    UPLOAD_DIR: str = "./storage/uploads"
    OUTPUT_DIR: str = "./storage/outputs"
    TEMP_DIR: str = "./storage/temp"

    # Processing defaults
    DEFAULT_CLIP_COUNT: int = 3
    DEFAULT_CLIP_LENGTH: int = 30
    MAX_CLIP_COUNT: int = 3
    MAX_FILE_SIZE_MB: int = 500

    # Task settings
    TASK_MAX_RETRIES: int = 3
    TASK_RETRY_DELAY: int = 60  # seconds

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
