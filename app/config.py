"""Application settings and environment configuration."""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Groq API Keys
    GROQ_API_KEY_1: Optional[str] = None
    GROQ_API_KEY_2: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None  # Single-key fallback

    # Groq Vision Model (Active Groq vision model)
    GROQ_VISION_MODEL: str = "qwen/qwen3.8-27b"

    # Timeouts (in seconds)
    IMAGE_DOWNLOAD_TIMEOUT_SECONDS: float = 12.0
    GROQ_REQUEST_TIMEOUT_SECONDS: float = 35.0

    # Retry Settings
    MAX_RETRIES_PER_KEY: int = 2
    RETRY_BACKOFF_FACTOR: float = 1.5

    # Max image dimensions to optimize vision request payload
    MAX_IMAGE_DIMENSION: int = 1280
    MAX_IMAGE_SIZE_BYTES: int = 3 * 1024 * 1024  # 3MB limit before compression

    @property
    def groq_keys(self) -> List[str]:
        """Returns ordered list of available non-empty Groq API keys."""
        keys = []
        if self.GROQ_API_KEY_1 and self.GROQ_API_KEY_1.strip():
            keys.append(self.GROQ_API_KEY_1.strip())
        if self.GROQ_API_KEY_2 and self.GROQ_API_KEY_2.strip():
            if self.GROQ_API_KEY_2.strip() not in keys:
                keys.append(self.GROQ_API_KEY_2.strip())
        # Fallback to single GROQ_API_KEY if neither key 1 nor key 2 is provided
        if not keys and self.GROQ_API_KEY and self.GROQ_API_KEY.strip():
            keys.append(self.GROQ_API_KEY.strip())
        return keys


settings = Settings()
