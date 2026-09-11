from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    app_env: str = Field(default="development")
    log_level: str = Field(default="INFO")

    llm_provider: str = Field(default="ollama")
    llm_model: str = Field(default="nemotron-3-super:cloud")

    ollama_host: str = Field(
        default="http://localhost:11434"
    )

    request_timeout_seconds: int = Field(
        default=30,
        ge=1,
    )

    max_retries: int = Field(
        default=2,
        ge=0,
    )

    max_pages_per_domain: int = Field(
        default=6,
        ge=1,
    )

    max_concurrent_domains: int = Field(
        default=3,
        ge=1,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached application settings instance."""

    return Settings()