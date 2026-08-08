"""
Centralised application configuration.

All settings are read from environment variables (or a local .env file),
so the same image runs unchanged in development, CI, and production.
Never hardcode credentials or environment-specific values elsewhere.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Application ---
    PROJECT_NAME: str = "VARUNA AI"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"  # development | staging | production
    API_V1_PREFIX: str = "/api/v1"

    # --- CORS (comma-separated list of allowed frontend origins) ---
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # --- Database (PostgreSQL + PostGIS) ---
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "varuna"
    POSTGRES_PASSWORD: str = "varuna"
    POSTGRES_DB: str = "varuna_ai"

    # --- Model artifacts (populated in later phases) ---
    # Directory where trained model weights/artifacts are stored and loaded from.
    MODEL_ARTIFACT_DIR: str = "../ai_models/artifacts"

    # --- External data source credentials (Phase 2) ---
    # MOSDAC and NASA Earthdata require registered accounts; keys live in .env only.
    MOSDAC_API_USER: str | None = None
    MOSDAC_API_PASSWORD: str | None = None
    NASA_EARTHDATA_TOKEN: str | None = None

    @property
    def cors_origins(self) -> list[str]:
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",") if origin.strip()]

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
