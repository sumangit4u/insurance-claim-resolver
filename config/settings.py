"""Application settings via pydantic-settings.

Pattern: same as Inventra's config/settings.py but adapted for insurance claims.
All values come from environment variables or .env file.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. Reads from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Google AI ---
    google_api_key: str = Field(default="", description="Gemini API key")

    # --- GCP (not yet available) ---
    gcp_project_id: str = Field(default="", description="GCP project ID")
    gcp_region: str = Field(default="us-central1", description="GCP region")
    vertex_search_datastore_id: str = Field(
        default="", description="Vertex AI Search datastore ID"
    )
    firestore_database_id: str = Field(
        default="(default)", description="Firestore database ID"
    )

    # --- Application ---
    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # --- Model names (override via env if needed) ---
    gemini_model: str = Field(default="gemini-2.5-flash")
    gemini_pro_model: str = Field(default="gemini-2.5-pro")
    embedding_model: str = Field(default="models/gemini-embedding-001")

    @property
    def is_production(self) -> bool:
        """True when running in Cloud Run / production."""
        return self.environment.lower() == "production"

    @property
    def gcp_ready(self) -> bool:
        """True when GCP credentials and project are configured."""
        return bool(self.gcp_project_id and self.gcp_project_id != "your_project_id")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached Settings singleton."""
    return Settings()
