"""Application configuration.

Settings are loaded from environment variables and an optional ``.env``
file using pydantic-settings, so configuration is validated and typed
rather than read ad-hoc from ``os.environ``.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from extractkit.exceptions import ConfigError


class Settings(BaseSettings):
    """Runtime configuration for extractkit.

    Attributes:
        openai_api_key: Secret key for the OpenAI API.
        model: Chat model used for extraction.
        rpm_limit: Soft cap on requests per minute to avoid rate limits.
        request_timeout: Per-request timeout in seconds.
        max_retries: How many times to retry a failed API call.
    """

    model_config = SettingsConfigDict(
        env_prefix="EXTRACTKIT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # The API key keeps its conventional name; the alias bypasses the
    # EXTRACTKIT_ prefix so a standard OPENAI_API_KEY env var is picked up.
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    model: str = Field(default="gpt-4o-mini")
    rpm_limit: int = Field(default=60, ge=1)
    request_timeout: float = Field(default=120.0, gt=0)
    max_retries: int = Field(default=5, ge=0)

    def validate_ready(self) -> None:
        """Check that the settings are usable for a real run.

        Raises:
            ConfigError: If the OpenAI API key is missing.
        """
        if not self.openai_api_key:
            raise ConfigError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env "
                "and add your key, or export it in your shell."
            )


def load_settings() -> Settings:
    """Load and return application settings.

    Returns:
        A populated ``Settings`` instance.
    """
    return Settings()
