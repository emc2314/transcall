import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_FILE = Path(os.getenv("TRANSCALL_CONFIG_PATH", "config.json"))
logger = logging.getLogger("Config")


class ModelConfig(BaseModel):
    provider: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    deployment: Optional[str] = None
    api_version: Optional[str] = None
    model: Optional[str] = None
    timeout_seconds: Optional[int] = 120
    credentials_env: Optional[str] = None
    credentials_json: Optional[str] = None
    project_id: Optional[str] = None
    location: Optional[str] = None
    api_endpoint: Optional[str] = None

    model_config = SettingsConfigDict(extra="allow")


class ServiceSettings(BaseSettings):
    client_api_key: Optional[str] = None
    client_api_key_env: Optional[str] = None
    models: Dict[str, ModelConfig] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="TRANSCALL_",
        env_nested_delimiter="__",
        extra="allow",
    )

    @classmethod
    def load_from_file(cls, path: Path = CONFIG_FILE) -> "ServiceSettings":
        data: Dict = {}
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as fp:
                    data = json.load(fp)
            else:
                logger.warning(
                    "Config file %s not found; defaulting to empty config", path
                )
        except Exception as exc:
            logger.error("Failed to load config file %s: %s", path, exc)
        return cls(**data)


class ConfigManager:
    def __init__(self, settings: Optional[ServiceSettings] = None):
        self.settings = settings or ServiceSettings.load_from_file()

    @classmethod
    def load(cls, path: Path = CONFIG_FILE) -> "ConfigManager":
        settings = ServiceSettings.load_from_file(path)
        return cls(settings)

    def get_model_config(self, model_name: str) -> Optional[Dict]:
        model = self.settings.models.get(model_name)
        if not model:
            return None

        resolved = model.model_copy(deep=True)

        if resolved.api_key_env:
            env_val = os.getenv(resolved.api_key_env)
            if env_val:
                resolved.api_key = env_val.strip()

        if resolved.credentials_env:
            creds_val = os.getenv(resolved.credentials_env)
            if creds_val:
                resolved.credentials_json = creds_val

        return resolved.model_dump()

    def get_client_api_key(self) -> Optional[str]:
        key = self.settings.client_api_key
        if self.settings.client_api_key_env:
            env_val = os.getenv(self.settings.client_api_key_env)
            if env_val:
                key = env_val

        return key.strip() if isinstance(key, str) and key.strip() else None
