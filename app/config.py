import json
import logging
import os
from typing import Dict, Optional, Any

CONFIG_FILE = "config.json"
logger = logging.getLogger("ImageService")


class ConfigManager:
    _instance = None
    _config: Dict[str, Any] = {}
    _client_api_key: Optional[str] = None

    @classmethod
    def load(cls):
        try:
            with open(CONFIG_FILE, "r") as f:
                cls._config = json.load(f)
                api_key_value = cls._config.get("client_api_key")
                api_key_env = cls._config.get("client_api_key_env")
                resolved_key = None
                if isinstance(api_key_value, (str, int, float)):
                    resolved_key = str(api_key_value).strip()
                if api_key_env:
                    env_val = os.environ.get(str(api_key_env))
                    if env_val:
                        resolved_key = env_val.strip()
                cls._client_api_key = resolved_key or None
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            cls._config = {"models": {}}
            cls._client_api_key = None

    @classmethod
    def get_model_config(cls, model_name: str) -> Optional[Dict]:
        if not cls._config:
            cls.load()

        model_conf = cls._config.get("models", {}).get(model_name)
        if not model_conf:
            return None

        # Resolve API Key from environment variable
        # We create a shallow copy to avoid mutating the cached config permanently with the resolved key
        # (though mutating it might be fine, but clean separation is safer)
        resolved_conf = model_conf.copy()
        env_var_name = resolved_conf.get("api_key_env")
        if env_var_name:
            api_key = os.environ.get(env_var_name)
            if not api_key:
                logger.warning(
                    f"Environment variable '{env_var_name}' not found for model '{model_name}'"
                )
            resolved_conf["api_key"] = api_key
        else:
            # Fallback: if 'api_key' was directly in json (legacy support), keep it.
            # If neither exists, api_key will be None or whatever was in json.
            pass

        # Resolve Credentials JSON from environment variable (for Vertex AI)
        cred_env_name = resolved_conf.get("credentials_env")
        if cred_env_name:
            creds = os.environ.get(cred_env_name)
            if not creds:
                logger.warning(
                    f"Environment variable '{cred_env_name}' not found for model '{model_name}'"
                )
            resolved_conf["credentials_json"] = creds

        return resolved_conf

    @classmethod
    def get_client_api_key(cls) -> Optional[str]:
        if not cls._config:
            cls.load()
        return cls._client_api_key


# Initialize on module import
ConfigManager.load()
