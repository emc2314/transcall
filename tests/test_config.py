import json
from pathlib import Path
from typing import Any, Dict, Callable

import pytest

from app.config import ConfigManager


@pytest.fixture()
def mock_config_data() -> Dict[str, Any]:
    return {
        "client_api_key": "default-client-key",
        "models": {
            "gpt-test": {
                "provider": "openai",
                "base_url": "https://api.openai.com/v1",
                "api_key": "default-key",
                "api_key_env": "GPT_TEST_KEY",
            },
            "gemini-test": {
                "provider": "gemini",
                "base_url": "https://gemini.google.com",
                "api_key": "gemini-key",
                "credentials_env": "GOOGLE_CREDS_JSON",
            },
        },
    }


@pytest.fixture()
def config_file(tmp_path, mock_config_data) -> Path:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(mock_config_data))
    return config_path


@pytest.fixture()
def load_manager(monkeypatch) -> Callable[[Path], ConfigManager]:
    def _loader(path: Path) -> ConfigManager:
        monkeypatch.setenv("TRANSCALL_CONFIG_PATH", str(path))
        return ConfigManager.load(path)

    return _loader


def test_load_from_file(config_file, load_manager):
    manager = load_manager(config_file)
    assert manager.settings.client_api_key == "default-client-key"
    assert manager.settings.models["gpt-test"].provider == "openai"


def test_get_model_config_basic(config_file, load_manager):
    manager = load_manager(config_file)
    config = manager.get_model_config("gpt-test")
    assert config is not None
    assert config["provider"] == "openai"
    assert config["api_key"] == "default-key"


def test_model_config_is_copy(config_file, load_manager):
    manager = load_manager(config_file)
    config = manager.get_model_config("gpt-test")
    config["api_key"] = "mutated"

    fresh = manager.get_model_config("gpt-test")
    assert fresh["api_key"] == "default-key"


def test_get_model_config_env_override(config_file, load_manager, monkeypatch):
    monkeypatch.setenv("GPT_TEST_KEY", "env-overridden-key")
    manager = load_manager(config_file)

    config = manager.get_model_config("gpt-test")
    assert config["api_key"] == "env-overridden-key"


def test_get_model_config_credentials_override(
    config_file, load_manager, monkeypatch
):
    manager = load_manager(config_file)
    creds_json = '{"type": "service_account"}'
    monkeypatch.setenv("GOOGLE_CREDS_JSON", creds_json)

    config = manager.get_model_config("gemini-test")
    assert config["credentials_json"] == creds_json


def test_get_client_api_key(config_file, load_manager):
    manager = load_manager(config_file)
    assert manager.get_client_api_key() == "default-client-key"


def test_get_client_api_key_env_override(tmp_path, load_manager, monkeypatch):
    custom_path = tmp_path / "custom.json"
    custom_path.write_text(
        json.dumps(
            {"client_api_key": "default", "client_api_key_env": "CLIENT_KEY_ENV"}
        )
    )

    monkeypatch.setenv("CLIENT_KEY_ENV", "env-client-key")
    manager = load_manager(custom_path)
    assert manager.get_client_api_key() == "env-client-key"


def test_missing_config_file(tmp_path, load_manager):
    non_existent_path = tmp_path / "non_existent.json"
    manager = load_manager(non_existent_path)
    assert manager.settings.models == {}


def test_invalid_config_file(tmp_path, load_manager):
    broken_path = tmp_path / "broken.json"
    broken_path.write_text("{ invalid json")

    manager = load_manager(broken_path)
    assert manager.settings.models == {}


def test_get_non_existent_model(config_file, load_manager):
    manager = load_manager(config_file)
    assert manager.get_model_config("unknown-model") is None
