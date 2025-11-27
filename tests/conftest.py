import json
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).parent / "fixtures"


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


@pytest.fixture
def openai_generation_request(fixtures_dir):
    return _load_json(fixtures_dir / "openai_generation_request.json")


@pytest.fixture
def openai_generation_response(fixtures_dir):
    return _load_json(fixtures_dir / "openai_generation_response.json")


@pytest.fixture
def gemini_generate_content_request(fixtures_dir):
    return _load_json(fixtures_dir / "gemini_generate_content_request.json")


@pytest.fixture
def gemini_generate_content_response(fixtures_dir):
    return _load_json(fixtures_dir / "gemini_generate_content_response.json")


@pytest.fixture
def openai_config():
    return {
        "provider": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": "test-key",
        "timeout_seconds": 30,
    }


@pytest.fixture
def gemini_config():
    return {
        "provider": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key": "gemini-key",
        "timeout_seconds": 30,
    }
