import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch
from main import app, get_image_service, verify_api_key
from app.schemas import UnifiedImageResponse, UnifiedImageResponseItem

# 说明：该文件依赖 FastAPI TestClient 运行在同步测试环境中。


@pytest.fixture
def mock_image_service():
    """
    创建一个 Mock 的 Service，避免真实调用 OpenAI/Google
    """
    service_mock = MagicMock()
    # 设置 process_request 为异步方法 (AsyncMock)
    service_mock.process_request = AsyncMock()
    return service_mock


@pytest.fixture(autouse=True)
def mock_config_manager():
    """
    Mock main.py 中的全局 config_manager。
    这对于绕过 main.py 中的 `if not config: raise 400` 检查至关重要。
    """
    with patch("main.config_manager") as mock_cfg:
        # 让 get_model_config 永远返回一个有效的配置字典
        # 这样 RequestMapper 也能正常工作
        def side_effect(model_name):
            provider = "gemini" if model_name.startswith("gemini") else "openai"
            config = {
                "provider": provider,
                "api_key": "sk-dummy",
                "base_url": "https://dummy.url",
            }
            if provider == "gemini":
                config["model"] = model_name
            return config

        mock_cfg.get_model_config.side_effect = side_effect
        yield mock_cfg


@pytest.fixture
def api_client(mock_config_manager, mock_image_service):
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def override_dependencies(mock_image_service):
    """
    自动应用依赖覆盖：
    1. 替换真实的 ImageGenerationService 为 Mock 对象
    2. 禁用 API Key 验证 (方便测试)
    """
    app.dependency_overrides[get_image_service] = lambda: mock_image_service
    app.dependency_overrides[verify_api_key] = lambda: None
    yield
    # 清理覆盖
    app.dependency_overrides = {}


def test_openai_generations_endpoint_structure(mock_image_service, api_client):
    """
    测试 /v1/images/generations 接口能否正确接收 OpenAI SDK 格式的请求，
    并返回符合 OpenAI 格式的响应。
    """
    # 1. 准备 Mock Service 的返回结果 (Unified 格式)
    mock_image_service.process_request.return_value = UnifiedImageResponse(
        created=1700000000,
        images=[
            UnifiedImageResponseItem(
                url="https://example.com/image1.png",
                revised_prompt="A revised prompt for image 1",
                mime_type="image/png",
            )
        ],
        usage_source="openai",
        usage={"prompt_tokens": 15, "total_tokens": 15},
    )

    # 2. 模拟 OpenAI SDK 发送的 Payload
    sdk_payload = {
        "model": "gpt-image-1",
        "prompt": "A cybernetic cat",
        "n": 1,
        "size": "1024x1024",
        "quality": "hd",
        "style": "vivid",
    }

    # 3. 发起请求
    response = api_client.post("/v1/images/generations", json=sdk_payload)

    # 4. 验证 Service 层是否被正确调用 (参数映射检查)
    assert (
        response.status_code == 200
    ), f"Expected 200, got {response.status_code}: {response.text}"

    # 检查 Mock 被调用时的参数
    mock_image_service.process_request.assert_called_once()
    # 获取调用时的参数对象 (UnifiedImageRequest)
    called_req = mock_image_service.process_request.call_args[0][0]

    # 验证 RequestMapper 是否正确工作
    assert called_req.prompt == "A cybernetic cat"
    assert called_req.target_model == "gpt-image-1"
    assert called_req.style == "vivid"
    assert called_req.quality == "hd"

    # 5. 验证响应格式 (ResponseMapper 检查)
    resp_json = response.json()

    # 必须包含 OpenAI SDK 期望的顶级字段
    assert "created" in resp_json
    assert "data" in resp_json
    assert len(resp_json["data"]) == 1

    item = resp_json["data"][0]
    assert item["url"] == "https://example.com/image1.png"
    assert item["revised_prompt"] == "A revised prompt for image 1"

    # 验证 usage 字段透传
    assert "usage" in resp_json
    assert resp_json["usage"]["prompt_tokens"] == 15


def test_gemini_endpoint_structure(mock_image_service, api_client):
    """
    测试 Gemini 风格的 Endpoint
    """
    mock_image_service.process_request.return_value = UnifiedImageResponse(
        created=1700000000,
        images=[
            UnifiedImageResponseItem(
                mime_type="image/png",
                b64_json="base64_encoded_data_here",
                finish_reason="STOP",
            )
        ],
        usage_source="gemini",
        usage={"totalTokenCount": 100},
    )

    gemini_payload = {
        "contents": [{"parts": [{"text": "Draw a robot"}]}],
        "generationConfig": {"candidateCount": 1},
    }

    response = api_client.post(
        "/v1beta/models/gemini-pro-vision:generateContent", json=gemini_payload
    )

    assert (
        response.status_code == 200
    ), f"Expected 200, got {response.status_code}: {response.text}"

    # 验证 Service 调用
    mock_image_service.process_request.assert_called_once()
    called_req = mock_image_service.process_request.call_args[0][0]
    assert called_req.prompt == "Draw a robot"
    assert called_req.target_model == "gemini-pro-vision"

    # 验证 Gemini 格式响应
    resp_json = response.json()
    assert "candidates" in resp_json
    assert (
        resp_json["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
        == "base64_encoded_data_here"
    )
