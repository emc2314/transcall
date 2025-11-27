import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from main import app, get_image_service, verify_api_key
from app.schemas import UnifiedImageResponse, UnifiedImageResponseItem, UnifiedImageRequest, UnifiedMessage, UnifiedContentPart
from app.mappers import RequestMapper, ResponseMapper

# TestClient setup
client = TestClient(app)

@pytest.fixture
def mock_image_service():
    service_mock = MagicMock()
    service_mock.process_request = AsyncMock()
    return service_mock

@pytest.fixture(autouse=True)
def mock_config_manager():
    with patch("main.config_manager") as mock_cfg:
        def side_effect(model_name):
            return {
                "provider": "gemini",
                "api_key": "gemini-dummy-key",
                "base_url": "https://generativelanguage.googleapis.com/v1beta",
            }
        mock_cfg.get_model_config.side_effect = side_effect
        yield mock_cfg

@pytest.fixture(autouse=True)
def override_dependencies(mock_image_service):
    app.dependency_overrides[get_image_service] = lambda: mock_image_service
    app.dependency_overrides[verify_api_key] = lambda: None
    yield
    app.dependency_overrides = {}

DUMMY_RED_DOT_PNG_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mNk+A8AAQUhBG9OmZ0AAAAASUVORK5CYII="
DUMMY_RED_DOT_PNG_BYTES = base64.b64decode(DUMMY_RED_DOT_PNG_BASE64)

def test_gemini_multiturn_conversation(mock_image_service):
    # --- 第一轮模拟: 模型返回图片 --- 
    model_first_response_unified = UnifiedImageResponse(
        created=1700000000,
        images=[
            UnifiedImageResponseItem(
                mime_type="image/png",
                b64_json=DUMMY_RED_DOT_PNG_BASE64,
                index=0 
            ),
            UnifiedImageResponseItem(
                revised_prompt="Here is the cat you requested.",
                index=0 
            ),
        ],
        usage_source="gemini",
        prompt_feedback={"blockReason": None},
        model_version="gemini-pro-vision-001",
    )

    model_first_response_gemini_payload = ResponseMapper.unified_to_gemini_format(
        model_first_response_unified
    )
    model_response_contents = model_first_response_gemini_payload["candidates"][0]["content"]
    
    # --- 第二轮模拟: 用户带历史记录发送修改意见 --- 
    user_second_request_payload = {
        "contents": [
            {"role": "user", "parts": [{"text": "画一只猫"}]},
            {"role": "model", "parts": model_response_contents["parts"]},
            {"role": "user", "parts": [{"text": "把猫的颜色改成蓝色"}]},
        ],
        "generationConfig": {"candidateCount": 1},
    }

    model_second_response_unified = UnifiedImageResponse(
        created=1700000001,
        images=[
            UnifiedImageResponseItem(
                mime_type="image/png",
                b64_json=DUMMY_RED_DOT_PNG_BASE64,
                index=0 
            ),
            UnifiedImageResponseItem(
                revised_prompt="Okay, here's your cat in blue.", 
                index=0 
            ),
        ],
        usage_source="gemini",
        prompt_feedback={"blockReason": None},
        model_version="gemini-pro-vision-001",
    )
    mock_image_service.process_request.return_value = model_second_response_unified

    # --- 发起请求 --- 
    response = client.post(
        "/v1beta/models/gemini-pro-vision:generateContent",
        json=user_second_request_payload
    )

    assert response.status_code == 200, f"Failed with {response.status_code}: {response.text}"
    resp_json = response.json()
    
    assert "candidates" in resp_json
    assert len(resp_json["candidates"]) == 1
    candidate_parts = resp_json["candidates"][0]["content"]["parts"]
    assert len(candidate_parts) == 2 

    # 验证 Service 调用
    mock_image_service.process_request.assert_called_once()
    called_req: UnifiedImageRequest = mock_image_service.process_request.call_args[0][0]
    
    # 验证 Prompt Property (计算属性)
    expected_flat_prompt = "画一只猫 Here is the cat you requested. 把猫的颜色改成蓝色"
    assert called_req.prompt == expected_flat_prompt
    
    # 验证 messages 结构
    assert len(called_req.messages) == 3
    
    assert called_req.messages[0].role == "user"
    assert called_req.messages[0].parts[0].text == "画一只猫"
    
    assert called_req.messages[1].role == "model"
    
    model_parts = called_req.messages[1].parts
    text_part = next((p for p in model_parts if p.text), None)
    image_part = next((p for p in model_parts if p.image_data), None)
    
    assert text_part is not None
    assert text_part.text == "Here is the cat you requested."
    
    assert image_part is not None
    # UnifiedContentPart stores raw bytes and separate mime type
    assert image_part.image_mime_type == "image/png"
    assert image_part.image_data == DUMMY_RED_DOT_PNG_BYTES
    
    assert called_req.messages[2].role == "user"
    assert called_req.messages[2].parts[0].text == "把猫的颜色改成蓝色"