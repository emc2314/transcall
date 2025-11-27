import json

import pytest

from app import exceptions as exc
from app.error_handlers import transcall_exception_handler


@pytest.mark.asyncio
async def test_upstream_rate_limit_hint():
    err = exc.UpstreamResponseError("openai", 429, "Rate limit exceeded")
    response = await transcall_exception_handler(None, err)
    assert response.status_code == 429
    body = json.loads(response.body)
    assert body["error"] == "UPSTREAM_RATE_LIMIT"


@pytest.mark.asyncio
async def test_upstream_safety_hint():
    err = exc.UpstreamResponseError("openai", 400, "Safety violation detected")
    response = await transcall_exception_handler(None, err)
    assert response.status_code == 400
    body = json.loads(response.body)
    assert body["error"] == "UPSTREAM_SAFETY_BLOCKED"


@pytest.mark.asyncio
async def test_timeout_error_passthrough():
    err = exc.UpstreamTimeoutError("openai", "https://example.com")
    response = await transcall_exception_handler(None, err)
    assert response.status_code == 504
    body = json.loads(response.body)
    assert body["error"] == "UPSTREAM_TIMEOUT"
