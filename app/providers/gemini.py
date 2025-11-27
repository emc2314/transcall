import logging
from typing import Dict

import httpx

from app.exceptions import (
    UpstreamConnectionError,
    UpstreamResponseError,
    UpstreamTimeoutError,
)
from app.mappers import RequestMapper, ResponseMapper
from app.providers.base import Provider
from app.schemas import UnifiedImageRequest, UnifiedImageResponse
from app.logging_utils import log_debug_payload

logger = logging.getLogger("GeminiProvider")


class GeminiProvider(Provider):
    async def generate(self, req: UnifiedImageRequest) -> UnifiedImageResponse:
        config = self.config
        upstream_model = config.get("model") or req.target_model

        url = f"{config['base_url']}/models/{upstream_model}:generateContent"
        params_qs = {"key": config["api_key"]}
        headers = {"Content-Type": "application/json"}
        payload = RequestMapper.unified_to_gemini_payload(req)

        log_debug_payload(
            logger,
            f"[UPSTREAM REQUEST][gemini] POST {url}",
            headers=headers,
            body={"params": params_qs, "json": payload},
        )

        timeout = httpx.Timeout(config.get("timeout_seconds", 120))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url, params=params_qs, json=payload, headers=headers
                )
        except httpx.TimeoutException:
            raise UpstreamTimeoutError("gemini", url)
        except httpx.RequestError as e:
            raise UpstreamConnectionError("gemini", str(e), url)

        log_debug_payload(
            logger,
            f"[UPSTREAM RESPONSE][gemini] {resp.status_code} {url}",
            headers=resp.headers,
            body=resp.text,
        )

        if resp.status_code >= 400:
            raise UpstreamResponseError(
                "gemini", resp.status_code, resp.text, payload=resp.text
            )

        return ResponseMapper.gemini_to_unified(resp.json())
