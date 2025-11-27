import logging
from typing import Dict

import httpx

from app.auth import GoogleAuthManager
from app.exceptions import (
    InvalidModelConfigError,
    ProviderAuthenticationError,
    UpstreamConnectionError,
    UpstreamResponseError,
    UpstreamTimeoutError,
)
from app.mappers import RequestMapper, ResponseMapper
from app.providers.base import Provider
from app.schemas import UnifiedImageRequest, UnifiedImageResponse
from app.logging_utils import log_debug_payload

logger = logging.getLogger("VertexProvider")


class VertexAIProvider(Provider):
    async def generate(self, req: UnifiedImageRequest) -> UnifiedImageResponse:
        config = self.config
        creds_json = config.get("credentials_json")
        if not creds_json:
            raise InvalidModelConfigError(
                "Missing 'credentials_json' or configured env var for Vertex AI model.",
                meta={"model": req.target_model},
            )

        try:
            access_token = GoogleAuthManager.get_access_token(creds_json)
        except Exception as e:
            logger.error(f"Failed to get Vertex AI access token: {e}")
            raise ProviderAuthenticationError(
                "Failed to authenticate with Vertex AI.",
                meta={"provider": "vertexai"},
            ) from e

        project_id = config.get("project_id")
        location = config.get("location")
        if not project_id or not location:
            raise InvalidModelConfigError(
                "Vertex AI config requires 'project_id' and 'location'.",
                meta={"model": req.target_model},
            )

        api_endpoint = config.get("api_endpoint") or "aiplatform.googleapis.com"
        upstream_model = config.get("model") or req.target_model
        url = f"https://{api_endpoint}/v1/projects/{project_id}/locations/{location}/publishers/google/models/{upstream_model}:generateContent"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        payload = RequestMapper.unified_to_gemini_payload(req)

        log_debug_payload(
            logger,
            f"[UPSTREAM REQUEST][vertexai] POST {url}",
            headers=headers,
            body={"json": payload},
        )

        timeout = httpx.Timeout(config.get("timeout_seconds", 120))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException:
            raise UpstreamTimeoutError("vertexai", url)
        except httpx.RequestError as e:
            raise UpstreamConnectionError("vertexai", str(e), url)

        log_debug_payload(
            logger,
            f"[UPSTREAM RESPONSE][vertexai] {resp.status_code} {url}",
            headers=resp.headers,
            body=resp.text,
        )

        if resp.status_code >= 400:
            raise UpstreamResponseError(
                "vertexai", resp.status_code, resp.text, payload=resp.text
            )

        return ResponseMapper.gemini_to_unified(resp.json())
