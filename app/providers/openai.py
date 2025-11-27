import logging
from typing import Any, Dict, List, Optional

import httpx

from app.exceptions import (
    InvalidModelConfigError,
    UpstreamConnectionError,
    UpstreamResponseError,
    UpstreamTimeoutError,
)
from app.logging_utils import format_binary_content, log_debug_payload
from app.mappers import RequestMapper, ResponseMapper
from app.providers.base import Provider
from app.schemas import UnifiedImageRequest, UnifiedImageResponse

logger = logging.getLogger("OpenAIProvider")


class OpenAIProvider(Provider):
    async def generate(self, req: UnifiedImageRequest) -> UnifiedImageResponse:
        config = self.config
        url_base = config["base_url"].rstrip("/")
        api_key = config["api_key"]
        deployment = config.get("deployment")
        api_version = config.get("api_version")

        if (deployment and not api_version) or (api_version and not deployment):
            raise InvalidModelConfigError(
                "Azure OpenAI configuration requires both 'deployment' and 'api_version'.",
                meta={"model": req.target_model},
            )

        is_azure = bool(deployment and api_version)
        if is_azure:
            headers = {"api-key": api_key}
            images_base = f"{url_base}/openai/deployments/{deployment}/images"
            params = {"api-version": api_version}
        else:
            headers = {"Authorization": f"Bearer {api_key}"}
            images_base = f"{url_base}/images"
            params = None

        upstream_model = config.get("model") or req.target_model
        timeout = httpx.Timeout(config.get("timeout_seconds", 120))

        # Check if request contains images (Edits flow)
        has_images = False
        for msg in req.messages:
            for part in msg.parts:
                if part.image_data:
                    has_images = True
                    break
            if has_images:
                break

        if has_images:
            return await self._call_edits(
                req, headers, params, images_base, upstream_model, timeout
            )

        return await self._call_generations(
            req, headers, params, images_base, upstream_model, timeout
        )

    async def _call_edits(
        self,
        req: UnifiedImageRequest,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]],
        images_base: str,
        upstream_model: str,
        timeout: httpx.Timeout,
    ) -> UnifiedImageResponse:
        url = f"{images_base}/edits"
        payload = RequestMapper.unified_to_openai_payload(req)
        payload["model"] = upstream_model
        data = {k: str(v) for k, v in payload.items() if v is not None}

        files_list_for_httpx = []
        files_log: List[Dict[str, Any]] = []
        
        # Extract images from messages for multipart upload
        img_idx = 0
        for msg in req.messages:
            for part in msg.parts:
                if part.image_data:
                    field_name = "image" if img_idx == 0 else f"image[{img_idx}]" # Rough heuristic for field naming
                    # Note: OpenAI edits usually take ONE image and ONE mask. 
                    # If multiple images are present, we might need logic to map them to 'image' and 'mask' if mask not explicit.
                    # For now, let's assume first image is 'image'.
                    # If there are multiple, OpenAI API might complain or we map them sequentially.
                    # Let's stick to standard single image edit for safety unless we have strict multi-image logic.
                    
                    if img_idx > 0 and not req.mask_image_bytes:
                         # If we have a second image and no explicit mask, maybe the second image IS the mask?
                         # This is ambiguous. For strictness, we only send the first image as 'image'.
                         logger.warning("OpenAI Edits: Multiple input images detected but only one supported as base image. Ignoring extras.")
                         continue

                    filename = f"image_{img_idx}.png"
                    mime = part.image_mime_type or "image/png"
                    files_list_for_httpx.append(
                        ("image", (filename, part.image_data, mime))
                    )
                    files_log.append(
                        {
                            "field": "image",
                            "filename": filename,
                            "size_bytes": len(part.image_data),
                            "content": format_binary_content(part.image_data),
                        }
                    )
                    img_idx += 1

        if req.mask_image_bytes:
            files_list_for_httpx.append(
                (
                    "mask",
                    (
                        "mask.png",
                        req.mask_image_bytes,
                        req.mask_image_mime or "image/png",
                    ),
                )
            )
            files_log.append(
                {
                    "field": "mask",
                    "filename": "mask.png",
                    "size_bytes": len(req.mask_image_bytes),
                    "content": format_binary_content(req.mask_image_bytes),
                }
            )

        log_debug_payload(
            logger,
            f"[UPSTREAM REQUEST][openai] POST {url}",
            headers=headers,
            body={"params": params, "form_data": data, "files": files_log},
        )

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url, headers=headers, params=params, data=data, files=files_list_for_httpx
                )
        except httpx.TimeoutException:
            raise UpstreamTimeoutError("openai", url)
        except httpx.RequestError as e:
            raise UpstreamConnectionError("openai", str(e), url)

        log_debug_payload(
            logger,
            f"[UPSTREAM RESPONSE][openai] {resp.status_code} {url}",
            headers=resp.headers,
            body=resp.text,
        )

        if resp.status_code >= 400:
            raise UpstreamResponseError("openai", resp.status_code, resp.text, payload=resp.text)

        return ResponseMapper.openai_to_unified(resp.json())

    async def _call_generations(
        self,
        req: UnifiedImageRequest,
        headers: Dict[str, str],
        params: Optional[Dict[str, Any]],
        images_base: str,
        upstream_model: str,
        timeout: httpx.Timeout,
    ) -> UnifiedImageResponse:
        url = f"{images_base}/generations"
        payload = RequestMapper.unified_to_openai_payload(req)
        payload["model"] = upstream_model

        log_debug_payload(
            logger,
            f"[UPSTREAM REQUEST][openai] POST {url}",
            headers=headers,
            body={"params": params, "json": payload},
        )

        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url, headers=headers, params=params, json=payload
                )
        except httpx.TimeoutException:
            raise UpstreamTimeoutError("openai", url)
        except httpx.RequestError as e:
            raise UpstreamConnectionError("openai", str(e), url)

        log_debug_payload(
            logger,
            f"[UPSTREAM RESPONSE][openai] {resp.status_code} {url}",
            headers=resp.headers,
            body=resp.text,
        )

        if resp.status_code >= 400:
            raise UpstreamResponseError("openai", resp.status_code, resp.text, payload=resp.text)

        return ResponseMapper.openai_to_unified(resp.json())