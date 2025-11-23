import logging
from typing import Dict, Any, List
import httpx
from fastapi import HTTPException
from app.schemas import UnifiedImageRequest, UnifiedImageResponse
from app.config import ConfigManager
from app.mappers import RequestMapper, ResponseMapper
from app.logging_utils import log_debug_payload, format_binary_content
from app.auth import GoogleAuthManager

logger = logging.getLogger("Service")


class ImageGenerationService:

    @staticmethod
    async def process_request(req: UnifiedImageRequest) -> UnifiedImageResponse:
        # 1. Get Provider Config
        config = ConfigManager.get_model_config(req.target_model)
        if not config:
            raise HTTPException(
                status_code=400, detail=f"Model '{req.target_model}' not supported."
            )

        provider = config.get("provider")

        if provider == "openai":
            return await ImageGenerationService._call_openai(req, config)
        elif provider == "gemini":
            return await ImageGenerationService._call_gemini(req, config)
        elif provider == "vertexai":
            return await ImageGenerationService._call_vertexai(req, config)
        else:
            raise HTTPException(
                status_code=500, detail=f"Unknown provider '{provider}'"
            )

    @staticmethod
    async def _call_openai(
        req: UnifiedImageRequest, config: Dict[str, Any]
    ) -> UnifiedImageResponse:
        url_base = config["base_url"].rstrip("/")
        api_key = config["api_key"]
        deployment = config.get("deployment")
        api_version = config.get("api_version")

        if (deployment and not api_version) or (api_version and not deployment):
            raise HTTPException(
                status_code=500,
                detail="Azure OpenAI configuration requires both 'deployment' and 'api_version'.",
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

        # Determine upstream model name (allows aliasing in config)
        upstream_model = config.get("model") or req.target_model

        # Determine endpoint based on content (presence of input images)
        timeout_seconds = config.get("timeout_seconds", 120)
        timeout = httpx.Timeout(timeout_seconds)

        if req.input_image_bytes_list:
            # Edits Endpoint
            url = f"{images_base}/edits"

            # Construct Multipart for ALL images and optional mask
            files_list_for_httpx = []
            field_names = req.input_image_field_names or []
            files_log: List[Dict[str, Any]] = []
            if req.input_image_bytes_list and req.input_image_mime_list:
                for i, (img_bytes, img_mime) in enumerate(
                    zip(req.input_image_bytes_list, req.input_image_mime_list)
                ):
                    field_name = (
                        field_names[i]
                        if i < len(field_names) and field_names[i]
                        else "image"
                    )
                    filename = f"image_{i}.png"
                    files_list_for_httpx.append(
                        (field_name, (filename, img_bytes, img_mime or "image/png"))
                    )
                    files_log.append(
                        {
                            "field": field_name,
                            "filename": filename,
                            "size_bytes": len(img_bytes),
                            "content": format_binary_content(img_bytes),
                        }
                    )

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

            # Data params (convert to string for multipart form data)
            payload = RequestMapper.unified_to_openai_payload(req)
            # Override model with the configured upstream name
            payload["model"] = upstream_model
            data = {k: str(v) for k, v in payload.items() if v is not None}

            log_debug_payload(
                logger,
                f"[UPSTREAM REQUEST][openai] POST {url}",
                headers=headers,
                body={"params": params, "form_data": data, "files": files_log},
            )

            logger.info(f"Calling OpenAI Edits: {url}")
            timeout = httpx.Timeout(config.get("timeout_seconds", 120))
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        url,
                        headers=headers,
                        params=params,
                        data=data,
                        files=files_list_for_httpx,
                    )
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=504, detail="Upstream service timed out"
                )
            except httpx.RequestError as e:
                raise HTTPException(
                    status_code=502, detail=f"Upstream connection error: {str(e)}"
                )

        else:
            # Generations Endpoint
            url = f"{images_base}/generations"
            payload = RequestMapper.unified_to_openai_payload(req)
            # Override model with the configured upstream name
            payload["model"] = upstream_model

            log_debug_payload(
                logger,
                f"[UPSTREAM REQUEST][openai] POST {url}",
                headers=headers,
                body={"params": params, "json": payload},
            )

            logger.info(f"Calling OpenAI Generations: {url}")
            timeout = httpx.Timeout(config.get("timeout_seconds", 120))
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    resp = await client.post(
                        url,
                        headers=headers,
                        params=params,
                        json=payload,
                    )
            except httpx.TimeoutException:
                raise HTTPException(
                    status_code=504, detail="Upstream service timed out"
                )
            except httpx.RequestError as e:
                raise HTTPException(
                    status_code=502, detail=f"Upstream connection error: {str(e)}"
                )

        log_debug_payload(
            logger,
            f"[UPSTREAM RESPONSE][openai] {resp.status_code} {url}",
            headers=resp.headers,
            body=resp.text,
        )

        if resp.status_code >= 400:
            logger.error(f"OpenAI Error: {resp.text}")
            raise HTTPException(
                status_code=resp.status_code, detail=f"Upstream Error: {resp.text}"
            )

        return ResponseMapper.openai_to_unified(resp.json())

    @staticmethod
    async def _call_gemini(
        req: UnifiedImageRequest, config: Dict[str, Any]
    ) -> UnifiedImageResponse:
        # Allow aliasing via config
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

        logger.info(f"Calling Gemini: {url}")
        timeout = httpx.Timeout(config.get("timeout_seconds", 120))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    url, params=params_qs, json=payload, headers=headers
                )
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Upstream service timed out")
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502, detail=f"Upstream connection error: {str(e)}"
            )

        log_debug_payload(
            logger,
            f"[UPSTREAM RESPONSE][gemini] {resp.status_code} {url}",
            headers=resp.headers,
            body=resp.text,
        )

        if resp.status_code >= 400:
            logger.error(f"Gemini Error: {resp.text}")
            raise HTTPException(
                status_code=resp.status_code, detail=f"Upstream Error: {resp.text}"
            )

        return ResponseMapper.gemini_to_unified(resp.json())

    @staticmethod
    async def _call_vertexai(
        req: UnifiedImageRequest, config: Dict[str, Any]
    ) -> UnifiedImageResponse:
        # 1. Get Credentials and Token
        creds_json = config.get("credentials_json")
        if not creds_json:
            raise HTTPException(
                status_code=500,
                detail="Missing 'credentials_json' or configured env var for Vertex AI model.",
            )

        try:
            access_token = GoogleAuthManager.get_access_token(creds_json)
        except Exception as e:
            logger.error(f"Failed to get Vertex AI access token: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to authenticate with Vertex AI."
            )

        # 2. Construct URL
        # Format: https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent
        project_id = config.get("project_id")
        location = config.get("location")

        if not project_id or not location:
            raise HTTPException(
                status_code=500,
                detail="Vertex AI config requires 'project_id' and 'location'.",
            )

        api_endpoint = config.get("api_endpoint")
        if not api_endpoint:
            api_endpoint = f"aiplatform.googleapis.com"

        # Use target_model from request, but allow override via config (e.g. "gemini-pro-vision")
        upstream_model = config.get("model") or req.target_model

        # The user example used :generateContent, so we hardcode that for now as this service is for generation.
        method = "generateContent"

        url = f"https://{api_endpoint}/v1/projects/{project_id}/locations/{location}/publishers/google/models/{upstream_model}:{method}"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }

        # 3. Construct Payload (Same as Gemini)
        payload = RequestMapper.unified_to_gemini_payload(req)

        log_debug_payload(
            logger,
            f"[UPSTREAM REQUEST][vertexai] POST {url}",
            headers=headers,
            body={"json": payload},
        )

        logger.info(f"Calling Vertex AI: {url}")
        timeout = httpx.Timeout(config.get("timeout_seconds", 120))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=payload, headers=headers)
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Upstream service timed out")
        except httpx.RequestError as e:
            raise HTTPException(
                status_code=502, detail=f"Upstream connection error: {str(e)}"
            )

        log_debug_payload(
            logger,
            f"[UPSTREAM RESPONSE][vertexai] {resp.status_code} {url}",
            headers=resp.headers,
            body=resp.text,
        )

        if resp.status_code >= 400:
            logger.error(f"Vertex AI Error: {resp.text}")
            raise HTTPException(
                status_code=resp.status_code, detail=f"Upstream Error: {resp.text}"
            )

        # 4. Map Response (Same as Gemini)
        return ResponseMapper.gemini_to_unified(resp.json())
