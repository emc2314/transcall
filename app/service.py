import logging
from typing import Dict, Any, List
import httpx
from fastapi import HTTPException
from app.schemas import UnifiedImageRequest, UnifiedImageResponse
from app.config import ConfigManager
from app.mappers import RequestMapper, ResponseMapper
from app.logging_utils import log_debug_payload, format_binary_content

logger = logging.getLogger("Service")

class ImageGenerationService:
    
    @staticmethod
    async def process_request(req: UnifiedImageRequest) -> UnifiedImageResponse:
        # 1. Get Provider Config
        config = ConfigManager.get_model_config(req.target_model)
        if not config:
            raise HTTPException(status_code=400, detail=f"Model '{req.target_model}' not supported.")
        
        provider = config.get("provider")
        
        if provider == "openai":
            return await ImageGenerationService._call_openai(req, config)
        elif provider == "gemini":
            return await ImageGenerationService._call_gemini(req, config)
        else:
            raise HTTPException(status_code=500, detail=f"Unknown provider '{provider}'")

    @staticmethod
    async def _call_openai(req: UnifiedImageRequest, config: Dict[str, Any]) -> UnifiedImageResponse:
        url_base = config["base_url"].rstrip("/")
        api_key = config["api_key"]
        deployment = config.get("deployment")
        api_version = config.get("api_version")

        if (deployment and not api_version) or (api_version and not deployment):
            raise HTTPException(
                status_code=500,
                detail="Azure OpenAI configuration requires both 'deployment' and 'api_version'."
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
                for i, (img_bytes, img_mime) in enumerate(zip(req.input_image_bytes_list, req.input_image_mime_list)):
                    field_name = field_names[i] if i < len(field_names) and field_names[i] else 'image'
                    filename = f'image_{i}.png'
                    files_list_for_httpx.append((field_name, (filename, img_bytes, img_mime or 'image/png')))
                    files_log.append({
                        "field": field_name,
                        "filename": filename,
                        "size_bytes": len(img_bytes),
                        "content": format_binary_content(img_bytes)
                    })

            if req.mask_image_bytes:
                files_list_for_httpx.append(('mask', ('mask.png', req.mask_image_bytes, req.mask_image_mime or 'image/png')))
                files_log.append({
                    "field": "mask",
                    "filename": "mask.png",
                    "size_bytes": len(req.mask_image_bytes),
                    "content": format_binary_content(req.mask_image_bytes)
                })
            
            # Data params (convert to string for multipart form data)
            payload = RequestMapper.unified_to_openai_payload(req)
            data = {k: str(v) for k, v in payload.items() if v is not None}

            log_debug_payload(
                logger,
                f"[UPSTREAM REQUEST][openai] POST {url}",
                headers=headers,
                body={
                    "params": params,
                    "form_data": data,
                    "files": files_log
                }
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
                raise HTTPException(status_code=504, detail="Upstream service timed out")
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Upstream connection error: {str(e)}")

        else:
            # Generations Endpoint
            url = f"{images_base}/generations"
            payload = RequestMapper.unified_to_openai_payload(req)

            log_debug_payload(
                logger,
                f"[UPSTREAM REQUEST][openai] POST {url}",
                headers=headers,
                body={
                    "params": params,
                    "json": payload
                }
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
                raise HTTPException(status_code=504, detail="Upstream service timed out")
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail=f"Upstream connection error: {str(e)}")
        
        log_debug_payload(
            logger,
            f"[UPSTREAM RESPONSE][openai] {resp.status_code} {url}",
            headers=resp.headers,
            body=resp.text
        )

        if resp.status_code >= 400:
            logger.error(f"OpenAI Error: {resp.text}")
            raise HTTPException(status_code=resp.status_code, detail=f"Upstream Error: {resp.text}")
            
        return ResponseMapper.openai_to_unified(resp.json())

    @staticmethod
    async def _call_gemini(req: UnifiedImageRequest, config: Dict[str, Any]) -> UnifiedImageResponse:
        url = f"{config['base_url']}/models/{req.target_model}:generateContent"
        params_qs = {"key": config['api_key']}
        headers = {"Content-Type": "application/json"}
        
        payload = RequestMapper.unified_to_gemini_payload(req)

        log_debug_payload(
            logger,
            f"[UPSTREAM REQUEST][gemini] POST {url}",
            headers=headers,
            body={
                "params": params_qs,
                "json": payload
            }
        )
        
        logger.info(f"Calling Gemini: {url}")
        timeout = httpx.Timeout(config.get("timeout_seconds", 120))
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, params=params_qs, json=payload, headers=headers)
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Upstream service timed out")
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upstream connection error: {str(e)}")
            
        log_debug_payload(
            logger,
            f"[UPSTREAM RESPONSE][gemini] {resp.status_code} {url}",
            headers=resp.headers,
            body=resp.text
        )

        if resp.status_code >= 400:
            logger.error(f"Gemini Error: {resp.text}")
            raise HTTPException(status_code=resp.status_code, detail=f"Upstream Error: {resp.text}")

        return ResponseMapper.gemini_to_unified(resp.json())
