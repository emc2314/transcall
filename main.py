import logging
import os
import sys
from typing import List, Optional, Any, Dict, Tuple, Set
from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, Response

from app.config import ConfigManager
from app.schemas import UnifiedImageRequest
from app.mappers import RequestMapper, ResponseMapper
from app.service import ImageGenerationService
from app.logging_utils import log_debug_payload, format_structured


def _configure_logging() -> logging.Logger:
    """
    Configure logging so that it defaults to stdout (Docker captures stdout/stderr),
    but allows overriding via LOG_DESTINATION env var.

    LOG_DESTINATION options:
      - "stdout" (default): stream to stdout (docker compose logs)
      - "stderr": stream to stderr
      - any other value: treated as a file path to append logs
    """
    log_dest = os.getenv("LOG_DESTINATION", "stdout").strip().lower()
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level_name, logging.INFO)
    log_kwargs: Dict[str, Any] = {
        "level": level,
        "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    }

    if log_dest in {"stdout", "console"}:
        log_kwargs["handlers"] = [logging.StreamHandler(sys.stdout)]
    elif log_dest == "stderr":
        log_kwargs["handlers"] = [logging.StreamHandler(sys.stderr)]
    else:
        # treat LOG_DESTINATION as a file path
        log_kwargs["filename"] = log_dest
        log_kwargs["filemode"] = "a"

    logging.basicConfig(**log_kwargs)
    return logging.getLogger("API")


logger = _configure_logging()

app = FastAPI()


@app.middleware("http")
async def debug_raw_io_middleware(request: Request, call_next):
    if logger.isEnabledFor(logging.DEBUG):
        body = await request.body()
        log_debug_payload(
            logger,
            f"[CLIENT REQUEST] {request.method} {request.url.path}",
            headers=request.headers,
            body=body,
            treat_body_as_binary=True,
        )

        body_consumed = False

        async def receive():
            nonlocal body_consumed
            if not body_consumed:
                body_consumed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        request = Request(request.scope, receive=receive)
        response = await call_next(request)

        resp_body = b""
        async for chunk in response.body_iterator:
            resp_body += chunk

        log_debug_payload(
            logger,
            f"[SERVER RESPONSE] {request.method} {request.url.path} -> {response.status_code}",
            headers=response.headers,
            body=resp_body,
            treat_body_as_binary=True,
        )

        return Response(
            content=resp_body,
            status_code=response.status_code,
            headers=dict(response.headers),
            media_type=response.media_type,
        )

    return await call_next(request)


def _coerce_bool(value: Optional[Any]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return False


def _has_positive_value(value: Optional[Any]) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return bool(value)
    try:
        return int(value) > 0
    except (ValueError, TypeError):
        return False


def _enforce_non_streaming(
    stream_value: Optional[Any], partial_images_value: Optional[Any]
):
    if _coerce_bool(stream_value):
        raise HTTPException(
            status_code=501,
            detail="Streaming responses are not supported by this service yet.",
        )
    if _has_positive_value(partial_images_value):
        raise HTTPException(
            status_code=501,
            detail="partial_images requires streaming, which is not supported by this service yet.",
        )


@app.post("/images/generations")
@app.post("/v1/images/generations")
async def openai_generations(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    logger.info("Params (OpenAI Gen): %s", format_structured(body))

    _enforce_non_streaming(body.get("stream"), body.get("partial_images"))

    # 1. Determine Target Model from Request
    requested_model = body.get("model", "gpt-image-1")

    # 2. Check Config to see who provides this model
    config = ConfigManager.get_model_config(requested_model)
    if not config:
        raise HTTPException(
            status_code=400, detail=f"Model '{requested_model}' not supported."
        )

    # 3. Map to Unified Request
    unified_req = RequestMapper.openai_gen_to_unified(
        body, config, provider=config["provider"]
    )

    # 4. Process
    unified_resp = await ImageGenerationService.process_request(unified_req)

    # 5. Map back to OpenAI format (since this is the OpenAI endpoint)
    final_resp = ResponseMapper.unified_to_openai_format(unified_resp)

    return JSONResponse(content=final_resp)


@app.post("/images/edits")
@app.post("/v1/images/edits")
async def openai_edits(
    model: str = Form(...),
    prompt: str = Form(...),
    image_primary: List[UploadFile] = File(default_factory=list, alias="image"),
    image_alt: List[UploadFile] = File(default_factory=list, alias="image[]"),
    mask: Optional[UploadFile] = File(None),
    n: Optional[int] = Form(None),
    size: Optional[str] = Form(None),
    response_format: Optional[str] = Form(None),
    user: Optional[str] = Form(None),
    # Extended OpenAI-specific Parameters
    background: Optional[str] = Form(None),
    moderation: Optional[str] = Form(None),
    quality: Optional[str] = Form(None),
    output_format: Optional[str] = Form(None),
    output_compression: Optional[int] = Form(None),
    partial_images: Optional[int] = Form(None),
    stream: Optional[bool] = Form(None),
    input_fidelity: Optional[str] = Form(None),
):
    # Track optional fields actually provided
    image_entries: List[Tuple[UploadFile, str]] = []
    for img in image_primary or []:
        image_entries.append((img, "image"))
    for img in image_alt or []:
        image_entries.append((img, "image[]"))
    effective_size = size
    effective_n = n if n is not None else 1

    optional_field_values = {
        "n": n,
        "size": size,
        "response_format": response_format,
        "user": user,
        "background": background,
        "moderation": moderation,
        "quality": quality,
        "output_format": output_format,
        "output_compression": output_compression,
        "partial_images": partial_images,
        "stream": stream,
        "input_fidelity": input_fidelity,
    }

    provided_optional_fields: Set[str] = set()
    for field, value in optional_field_values.items():
        if value is None:
            continue
        provided_optional_fields.add(field)

    # Log params (excluding binary data for brevity)
    log_size = size if size is not None else "auto (default)"
    logger.info(
        f"Params (OpenAI Edit): model={model}, prompt={prompt}, size={log_size}, "
        f"num_images={len(image_entries)}, has_mask={mask is not None}"
    )

    _enforce_non_streaming(stream, partial_images)

    config = ConfigManager.get_model_config(model)
    if not config:
        raise HTTPException(status_code=400, detail=f"Model '{model}' not supported.")

    # Read all image files and mask
    image_data_list: List[Tuple[bytes, str]] = []
    image_field_names: List[str] = []
    for img_file, field_name in image_entries:
        content = await img_file.read()
        image_data_list.append(
            (content, img_file.content_type or "application/octet-stream")
        )
        image_field_names.append(field_name)

    mask_data: Optional[Tuple[bytes, str]] = None
    if mask:
        mask_content = await mask.read()
        mask_data = (mask_content, mask.content_type or "application/octet-stream")

    # Construct Params Dict for Mapper (includes all OpenAI-specific parameters)
    params: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "n": effective_n,
        "size": effective_size,
        "response_format": response_format,
        "user": user,
        "background": background,
        "moderation": moderation,
        "quality": quality,
        "output_format": output_format,
        "output_compression": output_compression,
        "partial_images": partial_images,
        "stream": stream,
        "input_fidelity": input_fidelity,
    }

    # Map to Unified
    unified_req = RequestMapper.openai_edit_to_unified(
        params,
        image_data_list,
        mask_data,
        config,
        provider=config["provider"],
        provided_fields=provided_optional_fields,
        image_field_names=image_field_names,
    )

    # Process
    unified_resp = await ImageGenerationService.process_request(unified_req)

    # Map Output
    final_resp = ResponseMapper.unified_to_openai_format(unified_resp)

    return JSONResponse(content=final_resp)


@app.post("/v1beta/models/{model_name}:generateContent")
@app.post("/v1beta1/models/{model_name}:generateContent")
@app.post("/v1/models/{model_name}:generateContent")
# Support Vertex AI style paths (simplified for proxying, ignoring project/location/publisher parts)
@app.post("/v1beta1/publishers/google/models/{model_name}:generateContent")
@app.post("/v1/publishers/google/models/{model_name}:generateContent")
@app.post(
    "/v1beta1/projects/{project}/locations/{location}/publishers/google/models/{model_name}:generateContent"
)
@app.post(
    "/v1/projects/{project}/locations/{location}/publishers/google/models/{model_name}:generateContent"
)
async def gemini_generate_content(
    model_name: str,
    request: Request,
    project: Optional[str] = None,
    location: Optional[str] = None,
):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    logger.info("Params (Gemini): %s", format_structured(body))

    # Config Check
    config = ConfigManager.get_model_config(model_name)
    if not config:
        raise HTTPException(
            status_code=400, detail=f"Model '{model_name}' not supported."
        )

    # Map to Unified
    unified_req = RequestMapper.gemini_content_to_unified(
        body, model_name, config, provider=config["provider"]
    )

    if config["provider"] == "gemini":
        # Use the Gemini model from the URL; OpenAI-backed entries retain their mapped target
        unified_req.target_model = model_name

    # Process
    unified_resp = await ImageGenerationService.process_request(unified_req)

    # Map Output to Gemini Format
    final_resp = ResponseMapper.unified_to_gemini_format(unified_resp)

    return JSONResponse(content=final_resp)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
