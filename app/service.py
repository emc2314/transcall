import logging
from typing import Dict, Type, Final

from app import exceptions as exc
from app.config import ConfigManager
from app.providers.base import Provider
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider
from app.providers.vertex import VertexAIProvider
from app.schemas import UnifiedImageRequest, UnifiedImageResponse, ProviderName

logger = logging.getLogger("Service")


class ImageGenerationService:
    _PROVIDER_MAP: Final[Dict[ProviderName, Type[Provider]]] = {
        "openai": OpenAIProvider,
        "gemini": GeminiProvider,
        "vertexai": VertexAIProvider,
    }

    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager

    async def process_request(self, req: UnifiedImageRequest) -> UnifiedImageResponse:
        config = self.config_manager.get_model_config(req.target_model)
        if not config:
            raise exc.ModelNotSupportedError(req.target_model)

        provider_name = config.get("provider")
        if provider_name not in self._PROVIDER_MAP:
            raise exc.ProviderNotRegisteredError(provider_name or "<unset>")
        provider_cls = self._PROVIDER_MAP[provider_name]

        provider: Provider = provider_cls(config)
        try:
            return await provider.generate(req)
        except exc.TranscallError:
            raise
        except Exception as err:  # pragma: no cover - defensive
            logger.exception("Unhandled provider error")
            raise exc.InternalServiceError("Unhandled service error.") from err
