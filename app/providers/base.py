from abc import ABC, abstractmethod
from typing import Dict

from app.schemas import UnifiedImageRequest, UnifiedImageResponse


class Provider(ABC):
    """Strategy interface for provider-specific implementations."""

    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    async def generate(self, req: UnifiedImageRequest) -> UnifiedImageResponse:
        """Execute the upstream call."""
