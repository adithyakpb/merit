"""
Base API Client Interface

This module defines the base interfaces for all API clients and configurations in the MERIT system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union

from merit.core.logging import get_logger
from .errors import MeritAPIInvalidRequestError

logger = get_logger(__name__)


from pydantic import BaseModel, Field, ConfigDict


class EmbeddingResponse(BaseModel):
    """A standardized container for embedding results."""
    embeddings: List[List[float]] = Field(..., description="The list of embedding vectors.")
    provider_metadata: Dict[str, Any] = Field({}, description="Provider-specific metadata, like usage stats or model info.")

class TextGenerationResponse(BaseModel):
    """A standardized container for text generation results."""
    text: str = Field(..., description="The generated text.")
    provider_metadata: Dict[str, Any] = Field({}, description="Provider-specific metadata, like finish reasons or token counts.")
    
    
class BaseAPIClientConfig(BaseModel):
    """
    Abstract base class for API client configurations using Pydantic.
    
    This class defines the interface and common functionality for all API client
    configurations in the system.
    """
    
    model_config = ConfigDict(extra='allow', arbitrary_types_allowed=True)

    api_key: Optional[str] = Field(None, description="API key for authentication.")
    base_url: Optional[str] = Field(None, description="Base URL for the API.")
    enable_retries: bool = Field(True, description="Enable automatic retry functionality.")
    enable_throttling: bool = Field(True, description="Enable adaptive throttling functionality.")
    max_retries: int = Field(3, description="Maximum number of retry attempts.")
    backoff_factor: float = Field(0.5, description="Exponential backoff factor for retries.")
    initial_delay: float = Field(0.5, description="Initial delay for adaptive throttling in seconds.")
    min_delay: float = Field(0.05, description="Minimum delay for adaptive throttling in seconds.")
    max_delay: float = Field(2.0, description="Maximum delay for adaptive throttling in seconds.")

    def validate_required_fields(self, required_fields: List[str]):
        """
        Validates that a given list of fields are not None.
        
        Args:
            required_fields: A list of field names to validate.
            
        Raises:
            MeritAPIInvalidRequestError: If any of the required fields are None.
        """
        missing_fields = [
            field for field in required_fields if getattr(self, field, None) is None
        ]
        if missing_fields:
            raise MeritAPIInvalidRequestError(
                f"Missing required configuration parameters: {', '.join(missing_fields)}",
                details={"missing_parameters": missing_fields}
            )


class BaseAPIClient(ABC):
    """
    Abstract base class for API clients.
    
    This class defines the interface that all API clients must implement.
    """

    config: BaseAPIClientConfig

    def __init__(self, config: BaseAPIClientConfig):
        self.config = config
        logger.info(f"BaseAPIClient initialized with config: {config.__class__.__name__}")

    @property
    @abstractmethod
    def is_authenticated(self) -> bool:
        """
        Check if the client is authenticated.
        
        Returns:
            bool: True if the client is authenticated, False otherwise.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(self, texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
        """
        Get embeddings for the given texts.
        
        Args:
            texts: A string or list of strings to get embeddings for.
            **kwargs: Additional arguments for the API call.

        Returns:
            EmbeddingResponse: A standardized response object containing embeddings and metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> TextGenerationResponse:
        """
        Generate text based on the given prompt.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments for the API call.

        Returns:
            TextGenerationResponse: A standardized response object containing the generated text and metadata.
        """
        raise NotImplementedError

    @abstractmethod
    async def get_embeddings_async(self, texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
        """
        Asynchronously get embeddings for the given texts.
        
        Args:
            texts: A string or list of strings to get embeddings for.
            **kwargs: Additional arguments for the API call.

        Returns:
            EmbeddingResponse: A standardized response object containing embeddings and metadata.
        """
        raise NotImplementedError

    @abstractmethod
    async def generate_text_async(self, prompt: str, **kwargs) -> TextGenerationResponse:
        """
        Asynchronously generate text based on the given prompt.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments for the API call.

        Returns:
            TextGenerationResponse: A standardized response object containing the generated text and metadata.
        """
        raise NotImplementedError
