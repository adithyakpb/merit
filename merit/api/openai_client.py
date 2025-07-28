"""
OpenAI API Client for MERIT

This module provides a client for the OpenAI API that properly implements
the AIAPIClient interface with full MERIT system integration.
"""

import os
from typing import Dict, Any, List, Optional, Union, Tuple
from dotenv import load_dotenv

from openai import (
    OpenAI as SDKOpenAIClient,
    AzureOpenAI as SDKAzureOpenAIClient,
    AsyncOpenAI,
    AsyncAzureOpenAI,
    APIError,
    AuthenticationError,
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    BadRequestError,
) # Other errors can be added as needed

from .base import BaseAPIClient, BaseAPIClientConfig, EmbeddingResponse, TextGenerationResponse
from .run_config import RetryConfig, with_retry, adaptive_throttle # Import decorators
from ..core.logging import get_logger
import json # For safe logging of RetryConfig
from ..core.utils import parse_json # May still be needed for prompts, but not for SDK response parsing

from .errors import (
    MeritAPIAuthenticationError,
    MeritAPIConnectionError,
    MeritAPIServerError,
    MeritAPITimeoutError,
    MeritAPIRateLimitError,
    MeritAPIResourceNotFoundError,
    MeritAPIInvalidRequestError
)

logger = get_logger(__name__)

# Default base URL for OpenAI API (SDK handles this, but good for reference or overrides)
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# from merit.api.client passed by create_ai_client, or direct parameters.

class OpenAIClientConfig(BaseAPIClientConfig):
    """
    Configuration specific to the OpenAI client, using the OpenAI SDK.
    Inherits from the generic AIAPIClientConfig.
    
    For Azure OpenAI:
    - Set api_type to 'azure'
    - Set api_version (required for Azure)
    - Set base_url to your Azure endpoint URL
    - Provide api_key (your Azure OpenAI API key)
    """
    embedding_model: str = "text-embedding-ada-002"
    organization_id: Optional[str] = None
    request_timeout: Optional[Union[float, Tuple[float, float]]] = 60.0 
    max_sdk_retries: Optional[int] = 2
    api_type: str = "openai"
    api_version: Optional[str] = None

class OpenAIClient(BaseAPIClient):
    """
    OpenAI API client implementation using the official OpenAI Python SDK.
    
    This class provides methods to interact with the OpenAI API for tasks such as
    text generation and embeddings.
    """
    
    def __init__(self, config: OpenAIClientConfig):
        """
        Initialize the OpenAI client using an OpenAIClientConfig object.
        
        Args:
            config: An OpenAIClientConfig instance containing all necessary configurations.
        """
        super().__init__(config)
        self.config: OpenAIClientConfig = config

        if not self.config.api_key:
            raise MeritAPIAuthenticationError("OpenAI API key is not set in the configuration.")

        common_args = {
            "api_key": self.config.api_key,
            "timeout": self.config.request_timeout,
            "max_retries": self.config.max_sdk_retries 
        }
        
        if self.config.organization_id:
            common_args["organization"] = self.config.organization_id

        if self.config.api_type == "azure":
            self.config.validate_required_fields(["api_version", "base_url"])
            logger.info(f"Initializing AzureOpenAI client with endpoint: {self.config.base_url} and API version: {self.config.api_version}")
            self.sdk_client = SDKAzureOpenAIClient(
                azure_endpoint=self.config.base_url,
                api_version=self.config.api_version,
                **common_args
            )
            self.async_sdk_client = AsyncAzureOpenAI(
                azure_endpoint=self.config.base_url,
                api_version=self.config.api_version,
                **common_args
            )
        else:
            logger.info(f"Initializing OpenAI client with base_url: {self.config.base_url}")
            self.sdk_client = SDKOpenAIClient(
                base_url=self.config.base_url,
                **common_args
            )
            self.async_sdk_client = AsyncOpenAI(
                base_url=self.config.base_url,
                **common_args
            )
        
        logger.info(f"Initialized OpenAIClient with SDK. Model: {self.config.model}, Embedding Model: {self.config.embedding_model}")
    
    def _apply_retry_decorators(self):
        """
        Apply retry decorators to API methods after initialization is complete.
        This is called at the end of __init__ to ensure methods are fully defined before wrapping.
        """
        if not self.retry_config:
            logger.warning("No retry configuration available, skipping decorator application")
            return
            
        rc = self.retry_config
        
        try:
            # Wrap generate_text
            if hasattr(self, 'generate_text'):
                _original_generate_text = self.generate_text
                self.generate_text = adaptive_throttle(
                    with_retry(
                        max_retries=rc.max_retries,
                        backoff_factor=rc.backoff_factor,
                        jitter=rc.jitter,
                        retry_on=rc.retry_on_exceptions,
                        retry_status_codes=rc.retry_status_codes
                    )(_original_generate_text)
                )
                logger.info("Applied retry decorators to generate_text method")
                
            # Wrap get_embeddings
            if hasattr(self, 'get_embeddings'):
                _original_get_embeddings = self.get_embeddings
                self.get_embeddings = adaptive_throttle(
                    with_retry(
                        max_retries=rc.max_retries,
                        backoff_factor=rc.backoff_factor,
                        jitter=rc.jitter,
                        retry_on=rc.retry_on_exceptions,
                        retry_status_codes=rc.retry_status_codes
                    )(_original_get_embeddings)
                )
                logger.info("Applied retry decorators to get_embeddings method")
                
            # Log successful application
            if hasattr(rc, 'model_dump'):
                try:
                    loggable_rc_dict = rc.model_dump()
                    if 'retry_on_exceptions' in loggable_rc_dict and isinstance(loggable_rc_dict['retry_on_exceptions'], list):
                        loggable_rc_dict['retry_on_exceptions'] = [exc.__name__ for exc in loggable_rc_dict['retry_on_exceptions']]
                    logger.info(f"Applied adaptive retry to OpenAIClient methods with effective config: {json.dumps(loggable_rc_dict, indent=2, default=str)}")
                except Exception as e:
                    logger.warning(f"Failed to log retry config details: {e}")
        except Exception as e:
            logger.error(f"Error applying retry decorators: {e}", exc_info=True)
            # Continue without decorators rather than failing
            
    @property
    def is_authenticated(self) -> bool:
        """
        Check if the client is authenticated.
        
        Returns:
            bool: True if the client has a valid API key, False otherwise.
        """
        return self.api_key is not None
    
    def login(self) -> bool:
        """
        OpenAI uses API key authentication, so login is not applicable.
        
        Returns:
            bool: True if API key is present, False otherwise.
        """
        return self.is_authenticated
    
    def get_token(self) -> Optional[str]:
        """
        Get the API key (token equivalent for OpenAI).
        
        Returns:
            Optional[str]: The API key, or None if not set.
        """
        return self.api_key
    
    def get_embeddings(self, texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
        """
        Get embeddings for the given texts using the OpenAI SDK.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []

        embedding_model_to_use = kwargs.pop("model", self.config.embedding_model)
        logger.info(f"[SDK] Getting embeddings for {len(texts)} texts using model {embedding_model_to_use}")

        try:
            response = self.sdk_client.embeddings.create(
                model=embedding_model_to_use,
                input=texts,
                **kwargs
            )
            embeddings = [embedding_obj.embedding for embedding_obj in response.data]
            logger.info(f"[SDK] Successfully got embeddings for {len(texts)} texts.")
            return EmbeddingResponse(
                embeddings=embeddings,
                provider_metadata=response.model_dump()
            )
        except APIError as e:
            if isinstance(e, RateLimitError):
                raise MeritAPIRateLimitError from e
            elif isinstance(e, AuthenticationError):
                raise MeritAPIAuthenticationError from e
            elif isinstance(e, BadRequestError):
                raise MeritAPIInvalidRequestError from e
            else:
                raise MeritAPIServerError from e

    def generate_text(self, prompt: str, **kwargs) -> TextGenerationResponse:
        """
        Generate text based on the given prompt using the OpenAI SDK's chat completions.
        """
        logger.info(f"[SDK] OpenAIClient.generate_text called. Model: {self.config.model}")

        messages = kwargs.pop("messages", [{"role": "user", "content": prompt}])
        
        params = {
            "model": self.config.model,
            "messages": messages,
            **kwargs
        }

        try:
            completion = self.sdk_client.chat.completions.create(**params)
            generated_content = completion.choices[0].message.content or ""
            logger.info(f"[SDK] Successfully generated text using model {params.get('model')}.")
            return TextGenerationResponse(
                text=generated_content,
                provider_metadata=completion.model_dump()
            )
        except APIError as e:
            if isinstance(e, RateLimitError):
                raise MeritAPIRateLimitError from e
            elif isinstance(e, AuthenticationError):
                raise MeritAPIAuthenticationError from e
            elif isinstance(e, BadRequestError):
                raise MeritAPIInvalidRequestError from e
            else:
                raise MeritAPIServerError from e

    async def get_embeddings_async(self, texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
        """
        Asynchronously get embeddings for the given texts using the OpenAI SDK.
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return EmbeddingResponse(embeddings=[])

        embedding_model_to_use = kwargs.pop("model", self.config.embedding_model)
        logger.info(f"[SDK] Getting async embeddings for {len(texts)} texts using model {embedding_model_to_use}")

        try:
            response = await self.async_sdk_client.embeddings.create(
                model=embedding_model_to_use,
                input=texts,
                **kwargs
            )
            embeddings = [embedding_obj.embedding for embedding_obj in response.data]
            logger.info(f"[SDK] Successfully got async embeddings for {len(texts)} texts.")
            return EmbeddingResponse(
                embeddings=embeddings,
                provider_metadata=response.model_dump()
            )
        except APIError as e:
            if isinstance(e, RateLimitError):
                raise MeritAPIRateLimitError from e
            elif isinstance(e, AuthenticationError):
                raise MeritAPIAuthenticationError from e
            elif isinstance(e, BadRequestError):
                raise MeritAPIInvalidRequestError from e
            else:
                raise MeritAPIServerError from e

    async def generate_text_async(self, prompt: str, **kwargs) -> TextGenerationResponse:
        """
        Asynchronously generate text based on the given prompt using the OpenAI SDK's chat completions.
        """
        logger.info(f"[SDK] OpenAIClient.generate_text_async called. Model: {self.config.model}")

        messages = kwargs.pop("messages", [{"role": "user", "content": prompt}])
        
        params = {
            "model": self.config.model,
            "messages": messages,
            **kwargs
        }

        try:
            completion = await self.async_sdk_client.chat.completions.create(**params)
            generated_content = completion.choices[0].message.content or ""
            logger.info(f"[SDK] Successfully generated async text using model {params.get('model')}.")
            return TextGenerationResponse(
                text=generated_content,
                provider_metadata=completion.model_dump()
            )
        except APIError as e:
            if isinstance(e, RateLimitError):
                raise MeritAPIRateLimitError from e
            elif isinstance(e, AuthenticationError):
                raise MeritAPIAuthenticationError from e
            elif isinstance(e, BadRequestError):
                raise MeritAPIInvalidRequestError from e
            else:
                raise MeritAPIServerError from e
    
    def create_chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """
        Create a chat completion with multiple messages.
        
        This is more flexible than generate_text() which only supports a single prompt.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys.
            **kwargs: Additional parameters to pass to the API.
            
        Returns:
            Dict[str, Any]: The complete API response.
        """
        # Set default parameters
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
        }
        
        # Update with any provided kwargs
        params.update(kwargs)
        
        try:
            logger.info(f"Creating chat completion with model {self.model}")
            
            # Use the SDK client for chat completions
            response = self.sdk_client.chat.completions.create(**params)
            return response.model_dump() # Convert the SDK response object to a dict
        
        except Exception as e:
            merit_error = self._convert_requests_error(e, "chat/completions")
            logger.error(f"Failed to create chat completion: {merit_error}")
            return {"error": str(merit_error)}
    
    def list_models(self) -> List[str]:
        """
        List available models from OpenAI.
        
        Returns:
            List[str]: List of model IDs.
        """
        try:
            logger.info("Listing available models")
            
            # Use the SDK client to list models
            response = self.sdk_client.models.list()
            return [model.id for model in response.data]
        
        except Exception as e:
            merit_error = self._convert_requests_error(e, "models")
            logger.error(f"Failed to list models: {merit_error}")
            return []
