"""
Gemini API Client for MERIT

This module provides a client for the Google Gemini API that properly implements
the AIAPIClient interface with full MERIT system integration.
"""

import os
from typing import List, Dict, Optional, Union

try:
    from google import genai
except ImportError:
    pass

from .base import BaseAPIClient, BaseAPIClientConfig, EmbeddingResponse, TextGenerationResponse
from ..core.logging import get_logger
from ..core.cache import cache_embeddings
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


class GeminiClientConfig(BaseAPIClientConfig):
    """
    Configuration class for Gemini API clients.
    
    This class handles configuration for Gemini API clients and can be initialized
    from different sources including environment variables, config files,
    or explicit parameters.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        generation_model: str = "gemini-2.0-flash-exp",
        embedding_model: str = "text-embedding-004",
        max_output_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 40,
        **kwargs
    ):
        """
        Initialize the Gemini API client configuration.
        
        Args:
            api_key: Gemini API key.
            base_url: Base URL for the Gemini API. Default is "https://generativelanguage.googleapis.com/v1beta".
            generation_model: Model to use for text generation. Default is "gemini-2.0-flash-exp".
            embedding_model: Model to use for embeddings. Default is "text-embedding-004".
            max_output_tokens: Maximum number of tokens to generate.
            temperature: Temperature for text generation.
            top_p: Top-p value for text generation.
            top_k: Top-k value for text generation.
            **kwargs: Additional configuration parameters.
        """
        #NOTE is this needed?
        if base_url is None:
            base_url = "https://generativelanguage.googleapis.com/v1beta"
            
        super().__init__(api_key=api_key, base_url=base_url, model=generation_model, **kwargs)
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
    
    @classmethod
    def get_supported_env_vars(cls) -> List[str]:
        """
        Get the list of supported environment variable names.
        
        Returns:
            List[str]: List of supported environment variable names.
        """
        # Add Gemini-specific environment variables
        return super().get_supported_env_vars() + ["GOOGLE_API_KEY", "GEMINI_API_KEY"]


class GeminiClient(BaseAPIClient):
    def _lazy_import_genai(self):
        try:
            from google import genai
            from google.genai import types
            return genai, types
        except ImportError as e:
            raise ImportError(
                "Google's Generative AI SDK is not installed. Please install it with `pip install merit[google]`"
            ) from e

    """
    A client for the Google Gemini API.
    
    This client implements the AIAPIClient interface and uses the
    Google Generative AI (genai) library to interact with the Gemini models
    with full MERIT system integration.
    """
    
    def __init__(self, config: GeminiClientConfig):
        """
        Initialize the Gemini client.
        
        Args:
            config: The Gemini client configuration.
        """
        super().__init__(config)
        self.config: GeminiClientConfig = config

        # Lazy import genai and configure it
        genai, _ = self._lazy_import_genai()

        if not self.config.api_key:
            self.config.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

        if not self.config.api_key:
            raise MeritAPIAuthenticationError(
                "Gemini API key not found. "
                "Set it in your config or as GOOGLE_API_KEY/GEMINI_API_KEY environment variable."
            )

        # Initialize the Gemini client
        try:
            genai.configure(api_key=self.config.api_key)
            self.client = genai.GenerativeModel(self.config.generation_model)
            logger.info(f"GeminiClient initialized for model: {self.config.generation_model}")
        except Exception as e:
            raise MeritAPIConnectionError(f"Failed to configure Gemini client: {e}") from e
    
    def _convert_gemini_error(self, error: Exception, endpoint: str = "") -> Exception:
        """
        Convert Google Gemini API exceptions to MeritAPI errors.
        
        Args:
            error: The Gemini exception to convert.
            endpoint: The API endpoint that failed (for context).
            
        Returns:
            Exception: The appropriate MeritAPI error.
        """
        details = {"original_error": str(error), "endpoint": endpoint}
        error_str = str(error).lower()
        
        # Check for authentication errors
        if "api key" in error_str or "authentication" in error_str or "unauthorized" in error_str:
            return MeritAPIAuthenticationError(
                "Gemini API authentication failed",
                details=details
            )
        
        # Check for rate limiting
        if "rate limit" in error_str or "quota" in error_str or "too many requests" in error_str:
            return MeritAPIRateLimitError(
                "Gemini API rate limit exceeded",
                details=details
            )
        
        # Check for invalid requests
        if "invalid" in error_str or "bad request" in error_str:
            return MeritAPIInvalidRequestError(
                "Invalid request to Gemini API",
                details=details
            )
        
        # Check for not found errors
        if "not found" in error_str or "model not found" in error_str:
            return MeritAPIResourceNotFoundError(
                "Gemini API resource not found",
                details=details
            )
        
        # Check for server errors
        if "server error" in error_str or "internal error" in error_str:
            return MeritAPIServerError(
                "Gemini API server error",
                details=details
            )
        
        # Check for timeout errors
        if "timeout" in error_str or "deadline" in error_str:
            return MeritAPITimeoutError(
                "Gemini API request timed out",
                details=details
            )
        
        # Check for connection errors
        if "connection" in error_str or "network" in error_str:
            return MeritAPIConnectionError(
                "Failed to connect to Gemini API",
                details=details
            )
        
        # Default to server error for unknown Gemini exceptions
        return MeritAPIServerError(
            "Gemini API error occurred",
            details=details
        )

    def _handle_api_error(self, error: Exception, strict: Optional[bool] = None) -> None:
        """
        Central error handling with strict mode support.
        
        Args:
            error: The exception that occurred.
            strict: Override for strict mode. If None, uses client's strict setting.
            
        Returns:
            None in graceful mode, raises exception in strict mode.
            
        Raises:
            Exception: The original error in strict mode.
        """
        use_strict = strict if strict is not None else self.config.strict
        if use_strict:
            logger.error(f"API call failed in strict mode: {error}")
            raise error
        else:
            logger.warning(f"API call failed in graceful mode: {error}")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for Gemini API requests.
        
        Returns:
            Dict[str, str]: The headers.
        """
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.config.api_key
        }
    
    @property
    def is_authenticated(self) -> bool:
        """
        Check if the client is authenticated.
        
        Returns:
            bool: True if the client has a valid API key, False otherwise.
        """
        #TODO logic to validate the api key
        return self.config.api_key is not None
    
    def _get_token(self) -> Optional[str]:
        """
        Get the API key (token equivalent for Gemini).
        
        Returns:
            Optional[str]: The API key, or None if not set.
        """
        return self.config.api_key
    
    def generate_text(self, prompt: str, **kwargs) -> TextGenerationResponse:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments including 'max_tokens', 'temperature', 'system_prompt'.
            
        Returns:
            TextGenerationResponse: The standardized response object.
        """
        genai, _ = self._lazy_import_genai()
        try:
            max_tokens = kwargs.get('max_tokens', self.config.max_output_tokens)
            temperature = kwargs.get('temperature', self.config.temperature)
            
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )

            logger.info(f"Generating text with model {self.config.generation_model}")
            
            response = self.client.models.generate_content(
                contents=prompt,
                generation_config=generation_config
            )
            
            return TextGenerationResponse(
                text=response.text,
                provider_metadata=response.to_dict()
            )
            
        except Exception as e:
            merit_error = self._convert_gemini_error(e, "generate_content")
            self._handle_api_error(merit_error)
            return TextGenerationResponse(text="", provider_metadata={"error": str(merit_error)})

    @cache_embeddings
    def get_embeddings(self, texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
        """
        Get embeddings for the given texts.
        
        Args:
            texts: A string or list of strings to get embeddings for.
            
        Returns:
            EmbeddingResponse: The standardized response object.
        """
        genai, _ = self._lazy_import_genai()
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            logger.info(f"Getting embeddings for {len(texts)} texts using model {self.config.embedding_model}")
            
            result = genai.embed_content(
                model=self.config.embedding_model,
                content=texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            return EmbeddingResponse(
                embeddings=result['embedding'],
                provider_metadata=result
            )
            
        except Exception as e:
            merit_error = self._convert_gemini_error(e, "embed_content")
            self._handle_api_error(merit_error)
            return EmbeddingResponse(embeddings=[[] for _ in texts], provider_metadata={"error": str(merit_error)})

    async def generate_text_async(self, prompt: str, **kwargs) -> TextGenerationResponse:
        """
        Asynchronously generate text based on a prompt.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments including 'max_tokens', 'temperature', 'system_prompt'.
            
        Returns:
            TextGenerationResponse: The standardized response object.
        """
        genai, _ = self._lazy_import_genai()
        try:
            max_tokens = kwargs.get('max_tokens', self.config.max_output_tokens)
            temperature = kwargs.get('temperature', self.config.temperature)
            
            generation_config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
            )

            logger.info(f"Asynchronously generating text with model {self.config.generation_model}")
            
            response = await self.client.generate_content_async(
                contents=prompt,
                generation_config=generation_config
            )
            
            return TextGenerationResponse(
                text=response.text,
                provider_metadata=response.to_dict()
            )
            
        except Exception as e:
            merit_error = self._convert_gemini_error(e, "generate_content")
            self._handle_api_error(merit_error)
            return TextGenerationResponse(text="", provider_metadata={"error": str(merit_error)})

    async def get_embeddings_async(self, texts: Union[str, List[str]], **kwargs) -> EmbeddingResponse:
        """
        Asynchronously get embeddings for the given texts.
        
        Args:
            texts: A string or list of strings to get embeddings for.
            
        Returns:
            EmbeddingResponse: The standardized response object.
        """
        genai, _ = self._lazy_import_genai()
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            logger.info(f"Asynchronously getting embeddings for {len(texts)} texts using model {self.config.embedding_model}")
            
            result = await genai.embed_content_async(
                model=self.config.embedding_model,
                content=texts,
                task_type="RETRIEVAL_DOCUMENT"
            )
            
            return EmbeddingResponse(
                embeddings=result['embedding'],
                provider_metadata=result
            )
            
        except Exception as e:
            merit_error = self._convert_gemini_error(e, "embed_content")
            self._handle_api_error(merit_error)
            return EmbeddingResponse(embeddings=[[] for _ in texts], provider_metadata={"error": str(merit_error)})
    
    
    def list_models(self) -> List[str]:
        """
        List available models from Gemini.
        
        Returns:
            List[str]: List of model names.
        """
        genai, _ = self._lazy_import_genai()
        try:
            logger.info("Listing available models")
            
            models = []
            for m in genai.list_models():
                if 'generateContent' in m.supported_generation_methods:
                    models.append(m.name)
            
            return models
            
        except Exception as e:
            merit_error = self._convert_gemini_error(e, "list_models")
            logger.error(f"Failed to list models: {merit_error}")
            return []
    
    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in the given text.
        
        Args:
            text: Text to count tokens for.
            model: Model to use for token counting. If None, uses the default generation model.
            
        Returns:
            int: Number of tokens.
        """
        try:
            count_model = model or self.config.generation_model
            logger.info(f"Counting tokens with model {count_model}")
            
            response = self.client.count_tokens(contents=text)
            
            return response.total_tokens
            
        except Exception as e:
            merit_error = self._convert_gemini_error(e, "count_tokens")
            logger.error(f"Failed to count tokens: {merit_error}")
            return 0
