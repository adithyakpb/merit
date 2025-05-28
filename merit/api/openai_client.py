"""
OpenAI API Client for MERIT

This module provides a client for the OpenAI API that properly implements
the AIAPIClient interface with full MERIT system integration.
"""

import os
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv
import requests

from .client import AIAPIClient, AIAPIClientConfig
from .base import validate_embeddings_response, validate_text_response
from ..core.logging import get_logger
from ..core.cache import cache_embeddings
from ..core.utils import parse_json
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


class OpenAIClientConfig(AIAPIClientConfig):
    """
    Configuration class for OpenAI API clients.
    
    This class handles configuration for OpenAI API clients and can be initialized
    from different sources including environment variables, config files,
    or explicit parameters.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        organization_id: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        strict: bool = False,
        **kwargs
    ):
        """
        Initialize the OpenAI API client configuration.
        
        Args:
            api_key: OpenAI API key.
            base_url: Base URL for the OpenAI API. Default is "https://api.openai.com/v1".
            organization_id: OpenAI organization ID.
            model: Model to use for text generation. Default is "gpt-3.5-turbo".
            embedding_model: Model to use for embeddings. Default is "text-embedding-ada-002".
            **kwargs: Additional configuration parameters.
        """
        if base_url is None:
            base_url = "https://api.openai.com/v1"
            
        super().__init__(api_key=api_key, base_url=base_url, model=model, **kwargs)
        self.organization_id = organization_id
        self.embedding_model = embedding_model
    
    @classmethod
    def get_supported_env_vars(cls) -> List[str]:
        """
        Get the list of supported environment variable names.
        
        Returns:
            List[str]: List of supported environment variable names.
        """
        # Add OpenAI-specific environment variables
        return super().get_supported_env_vars() + ["OPENAI_API_KEY", "OPENAI_ORGANIZATION"]


class OpenAIClient(AIAPIClient):
    """
    OpenAI API client implementation.
    
    This client provides access to OpenAI's API for embeddings and text generation
    with full MERIT system integration including caching, rate limiting, and validation.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        organization_id: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        embedding_model: str = "text-embedding-ada-002",
        base_url: str = "https://api.openai.com/v1",
        config: Optional[Union[OpenAIClientConfig, Dict[str, Any]]] = None,
        env_file: Optional[str] = None,
        required_vars: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the OpenAI client.
        
        Args:
            api_key: OpenAI API key. If not provided, will look for OPENAI_API_KEY environment variable.
            organization_id: OpenAI organization ID. If not provided, will look for OPENAI_ORGANIZATION environment variable.
            model: Model to use for text generation. Default is "gpt-3.5-turbo".
            embedding_model: Model to use for embeddings. Default is "text-embedding-ada-002".
            base_url: Base URL for the OpenAI API. Default is "https://api.openai.com/v1".
            config: Configuration object or dictionary.
            env_file: Path to .env file containing environment variables.
            required_vars: List of environment variable names that are required when loading from environment.
            **kwargs: Additional parameters.
        """
        # Check for OpenAI-specific environment variables first
        if env_file is not None or api_key is None:
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if openai_api_key:
                api_key = openai_api_key
                
            openai_org = os.getenv("OPENAI_ORGANIZATION")
            if openai_org and organization_id is None:
                organization_id = openai_org
        
        # Initialize the parent class with proper inheritance
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            env_file=env_file,
            config=config,
            required_vars=required_vars,
            **kwargs
        )
        
        # Set OpenAI-specific attributes
        self.organization_id = organization_id
        self.embedding_model = embedding_model
        
        # Override from config if provided
        if config is not None:
            if isinstance(config, OpenAIClientConfig):
                if config.organization_id is not None:
                    self.organization_id = config.organization_id
                if config.embedding_model is not None:
                    self.embedding_model = config.embedding_model
            elif isinstance(config, dict):
                if config.get('organization_id') is not None:
                    self.organization_id = config.get('organization_id')
                if config.get('embedding_model') is not None:
                    self.embedding_model = config.get('embedding_model')
        
        logger.info(f"Initialized OpenAIClient with model={self.model}, embedding_model={self.embedding_model}")
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for OpenAI API requests.
        
        Returns:
            Dict[str, str]: The headers.
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id
        
        return headers
    
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
    
    @cache_embeddings
    @validate_embeddings_response
    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Get embeddings for the given texts using OpenAI's embeddings API.
        
        Args:
            texts: A string or list of strings to get embeddings for.
            
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
            
        Note:
            This method transforms OpenAI's response format to match the AIAPIClient format.
            OpenAI returns: {"data": [{"embedding": [...]}, ...]}
            AIAPIClient expects: [[...], ...]
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            logger.info(f"Getting embeddings for {len(texts)} texts using model {self.embedding_model}")
            
            response = requests.post(
                f"{self.base_url}/embeddings",
                headers=self._get_headers(),
                json={
                    "input": texts,
                    "model": self.embedding_model
                }
            )
            
            response.raise_for_status()
            data = parse_json(response.text)
            
            # Extract embeddings from the response and format to match AIAPIClient
            if "data" in data:
                embeddings = [item["embedding"] for item in data["data"]]
                return embeddings
            else:
                logger.error("No embeddings in response")
                return [[] for _ in texts]
        
        except Exception as e:
            merit_error = self._convert_requests_error(e, "embeddings")
            return self._handle_api_error(merit_error) or [[] for _ in texts]
    
    @validate_text_response
    def generate_text(self, prompt: str, strict: Optional[bool] = None, **kwargs) -> str:
        """
        Generate text based on the given prompt using OpenAI's chat completions API.
        
        Args:
            prompt: The prompt to generate text from.
            strict: Override for strict mode. If None, uses client's strict setting.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            str: The generated text.
            
        Note:
            This method transforms OpenAI's response format to match the AIAPIClient format.
            OpenAI returns: {"choices": [{"message": {"content": "..."}}]}
            AIAPIClient expects: "..."
        """
        # Set default parameters
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        
        # Update with any provided kwargs (excluding strict)
        api_kwargs = {k: v for k, v in kwargs.items() if k != 'strict'}
        params.update(api_kwargs)
        
        # If messages were provided directly, use those instead of creating from prompt
        if "messages" in kwargs:
            params["messages"] = kwargs["messages"]
        
        try:
            logger.info(f"Generating text with model {self.model}")
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=params
            )
            
            response.raise_for_status()
            data = parse_json(response.text)
            
            # Extract text from the response and format to match AIAPIClient
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                error = ValueError("No text in response")
                return self._handle_api_error(error, strict)
        
        except Exception as e:
            merit_error = self._convert_requests_error(e, "chat/completions")
            return self._handle_api_error(merit_error, strict) or ""
    
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
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=params
            )
            
            response.raise_for_status()
            return parse_json(response.text)
        
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
            
            response = requests.get(
                f"{self.base_url}/models",
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            data = parse_json(response.text)
            
            return [model["id"] for model in data.get("data", [])]
        
        except Exception as e:
            merit_error = self._convert_requests_error(e, "models")
            logger.error(f"Failed to list models: {merit_error}")
            return []
