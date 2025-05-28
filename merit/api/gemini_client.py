"""
Gemini API Client for MERIT

This module provides a client for the Google Gemini API that properly implements
the AIAPIClient interface with full MERIT system integration.
"""

import os
from typing import List, Dict, Any, Optional, Union
from dotenv import load_dotenv

from google import genai
from google.genai import types

from .client import AIAPIClient, AIAPIClientConfig
from .base import validate_embeddings_response, validate_text_response
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


class GeminiClientConfig(AIAPIClientConfig):
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


class GeminiClient(AIAPIClient):
    """
    A client for the Google Gemini API.
    
    This client implements the AIAPIClient interface and uses the
    Google Generative AI (genai) library to interact with the Gemini models
    with full MERIT system integration.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        generation_model: str = "gemini-2.0-flash-exp",
        embedding_model: str = "text-embedding-004",
        max_output_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 40,
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        config: Optional[Union[GeminiClientConfig, Dict[str, Any]]] = None,
        env_file: Optional[str] = None,
        required_vars: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: The API key for the Gemini API. If not provided, will look for GOOGLE_API_KEY or GEMINI_API_KEY environment variable.
            generation_model: The model to use for text generation.
            embedding_model: The model to use for embeddings.
            max_output_tokens: The maximum number of tokens to generate.
            temperature: The temperature for text generation.
            top_p: The top-p value for text generation.
            top_k: The top-k value for text generation.
            base_url: Base URL for the Gemini API.
            config: Configuration object or dictionary.
            env_file: Path to .env file containing environment variables.
            required_vars: List of environment variable names that are required when loading from environment.
            **kwargs: Additional parameters.
        """
        # Check for Gemini-specific environment variables first
        if env_file is not None or api_key is None:
            gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if gemini_api_key:
                api_key = gemini_api_key
        
        # Ensure we have an API key
        if not api_key:
            raise MeritAPIAuthenticationError(
                "API key is required for Gemini client",
                details={
                    "required_env_vars": ["GOOGLE_API_KEY", "GEMINI_API_KEY"],
                    "help": "Provide api_key parameter or set GOOGLE_API_KEY/GEMINI_API_KEY environment variable"
                }
            )
        
        # Initialize the parent class with proper inheritance
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=generation_model,
            env_file=env_file,
            config=config,
            required_vars=required_vars,
            **kwargs
        )
        
        # Set Gemini-specific attributes
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        
        # Override from config if provided
        if config is not None:
            if isinstance(config, GeminiClientConfig):
                if config.generation_model is not None:
                    self.generation_model = config.generation_model
                if config.embedding_model is not None:
                    self.embedding_model = config.embedding_model
                if config.max_output_tokens is not None:
                    self.max_output_tokens = config.max_output_tokens
                if config.temperature is not None:
                    self.temperature = config.temperature
                if config.top_p is not None:
                    self.top_p = config.top_p
                if config.top_k is not None:
                    self.top_k = config.top_k
            elif isinstance(config, dict):
                self.generation_model = config.get('generation_model', self.generation_model)
                self.embedding_model = config.get('embedding_model', self.embedding_model)
                self.max_output_tokens = config.get('max_output_tokens', self.max_output_tokens)
                self.temperature = config.get('temperature', self.temperature)
                self.top_p = config.get('top_p', self.top_p)
                self.top_k = config.get('top_k', self.top_k)
        
        # Initialize the Gemini client
        try:
            self.client = genai.Client(api_key=self.api_key)
            logger.info(f"Initialized GeminiClient with generation_model={self.generation_model}, embedding_model={self.embedding_model}")
        except Exception as e:
            merit_error = self._convert_gemini_error(e, "client_initialization")
            logger.error(f"Failed to initialize Gemini client: {merit_error}")
            raise merit_error
    
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
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get headers for Gemini API requests.
        
        Returns:
            Dict[str, str]: The headers.
        """
        return {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
    
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
        Gemini uses API key authentication, so login is not applicable.
        
        Returns:
            bool: True if API key is present, False otherwise.
        """
        return self.is_authenticated
    
    def get_token(self) -> Optional[str]:
        """
        Get the API key (token equivalent for Gemini).
        
        Returns:
            Optional[str]: The API key, or None if not set.
        """
        return self.api_key
    
    @validate_text_response
    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The prompt to generate text from.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature for text generation.
            system_prompt: A system prompt to use.
            **kwargs: Additional arguments.
            
        Returns:
            str: The generated text.
        """
        try:
            # Set up the generation config
            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens or self.max_output_tokens,
                temperature=temperature or self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
            )
            
            # Add system prompt if provided
            if system_prompt:
                config.system_instruction = system_prompt
            
            logger.info(f"Generating text with model {self.generation_model}")
            
            # Generate the content
            response = self.client.models.generate_content(
                model=self.generation_model,
                contents=prompt,
                config=config
            )
            
            # Return the generated text
            return response.text
            
        except Exception as e:
            merit_error = self._convert_gemini_error(e, "generate_content")
            return self._handle_api_error(merit_error) or ""
    
    @cache_embeddings
    @validate_embeddings_response
    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Get embeddings for the given texts.
        
        Args:
            texts: A string or list of strings to get embeddings for.
            
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
            For a single input text, still returns a list containing one embedding.
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        try:
            logger.info(f"Getting embeddings for {len(texts)} texts using model {self.embedding_model}")
            
            for text in texts:
                result = self.client.models.embed_content(
                    model=self.embedding_model,
                    contents=text
                )
                
                # Convert ContentEmbedding to a list of floats
                if hasattr(result.embeddings, 'values'):
                    # If it's a ContentEmbedding object with a 'values' attribute
                    if hasattr(result.embeddings.values, 'tolist'):
                        embedding_values = result.embeddings.values.tolist()
                    else:
                        embedding_values = list(result.embeddings.values)
                elif isinstance(result.embeddings, list) and len(result.embeddings) == 1:
                    # If it's a list containing a single embedding
                    if hasattr(result.embeddings[0], 'values'):
                        # If the embedding has a 'values' attribute
                        if hasattr(result.embeddings[0].values, 'tolist'):
                            embedding_values = result.embeddings[0].values.tolist()
                        else:
                            embedding_values = list(result.embeddings[0].values)
                    else:
                        # If the embedding is already a list
                        embedding_values = result.embeddings[0]
                else:
                    # Fallback to using the embeddings as is
                    if hasattr(result.embeddings, 'tolist'):
                        embedding_values = result.embeddings.tolist()
                    else:
                        embedding_values = list(result.embeddings)
                
                embeddings.append(embedding_values)
            
            return embeddings
            
        except Exception as e:
            merit_error = self._convert_gemini_error(e, "embed_content")
            return self._handle_api_error(merit_error) or [[] for _ in texts]
    
    def create_chat_session(self, model: Optional[str] = None) -> Any:
        """
        Create a chat session for multi-turn conversations.
        
        Args:
            model: Model to use for the chat session. If None, uses the default generation model.
            
        Returns:
            Chat session object.
        """
        try:
            chat_model = model or self.generation_model
            logger.info(f"Creating chat session with model {chat_model}")
            
            chat = self.client.chats.create(model=chat_model)
            return chat
            
        except Exception as e:
            merit_error = self._convert_gemini_error(e, "create_chat")
            logger.error(f"Failed to create chat session: {merit_error}")
            return None
    
    def list_models(self) -> List[str]:
        """
        List available models from Gemini.
        
        Returns:
            List[str]: List of model names.
        """
        try:
            logger.info("Listing available models")
            
            models = []
            for model in self.client.models.list():
                models.append(model.name)
            
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
            count_model = model or self.generation_model
            logger.info(f"Counting tokens with model {count_model}")
            
            response = self.client.models.count_tokens(
                model=count_model,
                contents=text
            )
            
            return response.total_tokens
            
        except Exception as e:
            merit_error = self._convert_gemini_error(e, "count_tokens")
            logger.error(f"Failed to count tokens: {merit_error}")
            return 0
