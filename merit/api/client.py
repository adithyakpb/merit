"""
API Client Implementation

This module provides API client implementations that can be extended
for specific APIs, including a generic AI API client and an OpenAI client.
"""

import os
import requests
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

from .base import BaseAPIClient, BaseAPIClientConfig
from ..api.base import validate_embeddings_response, validate_text_response
from ..core.logging import get_logger
from ..core.cache import cache_embeddings, is_caching_available
from ..core.utils import parse_json


logger = get_logger(__name__)


class AIAPIClientConfig(BaseAPIClientConfig):
    """
    Configuration class for API clients.
    
    This class handles configuration for API clients and can be initialized
    from different sources including environment variables, config files,
    or explicit parameters.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        login_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        environment: Optional[str] = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the API client configuration.
        
        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API.
            login_url: URL for login authentication.
            username: Username for authentication.
            password: Password for authentication.
            client_id: Client ID for the API.
            user_id: User ID for the API.
            environment: Environment for the API (e.g., "qa", "prod").
            model: Model to use for text generation.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(api_key=api_key, base_url=base_url)
        self.login_url = login_url
        self.username = username
        self.password = password
        self.client_id = client_id
        self.user_id = user_id
        self.environment = environment
        self.model = model
        self._additional_params.update(kwargs)


class AIAPIClient(BaseAPIClient):
    """
    Generic API client implementation.
    
    This class provides a generic implementation of the BaseAPIClient interface
    that can be extended for specific APIs.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        login_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        client_id: Optional[str] = None,
        user_id: Optional[str] = None,
        environment: Optional[str] = None,
        model: Optional[str] = None,
        token: Optional[str] = None,
        config: Optional[Union[AIAPIClientConfig, Dict[str, Any]]] = None,
        env_file: Optional[str] = None,
        required_vars: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the generic API client.
        
        This constructor supports three initialization methods:
        1. Direct parameters: Pass individual parameters directly
        2. Configuration object: Pass a config object or dictionary
        3. Environment variables: Set env_file and/or required_vars to load from environment
        
        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API.
            login_url: URL for login authentication.
            username: Username for authentication.
            password: Password for authentication.
            client_id: Client ID for the API.
            user_id: User ID for the API.
            environment: Environment for the API (e.g., "qa", "prod").
            model: Model to use for text generation.
            token: Authentication token (if already available).
            config: Configuration object or dictionary containing client configuration.
            env_file: Path to the .env file to load. If provided, load configuration from environment.
            required_vars: List of environment variable names that are required when loading from environment.
                          If None, defaults to ["BASE_URL"].
            **kwargs: Additional parameters to store.
                          
        Raises:
            ValueError: If any required environment variables are missing when loading from environment.
            FileNotFoundError: If the specified env_file does not exist.
            TypeError: If initialization fails due to missing required parameters.
        """
        # Initialize attributes with default values
        self.api_key = api_key
        self.base_url = base_url
        self.login_url = login_url
        self.username = username
        self.password = password
        self.client_id = client_id
        self.user_id = user_id
        self.environment = environment
        self.model = model
        self._token = token
        self._additional_params = {}
        
        # Process config object if provided
        if config is not None:
            config_dict = config.to_dict() if hasattr(config, 'to_dict') else config
            self._update_attributes(config_dict)
        
        # Process environment variables if requested
        if env_file is not None or required_vars is not None:
            try:
                # Get constructor parameter names (excluding special params)
                import inspect
                param_names = [p for p in inspect.signature(self.__init__).parameters.keys() 
                              if p not in ('self', 'config', 'env_file', 'required_vars', 'kwargs')]
                
                # Convert to uppercase for environment variables
                env_vars = [p.upper() for p in param_names]
                
                # Load from environment
                env_values = self.load_from_env(
                    env_file=env_file,
                    required_vars=required_vars,
                    supported_vars=env_vars
                )
                
                # Update attributes from environment
                self._update_attributes(env_values)
                    
            except (ValueError, FileNotFoundError) as e:
                logger.error(f"Failed to initialize client from environment: {str(e)}")
                raise
        
        # Process any additional kwargs
        self._update_attributes(kwargs)
        
        logger.info(f"Initialized AIAPIClient with base_url={self.base_url}, login_url={self.login_url}")
    
    def _update_attributes(self, source_dict: Dict[str, Any]) -> None:
        """
        Update instance attributes from a dictionary.
        
        Args:
            source_dict: Dictionary containing attribute values.
        """
        for key, value in source_dict.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
            elif value is not None:
                # Store unknown parameters in _additional_params
                self._additional_params[key] = value
    
    @classmethod
    def get_supported_env_vars(cls) -> List[str]:
        """
        Get the list of supported environment variable names.
        
        Returns:
            List[str]: List of supported environment variable names.
        """
        # Get constructor parameter names (excluding special params)
        import inspect
        param_names = [p for p in inspect.signature(cls.__init__).parameters.keys() 
                      if p not in ('self', 'config', 'env_file', 'required_vars', 'kwargs')]
        
        # Convert to uppercase for environment variables
        return [p.upper() for p in param_names]
    
    def login(self) -> bool:
        """
        Authenticate with the API.
        
        Returns:
            bool: True if authentication was successful, False otherwise.
        """
        if not self.login_url or not self.username or not self.password:
            logger.error("Login failed: Missing login URL, username, or password")
            return False
        
        try:
            logger.info(f"Logging in with username={self.username} at {self.login_url}")
            
            response = requests.post(
                self.login_url,
                json={
                    "username": self.username,
                    "password": self.password
                },
                headers={"Content-Type": "application/json"}
            )
            
            response.raise_for_status()
            data = response.json()
            
            if "token" in data:
                self._token = data["token"]
                logger.info("Login successful")
                return True
            else:
                logger.error("Login failed: No token in response")
                return False
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Login failed: {str(e)}")
            return False
    
    @cache_embeddings
    @validate_embeddings_response
    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Get embeddings for the given texts.
        
        Args:
            texts: A string or list of strings to get embeddings for.
            
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
            
        Note:
            This method is decorated with @validate_embeddings_response to ensure
            that all implementations return data in the expected format:
            - A list of embeddings, where each embedding is a list of floats
            - For a single input text, still returns a list containing one embedding
            - If the API call fails, returns a list of empty lists matching the length of the input texts
        """
        if not self.is_authenticated:
            logger.warning("Not authenticated, attempting to login")
            if not self.login():
                raise ValueError("Authentication required for embeddings")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            logger.info(f"Getting embeddings for {len(texts)} texts")
            
            # This is a generic implementation that should be overridden by subclasses
            # to handle specific API formats
            response = requests.post(
                f"{self.base_url}/embeddings",
                json={"texts": texts},
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract embeddings from the response
            # This is a generic implementation that should be overridden by subclasses
            if "embeddings" in data:
                return data["embeddings"]
            else:
                logger.error("No embeddings in response")
                return [[] for _ in texts]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get embeddings: {str(e)}")
            return [[] for _ in texts]
    
    @validate_text_response
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the given prompt.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            str: The generated text.
            
        Note:
            This method is decorated with @validate_text_response to ensure
            that all implementations return data in the expected format:
            - A string containing the generated text
            - If the API call fails, returns an empty string
        """
        if not self.is_authenticated:
            logger.warning("Not authenticated, attempting to login")
            if not self.login():
                raise ValueError("Authentication required for text generation")
        
        try:
            logger.info(f"Generating text for prompt: {prompt[:50]}...")
            
            # This is a generic implementation that should be overridden by subclasses
            # to handle specific API formats
            response = requests.post(
                f"{self.base_url}/generate",
                json={"prompt": prompt, **kwargs},
                headers=self._get_headers()
            )
            
            response.raise_for_status()
            data = response.json()
            
            # Extract text from the response
            # This is a generic implementation that should be overridden by subclasses
            if "text" in data:
                return data["text"]
            else:
                logger.error("No text in response")
                return ""
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate text: {str(e)}")
            return ""
    
    @property
    def is_authenticated(self) -> bool:
        """
        Check if the client is authenticated.
        
        Returns:
            bool: True if the client is authenticated, False otherwise.
        """
        return self._token is not None
    
    def get_token(self) -> Optional[str]:
        """
        Get the authentication token.
        
        Returns:
            Optional[str]: The authentication token, or None if not authenticated.
        """
        return self._token
    
    def _get_headers(self) -> Dict[str, str]:
        """
        Get the headers for API requests.
        
        Returns:
            Dict[str, str]: The headers.
        """
        headers = {"Content-Type": "application/json"}
        
        if self.api_key:
            headers["x-api-key"] = self.api_key
        
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        
        return headers


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
    
    This client provides access to OpenAI's API for embeddings and text generation.
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
        
        # Initialize the parent class
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
            AIAPIClient expects: {"embeddings": [[...], ...]}
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
                # Return in the format expected by AIAPIClient
                return embeddings
            else:
                logger.error("No embeddings in response")
                return [[] for _ in texts]
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get embeddings: {str(e)}")
            return [[] for _ in texts]
    
    def generate_text(self, prompt: str, **kwargs) -> str:
        """
        Generate text based on the given prompt using OpenAI's chat completions API.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            str: The generated text.
            
        Note:
            This method transforms OpenAI's response format to match the AIAPIClient format.
            OpenAI returns: {"choices": [{"message": {"content": "..."}}]}
            AIAPIClient expects: {"text": "..."}
        """
        # Set default parameters
        params = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000,
        }
        
        # Update with any provided kwargs
        params.update(kwargs)
        
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
                # Return in the format expected by AIAPIClient
                return data["choices"][0]["message"]["content"]
            else:
                logger.error("No text in response")
                return ""
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to generate text: {str(e)}")
            return ""
    
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
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to create chat completion: {str(e)}")
            return {"error": str(e)}
    
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
        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []
