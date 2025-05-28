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
        strict: bool = False,
        enable_retries: bool = True,
        enable_throttling: bool = True,
        max_retries: int = 3,
        backoff_factor: float = 0.5,
        initial_delay: float = 0.5,
        min_delay: float = 0.05,
        max_delay: float = 2.0,
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
            strict: If True, raise exceptions on API failures. If False, return None/empty gracefully.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(
            api_key=api_key, 
            base_url=base_url,
            enable_retries=enable_retries,
            enable_throttling=enable_throttling,
            max_retries=max_retries,
            backoff_factor=backoff_factor,
            initial_delay=initial_delay,
            min_delay=min_delay,
            max_delay=max_delay
        )
        self.login_url = login_url
        self.username = username
        self.password = password
        self.client_id = client_id
        self.user_id = user_id
        self.environment = environment
        self.model = model
        self.strict = strict
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
        strict: bool = False,
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
        self.strict = strict
        
        # Retry and throttling configuration
        self.enable_retries = kwargs.get('enable_retries', True)
        self.enable_throttling = kwargs.get('enable_throttling', True)
        self.max_retries = kwargs.get('max_retries', 3)
        self.backoff_factor = kwargs.get('backoff_factor', 0.5)
        self.initial_delay = kwargs.get('initial_delay', 0.5)
        self.min_delay = kwargs.get('min_delay', 0.05)
        self.max_delay = kwargs.get('max_delay', 2.0)
        
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
        logger.info(f"Retry/throttling config: retries={self.enable_retries}, throttling={self.enable_throttling}")
        
        # Apply decorators dynamically based on configuration
        self._apply_decorators()
    
    def _apply_decorators(self) -> None:
        """
        Apply retry and throttling decorators dynamically based on configuration.
        
        This method wraps the original API methods with decorators if enabled,
        using the client's configuration parameters.
        """
        from .run_config import with_retry, adaptive_throttle, with_adaptive_retry
        
        # List of methods to potentially decorate
        api_methods = ['get_embeddings', 'generate_text']
        
        for method_name in api_methods:
            if hasattr(self, method_name):
                original_method = getattr(self, method_name)
                
                # Skip if already decorated (avoid double decoration)
                if hasattr(original_method, '_merit_decorated'):
                    continue
                
                decorated_method = original_method
                
                # Apply retry decorator if enabled
                if self.enable_retries:
                    decorated_method = with_retry(
                        max_retries=self.max_retries,
                        backoff_factor=self.backoff_factor
                    )(decorated_method)
                    logger.debug(f"Applied retry decorator to {method_name} (max_retries={self.max_retries})")
                
                # Apply throttling decorator if enabled
                if self.enable_throttling:
                    # Create a custom throttling decorator with client-specific parameters
                    from .run_config import AdaptiveDelay
                    
                    # We need to create a custom decorator that uses the client's throttling config
                    def create_custom_throttle(initial_delay, min_delay, max_delay):
                        def custom_throttle(func):
                            from functools import wraps
                            import time
                            
                            # Create an adaptive delay instance for this method
                            delay_instance = AdaptiveDelay(
                                initial_delay=initial_delay,
                                min_delay=min_delay,
                                max_delay=max_delay
                            )
                            
                            @wraps(func)
                            def wrapper(*args, **kwargs):
                                # Wait before making the API call
                                delay_instance.wait()
                                
                                try:
                                    result = func(*args, **kwargs)
                                    delay_instance.success()
                                    return result
                                except Exception as e:
                                    # Check if it's a rate limit error
                                    from .errors import MeritAPIRateLimitError
                                    if isinstance(e, MeritAPIRateLimitError) or \
                                       (hasattr(e, 'response') and hasattr(e.response, 'status_code') and e.response.status_code == 429):
                                        delay_instance.failure()
                                    raise
                            
                            return wrapper
                        return custom_throttle
                    
                    throttle_decorator = create_custom_throttle(
                        self.initial_delay,
                        self.min_delay,
                        self.max_delay
                    )
                    decorated_method = throttle_decorator(decorated_method)
                    logger.debug(f"Applied throttling decorator to {method_name} (initial_delay={self.initial_delay})")
                
                # Mark as decorated to avoid double decoration
                decorated_method._merit_decorated = True
                
                # Replace the original method with the decorated version
                setattr(self, method_name, decorated_method)
                
                if self.enable_retries or self.enable_throttling:
                    features = []
                    if self.enable_retries:
                        features.append(f"retries(max={self.max_retries})")
                    if self.enable_throttling:
                        features.append(f"throttling(delay={self.initial_delay}s)")
                    logger.info(f"Enhanced {method_name} with {', '.join(features)}")
    
    def _convert_requests_error(self, error: Exception, endpoint: str = "") -> Exception:
        """
        Convert requests exceptions to MeritAPI errors with precise detection and exception chaining.
        
        Args:
            error: The exception to convert (typically from requests).
            endpoint: The API endpoint that failed (for context).
            
        Returns:
            Exception: The appropriate MeritAPI error with original exception chained.
        """
        details = {
            "original_error": str(error),
            "original_type": type(error).__name__,
            "endpoint": endpoint
        }
        
        # Precise detection using isinstance() for known requests exceptions
        if isinstance(error, requests.exceptions.ConnectionError):
            merit_error = MeritAPIConnectionError(
                "Failed to connect to the API service",
                details=details
            )
            merit_error.__cause__ = error
            return merit_error
        elif isinstance(error, requests.exceptions.Timeout):
            merit_error = MeritAPITimeoutError(
                "API request timed out",
                details=details
            )
            merit_error.__cause__ = error
            return merit_error
        elif isinstance(error, requests.exceptions.HTTPError):
            if hasattr(error, 'response') and error.response is not None:
                status_code = error.response.status_code
                details["status_code"] = status_code
                
                if status_code == 400:
                    merit_error = MeritAPIInvalidRequestError(
                        "Invalid request parameters",
                        details=details
                    )
                    merit_error.__cause__ = error
                    return merit_error
                elif status_code in (401, 403):
                    merit_error = MeritAPIAuthenticationError(
                        "Authentication failed",
                        details=details
                    )
                    merit_error.__cause__ = error
                    return merit_error
                elif status_code == 404:
                    merit_error = MeritAPIResourceNotFoundError(
                        "API endpoint not found",
                        details=details
                    )
                    merit_error.__cause__ = error
                    return merit_error
                elif status_code == 429:
                    retry_after = None
                    if 'Retry-After' in error.response.headers:
                        try:
                            retry_after = int(error.response.headers['Retry-After'])
                        except (ValueError, TypeError):
                            pass
                    merit_error = MeritAPIRateLimitError(
                        "API rate limit exceeded",
                        details=details,
                        retry_after=retry_after
                    )
                    merit_error.__cause__ = error
                    return merit_error
                elif status_code >= 500:
                    merit_error = MeritAPIServerError(
                        "API server error",
                        details=details
                    )
                    merit_error.__cause__ = error
                    return merit_error
            
            # Fallback for HTTP errors without response
            merit_error = MeritAPIServerError(
                "HTTP error occurred",
                details=details
            )
            merit_error.__cause__ = error
            return merit_error
        elif isinstance(error, requests.exceptions.RequestException):
            # Other requests exceptions (like ReadTimeout, etc.)
            merit_error = MeritAPIConnectionError(
                "Request failed",
                details=details
            )
            merit_error.__cause__ = error
            return merit_error
        else:
            # Non-requests exceptions - preserve original error type info
            merit_error = MeritAPIServerError(
                f"Unexpected error occurred: {type(error).__name__}",
                details=details
            )
            merit_error.__cause__ = error
            return merit_error
    
    def _handle_api_error(self, error: Exception, strict: Optional[bool] = None) -> Optional[Any]:
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
        use_strict = strict if strict is not None else self.strict
        
        if use_strict:
            # Strict mode: ERROR level, then raise
            logger.error(f"API call failed (strict mode): {error}")
            raise error
        else:
            # Graceful mode: WARNING level, then return None
            logger.warning(f"API call failed (graceful mode): {error}")
            return None
    
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
            merit_error = self._convert_requests_error(e, "login")
            logger.error(f"Login failed: {merit_error}")
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
                raise MeritAPIAuthenticationError(
                    "Authentication required for embeddings",
                    details={"endpoint": "embeddings"}
                )
        
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
            merit_error = self._convert_requests_error(e, "embeddings")
            return self._handle_api_error(merit_error) or [[] for _ in texts]
    
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
                raise MeritAPIAuthenticationError(
                    "Authentication required for text generation",
                    details={"endpoint": "generate"}
                )
        
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
            merit_error = self._convert_requests_error(e, "generate")
            return self._handle_api_error(merit_error) or ""
    
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
