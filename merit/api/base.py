"""
Base API Client Interface

This module defines the base interfaces for all API clients and configurations in the MERIT system.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional,  Callable
import os
import inspect
from functools import wraps
from dotenv import load_dotenv

from merit.core.logging import get_logger

logger = get_logger(__name__)


def validate_embeddings_response(func: Callable) -> Callable:
    """
    Decorator to validate the response format of get_embeddings method.
    
    Args:
        func: The get_embeddings method to validate.
        
    Returns:
        Callable: The wrapped function that validates the response.
    """
    @wraps(func)
    def wrapper(self, texts, *args, **kwargs):
        # Call the original function
        result = func(self, texts, *args, **kwargs)
        
        # Validate the result
        if not isinstance(result, list):
            logger.error(f"Invalid response format from {func.__name__}: expected list, got {type(result)}")
            # Convert to expected format
            return []
        
        # Ensure each embedding is a list of floats
        for i, embedding in enumerate(result):
            if not isinstance(embedding, list):
                logger.error(f"Invalid embedding format at index {i}: expected list, got {type(embedding)}")
                result[i] = []
            else:
                # Check if all elements are numeric
                for j, value in enumerate(embedding):
                    if not isinstance(value, (int, float)):
                        logger.error(f"Invalid value in embedding {i} at position {j}: expected numeric, got {type(value)}")
                        embedding[j] = 0.0
        
        return result
    
    return wrapper


def validate_text_response(func: Callable) -> Callable:
    """
    Decorator to validate the response format of generate_text method.
    
    Args:
        func: The generate_text method to validate.
        
    Returns:
        Callable: The wrapped function that validates the response.
    """
    @wraps(func)
    def wrapper(self, prompt, *args, **kwargs):
        # Call the original function
        result = func(self, prompt, *args, **kwargs)
        
        # Validate the result
        if not isinstance(result, str):
            logger.error(f"Invalid response format from {func.__name__}: expected str, got {type(result)}")
            # Convert to expected format
            return ""
        
        return result
    
    return wrapper


import inspect

class BaseAPIClientConfig(ABC):
    """
    Abstract base class for API client configurations.
    
    This class defines the interface and common functionality for all API client
    configurations in the system.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the base API client configuration.
        
        Args:
            api_key: API key for authentication.
            base_url: Base URL for the API.
            **kwargs: Additional configuration parameters.
        """
        self.api_key = api_key
        self.base_url = base_url
        self._additional_params = kwargs
    
    @classmethod
    def _get_constructor_params(cls) -> List[str]:
        """
        Get parameter names from the constructor.
        
        Returns:
            List[str]: List of parameter names from the constructor.
        """
        params = list(inspect.signature(cls.__init__).parameters.keys())
        # Remove 'self' and 'kwargs'
        return [p for p in params if p not in ('self', 'kwargs')]
    
    @classmethod
    def get_supported_env_vars(cls) -> List[str]:
        """
        Get the list of supported environment variable names.
        
        Returns:
            List[str]: List of supported environment variable names.
        """
        # Convert constructor parameter names to uppercase for env vars
        return [param.upper() for param in cls._get_constructor_params()]
    
    @classmethod
    def from_env(
        cls, 
        env_file: Optional[str] = None, 
        required_vars: Optional[List[str]] = None
    ) -> 'BaseAPIClientConfig':
        """
        Create a configuration from environment variables.
        
        Args:
            env_file: Path to the .env file to load. If None, the default .env file is used.
            required_vars: List of environment variable names that are required.
                          If None, defaults to ["BASE_URL"].
            
        Returns:
            BaseAPIClientConfig: A new configuration instance.
            
        Raises:
            ValueError: If any required environment variables are missing.
            FileNotFoundError: If the specified env_file does not exist.
        """
        # Set default required variables if not specified
        if required_vars is None:
            required_vars = ["BASE_URL"]  # At minimum, BASE_URL is required
        
        # Load environment variables
        if env_file:
            # Check if the env file exists
            if not os.path.exists(env_file):
                raise FileNotFoundError(f"Environment file not found: {env_file}")
            loaded = load_dotenv(env_file)
        else:
            loaded = load_dotenv()
        
        if not loaded:
            logger.warning("No .env file loaded or file was empty")
        
        # Check for required environment variables
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Create instance with available environment variables
        config_params = {}
        for var_name in cls.get_supported_env_vars():
            value = os.getenv(var_name)
            if value is not None:
                # Log securely (don't log actual values of sensitive data)
                if var_name in ["API_KEY", "PASSWORD"]:
                    logger.debug(f"Found environment variable: {var_name}=********")
                else:
                    logger.debug(f"Found environment variable: {var_name}={value}")
                config_params[var_name.lower()] = value
        
        return cls(**config_params)
    
    def validate(self, required_params: Optional[List[str]] = None) -> None:
        """
        Validate that required configuration parameters are present.
        
        Args:
            required_params: List of parameter names that are required.
                           If None, defaults to ["base_url"].
                           
        Raises:
            ValueError: If any required parameters are missing.
        """
        if required_params is None:
            required_params = ["base_url"]
        
        missing_params = []
        for param in required_params:
            param_lower = param.lower()
            if not hasattr(self, param_lower) or getattr(self, param_lower) is None:
                if param_lower not in self._additional_params:
                    missing_params.append(param_lower)
        
        if missing_params:
            raise ValueError(f"Missing required configuration parameters: {', '.join(missing_params)}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary for client initialization.
        
        Returns:
            Dict[str, Any]: Dictionary of configuration parameters.
        """
        # Get all instance attributes that don't start with underscore
        config_dict = {
            key: value for key, value in self.__dict__.items() 
            if not key.startswith('_') and value is not None
        }
        
        # Add additional parameters
        config_dict.update(self._additional_params)
        return config_dict


class BaseAPIClient(ABC):
    """
    Abstract base class for API clients.
    
    This class defines the interface that all API clients must implement.
    """
    
    @classmethod
    def get_supported_env_vars(cls) -> List[str]:
        """
        Get the list of supported environment variable names.
        
        Returns:
            List[str]: List of supported environment variable names.
        """
        return ["API_KEY", "BASE_URL"]
    
    @staticmethod
    def load_from_env(
        env_file: Optional[str] = None, 
        required_vars: Optional[List[str]] = None,
        supported_vars: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Load configuration from environment variables.
        
        Args:
            env_file: Path to the .env file to load. If None, the default .env file is used.
            required_vars: List of environment variable names that are required.
                          If None, defaults to ["BASE_URL"].
            supported_vars: List of environment variable names to look for.
                           If None, defaults to ["API_KEY", "BASE_URL"].
            
        Returns:
            Dict[str, Any]: Dictionary of configuration parameters loaded from environment.
            
        Raises:
            ValueError: If any required environment variables are missing.
            FileNotFoundError: If the specified env_file does not exist.
        """
        # Set default required variables if not specified
        if required_vars is None:
            required_vars = ["BASE_URL"]  # At minimum, BASE_URL is required
        
        if supported_vars is None:
            supported_vars = ["API_KEY", "BASE_URL"]
        
        # Load environment variables
        if env_file:
            # Check if the env file exists
            if not os.path.exists(env_file):
                raise FileNotFoundError(f"Environment file not found: {env_file}")
            loaded = load_dotenv(env_file)
        else:
            loaded = load_dotenv()
            
        if not loaded:
            logger.warning("No .env file loaded or file was empty")
        
        # Check for required environment variables
        missing_vars = [var for var in required_vars if os.getenv(var) is None]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        
        # Get environment variables
        env_vars = {}
        for var_name in supported_vars:
            value = os.getenv(var_name)
            if value is not None:
                # Log securely (don't log actual values of sensitive data)
                if var_name in ["API_KEY", "PASSWORD"]:
                    logger.debug(f"Found environment variable: {var_name}=********")
                else:
                    logger.debug(f"Found environment variable: {var_name}={value}")
                env_vars[var_name.lower()] = value
        
        return env_vars
