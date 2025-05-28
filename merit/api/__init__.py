"""
MERIT API Client Module

This module provides API client implementations for interacting with various LLM APIs.
"""

from .base import BaseAPIClient, BaseAPIClientConfig, validate_embeddings_response, validate_text_response
from .client import AIAPIClient, AIAPIClientConfig
from .openai_client import OpenAIClient, OpenAIClientConfig
from .gemini_client import GeminiClient, GeminiClientConfig
from .run_config import AdaptiveDelay, adaptive_throttle
from .errors import (
    MeritAPIAuthenticationError,
    MeritAPIRateLimitError,
    MeritAPIConnectionError,
    MeritAPIResourceNotFoundError,
    MeritAPIServerError,
    MeritAPITimeoutError,
    MeritAPIInvalidRequestError
)
