"""
MERIT API Error Classes

This module defines specific API-related error classes for the MERIT system.
"""

from typing import Dict, Any, Optional
from merit.core.errors import MeritAPIError


class APIAuthenticationError(MeritAPIError):
    """Raised when API authentication fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "001", 
        request_id: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Authentication with the API service failed."
        help_text = (
            "Please check your API key and ensure it is valid. "
            "Verify that the API key has been correctly set in your configuration. "
            "If using environment variables, ensure MERIT_API_KEY or the service-specific "
            "API key variable (e.g., OPENAI_API_KEY) is correctly set."
        )
        super().__init__(message or default_message, code, request_id, details, help_text)


class APIRateLimitError(MeritAPIError):
    """Raised when API rate limit is exceeded."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "002", 
        request_id: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None, 
        retry_after: Optional[int] = None
    ):
        """
        Initialize the rate limit error.
        
        Args:
            message: The error message
            code: The error code
            request_id: The API request ID
            details: Additional error details
            retry_after: Seconds to wait before retrying
        """
        self.retry_after = retry_after
        default_message = "API rate limit exceeded."
        help_text = (
            "The API provider's rate limit has been reached. "
            f"{'Please retry after ' + str(retry_after) + ' seconds. ' if retry_after else 'Please retry after some time. '}"
            "Consider implementing request batching or increasing the delay between requests. "
            "If this error persists, you may need to upgrade your API plan for higher rate limits."
        )
        super().__init__(message or default_message, code, request_id, details, help_text)


class APIConnectionError(MeritAPIError):
    """Raised when connection to the API fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "003", 
        request_id: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to connect to the API service."
        help_text = (
            "Please check your internet connection. "
            "Verify that the API service is available and not experiencing downtime. "
            "If you're using a proxy or VPN, ensure it's configured correctly. "
            "Try again after a few minutes if this appears to be a temporary issue."
        )
        super().__init__(message or default_message, code, request_id, details, help_text)


class APIResourceNotFoundError(MeritAPIError):
    """Raised when a requested API resource is not found."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "004", 
        request_id: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "The requested API resource or endpoint was not found."
        help_text = (
            "Please check that you're using the correct API endpoint. "
            "Verify the resource identifier is valid. "
            "Ensure the API version you're using supports this resource. "
            "Check the API documentation for any recent changes to endpoints."
        )
        super().__init__(message or default_message, code, request_id, details, help_text)


class APIServerError(MeritAPIError):
    """Raised when the API server returns a 5xx error."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "005", 
        request_id: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "The API service is experiencing internal issues."
        help_text = (
            "This is a server-side error with the API provider. "
            "It's usually temporary - please try again after a few minutes. "
            "If the issue persists, check the service status page or contact the API provider."
        )
        super().__init__(message or default_message, code, request_id, details, help_text)


class APITimeoutError(MeritAPIError):
    """Raised when an API request times out."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "006", 
        request_id: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "The API request timed out."
        help_text = (
            "The request took too long to complete and was terminated. "
            "This could be due to high server load, a complex query, or network issues. "
            "Try a simpler request, or retry at a less busy time. "
            "If you're processing large amounts of data, consider breaking it into smaller batches."
        )
        super().__init__(message or default_message, code, request_id, details, help_text)


class APIInvalidRequestError(MeritAPIError):
    """Raised when an API request is invalid."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "007", 
        request_id: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "The API request was invalid."
        help_text = (
            "The request was rejected by the API service due to invalid parameters or format. "
            "Check your input parameters for any formatting issues or invalid values. "
            "Refer to the API documentation for the correct parameter format and constraints. "
            "Verify that you're using the correct API version for your request."
        )
        super().__init__(message or default_message, code, request_id, details, help_text)
