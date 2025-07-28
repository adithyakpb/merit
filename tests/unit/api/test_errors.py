"""
Tests for the MERIT API error handling.

This module tests the API-specific error classes defined in merit.api.errors.
"""

import pytest
import re
from datetime import datetime

from merit.core.errors import MeritBaseError, MeritAPIError
from merit.api.errors import (
    MeritAPIAuthenticationError,
    MeritAPIRateLimitError,
    MeritAPIConnectionError,
    MeritAPIResourceNotFoundError,
    MeritAPIServerError,
    MeritAPITimeoutError,
    MeritAPIInvalidRequestError
)


class TestAPIErrorInheritance:
    """Tests for API error inheritance."""
    
    def test_api_error_inheritance(self):
        """Test that API errors inherit correctly."""
        auth_error = MeritAPIAuthenticationError()
        rate_limit_error = MeritAPIRateLimitError()
        connection_error = MeritAPIConnectionError()
        resource_not_found_error = MeritAPIResourceNotFoundError()
        server_error = MeritAPIServerError()
        timeout_error = MeritAPITimeoutError()
        invalid_request_error = MeritAPIInvalidRequestError()
        
        # All should be instances of MeritAPIError
        assert isinstance(auth_error, MeritAPIError)
        assert isinstance(rate_limit_error, MeritAPIError)
        assert isinstance(connection_error, MeritAPIError)
        assert isinstance(resource_not_found_error, MeritAPIError)
        assert isinstance(server_error, MeritAPIError)
        assert isinstance(timeout_error, MeritAPIError)
        assert isinstance(invalid_request_error, MeritAPIError)
        
        # All should be instances of MeritBaseError
        assert isinstance(auth_error, MeritBaseError)
        assert isinstance(rate_limit_error, MeritBaseError)
        assert isinstance(connection_error, MeritBaseError)
        assert isinstance(resource_not_found_error, MeritBaseError)
        assert isinstance(server_error, MeritBaseError)
        assert isinstance(timeout_error, MeritBaseError)
        assert isinstance(invalid_request_error, MeritBaseError)


class TestAPIErrorCodes:
    """Tests for API error codes."""
    
    def test_api_error_codes(self):
        """Test that API errors have the correct error codes."""
        # Check each error type has the expected code
        auth_error = MeritAPIAuthenticationError()
        assert "API-001" in str(auth_error)
        
        rate_limit_error = MeritAPIRateLimitError()
        assert "API-002" in str(rate_limit_error)
        
        connection_error = MeritAPIConnectionError()
        assert "API-003" in str(connection_error)
        
        resource_not_found_error = MeritAPIResourceNotFoundError()
        assert "API-004" in str(resource_not_found_error)
        
        server_error = MeritAPIServerError()
        assert "API-005" in str(server_error)
        
        timeout_error = MeritAPITimeoutError()
        assert "API-006" in str(timeout_error)
        
        invalid_request_error = MeritAPIInvalidRequestError()
        assert "API-007" in str(invalid_request_error)


class TestAPIErrorMessages:
    """Tests for API error messages."""
    
    def test_default_messages(self):
        """Test that API errors provide meaningful default messages."""
        auth_error = MeritAPIAuthenticationError()
        assert "Authentication" in str(auth_error)
        
        rate_limit_error = MeritAPIRateLimitError()
        assert "rate limit" in str(rate_limit_error).lower()
        
        connection_error = MeritAPIConnectionError()
        assert "connect" in str(connection_error).lower()
        
        resource_not_found_error = MeritAPIResourceNotFoundError()
        assert "not found" in str(resource_not_found_error).lower()
        
        server_error = MeritAPIServerError()
        assert "internal issues" in str(server_error).lower()
        
        timeout_error = MeritAPITimeoutError()
        assert "timed out" in str(timeout_error).lower()
        
        invalid_request_error = MeritAPIInvalidRequestError()
        assert "invalid" in str(invalid_request_error).lower()
    
    def test_custom_messages(self):
        """Test that custom messages override defaults."""
        custom_message = "Custom error message"
        error = MeritAPIAuthenticationError(message=custom_message)
        assert custom_message in str(error)
        assert "Authentication with the API service failed" not in str(error)


class TestAPIErrorDetails:
    """Tests for API error details."""
    
    def test_request_id(self):
        """Test that request IDs are correctly included."""
        request_id = "req-12345"
        error = MeritAPIAuthenticationError(request_id=request_id)
        assert f"Request ID: {request_id}" in str(error)
    
    def test_details_storage(self):
        """Test that error details are correctly stored."""
        details = {"method": "GET", "endpoint": "/users"}
        error = MeritAPIResourceNotFoundError(details=details)
        assert error.details == details
        assert error.details["method"] == "GET"
        assert error.details["endpoint"] == "/users"


class TestRateLimitError:
    """Tests specific to APIRateLimitError."""
    
    def test_retry_after(self):
        """Test that APIRateLimitError correctly handles retry_after."""
        retry_after = 30  # seconds
        error = MeritAPIRateLimitError(retry_after=retry_after)
        
        # Should have retry_after attribute
        assert hasattr(error, "retry_after")
        assert error.retry_after == retry_after
        
        # Should include retry time in help text
        assert f"retry after {retry_after} seconds" in str(error).lower()


class TestAPIErrorHelpText:
    """Tests for API error help text."""
    
    def test_authentication_error_help(self):
        """Test that APIAuthenticationError provides useful help text."""
        error = MeritAPIAuthenticationError()
        help_text = str(error)
        
        # Should mention API key
        assert "api key" in help_text.lower()
        assert "check" in help_text.lower()
    
    def test_rate_limit_error_help(self):
        """Test that APIRateLimitError provides useful help text."""
        error = MeritAPIRateLimitError()
        help_text = str(error)
        
        # Should mention retry or rate limits
        assert "rate limit" in help_text.lower()
        assert "retry" in help_text.lower()
    
    def test_connection_error_help(self):
        """Test that APIConnectionError provides useful help text."""
        error = MeritAPIConnectionError()
        help_text = str(error)
        
        # Should mention internet connection
        assert "internet connection" in help_text.lower()
        assert "check" in help_text.lower()
