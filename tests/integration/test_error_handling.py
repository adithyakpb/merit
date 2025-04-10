"""
Integration tests for MERIT error handling.

This module demonstrates how the error handling and retry systems work together
in realistic scenarios.
"""

import pytest
from unittest.mock import patch

from merit.api.run_config import with_retry, adaptive_throttle, with_adaptive_retry
from merit.api.errors import APIRateLimitError, APIConnectionError
from merit.core.errors import MeritBaseError, ValidationError


class MockAPIService:
    """
    Mock API service that simulates various error scenarios.
    
    This class helps test how MERIT's error handling works in realistic API
    interaction scenarios.
    """
    
    def __init__(self, fail_modes=None):
        """
        Initialize the mock API service.
        
        Args:
            fail_modes: Dict with 'connection_fails', 'rate_limit_fails', and
                       'validation_fails' counts
        """
        self.fail_modes = fail_modes or {}
        self.call_history = []
        
        # Counters for different types of calls
        self.connection_calls = 0
        self.rate_limit_calls = 0
        self.validation_calls = 0
    
    def connection_test(self):
        """Simulate an API call that might have connection issues."""
        self.call_history.append(("connection_test",))
        self.connection_calls += 1
        
        # Check if we should simulate a connection failure
        if self.connection_calls <= self.fail_modes.get('connection_fails', 0):
            raise APIConnectionError(
                "Could not connect to API service"
            )
        
        return "Connection successful"
    
    def rate_limited_operation(self):
        """Simulate an API call that might be rate limited."""
        self.call_history.append(("rate_limited_operation",))
        self.rate_limit_calls += 1
        
        # Check if we should simulate a rate limit
        if self.rate_limit_calls <= self.fail_modes.get('rate_limit_fails', 0):
            retry_after = 1  # short for testing
            # Fix the string concatenation issue in the original message
            raise APIRateLimitError(
                "Rate limit exceeded", 
                retry_after=retry_after
            )
        
        return "Operation completed successfully"
    
    def validation_operation(self, data):
        """Simulate an API call that might have validation issues."""
        self.call_history.append(("validation_operation", data))
        self.validation_calls += 1
        
        # Check if we should simulate a validation failure
        if self.validation_calls <= self.fail_modes.get('validation_fails', 0):
            raise ValidationError(
                "Invalid data provided",
                details={"field": "name", "error": "Name is required"}
            )
        
        return f"Data validated: {data}"


class TestIntegrationScenarios:
    """Integration tests for error handling scenarios."""
    
    @pytest.fixture
    def mock_service(self):
        """Create a mock API service."""
        return MockAPIService()
    
    def test_simple_retry_success(self, mock_service):
        """Test that retries eventually succeed."""
        # Configure service to fail twice with connection errors
        service = MockAPIService(fail_modes={'connection_fails': 2})
        
        # Apply retry decorator to the method
        test_method = with_retry(max_retries=3)(service.connection_test)
        
        # Should succeed after retries
        with patch('time.sleep'):  # Mock sleep to speed up tests
            result = test_method()
            assert result == "Connection successful"
            assert service.connection_calls == 3  # 1 original + 2 retries
    
    def test_adaptive_throttle_rate_limit(self, mock_service):
        """Test that adaptive throttling handles rate limits."""
        # Configure service to fail with rate limit error
        service = MockAPIService(fail_modes={'rate_limit_fails': 1})
        
        # Helper function that doesn't get confused with method binding
        def test_fn(*args, **kwargs):
            # Ignore any passed-in self
            return service.rate_limited_operation()
            
        # Apply adaptive throttle decorator
        test_method = adaptive_throttle(test_fn)
        
        # First call should fail with rate limit error
        with patch('time.sleep'):  # Mock sleep to speed up tests
            with pytest.raises(APIRateLimitError):
                test_method()
        
        # Second call should succeed with increased delay
        with patch('time.sleep'):
            result = test_method()
            assert result == "Operation completed successfully"
    
    def test_combined_adaptive_retry(self, mock_service):
        """Test the combined adaptive retry decorator."""
        # Configure service to fail with multiple error types
        service = MockAPIService(fail_modes={
            'connection_fails': 1, 
            'rate_limit_fails': 1
        })
        
        # Helper function that doesn't get confused with method binding
        def test_fn(*args, **kwargs):
            # Ignore any passed-in self
            return service.connection_test()
            
        # Apply combined decorator
        test_method = with_adaptive_retry(max_retries=3)(test_fn)
        
        # Should handle both connection and rate limit issues
        with patch('time.sleep'):  # Mock sleep to speed up tests
            result = test_method()
            assert result == "Connection successful"
    
    def test_validation_error_no_retry(self, mock_service):
        """Test that validation errors are not retried (business logic errors)."""
        # Configure service to fail with validation error
        service = MockAPIService(fail_modes={'validation_fails': 1})
        
        # Apply retry decorator - but it shouldn't retry validation errors
        test_method = with_retry(max_retries=3)(service.validation_operation)
        
        # Should fail immediately without retry
        with patch('time.sleep'):  # Mock sleep to speed up tests
            with pytest.raises(ValidationError) as excinfo:
                test_method("test data")
            
            # Check error details
            error = excinfo.value
            assert error.details["field"] == "name"
            assert "required fields" in str(error).lower()
            
            # Should only be called once (no retries)
            assert service.validation_calls == 1
    
    def test_realistic_api_workflow(self, mock_service):
        """Test a realistic workflow with multiple API calls."""
        # Configure service to have occasional issues
        service = MockAPIService(fail_modes={
            'connection_fails': 1,
            'rate_limit_fails': 0  # Don't fail with rate limits for this test
        })
        
        # Create client class with decorated methods
        class APIClient:
            def __init__(self, service):
                self.service = service
            
            # Helper functions defined as standalone functions
            def connect(self):
                return self.service.connection_test()
            
            def perform_operation(self):
                return self.service.rate_limited_operation()
            
            def validate(self, data):
                return self.service.validation_operation(data)
            
            # Apply decorators
            connect = with_retry(max_retries=2)(connect)
            perform_operation = with_retry(max_retries=3)(perform_operation)
            validate = with_retry(max_retries=2)(validate)
        
        # Create client
        client = APIClient(service)
        
        # Execute workflow
        with patch('time.sleep'):  # Mock sleep to speed up tests
            # First connect (will retry once)
            connect_result = client.connect()
            assert connect_result == "Connection successful"
            
            # Then perform operation (will retry for rate limits)
            operation_result = client.perform_operation()
            assert operation_result == "Operation completed successfully"
            
            # Then validate data
            validation_result = client.validate("test data")
            assert validation_result == "Data validated: test data"
            
            # Check call counts
            assert service.connection_calls == 2  # 1 + 1 retry
            assert service.rate_limit_calls == 1  # Just 1 successful call
            assert service.validation_calls == 1  # No failures
