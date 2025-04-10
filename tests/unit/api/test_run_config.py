"""
Tests for the MERIT API run configuration.

This module tests the functionality in merit.api.run_config, including:
- Adaptive delay mechanism
- Retry mechanism
- Combined adaptive throttle and retry
"""

import pytest
import time
import threading
import requests
import merit.api.run_config
from unittest.mock import MagicMock, patch

from merit.api.run_config import (
    AdaptiveDelay,
    adaptive_throttle,
    with_retry,
    with_adaptive_retry
)
from merit.api.errors import (
    APIConnectionError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError
)


class TestAdaptiveDelay:
    """Tests for the AdaptiveDelay class."""
    
    def test_initialization(self):
        """Test that AdaptiveDelay initializes with correct default values."""
        delay = AdaptiveDelay()
        
        assert delay.current_delay == 0.5  # Default initial delay
        assert delay.min_delay == 0.05     # Default minimum delay
        assert delay.max_delay == 2.0      # Default maximum delay
        assert delay.decrease_factor == 0.9  # Default decrease factor
        assert delay.increase_factor == 1.5  # Default increase factor
        
        assert delay.total_requests == 0
        assert delay.total_failures == 0
        assert delay.total_wait_time == 0
    
    def test_custom_initialization(self):
        """Test that AdaptiveDelay accepts custom initialization values."""
        delay = AdaptiveDelay(
            initial_delay=1.0,
            min_delay=0.1,
            max_delay=5.0,
            decrease_factor=0.8,
            increase_factor=2.0
        )
        
        assert delay.current_delay == 1.0
        assert delay.min_delay == 0.1
        assert delay.max_delay == 5.0
        assert delay.decrease_factor == 0.8
        assert delay.increase_factor == 2.0
    
    def test_success_decreases_delay(self):
        """Test that successful calls decrease the delay."""
        delay = AdaptiveDelay(initial_delay=1.0, decrease_factor=0.9)
        
        # Record a success
        delay.success()
        
        # Delay should decrease
        assert delay.current_delay == 1.0 * 0.9
        assert delay.total_requests == 1
        assert delay.total_failures == 0
    
    def test_failure_increases_delay(self):
        """Test that failed calls increase the delay."""
        delay = AdaptiveDelay(initial_delay=1.0, increase_factor=1.5)
        
        # Record a failure
        delay.failure()
        
        # Delay should increase
        assert delay.current_delay == 1.0 * 1.5
        assert delay.total_requests == 1
        assert delay.total_failures == 1
    
    def test_minimum_delay_constraint(self):
        """Test that delay doesn't go below minimum."""
        delay = AdaptiveDelay(initial_delay=0.1, min_delay=0.05, decrease_factor=0.1)
        
        # Multiple successes should eventually hit minimum
        for _ in range(10):
            delay.success()
        
        # Delay should be constrained to minimum
        assert delay.current_delay == 0.05
    
    def test_maximum_delay_constraint(self):
        """Test that delay doesn't go above maximum."""
        delay = AdaptiveDelay(initial_delay=1.0, max_delay=2.0, increase_factor=10.0)
        
        # A failure with large increase factor should hit maximum
        delay.failure()
        
        # Delay should be constrained to maximum
        assert delay.current_delay == 2.0
    
    @patch('time.sleep')
    def test_wait_method(self, mock_sleep):
        """Test the wait method."""
        delay = AdaptiveDelay(initial_delay=0.5)
        
        # Call wait
        delay.wait()
        
        # Should have called sleep with current_delay
        mock_sleep.assert_called_once_with(0.5)


class TestAdaptiveThrottleDecorator:
    """Tests for the adaptive_throttle decorator."""
    
    @patch('merit.api.run_config.AdaptiveDelay')
    def test_adaptive_throttle_success(self, mock_adaptive_delay):
        """Test that adaptive_throttle handles successful calls correctly."""
        # Create a mock instance
        mock_instance = MagicMock()
        mock_adaptive_delay.return_value = mock_instance
        
        # Create a test function
        @adaptive_throttle
        def test_func(self):
            return "success"
        
        # Create a test class instance
        class TestClass:
            pass
        
        test_obj = TestClass()
        
        # Call the decorated function
        result = test_func(test_obj)
        
        # Check the result
        assert result == "success"
        
        # Check that adaptive delay methods were called
        mock_instance.wait.assert_called_once()
        mock_instance.success.assert_called_once()
        assert mock_instance.failure.call_count == 0
    
    def test_adaptive_throttle_error(self):
        """Test that adaptive_throttle handles errors correctly."""
        # Create a test function that raises an exception
        @adaptive_throttle
        def test_func(self):
            raise ValueError("Test error")
        
        # Create a test class instance with a mocked delay instance
        class TestClass:
            def __init__(self):
                self.delay = MagicMock()
                
            def get_delay(self):
                return self.delay
        
        test_obj = TestClass()
        
        # Save the original _class_delays and patch it with our mock
        original_class_delays = merit.api.run_config._class_delays.copy()
        merit.api.run_config._class_delays["TestClass"] = test_obj.delay
        
        try:
            # Call the decorated function (should raise)
            with pytest.raises(ValueError):
                test_func(test_obj)
            
            # The error should have been logged but not treated as a rate limit
            assert test_obj.delay.success.call_count == 0
            assert test_obj.delay.failure.call_count == 0
            
        finally:
            # Restore the original class delays
            merit.api.run_config._class_delays = original_class_delays
    
    def test_adaptive_throttle_rate_limit(self):
        """Test that adaptive_throttle handles rate limit errors correctly."""
        # Create a test function that raises a rate limit exception
        @adaptive_throttle
        def test_func(self):
            response = MagicMock()
            response.status_code = 429
            error = requests.exceptions.RequestException("Rate limit exceeded")
            error.response = response
            raise error
        
        # Create a test class instance with a mocked delay instance
        class TestClass:
            def __init__(self):
                self.delay = MagicMock()
        
        test_obj = TestClass()
        
        # Save the original _class_delays and patch it with our mock
        original_class_delays = merit.api.run_config._class_delays.copy()
        merit.api.run_config._class_delays["TestClass"] = test_obj.delay
        
        try:
            # Call the decorated function (should raise)
            with pytest.raises(requests.exceptions.RequestException):
                test_func(test_obj)
            
            # This was a rate limit error, so failure should be called
            assert test_obj.delay.wait.call_count > 0
            assert test_obj.delay.success.call_count == 0
            assert test_obj.delay.failure.call_count > 0
            
        finally:
            # Restore the original class delays
            merit.api.run_config._class_delays = original_class_delays


class TestRetryDecorator:
    """Tests for the with_retry decorator."""
    
    def test_retry_success(self):
        """Test that with_retry returns the result on success."""
        # Create a test function
        @with_retry()
        def test_func():
            return "success"
        
        # Call the decorated function
        result = test_func()
        
        # Check the result
        assert result == "success"
    
    def test_retry_non_retryable_error(self):
        """Test that with_retry doesn't retry on non-retryable errors."""
        # Create a mock function that tracks calls
        mock_func = MagicMock(side_effect=ValueError("Non-retryable error"))
        
        # Apply the decorator
        decorated_func = with_retry()(mock_func)
        
        # Call the decorated function (should raise)
        with pytest.raises(ValueError):
            decorated_func()
        
        # Should only be called once (no retries)
        assert mock_func.call_count == 1
    
    @patch('time.sleep')
    def test_retry_retryable_error(self, mock_sleep):
        """Test that with_retry retries on retryable errors."""
        # Create a function that fails twice then succeeds
        call_count = 0
        def test_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise APIConnectionError("Connection error")
            return "success after retry"
        
        # Apply the decorator
        decorated_func = with_retry(max_retries=3)(test_func)
        
        # Call the decorated function
        result = decorated_func()
        
        # Check the result
        assert result == "success after retry"
        assert call_count == 3  # Original call + 2 retries
    
    @patch('time.sleep')
    def test_retry_max_retries_exceeded(self, mock_sleep):
        """Test that with_retry raises after max retries."""
        # Create a function that always fails
        mock_func = MagicMock(side_effect=APIConnectionError("Persistent connection error"))
        
        # Apply the decorator
        decorated_func = with_retry(max_retries=2)(mock_func)
        
        # Call the decorated function (should raise after retries)
        with pytest.raises(APIConnectionError):
            decorated_func()
        
        # Should be called max_retries + 1 times (original + retries)
        assert mock_func.call_count == 3
    
    @patch('time.sleep')
    def test_retry_backoff(self, mock_sleep):
        """Test that with_retry uses exponential backoff."""
        # Create a function that always fails
        mock_func = MagicMock(side_effect=APIConnectionError("Connection error"))
        
        # Apply the decorator with specific backoff factor
        decorated_func = with_retry(max_retries=2, backoff_factor=1.0, jitter=False)(mock_func)
        
        # Call the decorated function (will raise after retries)
        with pytest.raises(APIConnectionError):
            decorated_func()
        
        # Check that sleep was called with exponential backoff
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)  # First retry: 1.0 * (2^0)
        mock_sleep.assert_any_call(2.0)  # Second retry: 1.0 * (2^1)


class TestCombinedDecorators:
    """Tests for the with_adaptive_retry decorator."""
    
    def test_with_adaptive_retry_composition(self):
        """Test that with_adaptive_retry correctly composes the decorators."""
        # Create a test function with the combined decorator
        @with_adaptive_retry()
        def test_func(self):
            return "success"
        
        # The test is that this doesn't raise any exceptions
        # We can't easily verify the actual composition in a unit test
        assert callable(test_func)
