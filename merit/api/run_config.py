"""
Run Configuration for API Calls

This module provides Adaptive Delay, Exponential Backoff, and Retry functionality for API calls.
"""

import time
import threading
import os
import random
from typing import Callable, Any, Optional, Union, List, Tuple, Type
from functools import wraps

import requests

from merit.core.logging import get_logger
from merit.api.errors import (
    APIConnectionError,
    APIRateLimitError,
    APIServerError,
    APITimeoutError,
)

logger = get_logger(__name__)


class AdaptiveDelay:
    """
    Adaptive delay for API calls to avoid rate limiting.
    
    This class implements an adaptive delay mechanism that:
    1. Starts with a conservative delay
    2. Gradually reduces the delay as long as requests succeed
    3. Increases the delay when rate limits are hit
    4. Converges on the optimal delay value
    """
    def __init__(
        self, 
        initial_delay: float = None,
        min_delay: float = None,
        max_delay: float = None,
        decrease_factor: float = None,
        increase_factor: float = None,
        use_env: bool = False
    ):
        """
        Initialize the adaptive delay.
        
        Args:
            initial_delay: Initial delay in seconds (default: 0.5)
            min_delay: Minimum delay in seconds (default: 0.05)
            max_delay: Maximum delay in seconds (default: 2.0)
            decrease_factor: Factor to multiply delay by after success (default: 0.9)
            increase_factor: Factor to multiply delay by after failure (default: 1.5)
            use_env: Whether to load values from environment variables (default: False)
        """
        # If use_env is True, load from environment variables with fallbacks to defaults
        if use_env:
            self.current_delay = float(os.getenv("API_INITIAL_DELAY", "0.5"))
            self.min_delay = float(os.getenv("API_MIN_DELAY", "0.05"))
            self.max_delay = float(os.getenv("API_MAX_DELAY", "2.0"))
            self.decrease_factor = float(os.getenv("API_DECREASE_FACTOR", "0.9"))
            self.increase_factor = float(os.getenv("API_INCREASE_FACTOR", "1.5"))
        else:
            # Use provided values or defaults
            self.current_delay = initial_delay if initial_delay is not None else 0.5
            self.min_delay = min_delay if min_delay is not None else 0.05
            self.max_delay = max_delay if max_delay is not None else 2.0
            self.decrease_factor = decrease_factor if decrease_factor is not None else 0.9
            self.increase_factor = increase_factor if increase_factor is not None else 1.5
        
        # Statistics
        self.total_requests = 0
        self.total_failures = 0
        self.total_wait_time = 0
        self.start_time = time.time()
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        # Log initial settings
        logger.info(f"AdaptiveDelay initialized: initial={self.current_delay:.3f}s, min={self.min_delay:.3f}s, max={self.max_delay:.3f}s")
    
    def wait(self):
        """Wait for the current delay period and log the delay."""
        with self.lock:
            # Log current delay before waiting
            logger.info(f"API call delay: {self.current_delay:.3f}s (request #{self.total_requests+1})")
            
            # Record statistics
            wait_start = time.time()
        
        # Release lock during the actual waiting
        time.sleep(self.current_delay)
        
        with self.lock:
            # Update statistics after waiting
            actual_wait = time.time() - wait_start
            self.total_wait_time += actual_wait
            
            # Log actual wait time if it differs significantly from expected
            if abs(actual_wait - self.current_delay) > 0.01:
                logger.debug(f"Actual wait time: {actual_wait:.3f}s (vs expected {self.current_delay:.3f}s)")
    
    def success(self):
        """Record a successful API call and potentially decrease delay."""
        with self.lock:
            self.total_requests += 1
            
            # Decrease delay slightly after success
            old_delay = self.current_delay
            self.current_delay = max(
                self.min_delay,
                self.current_delay * self.decrease_factor
            )
            
            # Log statistics periodically or when delay changes
            if self.total_requests % 5 == 0 or old_delay != self.current_delay:
                elapsed = time.time() - self.start_time
                success_rate = (self.total_requests - self.total_failures) / max(1, self.total_requests)
                avg_delay = self.total_wait_time / max(1, self.total_requests)
                
                logger.info(
                    f"Stats: success_rate={success_rate:.2%}, "
                    f"avg_delay={avg_delay:.3f}s, "
                    f"new_delay={self.current_delay:.3f}s, "
                    f"requests={self.total_requests}, "
                    f"elapsed={elapsed:.1f}s"
                )
    
    def failure(self):
        """Record a failed API call and increase delay."""
        with self.lock:
            self.total_requests += 1
            self.total_failures += 1
            
            # Increase delay after failure
            old_delay = self.current_delay
            self.current_delay = min(
                self.max_delay,
                self.current_delay * self.increase_factor
            )
            
            # Always log rate limit failures
            logger.warning(
                f"Rate limit hit! Increasing delay: {old_delay:.3f}s → {self.current_delay:.3f}s "
                f"(failure #{self.total_failures})"
            )


# Dictionary to store adaptive delays for each class
_class_delays = {}

def adaptive_throttle(f: Callable) -> Callable:
    """
    Decorator to apply adaptive rate throttling to a function.
    
    This decorator:
    1. Waits before making the API call based on the adaptive delay
    2. Records API call timing
    3. Handles success/failure tracking
    4. Provides detailed logging
    
    Usage:
        # For class methods
        @adaptive_throttle
        def my_method(self, ...):
            # Method implementation
            
        # For standalone functions
        @adaptive_throttle
        def my_function(...):
            # Function implementation
    """
    @wraps(f)
    def wrapper(*args, **kwargs) -> Any:
        # Determine if this is a class method or standalone function
        if args and hasattr(args[0], '__class__'):
            # It's likely a class method with 'self' as first argument
            instance = args[0]
            class_name = instance.__class__.__name__
            # Keep the same argument structure for the wrapped function
            call_args = args
        else:
            # It's a standalone function or a class method called without instance
            class_name = f"Function_{f.__name__}"
            # Keep the same argument structure for the wrapped function
            call_args = args
        
        # Get or create the adaptive delay for this class/function
        if class_name not in _class_delays:
            _class_delays[class_name] = AdaptiveDelay(use_env=True)
            logger.info(f"Created adaptive delay for {class_name}")
        
        adaptive_delay = _class_delays[class_name]
        
        # Wait before making the API call
        adaptive_delay.wait()
        
        # Record API call start time
        call_start = time.time()
        
        try:
            # Make the API call
            result = f(*call_args, **kwargs)
            
            # Record API call duration
            call_duration = time.time() - call_start
            logger.info(f"API call to {f.__name__} completed in {call_duration:.3f}s")
            
            # Record successful API call
            adaptive_delay.success()
            
            return result
            
        except Exception as e:
            # Check if it's a rate limit error
            is_rate_limit = False
            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                is_rate_limit = e.response.status_code in (429, 503)
            elif "Service Temporarily Unavailable" in str(e):
                is_rate_limit = True
            
            if is_rate_limit:
                # Record failure to adjust delay
                adaptive_delay.failure()
                logger.error(f"Rate limit error in {f.__name__}: {str(e)}")
            else:
                # Log non-rate-limit errors
                logger.error(f"API error in {f.__name__} (not rate limit): {str(e)}")
            
            # Re-raise the exception
            raise
    
    return wrapper


def with_retry(
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    jitter: bool = True,
    retry_on: Optional[Union[Type[Exception], Tuple[Type[Exception], ...], List[Type[Exception]]]] = None,
    retry_status_codes: Optional[List[int]] = None,
):
    """
    Decorator that adds retry functionality to API methods.
    
    This decorator handles transient failures by automatically retrying failed API calls
    with exponential backoff. It works alongside the adaptive_throttle decorator to provide
    a comprehensive solution for handling API rate limits and transient errors.
    
    Args:
        max_retries: Maximum number of retries (default: 3)
        backoff_factor: Backoff factor for exponential backoff (default: 0.5)
        jitter: Whether to add random jitter to the backoff time (default: True)
        retry_on: Exception types to retry on (default: ConnectionError, Timeout, ServerError)
        retry_status_codes: HTTP status codes to retry on (default: 429, 500, 502, 503, 504)
        
    Returns:
        Decorated function with retry logic
    
    Usage:
        @with_retry(max_retries=5, backoff_factor=1.0)
        def my_api_function(self, ...):
            # Function implementation
    """
    # Default exceptions to retry on
    if retry_on is None:
        retry_on = (
            APIConnectionError, 
            APITimeoutError, 
            APIServerError,
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ReadTimeout,
        )
    
    # Default status codes to retry on
    if retry_status_codes is None:
        retry_status_codes = [429, 500, 502, 503, 504]
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            retries = 0
            last_exception = None
            
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except retry_on as e:
                    # Check if it's an HTTP error with a status code
                    should_retry = True
                    retry_after = None
                    status_code = None
                    
                    if isinstance(e, requests.exceptions.HTTPError) and hasattr(e, 'response'):
                        status_code = e.response.status_code
                        should_retry = status_code in retry_status_codes
                        
                        # Check for Retry-After header
                        if should_retry and status_code == 429 and 'Retry-After' in e.response.headers:
                            try:
                                retry_after = int(e.response.headers['Retry-After'])
                            except (ValueError, TypeError):
                                # If Retry-After header is not an integer, use the backoff formula
                                pass
                    
                    if retries >= max_retries or not should_retry:
                        # If we've exceeded max retries or shouldn't retry this error, re-raise
                        raise
                    
                    # Calculate backoff time
                    if retry_after is not None:
                        wait_time = retry_after
                    else:
                        wait_time = backoff_factor * (2 ** retries)
                        if jitter:
                            # Add random jitter (up to 25% of the backoff time)
                            wait_time += wait_time * random.uniform(0, 0.25)
                    
                    # Log the retry
                    logger.warning(
                        f"API call failed with {type(e).__name__}{' (Status: ' + str(status_code) + ')' if status_code else ''}, "
                        f"retrying in {wait_time:.2f} seconds (retry {retries+1}/{max_retries})"
                    )
                    
                    # Wait before retrying
                    time.sleep(wait_time)
                    retries += 1
                    last_exception = e
                except Exception as e:
                    # Don't retry on other exceptions
                    raise
            
            # If we get here, we've exhausted our retries
            if last_exception is not None:
                # Convert to API-specific error if it's a requests exception
                if isinstance(last_exception, requests.exceptions.ConnectionError):
                    raise APIConnectionError(
                        "Failed to connect to the API service after multiple retries.",
                        details={"original_error": str(last_exception), "retries": retries}
                    ) from last_exception
                elif isinstance(last_exception, requests.exceptions.Timeout):
                    raise APITimeoutError(
                        "API request timed out after multiple retries.",
                        details={"original_error": str(last_exception), "retries": retries}
                    ) from last_exception
                elif isinstance(last_exception, requests.exceptions.HTTPError) and hasattr(last_exception, 'response'):
                    # Handle rate limiting specifically
                    if last_exception.response.status_code == 429:
                        retry_after = None
                        if 'Retry-After' in last_exception.response.headers:
                            try:
                                retry_after = int(last_exception.response.headers['Retry-After'])
                            except (ValueError, TypeError):
                                pass
                        
                        raise APIRateLimitError(
                            "API rate limit exceeded and not resolved after multiple retries.",
                            details={"original_error": str(last_exception), "retries": retries},
                            retry_after=retry_after
                        ) from last_exception
                    elif last_exception.response.status_code >= 500:
                        raise APIServerError(
                            "API server error persisted after multiple retries.",
                            details={"original_error": str(last_exception), "retries": retries, 
                                    "status_code": last_exception.response.status_code}
                        ) from last_exception
                
                # If we can't convert it, re-raise the original
                raise last_exception
            
            # This should never happen, but just in case
            raise APIConnectionError(
                "Failed after multiple retries with an unknown error.",
                details={"retries": retries}
            )
        
        return wrapper
    
    return decorator


def with_adaptive_retry(
    max_retries: int = 3,
    backoff_factor: float = 0.5,
    jitter: bool = True,
):
    """
    Decorator that combines adaptive throttling and retry functionality.
    
    This decorator applies both adaptive_throttle and with_retry decorators
    to provide a comprehensive solution for API rate limiting and transient error handling.
    
    Args:
        max_retries: Maximum number of retries
        backoff_factor: Backoff factor for exponential backoff
        jitter: Whether to add random jitter to the backoff time
        
    Returns:
        Decorated function with adaptive throttling and retry logic
        
    Usage:
        @with_adaptive_retry(max_retries=5, backoff_factor=1.0)
        def my_api_function(self, ...):
            # Function implementation
    """
    def decorator(func: Callable):
        # Apply the decorators in the correct order:
        # 1. with_retry (innermost) - handles retries for transient errors
        # 2. adaptive_throttle (outermost) - handles rate limiting
        retry_func = with_retry(
            max_retries=max_retries, 
            backoff_factor=backoff_factor,
            jitter=jitter
        )(func)
        
        return adaptive_throttle(retry_func)
    
    return decorator
