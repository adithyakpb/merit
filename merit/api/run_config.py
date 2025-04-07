"""
Run Configuration for API Calls

This module provides Adaptive Delay and Exponential Backoff functionality for API calls.
"""

import time
import threading
import os
from typing import Callable, Any
from functools import wraps

from merit.core.logging import get_logger

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
        @adaptive_throttle
        def my_api_function(self, ...):
            # Function implementation
    """
    @wraps(f)
    def wrapper(self, *args, **kwargs) -> Any:
        # Get or create the adaptive delay for this class
        class_name = self.__class__.__name__
        if class_name not in _class_delays:
            _class_delays[class_name] = AdaptiveDelay(use_env=True)
            logger.info(f"Created adaptive delay for class {class_name}")
        
        adaptive_delay = _class_delays[class_name]
        
        # Wait before making the API call
        adaptive_delay.wait()
        
        # Record API call start time
        call_start = time.time()
        
        try:
            # Make the API call
            result = f(self, *args, **kwargs)
            
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
