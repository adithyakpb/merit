"""
MERIT Caching Module

This module provides caching functionality for the MERIT system.
It supports both in-memory caching and optional cachetools integration.
"""

import functools
import hashlib
import threading
from typing import Any, Callable, Dict, List, Optional, Union

import os
from typing import Optional

# Cache configuration
CACHE_ENABLED = os.environ.get("MERIT_CACHE_ENABLED", "1") == "1"
EMBEDDING_CACHE_SIZE = int(os.environ.get("MERIT_EMBEDDING_CACHE_SIZE", "1000"))
ANALYSIS_CACHE_SIZE = int(os.environ.get("MERIT_ANALYSIS_CACHE_SIZE", "100"))
ANALYSIS_CACHE_TTL = int(os.environ.get("MERIT_ANALYSIS_CACHE_TTL", "3600"))  # 1 hour
# Global flag to enable/disable caching (initialized from config)
_cache_enabled = CACHE_ENABLED

# Try to import cachetools, fall back to dummy implementation
try:
    from cachetools import LRUCache, TTLCache, cached
    from cachetools.keys import hashkey
    CACHETOOLS_AVAILABLE = True
except ImportError:
    CACHETOOLS_AVAILABLE = False
    
    # Fallback cache implementation
    class PassThroughCache:
        """A cache that doesn't actually cache - just passes through."""
        
        def __init__(self, *args, **kwargs):
            self.maxsize = kwargs.get('maxsize', 0)
            self.ttl = kwargs.get('ttl', 0)
        
        def __getitem__(self, key):
            raise KeyError(key)
        
        def __setitem__(self, key, value):
            pass
        
        def __contains__(self, key):
            return False
        
        def get(self, key, default=None):
            return default
        
        def clear(self):
            pass
        
        def __len__(self):
            return 0
    
    # Create pass-through versions of cachetools classes
    LRUCache = PassThroughCache
    TTLCache = PassThroughCache
    
    # Pass-through decorator that just calls the original function
    def cached(cache, key=None, lock=None):
        def decorator(func):
            return func
        return decorator

# Create locks for thread safety
embedding_cache_lock = threading.RLock()
analysis_cache_lock = threading.RLock()

# Create caches based on availability
if CACHETOOLS_AVAILABLE and _cache_enabled:
    embedding_cache = LRUCache(maxsize=EMBEDDING_CACHE_SIZE)
    analysis_cache = TTLCache(maxsize=ANALYSIS_CACHE_SIZE, ttl=ANALYSIS_CACHE_TTL)
else:
    embedding_cache = PassThroughCache(maxsize=EMBEDDING_CACHE_SIZE)
    analysis_cache = PassThroughCache(maxsize=ANALYSIS_CACHE_SIZE, ttl=ANALYSIS_CACHE_TTL)

def content_hash(content: Union[str, List[str], "ExampleSet"]) -> str:
    """
    Create a hash for content to use as a cache key.
    
    Args:
        content: The content to hash (string, list of strings, or ExampleSet)
        
    Returns:
        str: A hash string representing the content
    """
    if hasattr(content, "examples") and hasattr(content, "to_dict"):
        # It's an ExampleSet - extract inputs for hashing
        inputs = [example.input for example in content.examples]
        # Sort to ensure consistent hashing regardless of order
        content = "|".join(sorted(inputs))
    elif isinstance(content, list):
        # Check if list contains ExampleItem objects
        if content and hasattr(content[0], "input"):
            # Extract input strings from ExampleItem objects
            inputs = [example.input for example in content]
            content = "|".join(sorted(inputs))
        else:
            # Regular list of strings
            content = "|".join(sorted(content))
    
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def cache_embeddings(func: Callable) -> Callable:
    """
    Decorator to cache embeddings.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @functools.wraps(func)
    def wrapper(self, texts, *args, **kwargs):
        # If caching is disabled, just call the original function
        if not _cache_enabled or not CACHETOOLS_AVAILABLE:
            return func(self, texts, *args, **kwargs)
            
        # Handle single text case
        if isinstance(texts, str):
            texts = [texts]
            was_string = True
        else:
            was_string = False
        
        # Check cache for each text
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        with embedding_cache_lock:
            for i, text in enumerate(texts):
                # Create hash for this text
                text_hash = content_hash(text)
                # Check cache
                if text_hash in embedding_cache:
                    results.append(embedding_cache[text_hash])
                else:
                    results.append(None)  # Placeholder
                    texts_to_embed.append(text)
                    indices_to_embed.append(i)
        
        if texts_to_embed:
            # Call the original function for the texts that need embeddings
            new_embeddings = func(self, texts_to_embed, *args, **kwargs)
            
            # Update cache and results
            with embedding_cache_lock:
                for i, embedding in zip(indices_to_embed, new_embeddings):
                    text_hash = content_hash(texts[i])
                    embedding_cache[text_hash] = embedding
                    results[i] = embedding
        
        # Return single result if input was a string
        if was_string:
            return results[0]
        return results
    
    return wrapper

def cache_analysis(func: Callable) -> Callable:
    """
    Decorator to cache analysis results.
    
    Args:
        func: The function to decorate
        
    Returns:
        Callable: The decorated function
    """
    @functools.wraps(func)
    def wrapper(inputs, *args, **kwargs):
        # If caching is disabled, just call the original function
        if not _cache_enabled or not CACHETOOLS_AVAILABLE:
            return func(inputs, *args, **kwargs)
            
        # Create a cache key based on inputs and key parameters
        key_parts = [
            "analyze",
            content_hash(inputs),  # Use content_hash which now handles ExampleSet objects
            str(kwargs.get('use_llm', False)),
            str(kwargs.get('analysis_type', 'all'))
        ]
        cache_key = content_hash("|".join(key_parts))
        
        # Check cache
        with analysis_cache_lock:
            if cache_key in analysis_cache:
                return analysis_cache[cache_key]
        
        # Compute analysis
        result = func(inputs, *args, **kwargs)
        
        # Cache result
        with analysis_cache_lock:
            analysis_cache[cache_key] = result
        
        return result
    
    return wrapper

def enable_caching():
    """Enable caching globally."""
    global _cache_enabled
    _cache_enabled = True

def disable_caching():
    """Disable caching globally."""
    global _cache_enabled
    _cache_enabled = False
    # Clear existing caches
    embedding_cache.clear()
    analysis_cache.clear()

def is_caching_available():
    """
    Check if cachetools is available.
    
    Returns:
        bool: True if cachetools is available, False otherwise
    """
    return CACHETOOLS_AVAILABLE

def is_caching_enabled():
    """
    Check if caching is enabled.
    
    Returns:
        bool: True if caching is enabled and available, False otherwise
    """
    return _cache_enabled and CACHETOOLS_AVAILABLE

def get_cache_stats():
    """
    Get statistics about the caches.
    
    Returns:
        Dict[str, Any]: Cache statistics
    """
    if not CACHETOOLS_AVAILABLE:
        return {"available": False}
    
    return {
        "available": True,
        "enabled": _cache_enabled,
        "embedding_cache": {
            "size": len(embedding_cache),
            "maxsize": embedding_cache.maxsize,
            "usage": len(embedding_cache) / embedding_cache.maxsize if embedding_cache.maxsize > 0 else 0,
        },
        "analysis_cache": {
            "size": len(analysis_cache),
            "maxsize": analysis_cache.maxsize,
            "usage": len(analysis_cache) / analysis_cache.maxsize if analysis_cache.maxsize > 0 else 0,
            "ttl": getattr(analysis_cache, 'ttl', None),
        }
    }
