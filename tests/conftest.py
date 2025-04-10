"""
MERIT Test Configuration

This module provides fixtures and configuration for MERIT tests.
It includes support for multiple test environments and custom implementation testing.
"""

import pytest
import os
import json
import importlib
import inspect
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Type, List

from merit.core.models import Document
from merit.knowledge import KnowledgeBase

# Setup test paths
TEST_DIR = Path(__file__).parent
FIXTURES_DIR = TEST_DIR / "fixtures"

# Ensure fixtures directory exists
FIXTURES_DIR.mkdir(exist_ok=True)


class TestEnvironment(Enum):
    """Different test environments for various testing scenarios."""
    STANDARD = "standard"           # Regular testing environment
    MINIMAL = "minimal"             # Minimal configuration for edge cases
    PERFORMANCE = "performance"     # Performance-focused settings
    ERROR_PRONE = "error_prone"     # Environment likely to trigger errors
    CUSTOM = "custom"               # For custom user settings


# Environment configurations
ENV_CONFIGS = {
    TestEnvironment.STANDARD: {
        "MERIT_CACHE_ENABLED": "1",
        "MERIT_EMBEDDING_CACHE_SIZE": "100",
        "MERIT_ANALYSIS_CACHE_SIZE": "50",
        "MERIT_ANALYSIS_CACHE_TTL": "3600",
    },
    TestEnvironment.MINIMAL: {
        "MERIT_CACHE_ENABLED": "0",
        "MERIT_EMBEDDING_CACHE_SIZE": "10",
        "MERIT_ANALYSIS_CACHE_SIZE": "5",
        "MERIT_ANALYSIS_CACHE_TTL": "60",
    },
    TestEnvironment.PERFORMANCE: {
        "MERIT_CACHE_ENABLED": "1",
        "MERIT_EMBEDDING_CACHE_SIZE": "1000",
        "MERIT_ANALYSIS_CACHE_SIZE": "500",
        "MERIT_ANALYSIS_CACHE_TTL": "86400",
    },
    TestEnvironment.ERROR_PRONE: {
        "MERIT_CACHE_ENABLED": "1",
        "MERIT_EMBEDDING_CACHE_SIZE": "1",  # Too small, will cause evictions
        "MERIT_ANALYSIS_CACHE_SIZE": "1",
        "MERIT_ANALYSIS_CACHE_TTL": "1",    # 1 second TTL, items expire quickly
    }
}


@pytest.fixture
def test_environment(request):
    """
    Set up a specific test environment.
    
    Usage:
        @pytest.mark.parametrize("test_environment", 
                                [TestEnvironment.STANDARD, TestEnvironment.ERROR_PRONE], 
                                indirect=True)
        def test_something(test_environment):
            # Test with different environments
    
    Args:
        request: The pytest request object
        
    Returns:
        The environment name for reference
    """
    # Get the requested environment
    env = request.param if hasattr(request, 'param') else TestEnvironment.STANDARD
    
    # Save the old environment variables
    old_env = {}
    for key in set().union(*(config.keys() for config in ENV_CONFIGS.values())):
        if key in os.environ:
            old_env[key] = os.environ[key]
    
    # Set up the new environment
    if isinstance(env, TestEnvironment):
        config = ENV_CONFIGS[env]
    elif isinstance(env, dict):
        # Custom environment provided as a dictionary
        config = env
        env = TestEnvironment.CUSTOM
    else:
        raise ValueError(f"Invalid test environment: {env}")
    
    # Apply the configuration
    for key, value in config.items():
        os.environ[key] = value
    
    yield env.value if isinstance(env, TestEnvironment) else "custom"
    
    # Restore the old environment
    for key in config:
        if key in old_env:
            os.environ[key] = old_env[key]
        else:
            os.environ.pop(key, None)


@pytest.fixture(scope="session")
def base_test_environment():
    """
    Set up a base test environment for all tests.
    This is automatically used for the entire test session.
    """
    # Save the old environment variables
    old_env = {}
    for key, value in ENV_CONFIGS[TestEnvironment.STANDARD].items():
        if key in os.environ:
            old_env[key] = os.environ[key]
        os.environ[key] = value
    
    yield
    
    # Restore the old environment
    for key in ENV_CONFIGS[TestEnvironment.STANDARD]:
        if key in old_env:
            os.environ[key] = old_env[key]
        else:
            os.environ.pop(key, None)


# Fixture for sample documents
@pytest.fixture
def sample_documents():
    """
    Create a standard set of test documents.
    
    Returns:
        List[Document]: A list of Document objects for testing
    """
    return [
        Document(
            content="Paris is the capital of France. It is known for the Eiffel Tower.",
            metadata={"source": "geography", "topic": "France"},
            id="doc1"
        ),
        Document(
            content="The Eiffel Tower was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.",
            metadata={"source": "history", "topic": "Eiffel Tower"},
            id="doc2"
        ),
        Document(
            content="France is a country in Western Europe. It has a population of about 67 million people.",
            metadata={"source": "geography", "topic": "France"},
            id="doc3"
        )
    ]


@pytest.fixture
def sample_knowledge_base(sample_documents):
    """
    Create a knowledge base with sample documents.
    
    Args:
        sample_documents: The sample documents fixture
        
    Returns:
        KnowledgeBase: A knowledge base initialized with sample documents
    """
    return KnowledgeBase(documents=sample_documents)


# Mock LLM client for testing
class MockLLMClient:
    """
    Mock LLM client for testing purposes.
    
    This mock client can be used to simulate LLM API responses without making actual API calls.
    It supports both text generation and embedding generation.
    """
    
    def __init__(self, responses=None, embeddings=None):
        """
        Initialize the mock LLM client.
        
        Args:
            responses: Dictionary mapping prompts to response texts
            embeddings: Dictionary mapping texts to embedding vectors
        """
        self.responses = responses or {}
        self.embeddings = embeddings or {}
        self.call_history = []
    
    def generate_text(self, prompt, **kwargs):
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The input prompt
            **kwargs: Additional arguments
            
        Returns:
            str: Generated text
        """
        self.call_history.append(("generate_text", prompt, kwargs))
        
        # Check for exact match in responses
        if prompt in self.responses:
            return self.responses[prompt]
        
        # Check for partial matches
        for key, value in self.responses.items():
            if key in prompt:
                return value
        
        # Default response if no match
        return f"Mock response for: {prompt[:30]}..."
    
    def get_embeddings(self, texts):
        """
        Get embeddings for texts.
        
        Args:
            texts: List of texts to get embeddings for
            
        Returns:
            List[List[float]]: List of embedding vectors
        """
        self.call_history.append(("get_embeddings", texts))
        
        result = []
        for text in texts:
            if text in self.embeddings:
                result.append(self.embeddings[text])
            else:
                # Generate a deterministic fake embedding
                import hashlib
                seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % 10000
                import random
                rng = random.Random(seed)
                # Create a small vector for testing (real ones would be much larger)
                embedding = [rng.uniform(-1, 1) for _ in range(10)]
                result.append(embedding)
        
        return result


@pytest.fixture
def mock_llm_client():
    """
    Create a mock LLM client for testing.
    
    Returns:
        MockLLMClient: A mock LLM client
    """
    return MockLLMClient()


# Helper for testing custom implementations
def load_custom_implementation(module_path, class_name=None):
    """
    Load a custom implementation class or module for testing.
    
    This is useful for testing user-defined extensions to MERIT.
    
    Args:
        module_path: The import path to the module
        class_name: The name of the class to load (if any)
        
    Returns:
        The loaded module or class
    """
    try:
        module = importlib.import_module(module_path)
        if class_name:
            return getattr(module, class_name)
        return module
    except (ImportError, AttributeError) as e:
        pytest.skip(f"Could not load custom implementation: {str(e)}")


@pytest.fixture
def custom_metric_loader():
    """
    Fixture to load custom metrics for testing.
    
    Returns:
        Function to load custom metrics
    """
    def _load_metric(module_path, class_name):
        """Load a custom metric class."""
        return load_custom_implementation(module_path, class_name)
    
    return _load_metric


@pytest.fixture
def custom_evaluator_loader():
    """
    Fixture to load custom evaluators for testing.
    
    Returns:
        Function to load custom evaluators
    """
    def _load_evaluator(module_path, class_name):
        """Load a custom evaluator class."""
        return load_custom_implementation(module_path, class_name)
    
    return _load_evaluator
