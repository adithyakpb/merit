"""
MERIT Monitoring Collectors

This module provides data collectors for the MERIT monitoring system.
Collectors are responsible for gathering interaction data from various sources
such as logs, APIs, and message queues.
"""

from .collector import (
    BaseDataCollector,
    CollectionResult
)
from .log_collector import LogDataCollector
from .api_collector import APIProxyCollector, OpenAIProxyCollector, AnthropicProxyCollector

# Import Kafka collector if available
try:
    from .kafka_collector import KafkaCollector
    HAS_KAFKA = True
except ImportError:
    # Create a placeholder class that raises an error when instantiated
    class KafkaCollector:
        def __init__(self, *args, **kwargs):
            raise ImportError("KafkaCollector requires confluent-kafka package. Please install with 'pip install confluent-kafka'")
    HAS_KAFKA = False

__all__ = [
    'BaseDataCollector',
    'CollectionResult',
    'LogDataCollector', 
    'APIProxyCollector',
    'OpenAIProxyCollector',
    'AnthropicProxyCollector',
    'KafkaCollector',
    'HAS_KAFKA'
]
