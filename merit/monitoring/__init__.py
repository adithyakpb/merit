"""
MERIT Monitoring Module

A comprehensive monitoring system for LLM applications that provides non-invasive
monitoring, flexible integrations, and actionable insights.

This module enables monitoring of LLM applications with minimal integration effort,
supporting various data sources and visualization options.
"""

__version__ = "0.1.0"

# Core data models - now from a single models module
from .models import (
    # Base interaction models
    BaseInteractionComponent,
    BaseInteractionRequest, 
    BaseInteractionResponse,
    BaseInteraction,
    
    # LLM-specific models
    TokenInfo,
    LLMRequest,
    LLMResponse,
    LLMInteraction
)

# Data collection components - now from collectors package
from .collectors import (
    # Base collector interfaces
    BaseDataCollector,
    CollectionResult,
    
    # Concrete collectors
    LogDataCollector,
    APIProxyCollector,
    OpenAIProxyCollector,
    AnthropicProxyCollector,
    KafkaCollector
)

# Java Kafka adapter if available
try:
    from .collectors import JavaKafkaAdapter
except ImportError:
    pass

# Import storage components from the new top-level storage package
from ..storage import (
    BaseStorage,
    SQLiteStorage,
    FileStorage, #TODO
    DatabaseFactory
)

# Import monitoring service and LLM monitoring
from .service import MonitoringService, create_monitoring_service
from .llm import (
    LLMMonitor,
    get_monitor,
    track_generation,
    setup_monitoring
)

# Metrics system - import both shared foundation and monitoring-specific metrics
from ..metrics import (
    # Base metrics system
    BaseMetric,
    MetricContext,
    MetricCategory,
    
    # Monitoring-specific metrics
    MonitoringMetric,
    PerformanceMetric,
    UsageMetric,
    CostMetric,
    RequestVolumeMetric,
    LatencyMetric,
    TokenVolumeMetric,
    ErrorRateMetric,
    CostEstimateMetric,
    
    # Registry functions
    register_metric,
    get_metric,
    list_metrics,
    create_metric_instance
)

# Factory functions
def create_openai_collector(buffer_size: int = 1000, include_models: list = None, exclude_models: list = None):
    """
    Create and configure an OpenAI API collector.
    
    Args:
        buffer_size: Maximum number of interactions to buffer in memory
        include_models: List of model name patterns to include (e.g., ['gpt-4*'])
        exclude_models: List of model name patterns to exclude
        
    Returns:
        Configured OpenAIProxyCollector instance
    """
    from .collectors import OpenAIProxyCollector
    
    config = {
        "buffer_size": buffer_size,
        "enabled_providers": ["openai"]
    }
    
    if include_models:
        config["include_model_patterns"] = include_models
    if exclude_models:
        config["exclude_model_patterns"] = exclude_models
        
    collector = OpenAIProxyCollector(config)
    return collector

def create_log_collector(log_path: str, tail: bool = False, format: str = "json"):
    """
    Create and configure a log file collector.
    
    Args:
        log_path: Path to log file or directory of log files
        tail: Whether to watch files for new entries
        format: Format of log entries ("json" or "regex")
        
    Returns:
        Configured LogDataCollector instance
    """
    config = {
        "log_path": log_path,
        "tail": tail,
        "format": format
    }
    
    collector = LogDataCollector(config)
    return collector

def calculate_metrics(interactions, metrics=None):
    """
    Calculate metrics for a set of interactions.
    
    Args:
        interactions: List of interaction data
        metrics: List of metric names to calculate, or None for all monitoring metrics
        
    Returns:
        Dictionary mapping metric names to results
    """
    from ..metrics import list_metrics, create_metric_instance, MetricContext
    
    # If no metrics specified, use all monitoring metrics
    if metrics is None:
        metrics = list_metrics(context=MetricContext.MONITORING)
    
    results = {}
    for metric_name in metrics:
        try:
            # Create the metric instance
            metric = create_metric_instance(metric_name)
            
            # Calculate the metric
            result = metric(interactions)
            
            # Store the result
            results[metric_name] = result
        except Exception as e:
            # Log error but continue with other metrics
            from ..core.logging import get_logger
            logger = get_logger(__name__)
            logger.error(f"Error calculating metric {metric_name}: {e}")
            results[metric_name] = {"error": str(e)}
    
    return results
