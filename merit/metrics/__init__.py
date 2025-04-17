"""
MERIT Metrics Module

This module provides a shared foundation for metrics used across different MERIT
components, particularly evaluation and monitoring. It enables consistent
metric definitions, context-aware calculations, and a unified registry system.
"""

__version__ = "0.1.0"

# Base class and registry
from .base import (
    BaseMetric,
    MetricContext,
    MetricCategory,
    MetricRegistry,
    register_metric,
    get_metric,
    list_metrics,
    create_metric_instance,
    get_metric_metadata
)

# Context-specific base classes for monitoring
from .monitoring import (
    MonitoringMetric,
    PerformanceMetric,
    UsageMetric,
    CostMetric,
    RequestVolumeMetric,
    LatencyMetric,
    TokenVolumeMetric,
    ErrorRateMetric,
    CostEstimateMetric
)

# Classification metrics
from .llm_measured import (
    LLMMeasuredBaseMetric
)

# RAG metrics
from .rag import (
    CorrectnessMetric,
    FaithfulnessMetric,
    RelevanceMetric,
    CoherenceMetric,
    FluencyMetric,
    ContextPrecisionMetric
)
