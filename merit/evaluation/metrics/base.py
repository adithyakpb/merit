"""
Merit Evaluation Metrics Base Classes

This module provides base classes for metrics used in evaluating Large Language Model powered systems.
"""

from abc import ABC, abstractmethod
from ...core.logging import get_logger

logger = get_logger(__name__)

class BaseMetric(ABC):
    """Base class for all metrics."""
    name = "Base Metric"
    greater_is_better = True
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Calculate the metric."""
        raise NotImplementedError
    
    # Metric registry
_metrics_register = {}

def register_metric(name, metric_class):
    """
    Register a metric class.
    
    Args:
        name: The name of the metric
        metric_class: The metric class
    """
    _metrics_register[name] = metric_class

def get_metric(name):
    """
    Get a metric by name.
    
    Args:
        name: The name of the metric
        
    Returns:
        BaseMetric: An instance of the metric
    """
    if name not in _metrics_register:
        raise ValueError(f"Metric '{name}' not found in registry")
    return _metrics_register[name]()
