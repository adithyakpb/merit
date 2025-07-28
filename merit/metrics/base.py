"""
Base classes and registry for the MERIT metrics system.

This module provides the foundation for metrics across both monitoring and evaluation
contexts, enabling consistent metric definitions and shared functionality.
"""

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Set, Type, Callable, Union, Tuple
import inspect
from ..core.logging import get_logger

logger = get_logger(__name__)


class MetricContext(Enum):
    """Contexts in which a metric can be used."""
    MONITORING = auto()
    EVALUATION = auto()
    BOTH = auto()


class MetricCategory(Enum):
    """Categories for grouping related metrics."""
    PERFORMANCE = auto()  # Speed, throughput, latency
    QUALITY = auto()      # Accuracy, relevance, correctness
    USAGE = auto()        # Volume, token count, request patterns
    COST = auto()         # Financial metrics
    CUSTOM = auto()       # User-defined metrics


class BaseMetric(ABC):
    """
    Base class for all metrics in MERIT.
    
    This enhanced BaseMetric builds on the existing evaluation metrics foundation
    but adds support for context awareness and cross-component usage.
    
    Attributes:
        name: Display name for the metric
        description: Detailed description of what the metric measures
        greater_is_better: Whether higher values indicate better performance
        context: The context(s) in which this metric is intended to be used
        category: Category for grouping related metrics
        monitoring_requires: Model classes required for monitoring context
        evaluation_requires: Model classes required for evaluation context
    """
    name = "Base Metric"
    description = "Base class for all metrics"
    greater_is_better = True
    context = MetricContext.BOTH  # Default to usable in both contexts
    category = MetricCategory.CUSTOM
    
    # Model-based requirements for different contexts
    monitoring_requires = {}  # e.g., {"request": LLMRequest, "response": LLMResponse}
    evaluation_requires = {}  # e.g., {"input": Input, "response": Response}
    
    def __init__(self, mode: MetricContext = None):
        """
        Initialize metric with a specific mode.
        
        Args:
            mode: The context mode for this metric instance
        """
        if mode is None:
            self.mode = self.context
        else:
            self._validate_mode(mode)
            self.mode = mode
    
    def _validate_mode(self, mode: MetricContext):
        """Validate that the requested mode is supported by this metric."""
        if self.context == MetricContext.BOTH:
            if mode not in [MetricContext.MONITORING, MetricContext.EVALUATION]:
                raise ValueError(f"Invalid mode {mode} for metric {self.name}")
        elif self.context != mode:
            raise ValueError(f"Metric {self.name} does not support mode {mode}")
    
    @classmethod
    def for_monitoring(cls) -> 'BaseMetric':
        """Create instance for monitoring context."""
        return cls(mode=MetricContext.MONITORING)
    
    @classmethod
    def for_evaluation(cls) -> 'BaseMetric':
        """Create instance for evaluation context."""
        return cls(mode=MetricContext.EVALUATION)
    
    def preprocess_monitoring(self, request, response):
        """
        Preprocess data for monitoring context calculation.
        
        Args:
            request: LLMRequest object
            response: LLMResponse object
            
        Returns:
            Tuple of (processed_request, processed_response)
        """
        # Default implementation - no preprocessing
        return request, response
    
    def preprocess_evaluation(self, input_obj, response_obj, llm_client=None):
        """
        Preprocess data for evaluation context calculation.
        
        Args:
            input_obj: Input object
            response_obj: Response object
            llm_client: Optional LLM client for evaluation
            
        Returns:
            Tuple of (processed_input, processed_response, llm_client)
        """
        # Default implementation - no preprocessing
        return input_obj, response_obj, llm_client
    
    @abstractmethod
    def calculate_monitoring(self, request, response) -> Dict[str, Any]:
        """
        Calculate metric in monitoring context.
        
        Args:
            request: LLMRequest object (after preprocessing)
            response: LLMResponse object (after preprocessing)
            
        Returns:
            Dict containing metric result
        """
        raise NotImplementedError
    
    @abstractmethod
    def calculate_evaluation(self, input_obj, response_obj, llm_client=None) -> Dict[str, Any]:
        """
        Calculate metric in evaluation context.
        
        Args:
            input_obj: Input object (after preprocessing)
            response_obj: Response object (after preprocessing)
            llm_client: Optional LLM client for evaluation
            
        Returns:
            Dict containing metric result
        """
        raise NotImplementedError
    
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Calculate the metric based on the configured mode.
        
        This method handles the complete pipeline:
        1. Preprocessing the input data
        2. Calculating the metric
        
        Returns:
            Dict containing at minimum a "value" key with the metric value,
            and optionally additional keys with supporting information.
        """
        if self.mode == MetricContext.MONITORING:
            if len(args) < 2:
                raise ValueError("Monitoring mode requires request and response arguments")
            
            # Preprocess data
            processed_request, processed_response = self.preprocess_monitoring(args[0], args[1])
            
            # Calculate metric
            return self.calculate_monitoring(processed_request, processed_response)
            
        elif self.mode == MetricContext.EVALUATION:
            if len(args) < 2:
                raise ValueError("Evaluation mode requires input and response arguments")
            
            llm_client = args[2] if len(args) > 2 else kwargs.get('llm_client')
            
            # Preprocess data
            processed_input, processed_response, processed_client = self.preprocess_evaluation(
                args[0], args[1], llm_client
            )
            
            # Calculate metric
            return self.calculate_evaluation(processed_input, processed_response, processed_client)
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    @classmethod
    def get_requirements(cls) -> Dict[str, Any]:
        """
        Get the data requirements for this metric.
        
        Returns:
            Dictionary describing what data is required to calculate this metric.
        """
        return cls.requires
    
    @classmethod
    def is_applicable_to_context(cls, context: MetricContext) -> bool:
        """
        Check if this metric is applicable to a specific context.
        
        Args:
            context: The context to check against
            
        Returns:
            True if the metric is applicable to the given context
        """
        if cls.context == MetricContext.BOTH:
            return True
        return cls.context == context
    
    @classmethod
    def format_result(cls, result: Dict[str, Any], 
                     formatting: Optional[str] = None) -> Dict[str, Any]:
        """
        Format a metric result according to the specified format.
        
        Args:
            result: The raw metric result
            formatting: Optional formatting specification
            
        Returns:
            Formatted metric result
        """
        # Base implementation just returns the result as-is
        # Subclasses can override to provide specialized formatting
        return result


class MetricRegistry:
    """
    Registry for metrics with support for context filtering and discovery.
    
    This enhanced registry supports tagging metrics with context, category,
    and other metadata to facilitate discovery and appropriate usage.
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        # Primary registry - maps metric names to classes
        self._metrics: Dict[str, Type[BaseMetric]] = {}
        
        # Secondary indices for efficient filtering
        self._by_context: Dict[MetricContext, Set[str]] = {
            context: set() for context in MetricContext
        }
        self._by_category: Dict[MetricCategory, Set[str]] = {
            category: set() for category in MetricCategory
        }
    
    def register(self, metric_class: Type[BaseMetric], 
                name: Optional[str] = None, 
                context: Optional[MetricContext] = None,
                category: Optional[MetricCategory] = None) -> None:
        """
        Register a metric class.
        
        Args:
            metric_class: The metric class to register
            name: Optional custom name (defaults to metric_class.name)
            context: Optional context override (defaults to metric_class.context)
            category: Optional category override (defaults to metric_class.category)
        """
        # Determine the metric name
        metric_name = name or getattr(metric_class, 'name', metric_class.__name__)
        
        # Determine context and category
        metric_context = context or getattr(metric_class, 'context', MetricContext.BOTH)
        metric_category = category or getattr(metric_class, 'category', MetricCategory.CUSTOM)
        
        # Register the metric
        self._metrics[metric_name] = metric_class
        
        # Update secondary indices
        self._by_context[metric_context].add(metric_name)
        if metric_context == MetricContext.BOTH:
            # If a metric is in BOTH, it's also individually in each context
            self._by_context[MetricContext.MONITORING].add(metric_name)
            self._by_context[MetricContext.EVALUATION].add(metric_name)
            
        self._by_category[metric_category].add(metric_name)
        
        logger.debug(f"Registered metric '{metric_name}' in context {metric_context}")
    
    def get(self, name: str) -> Type[BaseMetric]:
        """
        Get a metric class by name.
        
        Args:
            name: The name of the metric
            
        Returns:
            The metric class
            
        Raises:
            KeyError: If the metric is not found
        """
        logger.debug(f"MetricRegistry.get: Attempting to get metric with name: '{name}'")
        logger.debug(f"MetricRegistry.get: Available keys in registry: {list(self._metrics.keys())}")
        if name not in self._metrics:
            logger.error(f"MetricRegistry.get: Metric '{name}' not found. Case-sensitive check.")
            # For debugging, let's check for a case-insensitive match
            for k in self._metrics.keys():
                if k.lower() == name.lower():
                    logger.warning(f"MetricRegistry.get: Found case-insensitive match for '{name}': '{k}'. This indicates a potential casing issue in the query or registration.")
                    break
        return self._metrics[name]
    
    def list(self, 
             context: Optional[MetricContext] = None,
             category: Optional[MetricCategory] = None) -> List[str]:
        """
        List available metrics, optionally filtered by context and/or category.
        
        Args:
            context: Optional context filter
            category: Optional category filter
            
        Returns:
            List of metric names matching the filters
        """
        # Start with all metrics
        result = set(self._metrics.keys())
        
        # Apply filters
        if context is not None:
            result = result.intersection(self._by_context[context])
            
        if category is not None:
            result = result.intersection(self._by_category[category])
            
        return sorted(result)
    
    def create_instance(self, name: str, *args, **kwargs) -> BaseMetric:
        """
        Create an instance of a registered metric.
        
        Args:
            name: The name of the metric
            *args, **kwargs: Arguments to pass to the constructor
            
        Returns:
            Metric instance
        """
        logger.debug(f"MetricRegistry.create_instance: Attempting to create instance for metric name: '{name}' with params: {kwargs}")
        metric_class = self.get(name)
        if metric_class:
            logger.debug(f"MetricRegistry.create_instance: Found metric class: {metric_class.__name__}")
            return metric_class(*args, **kwargs)
        else:
            # This case should ideally be caught by self.get(name) raising KeyError
            logger.error(f"MetricRegistry.create_instance: Metric class for '{name}' not found by get_metric prior to instantiation.")
            raise KeyError(f"Metric '{name}' not found in registry, cannot create instance.")
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata about a metric.
        
        Args:
            name: The name of the metric
            
        Returns:
            Dictionary with metadata
        """
        metric_class = self.get(name)
        
        # Extract metadata attributes
        return {
            "name": name,
            "description": getattr(metric_class, 'description', ""),
            "greater_is_better": getattr(metric_class, 'greater_is_better', True),
            "context": getattr(metric_class, 'context', MetricContext.BOTH),
            "category": getattr(metric_class, 'category', MetricCategory.CUSTOM),
            "requires": getattr(metric_class, 'requires', {})
        }


# Create a global registry instance
_registry = MetricRegistry()


def register_metric(metric_class: Type[BaseMetric], 
                   name: Optional[str] = None,
                   context: Optional[MetricContext] = None,
                   category: Optional[MetricCategory] = None) -> None:
    """
    Register a metric in the global registry.
    
    Args:
        metric_class: The metric class to register
        name: Optional custom name
        context: Optional context override
        category: Optional category override
    """
    _registry.register(metric_class, name, context, category)


def get_metric(name: str) -> Type[BaseMetric]:
    """
    Get a metric class from the global registry.
    
    Args:
        name: The name of the metric
        
    Returns:
        The metric class
    """
    return _registry.get(name)


def list_metrics(context: Optional[MetricContext] = None,
                category: Optional[MetricCategory] = None) -> List[str]:
    """
    List metrics in the global registry.
    
    Args:
        context: Optional context filter
        category: Optional category filter
        
    Returns:
        List of metric names
    """
    return _registry.list(context, category)


def create_metric_instance(name: str, *args, **kwargs) -> BaseMetric:
    """
    Create a metric instance from the global registry.
    
    Args:
        name: The name of the metric
        *args, **kwargs: Arguments to pass to the constructor
        
    Returns:
        Metric instance
    """
    return _registry.create_instance(name, *args, **kwargs)


def get_metric_metadata(name: str) -> Dict[str, Any]:
    """
    Get metadata about a metric from the global registry.
    
    Args:
        name: The name of the metric
        
    Returns:
        Dictionary with metadata
    """
    return _registry.get_metadata(name)
