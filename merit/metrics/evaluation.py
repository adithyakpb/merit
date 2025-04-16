"""
Evaluation-specific metric implementations and adapters for MERIT.

This module provides the bridge between the existing evaluation metrics
and the new shared metrics foundation, ensuring backward compatibility
while enabling cross-component use of metrics.
"""

from typing import Dict, Any, List, Optional, Union, Type, Callable, TypeVar
import inspect

from .base import (
    BaseMetric,
    MetricContext,
    MetricCategory,
    register_metric
)

# Use our own implementations instead of importing from evaluation metrics
from .classification import ClassificationPerformanceMetric, LLMMeasuredBaseMetric
EvalBaseMetric = BaseMetric  # For backward compatibility
from ..core.logging import get_logger

logger = get_logger(__name__)


class EvaluationMetric(BaseMetric):
    """
    Base class for evaluation-specific metrics.
    
    These metrics are designed to work with evaluation data and focus on
    quality assessment rather than operational monitoring.
    """
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY
    
    def evaluate_dataset(self, 
                        predictions: List[Any], 
                        references: List[Any],
                        **kwargs) -> Dict[str, Any]:
        """
        Evaluate a dataset with this metric.
        
        Args:
            predictions: List of model predictions
            references: List of reference answers
            **kwargs: Additional arguments for specific metrics
            
        Returns:
            Dict with evaluation results
        """
        # Default implementation calls the metric for each example
        results = []
        for pred, ref in zip(predictions, references):
            result = self(pred, ref, **kwargs)
            results.append(result)
            
        # Aggregate results
        agg_result = self._aggregate_results(results)
        return agg_result
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate individual example results.
        
        Args:
            results: List of per-example metric results
            
        Returns:
            Aggregated result
        """
        # Basic aggregation - average the values
        if not results:
            return {"value": None}
            
        # For metrics that return a value
        values = [r.get("value") for r in results if r.get("value") is not None]
        if values:
            avg_value = sum(values) / len(values)
            return {
                "value": avg_value,
                "count": len(results),
                "values": values,
                "min": min(values),
                "max": max(values)
            }
            
        return {"value": None, "count": len(results)}


class AccuracyMetric(EvaluationMetric):
    """
    Measures the accuracy of model predictions.
    
    This metric is useful for classification and QA tasks.
    """
    name = "Accuracy"
    description = "Proportion of correct predictions"
    greater_is_better = True
    
    def __call__(self, prediction: Any, reference: Any, **kwargs) -> Dict[str, Any]:
        """
        Calculate accuracy for a single example.
        
        Args:
            prediction: Model prediction
            reference: Reference answer
            **kwargs: Additional arguments
            
        Returns:
            Dict with accuracy result
        """
        # Basic implementation - exact match
        correct = prediction == reference
        return {
            "value": 1.0 if correct else 0.0,
            "correct": correct
        }


class ExactMatchMetric(AccuracyMetric):
    """
    Measures exact match between prediction and reference.
    
    This is a specialized accuracy metric for string comparison.
    """
    name = "Exact Match"
    description = "Proportion of predictions that exactly match the reference"
    
    def __call__(self, prediction: str, reference: str, **kwargs) -> Dict[str, Any]:
        """
        Calculate exact match for strings.
        
        Args:
            prediction: Predicted string
            reference: Reference string
            **kwargs: Additional arguments
            
        Returns:
            Dict with exact match result
        """
        # Normalize strings for comparison
        if kwargs.get("case_sensitive", False):
            pred_norm = prediction.strip()
            ref_norm = reference.strip()
        else:
            pred_norm = prediction.strip().lower()
            ref_norm = reference.strip().lower()
            
        correct = pred_norm == ref_norm
        return {
            "value": 1.0 if correct else 0.0,
            "correct": correct,
            "prediction": prediction,
            "reference": reference
        }


# Adapter for existing evaluation metrics
class EvaluationMetricAdapter(BaseMetric):
    """
    Adapter to use existing evaluation metrics with the new shared foundation.
    
    This class wraps an existing evaluation metric class and adapts its interface
    to work with the new shared metrics foundation, ensuring backward compatibility.
    """
    context = MetricContext.EVALUATION
    
    def __init__(self, eval_metric_class: Type[EvalBaseMetric]):
        """
        Initialize with an existing evaluation metric class.
        
        Args:
            eval_metric_class: Existing evaluation metric class to adapt
        """
        self.eval_metric_class = eval_metric_class
        self.eval_metric_instance = eval_metric_class()
        
        # Copy attributes from the wrapped metric
        self.name = getattr(eval_metric_class, 'name', eval_metric_class.__name__)
        self.description = getattr(eval_metric_class, 'description', "")
        self.greater_is_better = getattr(eval_metric_class, 'greater_is_better', True)
        
        # Determine category based on class hierarchy
        if issubclass(eval_metric_class, ClassificationPerformanceMetric):
            self.category = MetricCategory.QUALITY
        elif issubclass(eval_metric_class, LLMMeasuredBaseMetric):
            self.category = MetricCategory.QUALITY
        else:
            self.category = MetricCategory.CUSTOM
    
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Call the adapted evaluation metric.
        
        Args:
            *args, **kwargs: Arguments to pass to the evaluation metric
            
        Returns:
            Dict with the metric result
        """
        # Call the wrapped metric
        result = self.eval_metric_instance(*args, **kwargs)
        
        # If the result is already a dict, return it
        if isinstance(result, dict):
            return result
            
        # If the result is a scalar, wrap it in a dict
        if isinstance(result, (int, float, bool)):
            return {"value": result}
            
        # If the result is something else, convert it to a dict
        return {"value": result}


def adapt_evaluation_metric(eval_metric_class: Type[EvalBaseMetric], 
                           name: Optional[str] = None,
                           category: Optional[MetricCategory] = None) -> Type[BaseMetric]:
    """
    Adapt an existing evaluation metric to work with the new shared foundation.
    
    Args:
        eval_metric_class: Existing evaluation metric class to adapt
        name: Optional custom name
        category: Optional category override
        
    Returns:
        Adapted metric class
    """
    adapter = EvaluationMetricAdapter(eval_metric_class)
    
    # Register the adapted metric
    register_metric(
        eval_metric_class,
        name=name or adapter.name,
        context=MetricContext.EVALUATION,
        category=category or adapter.category
    )
    
    return adapter


# Register the new evaluation metrics
register_metric(AccuracyMetric)
register_metric(ExactMatchMetric)

# Import and adapt existing evaluation metrics
# This would typically be done by importing all metrics from the evaluation package
# and calling adapt_evaluation_metric for each one.
