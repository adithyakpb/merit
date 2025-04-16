"""
MERIT Classification Metrics

This module provides metrics for evaluating classification-based systems.
"""

from abc import abstractmethod
from .base import BaseMetric, MetricContext, MetricCategory, register_metric
from ..core.logging import get_logger
from ..core.models import TestItem, Input, Response    
from ..core.prompts import Prompt
from ..api.base import BaseAPIClient

logger = get_logger(__name__)

class ClassificationPerformanceMetric(BaseMetric):
    """Base class for classification performance metrics."""
    has_binary_counts = False
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY
    
    def __call__(self, model, dataset):
        """
        Calculate the metric for a model on a dataset.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            
        Returns:
            dict: A dictionary with the metric value and additional information
        """
        y_true, y_pred = self._get_predictions(model, dataset)
        value = self._calculate_metric(y_true, y_pred, model)
        affected_samples = self._calculate_affected_samples(y_true, y_pred, model)
        binary_counts = self._calculate_binary_counts(y_true, y_pred) if self.has_binary_counts else None
        
        return {
            "value": value,
            "affected_samples": affected_samples,
            "binary_counts": binary_counts
        }
    
    @abstractmethod
    def _calculate_metric(self, input: Input, response: Response, client: BaseAPIClient):
        """
        Calculate the metric value.
        
        Args:
            y_true: The true values
            y_pred: The predicted values
            model: The model
            
        Returns:
            float: The metric value
        """
        raise NotImplementedError
    
    def _calculate_affected_samples(self, y_true, y_pred, model):
        """
        Calculate the number of affected samples.
        
        Args:
            y_true: The true values
            y_pred: The predicted values
            model: The model
            
        Returns:
            int: The number of affected samples
        """
        return len(y_true)
    
    def _calculate_confusion_matrix(self, y_true, y_pred):
        """
        Calculate the values for the confusion matrix (TP, FP, TN, FN).
        
        Args:
            y_true: The true values
            y_pred: The predicted values
            
        Returns:
            dict: A dictionary with binary counts
        """
        # Default implementation for binary classification
        return {
            "tp": sum((y_t == 1 and y_p == 1) for y_t, y_p in zip(y_true, y_pred)),
            "fp": sum((y_t == 0 and y_p == 1) for y_t, y_p in zip(y_true, y_pred)),
            "tn": sum((y_t == 0 and y_p == 0) for y_t, y_p in zip(y_true, y_pred)),
            "fn": sum((y_t == 1 and y_p == 0) for y_t, y_p in zip(y_true, y_pred))
        }

class LLMMeasuredBaseMetric(BaseMetric):
    """Base class for metrics that are calculated using an LLM."""
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY
    
    def __call__(self, test_item: TestItem, response, client_llm_callable: callable, prompt: "Prompt"):
        """
        Calculate the metric for a TestItem and Response Pair using a LLM call.
        
        Args:
            test_item: The test item to evaluate
            response: The evaluated system's response
            client: The client to use for the evaluation
            prompt: The prompt to use for the evaluation
            
        Returns:
            dict: A dictionary with the metric value and additional information
        """
        value = self._calculate_metric(test_item, response, prompt, client_llm_callable)
        
        return {
            "value": value
        }
    
    @abstractmethod
    def _calculate_metric(self, test_item: TestItem, response: Response, prompt: Prompt, llm_callable: callable):
        """
        Calculate the metric value.
        
        Args:
            test_item: The test item to evaluate
            response: The evaluated system's response
            prompt: The prompt to use for the evaluation
            llm_callable: The LLM callable to use for the evaluation
            
        Returns:
            float: The metric value
        """
        raise NotImplementedError
