from .base import BaseMetric
from abc import abstractmethod
from ...core.logging import get_logger

logger = get_logger(__name__)
class ClassificationPerformanceMetric(BaseMetric):
    """Base class for classification performance metrics."""
    has_binary_counts = False
    
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
    
    def _get_predictions(self, model, dataset):
        """
        Get predictions from the model on the dataset.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            
        Returns:
            tuple: (y_true, y_pred)
        """
        # Default implementation assumes model has a predict method
        # and dataset has a target attribute
        try:
            y_true = dataset.target
            y_pred = model.predict(dataset)
            return y_true, y_pred
        except AttributeError:
            # Try to handle different model/dataset interfaces
            try:
                # Try to get predictions from model's predict method
                if hasattr(model, 'predict'):
                    y_pred = model.predict(dataset)
                else:
                    # Assume model is callable
                    y_pred = model(dataset)
                
                # Try to get true values from dataset
                if hasattr(dataset, 'target'):
                    y_true = dataset.target
                elif hasattr(dataset, 'y'):
                    y_true = dataset.y
                elif hasattr(dataset, 'labels'):
                    y_true = dataset.labels
                else:
                    # Assume dataset is a tuple of (X, y)
                    _, y_true = dataset
                
                return y_true, y_pred
            except Exception as e:
                logger.error(f"Failed to get predictions: {str(e)}")
                raise
    
    @abstractmethod
    def _calculate_metric(self, y_true, y_pred, model):
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
    
    def _calculate_binary_counts(self, y_true, y_pred):
        """
        Calculate binary counts (TP, FP, TN, FN).
        
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
