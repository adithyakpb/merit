"""
Merit Evaluator Classes

This module provides base classes for evaluators used in evaluating Large Language Model powered systems.
"""

import json
from abc import ABC, abstractmethod
from ...core.prompts import Prompt
from ...core.logging import get_logger
from ...core.models import EvaluationResult, EvaluationReport

logger = get_logger(__name__)

class BaseEvaluator(ABC):
    """Base class for all evaluators."""
    
    def __init__(self, metrics=None):
        """
        Initialize the evaluator.
        
        Args:
            metrics: List of metrics to evaluate
        """
        self.metrics = metrics or []
    
    @abstractmethod
    def evaluate(self, *args, **kwargs) -> EvaluationReport:
        """
        Evaluate the system.
        
        Returns:
            EvaluationReport: The evaluation report
        """
        raise NotImplementedError
    
    @abstractmethod
    def _evaluate_sample(self, *args, **kwargs) -> EvaluationResult:
        """
        Evaluate a single sample.
        
        Returns:
            EvaluationResult: The evaluation result
        """
        raise NotImplementedError
