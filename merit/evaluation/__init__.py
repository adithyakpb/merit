"""
MERIT Evaluation Module

This module provides functionality for evaluating LLM applications.
"""

from .evaluators.base import BaseEvaluator
from .evaluators.llm import LLMEvaluator
from .evaluators.rag import RAGEvaluator, evaluate_rag, Response
from .metrics.base import BaseMetric
from .metrics.classification import ClassificationPerformanceMetric
from .metrics.rag import (
    CorrectnessMetric, 
    FaithfulnessMetric, 
    RelevanceMetric, 
    CoherenceMetric, 
    FluencyMetric
)
