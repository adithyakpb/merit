"""
MERIT Evaluation Module

This module provides functionality for evaluating LLM applications.
"""

from .evaluators.base import BaseEvaluator
from .evaluators.llm import LLMEvaluator
from .evaluators.rag import RAGEvaluator, evaluate_rag, Response

# Import metrics from the central metrics module for backward compatibility
from ..metrics.base import BaseMetric
from ..metrics.llm_measured import LLMMeasuredBaseMetric
from ..metrics.rag import (
    CorrectnessMetric, 
    FaithfulnessMetric, 
    RelevanceMetric, 
    CoherenceMetric, 
    FluencyMetric
)
