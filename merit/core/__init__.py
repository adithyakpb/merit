"""
Merit Core Module

This module provides core functionality for the Merit framework.
"""

from .logging import get_logger
from .models import EvaluationResult, EvaluationReport, TestSet, TestInput
from .utils import parse_json
from .cache import cache_embeddings, is_caching_available
