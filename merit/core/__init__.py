"""
MERIT Core Module

This module provides core functionality for the MERIT framework.
"""

from .logging import get_logger
from .models import TestSet, TestItem
from .utils import parse_json, batch_iterator
from .cache import cache_embeddings, is_caching_available
