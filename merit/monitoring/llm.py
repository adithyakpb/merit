"""
LLM Monitoring Module

This module provides functionality for monitoring LLM interactions,
enabling collection and analysis of generation data including inputs,
outputs, timestamps, and other relevant metrics.
"""

import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Union, Callable

from .models import LLMInteraction, LLMRequest, LLMResponse, TokenInfo
from ..core.logging import get_logger

logger = get_logger(__name__)

class LLMMonitor:
    """
    Monitor for LLM generation activities.
    
    This class provides methods for tracking LLM interactions, aggregating
    metrics, and storing the data for further analysis.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the LLM monitor.
        
        Args:
            config: Configuration dictionary with the following options:
                - storage: Storage configuration for persisting interactions
                - callbacks: List of callback functions for real-time processing
                - enable_metrics: Whether to calculate metrics in real-time
        """
        self.config = config or {}
        self.storage = None
        self.callbacks = []
        self.enable_metrics = self.config.get("enable_metrics", True)
        
        # Set up storage if configured
        storage_config = self.config.get("storage")
        if storage_config:
            try:
                from ..storage import DatabaseFactory
                self.storage = DatabaseFactory.create(storage_config)
                logger.info(f"LLM monitor initialized with storage: {type(self.storage).__name__}")
            except Exception as e:
                logger.error(f"Error initializing storage: {e}")
        
        # Set up callbacks
        callbacks = self.config.get("callbacks", [])
        for callback in callbacks:
            self.add_callback(callback)
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a callback function for real-time interaction processing.
        
        Args:
            callback: Function that takes an interaction dictionary and does something with it
        """
        if callable(callback):
            self.callbacks.append(callback)
    
    def track(self, 
            prompt: str, 
            completion: str, 
            model: Optional[str] = None,
            input_tokens: Optional[int] = None,
            output_tokens: Optional[int] = None,
            start_time: Optional[datetime] = None,
            end_time: Optional[datetime] = None,
            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Track an LLM interaction with the given parameters.
        
        Args:
            prompt: The input prompt
            completion: The generated completion
            model: The model used for generation
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            start_time: When the generation was started
            end_time: When the generation was completed
            metadata: Additional metadata about the interaction
            
        Returns:
            The created interaction object
        """
        # Calculate timing information
        now = datetime.now()
        
        if start_time is None:
            start_time = now
            
        if end_time is None:
            end_time = now
            
        # Calculate latency
        if end_time and start_time:
            latency = (end_time - start_time).total_seconds()
        else:
            latency = None
            
        # Calculate tokens if not provided
        if input_tokens is None and prompt:
            # Rough token count based on whitespace splitting
            input_tokens = len(prompt.split())
            
        if output_tokens is None and completion:
            # Rough token count based on whitespace splitting
            output_tokens = len(completion.split())
            
        # Calculate total tokens
        total_tokens = 0
        if input_tokens is not None:
            total_tokens += input_tokens
            
        if output_tokens is not None:
            total_tokens += output_tokens
            
        # Create token info
        token_info = None
        if input_tokens is not None or output_tokens is not None:
            token_info = TokenInfo(
                input_tokens=input_tokens or 0,
                output_tokens=output_tokens or 0,
                total_tokens=total_tokens
            )
            
        # Create request
        request = LLMRequest(
            prompt=prompt,
            model=model,
            timestamp=start_time.isoformat()
        )
        
        # Create response - make sure we include request_id
        response = LLMResponse(
            request_id=request.id,
            completion=completion,
            model=model,
            tokens=token_info,
            latency=latency,
            timestamp=end_time.isoformat()
        )
        
        # Create full interaction
        interaction = LLMInteraction(
            timestamp=start_time.isoformat(),
            model=model,
            request=request,
            response=response,
            metadata=metadata or {}
        )
        
        # Convert to dictionary
        interaction_dict = interaction.to_dict()
        
        # Store the interaction
        self._store_interaction(interaction_dict)
        
        # Return the interaction
        return interaction_dict
    
    def track_from_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Track an LLM interaction from a dictionary.
        
        Args:
            data: Dictionary with interaction data
            
        Returns:
            The processed interaction dictionary
        """
        # Ensure required fields
        if "request" not in data and "prompt" in data:
            # Convert flat structure to nested
            data["request"] = {
                "prompt": data.pop("prompt")
            }
            
        if "response" not in data and "completion" in data:
            # Convert flat structure to nested
            data["response"] = {
                "completion": data.pop("completion")
            }
            
        # Add timestamp if missing
        if "timestamp" not in data:
            data["timestamp"] = datetime.now().isoformat()
            
        # Store the interaction
        self._store_interaction(data)
        
        return data
    
    def _store_interaction(self, interaction: Dict[str, Any]) -> None:
        """
        Store an interaction and process it with callbacks.
        
        Args:
            interaction: Interaction dictionary to store
        """
        # Process with callbacks
        for callback in self.callbacks:
            try:
                callback(interaction)
            except Exception as e:
                logger.error(f"Error in callback: {e}")
        
        # Store in storage if available
        if self.storage:
            try:
                self.storage.store([interaction])
            except Exception as e:
                logger.error(f"Error storing interaction: {e}")
    
    def get_recent_interactions(self, 
                               limit: int = 100, 
                               model: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recent interactions from storage.
        
        Args:
            limit: Maximum number of interactions to retrieve
            model: Filter by specific model
            
        Returns:
            List of interaction dictionaries
        """
        if not self.storage:
            return []
            
        query = {}
        if model:
            query["model"] = model
            
        try:
            return self.storage.query(query=query, limit=limit)
        except Exception as e:
            logger.error(f"Error retrieving interactions: {e}")
            return []
    
    def calculate_metrics(self, 
                        interactions: Optional[List[Dict[str, Any]]] = None,
                        metric_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate metrics from interactions.
        
        Args:
            interactions: List of interactions to analyze, or None to use storage
            metric_names: List of metric names to calculate, or None for all
            
        Returns:
            Dictionary mapping metric names to results
        """
        if not interactions and self.storage:
            # Get from storage
            try:
                interactions = self.storage.retrieve(limit=1000)
            except Exception as e:
                logger.error(f"Error retrieving interactions: {e}")
                return {}
                
        if not interactions:
            return {}
        
        try:
            from ..metrics import calculate_metrics
            return calculate_metrics(interactions, metric_names)
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def close(self) -> None:
        """Close the monitor and release resources."""
        if self.storage:
            try:
                self.storage.close()
            except Exception as e:
                logger.error(f"Error closing storage: {e}")


# Singleton instance for easy access
_default_monitor = None

def get_monitor() -> LLMMonitor:
    """
    Get the default monitor instance.
    
    Returns:
        Default LLMMonitor instance
    """
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = LLMMonitor()
    return _default_monitor

def track_generation(prompt: str, 
                    completion: str, 
                    model: Optional[str] = None,
                    **kwargs) -> Dict[str, Any]:
    """
    Track an LLM generation using the default monitor.
    
    Args:
        prompt: The input prompt
        completion: The generated completion
        model: The model used for generation
        **kwargs: Additional parameters to pass to track()
        
    Returns:
        The created interaction object
    """
    monitor = get_monitor()
    return monitor.track(prompt, completion, model, **kwargs)

def setup_monitoring(config: Dict[str, Any]) -> LLMMonitor:
    """
    Set up monitoring with the given configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configured LLMMonitor instance
    """
    global _default_monitor
    _default_monitor = LLMMonitor(config)
    return _default_monitor
