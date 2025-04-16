"""
Base interfaces for data collection in MERIT monitoring.

This module defines the abstract base classes and supporting types for
all data collectors, ensuring consistent behavior across different collection methods.
"""

import abc
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional, Callable, TypeVar, Generic, Union
from dataclasses import dataclass, field

from ..models import BaseInteraction, LLMInteraction


# Type for collection callbacks
CollectionCallback = Callable[[Union[Dict[str, Any], BaseInteraction]], None]


class CollectionStatus(Enum):
    """Status of a collection operation."""
    SUCCESS = "success"
    PARTIAL = "partial"  # Some data collected but with errors
    FAILURE = "failure"
    TIMEOUT = "timeout"


@dataclass
class CollectionResult:
    """
    Result of a collection operation.
    
    This standardizes the return value of collection operations,
    providing both the collected data and metadata about the operation.
    
    Attributes:
        status: Status of the collection operation
        data: The collected data (if any)
        items_processed: Number of items processed
        items_collected: Number of items successfully collected
        start_time: When the collection started
        end_time: When the collection ended
        error: Error message if status is not SUCCESS
    """
    status: CollectionStatus
    data: List[Dict[str, Any]] = field(default_factory=list)
    items_processed: int = 0
    items_collected: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        """Set end time if not provided."""
        if self.end_time is None:
            self.end_time = datetime.now()
    
    @property
    def duration(self) -> float:
        """Duration of the collection operation in seconds."""
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.items_processed == 0:
            return 0.0
        return (self.items_collected / self.items_processed) * 100.0


class BaseDataCollector(abc.ABC):
    """
    Abstract base class for all data collectors.
    
    Data collectors are responsible for gathering interaction data from
    various sources such as logs, API traffic, or databases.
    
    Implementations should override the abstract methods and may add
    additional functionality specific to their collection method.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the collector with configuration.
        
        Args:
            config: Configuration dictionary for the collector
        """
        self.config = config or {}
        self.callbacks: List[CollectionCallback] = []
        self._is_running = False
        
    def register_callback(self, callback: CollectionCallback) -> None:
        """
        Register a callback function for processing collected data.
        
        The callback will be called for each collected interaction with either
        a dictionary of raw data or a BaseInteraction object.
        
        Args:
            callback: Function to call with each collected data item
        """
        self.callbacks.append(callback)
    
    def _notify_callbacks(self, data: Union[Dict[str, Any], BaseInteraction]) -> None:
        """
        Notify all registered callbacks with collected data.
        
        Args:
            data: Collected data to pass to callbacks
        """
        for callback in self.callbacks:
            try:
                callback(data)
            except Exception as e:
                # Log error but continue with other callbacks
                print(f"Error in callback: {e}")
    
    @abc.abstractmethod
    def start(self) -> None:
        """
        Start collecting data.
        
        This method typically begins any background collection processes
        or opens necessary connections.
        """
        self._is_running = True
    
    @abc.abstractmethod
    def stop(self) -> None:
        """
        Stop collecting data.
        
        This method should clean up resources and close connections.
        """
        self._is_running = False
    
    @abc.abstractmethod
    def collect(self) -> CollectionResult:
        """
        Perform a collection operation.
        
        This method should collect data from the source and return a
        CollectionResult object with the results. It's typically called
        periodically or in response to events.
        
        Returns:
            CollectionResult with the results of the collection operation
        """
        pass
    
    @property
    def is_running(self) -> bool:
        """Whether the collector is currently running."""
        return self._is_running
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the collector.
        
        Returns:
            Dictionary with status information
        """
        return {
            "is_running": self.is_running,
            "collector_type": self.__class__.__name__,
            "config": self.config
        }
