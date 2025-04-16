"""
MERIT Monitoring Service

This module provides the main monitoring service that coordinates collectors,
metrics, storage, and UI components.
"""

import threading
import time
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import uuid

from ..core.logging import get_logger
from .collectors.collector import BaseDataCollector
from ..storage.persistence import BaseStorage
from ..storage import SQLiteStorage
from ..metrics import list_metrics, create_metric_instance, MetricContext

logger = get_logger(__name__)

class MonitoringService:
    """
    Central service for LLM monitoring.
    
    This service coordinates data collection, storage, and metrics calculation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the monitoring service.
        
        Args:
            config: Configuration for the service
                - collection_interval: How often to collect data in seconds (default: 60)
                - storage_type: Type of storage to use ("sqlite", "file", default: "sqlite")
                - storage_config: Configuration for the storage backend
                - retention_days: Number of days to retain data (default: 30)
                - purge_interval: How often to purge old data in seconds (default: 86400, 1 day)
                - metrics: List of metrics to enable (default: all)
        """
        self.config = config or {}
        self.collection_interval = float(self.config.get("collection_interval", 60))
        
        # Initialize storage
        storage_type = self.config.get("storage_type", "sqlite")
        storage_config = self.config.get("storage_config", {})
        
        from ..storage import DatabaseFactory
        storage_config_with_type = {"type": storage_type, **storage_config}
        self.storage = DatabaseFactory.create(storage_config_with_type)
        
        # Data retention settings
        self.retention_days = int(self.config.get("retention_days", 30))
        self.purge_interval = float(self.config.get("purge_interval", 86400))  # 1 day
        
        # Collectors
        self.collectors = []
        
        # Metrics
        self.metrics = {}
        self._init_metrics()
        
        # Internal state
        self._running = False
        self._collection_thread = None
        self._purge_thread = None
        self._last_collection = None
        self._last_purge = None
    
    def _init_metrics(self):
        """Initialize metrics."""
        # Get metrics from config or use all
        metric_names = self.config.get("metrics")
        
        if metric_names is None:
            # Use all monitoring metrics - list_metrics returns a list of strings
            metric_names = list_metrics(context=MetricContext.MONITORING)
        
        # Initialize metrics
        for name in metric_names:
            try:
                metric = create_metric_instance(name)
                if metric:
                    self.metrics[name] = metric
                    logger.info(f"Enabled metric: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize metric {name}: {str(e)}")
    
    def add_collector(self, collector: BaseDataCollector):
        """
        Add a data collector.
        
        Args:
            collector: The collector to add
        """
        if collector not in self.collectors:
            self.collectors.append(collector)
            logger.info(f"Added collector: {collector.__class__.__name__}")
    
    def remove_collector(self, collector: BaseDataCollector):
        """
        Remove a data collector.
        
        Args:
            collector: The collector to remove
        """
        if collector in self.collectors:
            self.collectors.remove(collector)
            logger.info(f"Removed collector: {collector.__class__.__name__}")
    
    def start(self):
        """Start the monitoring service."""
        if self._running:
            logger.warning("Monitoring service already running")
            return
        
        try:
            # Start collectors
            for collector in self.collectors:
                try:
                    collector.start()
                except Exception as e:
                    logger.error(f"Failed to start collector {collector.__class__.__name__}: {str(e)}")
            
            # Start background threads
            self._running = True
            
            # Start collection thread
            self._collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self._collection_thread.start()
            
            # Start purge thread
            self._purge_thread = threading.Thread(target=self._purge_loop, daemon=True)
            self._purge_thread.start()
            
            logger.info("Monitoring service started")
            
        except Exception as e:
            logger.error(f"Failed to start monitoring service: {str(e)}")
            self._running = False
            raise
    
    def stop(self):
        """Stop the monitoring service."""
        if not self._running:
            logger.warning("Monitoring service not running")
            return
        
        self._running = False
        
        # Stop collectors
        for collector in self.collectors:
            try:
                collector.stop()
            except Exception as e:
                logger.error(f"Failed to stop collector {collector.__class__.__name__}: {str(e)}")
        
        # Wait for threads to finish
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        
        if self._purge_thread:
            self._purge_thread.join(timeout=5.0)
        
        logger.info("Monitoring service stopped")
    
    def _collection_loop(self):
        """Background thread for collecting data."""
        while self._running:
            try:
                # Collect data
                self._collect_data()
                self._last_collection = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in collection loop: {str(e)}")
            
            # Wait for next collection
            time.sleep(self.collection_interval)
    
    def _purge_loop(self):
        """Background thread for purging old data."""
        while self._running:
            try:
                # Check if purge is due
                now = datetime.now()
                if (self._last_purge is None or 
                    (now - self._last_purge).total_seconds() >= self.purge_interval):
                    # Purge old data
                    count = self.storage.purge(retention_days=self.retention_days)
                    self._last_purge = now
                    
                    if count > 0:
                        logger.info(f"Purged {count} old interactions")
                
            except Exception as e:
                logger.error(f"Error in purge loop: {str(e)}")
            
            # Wait before checking again
            time.sleep(60)  # Check every minute
    
    def _collect_data(self):
        """Collect data from all collectors and store it."""
        all_interactions = []
        
        # Collect from each collector
        for collector in self.collectors:
            try:
                # Collect data
                result = collector.collect()
                
                if result and result.data:
                    # Add timestamp and ID if missing
                    for interaction in result.data:
                        if "timestamp" not in interaction:
                            interaction["timestamp"] = datetime.now().isoformat()
                        
                        if "id" not in interaction:
                            interaction["id"] = str(uuid.uuid4())
                    
                    # Add to all interactions
                    all_interactions.extend(result.data)
            
            except Exception as e:
                logger.error(f"Error collecting from {collector.__class__.__name__}: {str(e)}")
        
        # Store interactions
        if all_interactions:
            try:
                self.storage.store(all_interactions)
                logger.info(f"Stored {len(all_interactions)} interactions")
            except Exception as e:
                logger.error(f"Error storing interactions: {str(e)}")
    
    def get_metrics(self,
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   models: Optional[List[str]] = None,
                   metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate metrics for the specified time range and models.
        
        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            models: List of models to filter by
            metrics: List of metrics to calculate
            
        Returns:
            Dict[str, Any]: Dictionary of metric results
        """
        try:
            # Set default time range
            if start_time is None:
                start_time = datetime.now() - timedelta(days=1)
            
            if end_time is None:
                end_time = datetime.now()
            
            # Retrieve interactions using query parameters
            query = {}
            if start_time or end_time:
                query["timestamp"] = {}
                if start_time:
                    query["timestamp"]["$gte"] = start_time.isoformat()
                if end_time:
                    query["timestamp"]["$lte"] = end_time.isoformat()
                    
            if models:
                query["model"] = models[0]  # Using just the first model for simplicity
                
            interactions = self.storage.query(query=query, limit=1000)
            
            if not interactions:
                return {}
            
            # Determine which metrics to calculate
            metric_names = metrics or list(self.metrics.keys())
            results = {}
            
            # Calculate each metric
            for name in metric_names:
                if name in self.metrics:
                    try:
                        metric = self.metrics[name]
                        result = metric(interactions)
                        results[name] = result
                    except Exception as e:
                        logger.error(f"Error calculating metric {name}: {str(e)}")
                        results[name] = {"error": str(e)}
                else:
                    logger.warning(f"Metric not found: {name}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting metrics: {str(e)}")
            return {"error": str(e)}
    
    def get_interactions(self,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        models: Optional[List[str]] = None,
                        limit: int = 100,
                        offset: int = 0) -> List[Dict[str, Any]]:
        """
        Retrieve interactions from storage.
        
        Args:
            start_time: Start time for filtering
            end_time: End time for filtering
            models: List of models to filter by
            limit: Maximum number of interactions to retrieve
            offset: Number of interactions to skip
            
        Returns:
            List[Dict[str, Any]]: List of interactions
        """
        try:
            # Build query parameters
            query = {}
            if start_time or end_time:
                query["timestamp"] = {}
                if start_time:
                    query["timestamp"]["$gte"] = start_time.isoformat()
                if end_time:
                    query["timestamp"]["$lte"] = end_time.isoformat()
                    
            if models and len(models) > 0:
                query["model"] = models[0]  # Use first model for now
                
            return self.storage.query(query=query, limit=limit, offset=offset)
        except Exception as e:
            logger.error(f"Error getting interactions: {str(e)}")
            return []


def create_monitoring_service(config: Optional[Dict[str, Any]] = None) -> MonitoringService:
    """
    Create and configure a monitoring service with default collectors.
    
    Args:
        config: Configuration dictionary
            - collection_interval: How often to collect data in seconds (default: 60)
            - storage_type: Type of storage to use ("sqlite", "file", default: "sqlite")
            - storage_config: Configuration for the storage backend
            - collectors: List of collector configurations
                - type: Type of collector ("api", "log", "kafka")
                - config: Configuration for the collector
            
    Returns:
        MonitoringService: The configured monitoring service
    """
    service = MonitoringService(config)
    
    # Add collectors from config
    collector_configs = config.get("collectors", [])
    
    for collector_config in collector_configs:
        collector_type = collector_config.get("type")
        collector_config = collector_config.get("config", {})
        
        if collector_type == "api":
            from .collectors.api_collector import APIProxyCollector
            collector = APIProxyCollector(collector_config)
            service.add_collector(collector)
            
        elif collector_type == "log":
            from .collectors.log_collector import LogDataCollector
            collector = LogDataCollector(collector_config)
            service.add_collector(collector)
            
        elif collector_type == "kafka":
            from .collectors.kafka_collector import KafkaCollector
            collector = KafkaCollector(collector_config)
            service.add_collector(collector)
    
    return service
