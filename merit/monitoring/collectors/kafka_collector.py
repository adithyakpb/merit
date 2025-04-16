"""
MERIT Kafka Collector

This module provides a collector for monitoring LLM interactions via Kafka.
"""

import json
import threading
import time
from typing import Dict, List, Any, Optional, Callable
from confluent_kafka import Consumer, KafkaError, KafkaException

from .collector import BaseDataCollector, CollectionResult, CollectionStatus
from ..models import LLMInteraction, LLMRequest, LLMResponse, TokenInfo
from ...core.logging import get_logger

logger = get_logger(__name__)

class KafkaCollector(BaseDataCollector):
    """
    Collector that extracts LLM interaction data from Kafka topics.
    
    This collector can subscribe to Kafka topics that contain LLM interaction
    data and process them in real-time.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with Kafka configuration.
        
        Args:
            config: Configuration for the Kafka consumer
                - bootstrap_servers: Kafka bootstrap servers (default: localhost:9092)
                - topics: List of topics to subscribe to (default: ["llm-interactions"])
                - group_id: Consumer group ID (default: "merit-monitor")
                - poll_interval: How often to poll for messages in seconds (default: 1.0)
                - auto_offset_reset: Where to start consuming from (default: "latest")
                - session_timeout_ms: Consumer session timeout (default: 30000)
                - message_format: Format of messages (json or avro, default: "json")
                - transform_fn: Function to transform raw messages into LLMInteraction objects
        """
        super().__init__(config or {})
        self.bootstrap_servers = self.config.get("bootstrap_servers", "localhost:9092")
        self.topics = self.config.get("topics", ["llm-interactions"])
        self.group_id = self.config.get("group_id", "merit-monitor")
        self.poll_interval = float(self.config.get("poll_interval", 1.0))
        self.auto_offset_reset = self.config.get("auto_offset_reset", "latest")
        self.session_timeout_ms = int(self.config.get("session_timeout_ms", 30000))
        self.message_format = self.config.get("message_format", "json")
        self.transform_fn = self.config.get("transform_fn", self._default_transform)
        
        # Internal state
        self._consumer = None
        self._running = False
        self._thread = None
        self._messages = []
        self._lock = threading.Lock()
        
    def start(self):
        """Start consuming from Kafka topics."""
        if self._running:
            logger.warning("Kafka collector already running")
            return
        
        # Create consumer
        consumer_config = {
            'bootstrap.servers': self.bootstrap_servers,
            'group.id': self.group_id,
            'auto.offset.reset': self.auto_offset_reset,
            'session.timeout.ms': self.session_timeout_ms,
            'enable.auto.commit': True,
        }
        
        try:
            self._consumer = Consumer(consumer_config)
            self._consumer.subscribe(self.topics)
            logger.info(f"Subscribed to topics: {self.topics}")
            
            # Start background thread
            self._running = True
            self._thread = threading.Thread(target=self._consume_loop, daemon=True)
            self._thread.start()
            logger.info("Kafka collector started")
            
        except KafkaException as e:
            logger.error(f"Failed to start Kafka collector: {str(e)}")
            self._running = False
            raise
    
    def stop(self):
        """Stop consuming from Kafka topics."""
        if not self._running:
            logger.warning("Kafka collector not running")
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        
        if self._consumer:
            self._consumer.close()
            self._consumer = None
        
        logger.info("Kafka collector stopped")
    
    def collect(self) -> CollectionResult:
        """
        Process collected messages.
        
        Returns:
            CollectionResult: The collected messages since last call
        """
        with self._lock:
            messages = self._messages
            self._messages = []
        
        result = CollectionResult(
            status=CollectionStatus.SUCCESS,
            data=[],
            items_processed=0,
            items_collected=0
        )
        
        for msg in messages:
            result.items_processed += 1
            try:
                interaction = self._process_message(msg)
                if interaction:
                    # Convert to dictionary for storage if needed
                    if isinstance(interaction, LLMInteraction):
                        interaction_dict = interaction.to_dict()
                    else:
                        interaction_dict = interaction
                    
                    result.data.append(interaction_dict)
                    result.items_collected += 1
                    
                    # Notify callbacks
                    self._notify_callbacks(interaction)
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
        
        # Add metadata
        result.metadata = {
            "source": "kafka",
            "topics": self.topics,
            "count": result.items_collected
        }
        
        return result
    
    def _consume_loop(self):
        """Background thread for consuming messages."""
        while self._running:
            try:
                msg = self._consumer.poll(self.poll_interval)
                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        logger.debug(f"Reached end of partition {msg.partition()}")
                    else:
                        logger.error(f"Error consuming message: {msg.error()}")
                else:
                    # Process message
                    with self._lock:
                        self._messages.append(msg)
                    
            except Exception as e:
                logger.error(f"Error in Kafka consumer loop: {str(e)}")
                time.sleep(1)  # Avoid busy loop on error
    
    def _process_message(self, msg):
        """
        Process a Kafka message.
        
        Args:
            msg: Kafka message
            
        Returns:
            LLMInteraction: The processed interaction
        """
        try:
            # Extract message value
            value = msg.value()
            
            # Parse message based on format
            if self.message_format == "json":
                data = json.loads(value)
            elif self.message_format == "avro":
                # TODO: Implement Avro support
                logger.warning("Avro format not yet supported")
                return None
            else:
                logger.warning(f"Unsupported message format: {self.message_format}")
                return None
            
            # Transform raw message to LLMInteraction
            return self.transform_fn(data)
            
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")
            return None
    
    def _default_transform(self, data) -> LLMInteraction:
        """
        Default transformation function for Kafka messages.
        
        Args:
            data: Raw message data
            
        Returns:
            LLMInteraction: The transformed interaction
        """
        # Expect a structure similar to our LLMInteraction
        return LLMInteraction(
            timestamp=data.get("timestamp"),
            model=data.get("model"),
            request={
                "prompt": data.get("request", {}).get("prompt", "")
            },
            response={
                "completion": data.get("response", {}).get("completion", ""),
                "latency": data.get("response", {}).get("latency", 0.0),
                "status": data.get("response", {}).get("status", "unknown")
            },
            tokens={
                "input_tokens": data.get("tokens", {}).get("input_tokens", 0),
                "output_tokens": data.get("tokens", {}).get("output_tokens", 0),
                "total_tokens": data.get("tokens", {}).get("total_tokens", 0)
            },
            metadata=data.get("metadata", {})
        )
