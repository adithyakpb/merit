"""
MERIT Java Kafka Adapter

This module provides an adapter for existing Java Kafka implementations to work with MERIT.
It allows organizations with existing Java Kafka infrastructure to easily integrate
with MERIT's monitoring capabilities.
"""

import json
import requests
import time
from typing import Dict, Any, Optional, List
import threading

from .collector import BaseDataCollector, CollectionResult, CollectionStatus
from ..models import LLMInteraction, LLMRequest, LLMResponse, TokenInfo
from ...core.logging import get_logger

logger = get_logger(__name__)

class JavaKafkaAdapter(BaseDataCollector):
    """
    Adapter that connects to existing Java Kafka consumers via a REST interface.
    
    This adapter allows organizations to leverage their existing Java Kafka infrastructure
    while still using MERIT for monitoring. It works by connecting to a lightweight REST API
    that your Java application exposes, which serves processed Kafka messages.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize with Java Kafka adapter configuration.
        
        Args:
            config: Configuration for the adapter
                - endpoint_url: URL of the Java service REST endpoint (required)
                - api_key: API key for authentication (optional)
                - poll_interval: How often to poll for messages in seconds (default: 5.0)
                - batch_size: Maximum number of messages to fetch per request (default: 100)
                - timeout: Request timeout in seconds (default: 10.0)
                - transform_fn: Function to transform received data into LLMInteraction objects
        """
        super().__init__(config or {})
        
        self.endpoint_url = self.config.get("endpoint_url")
        if not self.endpoint_url:
            raise ValueError("endpoint_url is required for JavaKafkaAdapter")
            
        self.api_key = self.config.get("api_key")
        self.poll_interval = float(self.config.get("poll_interval", 5.0))
        self.batch_size = int(self.config.get("batch_size", 100))
        self.timeout = float(self.config.get("timeout", 10.0))
        self.transform_fn = self.config.get("transform_fn", self._default_transform)
        
        # Internal state
        self._running = False
        self._thread = None
        self._messages = []
        self._lock = threading.Lock()
        self._last_timestamp = None
    
    def start(self):
        """Start polling the Java Kafka service endpoint."""
        if self._running:
            logger.warning("Java Kafka adapter already running")
            return
        
        try:
            # Test the connection
            response = self._make_request(params={"test": True})
            if not response or response.get("status") != "ok":
                raise ValueError(f"Failed to connect to Java Kafka service: {response}")
            
            # Start background thread
            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
            logger.info("Java Kafka adapter started")
            
        except Exception as e:
            logger.error(f"Failed to start Java Kafka adapter: {str(e)}")
            self._running = False
            raise
    
    def stop(self):
        """Stop polling the Java Kafka service."""
        if not self._running:
            logger.warning("Java Kafka adapter not running")
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        
        logger.info("Java Kafka adapter stopped")
    
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
                interaction = self.transform_fn(msg)
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
            "source": "java_kafka",
            "endpoint": self.endpoint_url,
            "count": result.items_collected
        }
        
        return result
    
    def _poll_loop(self):
        """Background thread for polling the Java Kafka service."""
        while self._running:
            try:
                # Prepare request parameters
                params = {
                    "max_messages": self.batch_size
                }
                
                if self._last_timestamp:
                    params["since_timestamp"] = self._last_timestamp
                
                # Make request
                response = self._make_request(params=params)
                
                if response and "messages" in response:
                    messages = response.get("messages", [])
                    if messages:
                        # Update timestamp
                        if "latest_timestamp" in response:
                            self._last_timestamp = response["latest_timestamp"]
                        elif messages and "timestamp" in messages[-1]:
                            self._last_timestamp = messages[-1]["timestamp"]
                        
                        # Add messages to queue
                        with self._lock:
                            self._messages.extend(messages)
                        
                        logger.debug(f"Received {len(messages)} messages from Java Kafka service")
                
            except Exception as e:
                logger.error(f"Error polling Java Kafka service: {str(e)}")
            
            # Wait for next poll
            time.sleep(self.poll_interval)
    
    def _make_request(self, params=None):
        """
        Make a request to the Java Kafka service.
        
        Args:
            params: Request parameters
            
        Returns:
            dict: Response data
        """
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        
        try:
            response = requests.get(
                self.endpoint_url,
                params=params,
                headers=headers,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error response from Java Kafka service: {response.status_code} {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Error making request to Java Kafka service: {str(e)}")
            return None
    
    def _default_transform(self, data) -> LLMInteraction:
        """
        Default transformation function for Java Kafka messages.
        
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

class JavaKafkaOptions:
    """
    Helper class to generate Java code for integrating with MERIT.
    
    This class provides static methods to generate Java code snippets
    for integrating an existing Java Kafka consumer with MERIT.
    """
    
    @staticmethod
    def generate_rest_controller():
        """
        Generate a Spring Boot REST controller to interface with MERIT.
        
        Returns:
            str: Java code for a Spring Boot REST controller
        """
        return """
package com.yourcompany.llm.monitoring.merit;

import org.springframework.web.bind.annotation.*;
import org.springframework.beans.factory.annotation.Autowired;
import java.util.*;
import java.time.Instant;

@RestController
@RequestMapping("/api/merit")
public class MeritIntegrationController {

    @Autowired
    private KafkaMessageBuffer messageBuffer;
    
    /**
     * Endpoint for MERIT to retrieve Kafka messages
     */
    @GetMapping("/messages")
    public Map<String, Object> getMessages(
            @RequestParam(value = "max_messages", defaultValue = "100") int maxMessages,
            @RequestParam(value = "since_timestamp", required = false) String sinceTimestamp,
            @RequestParam(value = "test", required = false, defaultValue = "false") boolean isTest) {
        
        Map<String, Object> response = new HashMap<>();
        
        // Simple health check
        if (isTest) {
            response.put("status", "ok");
            response.put("version", "1.0.0");
            return response;
        }
        
        // Get messages from buffer
        Instant since = null;
        if (sinceTimestamp != null && !sinceTimestamp.isEmpty()) {
            try {
                since = Instant.parse(sinceTimestamp);
            } catch (Exception e) {
                response.put("error", "Invalid timestamp format");
                return response;
            }
        }
        
        List<Map<String, Object>> messages = messageBuffer.getMessages(since, maxMessages);
        
        // Set response
        response.put("messages", messages);
        if (!messages.isEmpty()) {
            response.put("latest_timestamp", Instant.now().toString());
            response.put("count", messages.size());
        }
        
        return response;
    }
}
"""
    
    @staticmethod
    def generate_kafka_buffer():
        """
        Generate a Java class for buffering Kafka messages.
        
        Returns:
            str: Java code for a Kafka message buffer
        """
        return """
package com.yourcompany.llm.monitoring.merit;

import org.springframework.stereotype.Component;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.time.Instant;

@Component
public class KafkaMessageBuffer {
    
    private static final Logger logger = LoggerFactory.getLogger(KafkaMessageBuffer.class);
    private static final int MAX_BUFFER_SIZE = 10000;
    
    private final Queue<Map<String, Object>> messageBuffer = new ConcurrentLinkedQueue<>();
    
    /**
     * Add a Kafka message to the buffer for MERIT to consume
     */
    public void addMessage(ConsumerRecord<String, String> record) {
        try {
            // Convert your Kafka record to the MERIT expected format
            Map<String, Object> meritFormat = convertToMeritFormat(record);
            
            // Add to buffer, maintain size limit
            messageBuffer.add(meritFormat);
            while (messageBuffer.size() > MAX_BUFFER_SIZE) {
                messageBuffer.poll();
            }
        } catch (Exception e) {
            logger.error("Error adding message to MERIT buffer", e);
        }
    }
    
    /**
     * Get messages from the buffer, optionally filtered by timestamp
     */
    public List<Map<String, Object>> getMessages(Instant sinceTimestamp, int maxMessages) {
        List<Map<String, Object>> result = new ArrayList<>();
        
        Iterator<Map<String, Object>> iterator = messageBuffer.iterator();
        while (iterator.hasNext() && result.size() < maxMessages) {
            Map<String, Object> message = iterator.next();
            
            // Filter by timestamp if needed
            if (sinceTimestamp != null) {
                String msgTimestamp = (String) message.get("timestamp");
                if (msgTimestamp != null) {
                    try {
                        Instant msgInstant = Instant.parse(msgTimestamp);
                        if (msgInstant.isBefore(sinceTimestamp)) {
                            continue;
                        }
                    } catch (Exception e) {
                        logger.warn("Invalid timestamp format in message", e);
                    }
                }
            }
            
            result.add(message);
            iterator.remove();
        }
        
        return result;
    }
    
    /**
     * Convert from your Kafka record format to MERIT expected format
     */
    private Map<String, Object> convertToMeritFormat(ConsumerRecord<String, String> record) {
        // TODO: Customize this method to convert your Kafka record format
        // to the format expected by MERIT
        
        Map<String, Object> meritFormat = new HashMap<>();
        meritFormat.put("timestamp", Instant.now().toString());
        
        // Example: Parse your JSON payload
        try {
            // Assuming record.value() contains a JSON string
            // Replace this with your actual parsing logic
            Map<String, Object> payload = parseJson(record.value());
            
            // Map fields according to MERIT's expected structure
            meritFormat.put("model", payload.getOrDefault("model_name", "unknown"));
            
            // Request data
            Map<String, Object> request = new HashMap<>();
            request.put("prompt", payload.getOrDefault("input_text", ""));
            meritFormat.put("request", request);
            
            // Response data
            Map<String, Object> response = new HashMap<>();
            response.put("completion", payload.getOrDefault("output_text", ""));
            response.put("latency", payload.getOrDefault("response_time_ms", 0));
            response.put("status", payload.getOrDefault("status", "unknown"));
            meritFormat.put("response", response);
            
            // Token data
            Map<String, Object> tokens = new HashMap<>();
            tokens.put("input_tokens", payload.getOrDefault("input_token_count", 0));
            tokens.put("output_tokens", payload.getOrDefault("output_token_count", 0));
            tokens.put("total_tokens", 
                ((Number)payload.getOrDefault("input_token_count", 0)).intValue() + 
                ((Number)payload.getOrDefault("output_token_count", 0)).intValue());
            meritFormat.put("tokens", tokens);
            
            // Additional metadata
            Map<String, Object> metadata = new HashMap<>();
            metadata.put("user_id", payload.getOrDefault("user_id", "anonymous"));
            metadata.put("session_id", payload.getOrDefault("session_id", ""));
            metadata.put("application", payload.getOrDefault("application_name", ""));
            meritFormat.put("metadata", metadata);
            
        } catch (Exception e) {
            logger.error("Error parsing Kafka message", e);
        }
        
        return meritFormat;
    }
    
    // Helper method to parse JSON (replace with your JSON library)
    private Map<String, Object> parseJson(String json) {
        // Replace with your JSON parsing logic
        // Example using Jackson:
        // return new ObjectMapper().readValue(json, Map.class);
        return new HashMap<>(); // Placeholder
    }
}
"""
    
    @staticmethod
    def generate_kafka_consumer():
        """
        Generate a Java Kafka consumer example that integrates with MERIT.
        
        Returns:
            str: Java code for a Kafka consumer
        """
        return """
package com.yourcompany.llm.monitoring.merit;

import org.springframework.kafka.annotation.KafkaListener;
import org.springframework.stereotype.Service;
import org.springframework.beans.factory.annotation.Autowired;
import org.apache.kafka.clients.consumer.ConsumerRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@Service
public class LlmInteractionConsumer {

    private static final Logger logger = LoggerFactory.getLogger(LlmInteractionConsumer.class);
    
    @Autowired
    private KafkaMessageBuffer meritBuffer;
    
    /**
     * Kafka listener for LLM interaction events
     */
    @KafkaListener(topics = "${llm.kafka.topics.interactions}", groupId = "${llm.kafka.group-id}")
    public void consume(ConsumerRecord<String, String> record) {
        try {
            logger.debug("Received message: {}", record.value());
            
            // Process the message for your internal systems
            // ... your existing logic here ...
            
            // Also buffer it for MERIT
            meritBuffer.addMessage(record);
            
        } catch (Exception e) {
            logger.error("Error processing Kafka message", e);
        }
    }
}
"""

    @staticmethod
    def generate_metrics_ingestion_steps():
        """
        Generate a list of steps to integrate MERIT with an existing Java Kafka setup.
        
        Returns:
            str: Integration steps
        """
        return """
# Integrating Your Java Kafka Consumer with MERIT

Follow these steps to connect your existing Java Kafka infrastructure with MERIT:

## 1. Add the MERIT Integration Components to Your Java Application

1. Add the `MeritIntegrationController` class to expose the REST API
2. Add the `KafkaMessageBuffer` class to buffer messages for MERIT
3. Update your Kafka consumer to add messages to the buffer

## 2. Configure Your Java Application

Add these properties to your application.properties or application.yml:

```yaml
# MERIT Integration Configuration
merit:
  buffer:
    max-size: 10000  # Maximum number of messages to buffer
  security:
    api-key: your-secret-key  # Optional API key for security
```

## 3. Configure MERIT to Connect to Your Java Service

Update your MERIT configuration to use the JavaKafkaAdapter:

```python
from merit.monitoring.collection.java_kafka_adapter import JavaKafkaAdapter

# Create the adapter
adapter = JavaKafkaAdapter(config={
    "endpoint_url": "http://your-java-service:8080/api/merit/messages",
    "api_key": "your-secret-key",  # If you enabled API key security
    "poll_interval": 5.0,  # How often to poll in seconds
    "batch_size": 100     # Max messages per request
})

# Add to your MERIT monitoring setup
from merit.monitoring import MonitoringService

monitoring_service = MonitoringService()
monitoring_service.add_collector(adapter)
monitoring_service.start()
```

## 4. Customize the Message Conversion

Modify the `convertToMeritFormat` method in `KafkaMessageBuffer` to properly convert your specific Kafka message format to the structure expected by MERIT.

## 5. Test the Integration

1. Start your Java application
2. Verify the REST endpoint is accessible: `curl http://your-java-service:8080/api/merit/messages?test=true`
3. Configure MERIT to use the JavaKafkaAdapter
4. Start MERIT and verify it can retrieve messages

## 6. Production Considerations

- Add proper authentication to the REST API
- Consider rate limiting for the API endpoint
- Implement proper error handling and logging
- Consider performance implications and adjust buffer size accordingly
"""
