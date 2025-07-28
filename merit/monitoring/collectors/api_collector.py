"""
API proxy collectors for MERIT monitoring.

This module provides collectors that can capture LLM interaction data by
proxying API calls to LLM providers like OpenAI, Anthropic, etc.
"""

import json
import time
import threading
import queue
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable, Set, Union
import re

from .collector import BaseDataCollector, CollectionResult, CollectionStatus
from ..models import LLMRequest, LLMResponse, LLMInteraction, TokenInfo


class APIProxyCollector(BaseDataCollector):
    """
    Collector that captures LLM interaction data by proxying API calls.
    
    This collector can be used in two ways:
    1. As a hook that other API client libraries can call
    2. As a middleware in web frameworks like Flask or FastAPI
    
    It records request and response data for LLM API calls, along with
    timing information and other metadata.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the API proxy collector.
        
        Args:
            config: Configuration dictionary with the following options:
                - buffer_size: Max number of interactions to store in memory (default: 1000)
                - include_model_patterns: List of model name patterns to include
                - exclude_model_patterns: List of model name patterns to exclude
                - enabled_providers: List of provider names to monitor (default: all)
        """
        super().__init__(config)
        self.buffer_size = config.get("buffer_size", 1000)
        self.include_model_patterns = config.get("include_model_patterns", [])
        self.exclude_model_patterns = config.get("exclude_model_patterns", [])
        self.enabled_providers = config.get("enabled_providers", ["openai", "anthropic", "google", "azure", "cohere"])
        
        # Compile model patterns
        self._include_patterns = [re.compile(pattern) for pattern in self.include_model_patterns]
        self._exclude_patterns = [re.compile(pattern) for pattern in self.exclude_model_patterns]
        
        # Internal state
        self._interaction_queue = queue.Queue(maxsize=self.buffer_size)
        self._active_requests: Dict[str, Dict[str, Any]] = {}  # request_id -> request_data
        self._processing_thread: Optional[threading.Thread] = None
        self._request_count = 0
        self._response_count = 0
    
    def start(self) -> None:
        """Start collecting API data in the background."""
        super().start()
        
        # Start background thread for processing interactions
        if not self._processing_thread or not self._processing_thread.is_alive():
            self._processing_thread = threading.Thread(
                target=self._process_interactions,
                daemon=True
            )
            self._processing_thread.start()
    
    def stop(self) -> None:
        """Stop collecting API data and clean up resources."""
        super().stop()
        
        # Stop processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            # Add None to the queue to signal thread to exit
            try:
                self._interaction_queue.put(None, block=False)
            except queue.Full:
                pass
                
            # Wait for thread to finish
            self._processing_thread.join(timeout=5.0)
    
    def collect(self) -> CollectionResult:
        """
        Collect all available interactions from the buffer.
        
        Returns:
            CollectionResult with the data collected
        """
        result = CollectionResult(
            status=CollectionStatus.SUCCESS,
            start_time=datetime.now()
        )
        
        interactions = []
        try:
            # Get all items from the queue without waiting
            while not self._interaction_queue.empty() and result.items_processed < self.buffer_size:
                try:
                    item = self._interaction_queue.get_nowait()
                    if item is not None:  # None is a signal to stop
                        interactions.append(item)
                        result.items_processed += 1
                        result.items_collected += 1
                        self._interaction_queue.task_done()
                except queue.Empty:
                    break
            
            # Add interactions to result
            result.data = interactions
            
            # Determine final status
            if result.items_processed == 0:
                result.status = CollectionStatus.SUCCESS  # No data is still success
        
        except Exception as e:
            result.status = CollectionStatus.FAILURE
            result.error = str(e)
        
        finally:
            result.end_time = datetime.now()
            return result
    
    def record_request(self, provider: str, endpoint: str, 
                       request_data: Dict[str, Any], 
                       request_id: Optional[str] = None) -> str:
        """
        Record an API request.
        
        This should be called by API client libraries before sending
        a request to the LLM provider.
        
        Args:
            provider: Name of the LLM provider (e.g., "openai", "anthropic")
            endpoint: API endpoint (e.g., "/v1/chat/completions")
            request_data: Complete request data including parameters
            request_id: Optional ID for the request (will be generated if not provided)
            
        Returns:
            The request ID (useful for linking to the response)
        """
        if not self.is_running:
            return request_id or ""
            
        if provider.lower() not in [p.lower() for p in self.enabled_providers]:
            return request_id or ""
        
        try:
            # Generate request ID if not provided
            if not request_id:
                import uuid
                request_id = str(uuid.uuid4())
                
            # Create request object with metadata
            req_obj = {
                "id": request_id,
                "provider": provider,
                "endpoint": endpoint,
                "timestamp": datetime.now().isoformat(),
                "data": request_data
            }
            
            # Extract model info for filtering
            model = self._extract_model_from_request(provider, request_data)
            if model:
                req_obj["model"] = model
                
                # Skip if model doesn't match filters
                if not self._should_include_model(model):
                    return request_id
            
            # Store in active requests map
            self._active_requests[request_id] = req_obj
            self._request_count += 1
            
            return request_id
            
        except Exception as e:
            # Log error but don't fail
            print(f"Error recording API request: {str(e)}")
            return request_id or ""
    
    def record_response(self, request_id: str, response_data: Dict[str, Any], 
                        status_code: int = 200, error: Optional[str] = None) -> None:
        """
        Record an API response.
        
        This should be called by API client libraries after receiving
        a response from the LLM provider.
        
        Args:
            request_id: ID of the corresponding request
            response_data: Complete response data
            status_code: HTTP status code
            error: Error message if the request failed
        """
        if not self.is_running:
            return
            
        if not request_id or request_id not in self._active_requests:
            return
            
        try:
            # Get the request data
            request_obj = self._active_requests.pop(request_id)
            
            # Create response object
            resp_obj = {
                "request_id": request_id,
                "timestamp": datetime.now().isoformat(),
                "status_code": status_code,
                "data": response_data
            }
            
            if error:
                resp_obj["error"] = error
            
            # Calculate latency
            try:
                req_time = datetime.fromisoformat(request_obj["timestamp"])
                resp_time = datetime.fromisoformat(resp_obj["timestamp"])
                latency = (resp_time - req_time).total_seconds()
                resp_obj["latency"] = latency
            except (ValueError, KeyError):
                # If timestamp is missing or invalid, skip latency calculation
                pass
            
            # Create complete interaction
            interaction = {
                "request": request_obj,
                "response": resp_obj
            }
            
            # Add to queue for processing
            try:
                self._interaction_queue.put(interaction, block=False)
                self._response_count += 1
            except queue.Full:
                # Queue is full, discard oldest item and try again
                try:
                    self._interaction_queue.get_nowait()
                    self._interaction_queue.put(interaction, block=False)
                except (queue.Empty, queue.Full):
                    pass
        
        except Exception as e:
            # Log error but don't fail
            print(f"Error recording API response: {str(e)}")
    
    def _process_interactions(self) -> None:
        """
        Background thread method for processing interactions.
        
        This continuously processes interactions from the queue and
        notifies callbacks.
        """
        while self.is_running:
            try:
                # Get the next interaction from the queue
                interaction = self._interaction_queue.get(block=True, timeout=1.0)
                
                # None is the signal to stop
                if interaction is None:
                    break
                
                # Process the interaction
                self._process_interaction(interaction)
                
                # Mark as done
                self._interaction_queue.task_done()
                
            except queue.Empty:
                # Timeout, check if we're still running
                continue
                
            except Exception as e:
                # Log error but continue
                print(f"Error processing API interaction: {str(e)}")
    
    def _process_interaction(self, interaction: Dict[str, Any]) -> None:
        """
        Process a single interaction.
        
        This extracts relevant data, converts to LLM models if possible,
        and notifies callbacks.
        
        Args:
            interaction: Raw interaction data with request and response
        """
        try:
            # Notify raw data callbacks
            self._notify_callbacks(interaction)
            
            # Convert to LLM interaction if possible
            llm_interaction = self._create_interaction(interaction)
            if llm_interaction:
                self._notify_callbacks(llm_interaction)
                
        except Exception as e:
            # Log error but don't fail
            print(f"Error processing interaction: {str(e)}")
    
    def _create_interaction(self, data: Dict[str, Any]) -> Optional[LLMInteraction]:
        """
        Convert raw API data to an LLMInteraction object.
        
        Args:
            data: Raw interaction data with request and response
            
        Returns:
            LLMInteraction object or None if conversion not possible
        """
        try:
            request_obj = data.get("request", {})
            response_obj = data.get("response", {})
            
            if not request_obj or not response_obj:
                return None
                
            # Get provider and endpoint
            provider = request_obj.get("provider", "")
            endpoint = request_obj.get("endpoint", "")
            
            # Get request and response data
            request_data = request_obj.get("data", {})
            response_data = response_obj.get("data", {})
            
            # Extract information based on provider and endpoint
            prompt, completion, token_info = self._extract_provider_specific_data(
                provider, endpoint, request_data, response_data
            )
            
            if not prompt:  # No valid LLM data found
                return None
            
            # Create request object
            request_id = request_obj.get("id", "")
            model = request_obj.get("model") or self._extract_model_from_request(provider, request_data)
            temperature = request_data.get("temperature")
            max_tokens = request_data.get("max_tokens")
            
            request = LLMRequest(
                id=request_id,
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Create response object
            status = "success" if response_obj.get("status_code", 200) < 400 else "error"
            latency = response_obj.get("latency")
            
            response = LLMResponse(
                request_id=request_id,
                completion=completion,
                model=model,
                tokens=token_info,
                status=status,
                latency=latency
            )
            
            # Create complete interaction
            return LLMInteraction(request=request, response=response)
        
        except Exception as e:
            # Log error but don't fail
            print(f"Error creating LLM interaction: {str(e)}")
            return None
    
    def _extract_provider_specific_data(self, provider: str, endpoint: str,
                                        request_data: Dict[str, Any],
                                        response_data: Dict[str, Any]) -> tuple[str, str, Optional[TokenInfo]]:
        """
        Extract provider-specific data from request and response.
        
        Different LLM providers use different request/response formats, so we need
        provider-specific logic to extract the relevant data.
        
        Args:
            provider: LLM provider name
            endpoint: API endpoint
            request_data: Request data
            response_data: Response data
            
        Returns:
            Tuple of (prompt, completion, token_info)
        """
        prompt = ""
        completion = ""
        token_info = None
        
        # Default token counts
        input_tokens = 0
        output_tokens = 0
        total_tokens = 0
        
        # Handle different providers
        provider = provider.lower()
        
        if provider == "openai":
            # OpenAI API
            if "/chat/completions" in endpoint:
                # Chat completions API
                messages = request_data.get("messages", [])
                prompt = json.dumps(messages)
                
                choices = response_data.get("choices", [])
                if choices and len(choices) > 0:
                    message = choices[0].get("message", {})
                    completion = message.get("content", "")
                
            elif "/completions" in endpoint:
                # Completions API
                prompt = request_data.get("prompt", "")
                
                choices = response_data.get("choices", [])
                if choices and len(choices) > 0:
                    completion = choices[0].get("text", "")
            
            # Token usage
            usage = response_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        
        elif provider == "anthropic":
            # Anthropic API
            if "/messages" in endpoint:
                # Messages API
                messages = request_data.get("messages", [])
                prompt = json.dumps(messages)
                system = request_data.get("system", "")
                if system:
                    prompt = f"System: {system}\n{prompt}"
                
                content = response_data.get("content", [])
                if content:
                    text_blocks = [block.get("text", "") for block in content 
                                   if block.get("type") == "text"]
                    completion = "\n".join(text_blocks)
            
            elif "/complete" in endpoint:
                # Complete API (older)
                prompt = request_data.get("prompt", "")
                completion = response_data.get("completion", "")
            
            # Token usage
            usage = response_data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        
        elif provider == "google":
            # Google (Vertex AI) API
            prompt = request_data.get("prompt", "")
            if not prompt and "contents" in request_data:
                contents = request_data.get("contents", [])
                prompt = json.dumps(contents)
            
            candidates = response_data.get("candidates", [])
            if candidates and len(candidates) > 0:
                content = candidates[0].get("content", {})
                parts = content.get("parts", [])
                texts = [part.get("text", "") for part in parts]
                completion = "\n".join(filter(None, texts))
            
            # Token usage - not always provided by Google
            usage = response_data.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            token_counts = response_data.get("tokenCounts", {})
            if token_counts:
                input_tokens = token_counts.get("inputTokenCount", input_tokens)
                output_tokens = token_counts.get("outputTokenCount", output_tokens)
        
        elif provider == "cohere":
            # Cohere API
            prompt = request_data.get("prompt", "")
            if not prompt and "message" in request_data:
                prompt = request_data.get("message", "")
            
            completion = response_data.get("text", "")
            if not completion and "generations" in response_data:
                generations = response_data.get("generations", [])
                if generations and len(generations) > 0:
                    completion = generations[0].get("text", "")
            
            # Token usage
            token_count = response_data.get("token_count", {})
            if token_count:
                input_tokens = token_count.get("prompt_tokens", 0)
                output_tokens = token_count.get("completion_tokens", 0)
                total_tokens = token_count.get("total_tokens", 0)
            
            meta = response_data.get("meta", {})
            if meta and "billing" in meta:
                billing = meta.get("billing", {})
                input_tokens = billing.get("input_tokens", input_tokens)
                output_tokens = billing.get("output_tokens", output_tokens)
        
        # Create token info if we have any token counts
        if input_tokens > 0 or output_tokens > 0:
            if total_tokens == 0:
                total_tokens = input_tokens + output_tokens
                
            token_info = TokenInfo(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens
            )
        
        return prompt, completion, token_info
    
    def _extract_model_from_request(self, provider: str, request_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract model name from request data.
        
        Args:
            provider: LLM provider name
            request_data: Request data
            
        Returns:
            Model name or None if not found
        """
        # Direct model field (most providers)
        if "model" in request_data:
            return request_data["model"]
        
        # Google (Vertex AI) specific
        if provider.lower() == "google":
            if "model" in request_data:
                return request_data["model"]
            # Check for Vertex AI model field
            model_name = request_data.get("modelId", "")
            if model_name:
                return model_name
        
        return None
    
    def _should_include_model(self, model: str) -> bool:
        """
        Check if a model should be included based on patterns.
        
        Args:
            model: Model name to check
            
        Returns:
            Whether the model should be included
        """
        if not model:
            return True  # Include by default if no model info
            
        # If include patterns specified, model must match at least one
        if self._include_patterns:
            if not any(pattern.search(model) for pattern in self._include_patterns):
                return False
        
        # If exclude patterns specified, model must not match any
        if self._exclude_patterns:
            if any(pattern.search(model) for pattern in self._exclude_patterns):
                return False
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the collector.
        
        Returns:
            Dictionary with status information
        """
        status = super().get_status()
        status.update({
            "request_count": self._request_count,
            "response_count": self._response_count,
            "pending_requests": len(self._active_requests),
            "queued_interactions": self._interaction_queue.qsize(),
            "enabled_providers": self.enabled_providers
        })
        return status


class OpenAIProxyCollector(APIProxyCollector):
    """
    Specialized collector for monitoring OpenAI API calls.
    
    This can be installed as a middleware in OpenAI client libraries
    to automatically capture all API calls without manual instrumentation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OpenAI proxy collector.
        
        Args:
            config: Configuration dictionary with collector options
        """
        # Override config with OpenAI-specific defaults
        config = config or {}
        config.setdefault("enabled_providers", ["openai"])
        super().__init__(config)
    
    def install_hooks(self) -> None:
        """
        Install hooks in the OpenAI client library.
        
        This method patches the OpenAI library to capture all API calls.
        It should be called once during application startup.
        """
        try:
            # Import OpenAI
            import openai
            import types
            
            # Get the current client or create a new one
            try:
                # OpenAI v1 API
                client = openai.OpenAI()
                
                # Save original methods
                original_post = client._client.post
                original_get = client._client.get
                
                # Create patched methods
                def patched_post(self, path, *args, **kwargs):
                    request_id = self._collector.record_request("openai", path, kwargs)
                    try:
                        response = original_post(path, *args, **kwargs)
                        self._collector.record_response(request_id, response, 200)
                        return response
                    except Exception as e:
                        self._collector.record_response(request_id, {}, 500, str(e))
                        raise
                
                def patched_get(self, path, *args, **kwargs):
                    request_id = self._collector.record_request("openai", path, kwargs)
                    try:
                        response = original_get(path, *args, **kwargs)
                        self._collector.record_response(request_id, response, 200)
                        return response
                    except Exception as e:
                        self._collector.record_response(request_id, {}, 500, str(e))
                        raise
                
                # Store collector reference
                client._client._collector = self
                
                # Patch methods
                client._client.post = types.MethodType(patched_post, client._client)
                client._client.get = types.MethodType(patched_get, client._client)
                
                print("Successfully installed hooks for OpenAI v1 API")
                
            except (AttributeError, ImportError):
                # Try OpenAI v0 API (legacy)
                original_post = openai.api_requestor.APIRequestor.request
                
                def patched_request(self, method, url, *args, **kwargs):
                    request_id = self._collector.record_request("openai", url, kwargs)
                    try:
                        result = original_post(self, method, url, *args, **kwargs)
                        self._collector.record_response(request_id, result[0], result[1])
                        return result
                    except Exception as e:
                        self._collector.record_response(request_id, {}, 500, str(e))
                        raise
                
                # Store collector reference
                openai.api_requestor.APIRequestor._collector = self
                
                # Patch method
                openai.api_requestor.APIRequestor.request = patched_request
                
                print("Successfully installed hooks for OpenAI v0 API (legacy)")
        
        except ImportError:
            raise ImportError(
                "OpenAI is not installed. Please install it with `pip install merit[openai]`"
            )
        except Exception as e:
            print(f"Error installing OpenAI hooks: {str(e)}")


class AnthropicProxyCollector(APIProxyCollector):
    """
    Specialized collector for monitoring Anthropic API calls.
    
    This can be installed as a middleware in Anthropic client libraries
    to automatically capture all API calls without manual instrumentation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Anthropic proxy collector.
        
        Args:
            config: Configuration dictionary with collector options
        """
        # Override config with Anthropic-specific defaults
        config = config or {}
        config.setdefault("enabled_providers", ["anthropic"])
        super().__init__(config)
    
    def install_hooks(self) -> None:
        """
        Install hooks in the Anthropic client library.
        
        This method patches the Anthropic library to capture all API calls.
        It should be called once during application startup.
        """
        try:
            # Import Anthropic
            import anthropic
            import types
            
            # Get the client
            client_cls = anthropic.Anthropic
            original_post = client_cls._client.post
            
            # Create patched method
            def patched_post(self, path, *args, **kwargs):
                request_id = self._collector.record_request("anthropic", path, kwargs)
                try:
                    response = original_post(path, *args, **kwargs)
                    self._collector.record_response(request_id, response, 200)
                    return response
                except Exception as e:
                    self._collector.record_response(request_id, {}, 500, str(e))
                    raise
            
            # Store collector reference and patch method
            def patch_client(client):
                client._client._collector = self
                client._client.post = types.MethodType(patched_post, client._client)
            
            # Patch the constructor to hook new clients
            original_init = client_cls.__init__
            
            def patched_init(self, *args, **kwargs):
                original_init(self, *args, **kwargs)
                patch_client(self)
            
            client_cls.__init__ = patched_init
            
            # Patch any existing client instances
            import gc
            for obj in gc.get_objects():
                if isinstance(obj, client_cls):
                    patch_client(obj)
            
            print("Successfully installed hooks for Anthropic API")
            
        except ImportError:
            print("Anthropic library not found, could not install hooks")
        except Exception as e:
            print(f"Error installing Anthropic hooks: {str(e)}")
