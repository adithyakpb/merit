"""
Models for MERIT monitoring.

This module defines the data models for the monitoring system, including:
- Base interaction models for request/response pairs
- LLM-specific interaction models with token tracking
- Data structures for monitoring and analysis

These models provide the foundation for representing, storing, and
analyzing monitored interactions across various systems.

Class Hierarchy:
- BaseInteractionComponent: Base class for components of an interaction
  - BaseInteractionRequest: Base class for interaction requests
  - BaseInteractionResponse: Base class for interaction responses
- BaseInteraction: Complete interaction containing a request and response pair
- TokenInfo: Information about token usage in LLM interactions
- LLMRequest: LLM-specific request
- LLMResponse: LLM-specific response
- LLMInteraction: Complete LLM interaction
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field


@dataclass
class BaseInteractionComponent:
    """
    Base class for components of an interaction (request or response).
    
    This is an abstract base class that defines common attributes and methods
    for both requests and responses. It should not be instantiated directly.
    
    Attributes:
        timestamp: When this component was created/sent/received
        metadata: Additional context-specific data as key-value pairs
    """
    # All fields have defaults since this is a base class
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary representation.
        
        Returns:
            Dictionary with component data
        """
        result = {
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.metadata:
            result["metadata"] = self.metadata
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseInteractionComponent':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary with component data
            
        Returns:
            Instance of the component class
        """
        # Handle timestamp conversion
        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except ValueError:
                timestamp = datetime.now()
        elif timestamp is None:
            timestamp = datetime.now()
        
        return cls(
            timestamp=timestamp,
            metadata=data.get("metadata", {})
        )


@dataclass
class BaseInteractionRequest(BaseInteractionComponent):
    """
    Base class for interaction requests.
    
    Represents a request sent to a service, with common request attributes.
    
    Attributes:
        id: Unique identifier for the request
        raw_content: The raw request content
        is_truncated: Whether the content was truncated
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    raw_content: str = ""
    is_truncated: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "id": self.id,
            "raw_content": self.raw_content,
        })
        
        if self.is_truncated:
            result["is_truncated"] = True
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseInteractionRequest':
        """Create from dictionary representation."""
        # Get base fields from parent class
        base_instance = super().from_dict(data)
        
        return cls(
            timestamp=base_instance.timestamp,
            metadata=base_instance.metadata,
            id=data.get("id", str(uuid.uuid4())),
            raw_content=data.get("raw_content", ""),
            is_truncated=data.get("is_truncated", False)
        )
    
    def get_size(self) -> int:
        """
        Get size of the request content in bytes.
        
        Returns:
            Size of raw_content in bytes
        """
        return len(self.raw_content.encode('utf-8'))


@dataclass
class BaseInteractionResponse(BaseInteractionComponent):
    """
    Base class for interaction responses.
    
    Represents a response received from a service, with common response attributes.
    
    Attributes:
        request_id: ID of the corresponding request
        raw_content: The raw response content
        is_truncated: Whether the content was truncated
        status: Response status (success, error, etc.)
        latency: Response time in seconds
    """
    # Required fields first - must come before any inherited default fields
    request_id: str = field(default="")  # Empty string default to fix parameter order
    # Optional fields with defaults
    raw_content: str = ""
    is_truncated: bool = False
    status: str = "success"  # success, error, timeout, etc.
    latency: Optional[float] = None
    
    def __post_init__(self):
        """Validate fields after initialization."""
        # Ensure request_id is provided
        if not self.request_id:
            raise ValueError("request_id is required for BaseInteractionResponse")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "request_id": self.request_id,
            "raw_content": self.raw_content,
            "status": self.status,
        })
        
        if self.is_truncated:
            result["is_truncated"] = True
            
        if self.latency is not None:
            result["latency"] = self.latency
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseInteractionResponse':
        """Create from dictionary representation."""
        # Get base fields from parent class
        base_instance = super().from_dict(data)
        
        # Make sure request_id is not empty
        request_id = data.get("request_id", "")
        if not request_id:
            # Use a placeholder if needed
            request_id = "unknown_request"
        
        return cls(
            timestamp=base_instance.timestamp,
            metadata=base_instance.metadata,
            request_id=request_id,
            raw_content=data.get("raw_content", ""),
            is_truncated=data.get("is_truncated", False),
            status=data.get("status", "success"),
            latency=data.get("latency")
        )
    
    def get_size(self) -> int:
        """
        Get size of the response content in bytes.
        
        Returns:
            Size of raw_content in bytes
        """
        return len(self.raw_content.encode('utf-8'))


@dataclass
class BaseInteraction:
    """
    Base class representing a complete interaction between a system and a service.
    
    This combines a request and response into a complete interaction
    and provides methods for analysis and serialization.
    
    Unlike BaseInteractionComponent, this is not a component itself but rather
    a container for a request-response pair.
    
    Attributes:
        request: The request sent to the service
        response: The response received from the service
    """
    request: BaseInteractionRequest
    response: BaseInteractionResponse
    
    def __post_init__(self):
        """Validate and link request and response."""
        # Ensure response is linked to this request
        if self.response.request_id != self.request.id:
            self.response.request_id = self.request.id
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert complete interaction to dictionary.
        
        Returns:
            Dictionary representation of the interaction
        """
        return {
            "request": self.request.to_dict(),
            "response": self.response.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseInteraction':
        """
        Create from dictionary representation.
        
        Args:
            data: Dictionary with interaction data
            
        Returns:
            Instance of BaseInteraction
        """
        request_data = data.get("request", {})
        response_data = data.get("response", {})
        
        request = BaseInteractionRequest.from_dict(request_data)
        response = BaseInteractionResponse.from_dict(response_data)
        
        return cls(request=request, response=response)
    
    def get_available_metrics(self) -> List[str]:
        """
        Return list of metrics that can be calculated from this interaction.
        
        This helps determine what analytics are possible with the available data.
        
        Returns:
            List of metric names that can be calculated
        """
        # Common metrics for all interaction types
        metrics = ["request_count"]
        
        if self.response.latency is not None:
            metrics.append("latency")
            
        return metrics

# LLM-specific models

@dataclass
class TokenInfo:
    """
    Information about token usage in an LLM interaction.
    
    This class tracks token usage metrics for LLM interactions,
    which is essential for both performance analysis and cost tracking.
    
    Attributes:
        input_tokens: Number of tokens in the input/prompt
        output_tokens: Number of tokens in the output/completion
        total_tokens: Total tokens used in the interaction
        is_estimated: Whether the token counts are estimated or actual
    """
    input_tokens: int
    output_tokens: int
    total_tokens: int = field(default=0)
    is_estimated: bool = False
    
    def __post_init__(self):
        """Calculate total tokens if not provided."""
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "is_estimated": self.is_estimated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TokenInfo':
        """Create from dictionary representation."""
        if not data:
            return None
            
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            is_estimated=data.get("is_estimated", False)
        )


@dataclass
class LLMRequest(BaseInteractionRequest):
    """
    LLM-specific request.
    
    Extends the base request with fields specific to LLM interactions.
    
    Attributes:
        input: The Input object containing user_input, prompt_prefix, prompt_suffix
        model: The requested model name
        temperature: Sampling temperature parameter
        max_tokens: Maximum tokens to generate
        stop_sequences: Sequences that will stop generation
        top_p: Nucleus sampling parameter
        frequency_penalty: Penalty for token frequency
        presence_penalty: Penalty for token presence
    """
    from ..core.models import Input
    
    input: Input = field(default_factory=lambda: Input(user_input=""))
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stop_sequences: List[str] = field(default_factory=list)
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    
    @property
    def user_input(self) -> str:
        """Get the raw user input"""
        return self.input.user_input
    
    def __post_init__(self):
        """Ensure raw_content is set if empty but input content is provided."""
        if not self.raw_content and self.input.content:
            self.raw_content = self.input.content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "input": self.input.to_dict(),
        })
        
        # Add optional fields if present
        if self.model:
            result["model"] = self.model
            
        if self.temperature is not None:
            result["temperature"] = self.temperature
            
        if self.max_tokens is not None:
            result["max_tokens"] = self.max_tokens
            
        if self.stop_sequences:
            result["stop_sequences"] = self.stop_sequences
            
        if self.top_p is not None:
            result["top_p"] = self.top_p
            
        if self.frequency_penalty is not None:
            result["frequency_penalty"] = self.frequency_penalty
            
        if self.presence_penalty is not None:
            result["presence_penalty"] = self.presence_penalty
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMRequest':
        """Create from dictionary representation."""
        from ..core.models import Input
        
        # Get base fields from parent class
        base_instance = super().from_dict(data)
        
        # Handle input data
        input_data = data.get("input", {})
        if isinstance(input_data, dict):
            input_obj = Input.from_dict(input_data)
        else:
            input_obj = Input(user_input=str(input_data))
            #check if this is correct
        return cls(
            # Base fields
            id=base_instance.id,
            timestamp=base_instance.timestamp,
            metadata=base_instance.metadata,
            raw_content=base_instance.raw_content,
            is_truncated=base_instance.is_truncated,
            
            # LLM-specific fields
            input=input_obj,
            model=data.get("model"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            stop_sequences=data.get("stop_sequences", []),
            top_p=data.get("top_p"),
            frequency_penalty=data.get("frequency_penalty"),
            presence_penalty=data.get("presence_penalty")
        )
    
    def estimate_tokens(self) -> int:
        """
        Estimate number of tokens in the prompt.
        
        This is a simplified estimation. For accurate counts,
        use a proper tokenizer for the specific model.
        
        Returns:
            Estimated token count
        """
        # Very simple approximation - words plus some overhead
        #TODO use the tokeniser  
        return len(self.input.content.split()) + 10


@dataclass
class LLMResponse(BaseInteractionResponse):
    """
    LLM-specific response.
    
    Extends the base response with fields specific to LLM responses.
    
    Attributes:
        request_id: ID of the corresponding request
        completion: The generated text/completion
        model: The actual model used (may differ from request)
        tokens: Token usage information
        finish_reason: Why the model stopped generating (length, stop, etc.)
    """
    # LLM-specific fields - all with defaults
    completion: str = ""
    model: Optional[str] = None
    tokens: Optional[TokenInfo] = None
    finish_reason: Optional[str] = None
    
    def __post_init__(self):
        """Ensure raw_content is set if empty but completion is provided."""
        super().__post_init__()  # Call the parent's __post_init__ first
        if not self.raw_content and self.completion:
            self.raw_content = self.completion
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = super().to_dict()
        result.update({
            "completion": self.completion,
        })
        
        # Add optional fields if present
        if self.model:
            result["model"] = self.model
            
        if self.tokens:
            result["tokens"] = self.tokens.to_dict()
            
        if self.finish_reason:
            result["finish_reason"] = self.finish_reason
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMResponse':
        """Create from dictionary representation."""
        # Get base fields from parent class
        base_instance = super().from_dict(data)
        
        # Parse token info if present
        tokens_data = data.get("tokens")
        tokens = TokenInfo.from_dict(tokens_data) if tokens_data else None
        
        return cls(
            # Base fields
            request_id=base_instance.request_id,
            timestamp=base_instance.timestamp,
            metadata=base_instance.metadata,
            raw_content=base_instance.raw_content,
            is_truncated=base_instance.is_truncated,
            status=base_instance.status,
            latency=base_instance.latency,
            
            # LLM-specific fields
            completion=data.get("completion", ""),
            model=data.get("model"),
            tokens=tokens,
            finish_reason=data.get("finish_reason")
        )
    
    def estimate_tokens(self) -> int:
        """
        Estimate number of tokens in the completion.
        
        This is a simplified estimation. For accurate counts,
        use a proper tokenizer for the specific model.
        
        Returns:
            Estimated token count
        """
        # Very simple approximation - words plus some overhead
        return len(self.completion.split()) + 5


@dataclass
class LLMInteraction(BaseInteraction):
    """
    LLM-specific interaction combining request and response.
    
    This class represents a complete interaction with a Large Language Model,
    providing additional methods specific to LLM analysis.
    
    Attributes:
        request: LLM-specific request
        response: LLM-specific response
    """
    request: LLMRequest
    response: LLMResponse
    
    def __init__(self, 
                 request: Optional[LLMRequest] = None, 
                 response: Optional[LLMResponse] = None,
                 prompt: Optional[str] = None,
                 completion: Optional[str] = None,
                 model: Optional[str] = None,
                 tokens: Optional[TokenInfo] = None,
                 latency: Optional[float] = None):
        """
        Initialize an LLM interaction with flexible parameters.
        
        This constructor supports both:
        1. Regular initialization with request and response objects
        2. Simplified initialization with basic fields
        
        Args:
            request: LLMRequest object
            response: LLMResponse object
            prompt: Direct prompt text (alternative to request)
            completion: Direct completion text (alternative to response)
            model: Model name to use if creating request/response
            tokens: Token info to use if creating response
            latency: Latency to use if creating response
        """
        import uuid
        
        # If request and response are provided, use them directly
        if request is not None and response is not None:
            self.request = request
            self.response = response
            return
            
        # Otherwise, we're using the simplified constructor
        if prompt is None:
            raise ValueError("Either request/response pair or prompt must be provided")
            
        # Generate an ID to link request and response
        request_id = str(uuid.uuid4())
        
        # Create request with Input object
        from ..core.models import Input
        input_obj = Input(user_input=prompt)
        self.request = LLMRequest(
            id=request_id,
            input=input_obj,
            model=model
        )
        
        # Create response (default to empty completion if not provided)
        self.response = LLMResponse(
            request_id=request_id,
            completion=completion or "",
            model=model,
            tokens=tokens,
            latency=latency
        )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LLMInteraction':
        """
        Create from dictionary representation.
        
        Override the base class method to ensure we use LLM-specific classes.
        
        Args:
            data: Dictionary with interaction data
            
        Returns:
            Instance of LLMInteraction
        """
        request_data = data.get("request", {})
        response_data = data.get("response", {})
        
        # Use LLM-specific request and response classes
        request = LLMRequest.from_dict(request_data)
        response = LLMResponse.from_dict(response_data)
        
        return cls(request=request, response=response)
    
    def get_total_tokens(self) -> Optional[int]:
        """
        Get total tokens used in the interaction.
        
        Returns:
            Total token count or None if not available
        """
        if self.response.tokens:
            return self.response.tokens.total_tokens
        return None
    
    def estimate_cost(self, pricing_info: Optional[Dict[str, Any]] = None) -> Optional[float]:
        """
        Estimate cost of the interaction based on model and tokens.
        
        Args:
            pricing_info: Dictionary with pricing information, or None to use defaults
            
        Returns:
            Estimated cost in USD or None if cannot be calculated
        """
        if not self.response.tokens:
            return None
            
        model = self.response.model or self.request.model
        if not model:
            return None
            
        # If no pricing info provided, use some reasonable defaults
        if not pricing_info:
            pricing_info = self._get_default_pricing(model)
            
        if not pricing_info:
            return None
            
        # Calculate cost based on token usage
        input_cost = self.response.tokens.input_tokens * pricing_info.get("input_price_per_token", 0)
        output_cost = self.response.tokens.output_tokens * pricing_info.get("output_price_per_token", 0)
        
        return input_cost + output_cost
    
    def _get_default_pricing(self, model: str) -> Dict[str, float]:
        """
        Get default pricing for common models.
        
        This is a very simplified pricing table and should be replaced
        with more accurate and up-to-date information in production.
        
        Args:
            model: Model name
            
        Returns:
            Dictionary with pricing information or None if unknown
        """
        model_lower = model.lower()
        
        # Very simplified pricing based on rough approximations
        # In a real implementation, this would come from a configuration or API
        if "gpt-4" in model_lower:
            if "8k" in model_lower:
                return {
                    "input_price_per_token": 0.00003,
                    "output_price_per_token": 0.00006
                }
            else:  # Default to 32k or others
                return {
                    "input_price_per_token": 0.00006,
                    "output_price_per_token": 0.00012
                }
        elif "gpt-3.5" in model_lower:
            return {
                "input_price_per_token": 0.0000015,
                "output_price_per_token": 0.000002
            }
        elif "claude" in model_lower:
            if "opus" in model_lower:
                return {
                    "input_price_per_token": 0.00005,
                    "output_price_per_token": 0.00015
                }
            elif "sonnet" in model_lower:
                return {
                    "input_price_per_token": 0.00001,
                    "output_price_per_token": 0.00003
                }
            else:  # haiku or other
                return {
                    "input_price_per_token": 0.000002,
                    "output_price_per_token": 0.000001
                }
        
        # Unknown model
        return None
    
    def get_available_metrics(self) -> List[str]:
        """
        Return list of metrics that can be calculated from this interaction.
        
        This helps determine what analytics are possible with the available data.
        
        Returns:
            List of metric names that can be calculated
        """
        metrics = super().get_available_metrics()
        
        # Add LLM-specific metrics
        if self.request.model or self.response.model:
            metrics.append("model_usage")
            
        if self.response.tokens:
            metrics.extend(["token_usage", "token_efficiency"])
            
            # Cost metrics require both tokens and model
            if self.request.model or self.response.model:
                metrics.append("cost")
                
        return metrics
