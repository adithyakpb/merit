# Creating Custom API Clients

This guide explains how to create custom API clients in MERIT. Custom API clients allow you to connect to specific AI APIs or services that aren't directly supported by MERIT's built-in clients.

## Why Create Custom API Clients?

You might want to create a custom API client for several reasons:

- Connect to a proprietary or internal API
- Use a new or unsupported AI service provider
- Add specialized functionality for your specific use case
- Implement custom authentication or request handling
- Optimize performance for your specific requirements

## Extending BaseAPIClient

The simplest way to create a custom API client is to extend the `BaseAPIClient` class. This is appropriate when you need to implement the core functionality from scratch.

```python
from merit.api.base import BaseAPIClient

class CustomAPIClient(BaseAPIClient):
    """A custom API client for a specific service."""
    
    def __init__(self, custom_param=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
    
    def get_embeddings(self, texts):
        """
        Get embeddings for the given texts.
        
        Args:
            texts: A string or list of strings to get embeddings for.
            
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        # Implement your custom embedding logic here
        # Make sure to return a list of embeddings (list of lists of floats)
        pass
    
    def generate_text(self, prompt, **kwargs):
        """
        Generate text from a prompt.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            str: The generated text.
        """
        # Implement your custom text generation logic here
        # Make sure to return a string
        pass
    
    def custom_method(self, data):
        """
        A custom method specific to this API client.
        
        Args:
            data: The data to process.
            
        Returns:
            Any: The result of the custom method.
        """
        # Implement your custom method here
        pass
```

## Extending AIAPIClient

If you're creating a client for an AI API that follows common patterns, it's often easier to extend the `AIAPIClient` class, which provides a generic implementation for AI APIs.

```python
from merit.api.client import AIAPIClient

class CustomAIAPIClient(AIAPIClient):
    """A custom client for a specific AI API."""
    
    def __init__(self, model=None, **kwargs):
        super().__init__(**kwargs)
        self.model = model or "default-model"
    
    def _get_headers(self):
        """
        Get the headers for API requests.
        
        Returns:
            dict: The headers for API requests.
        """
        headers = super()._get_headers()
        headers["X-Custom-Header"] = "custom-value"
        return headers
    
    def _process_text_generation_response(self, response):
        """
        Process the response from the text generation API.
        
        Args:
            response: The response from the API.
            
        Returns:
            str: The generated text.
        """
        # Extract the generated text from the API response
        # This will vary depending on the API's response format
        data = response.json()
        return data.get("generated_text", "")
    
    def _process_embeddings_response(self, response, texts):
        """
        Process the response from the embeddings API.
        
        Args:
            response: The response from the API.
            texts: The original input texts.
            
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        # Extract the embeddings from the API response
        # This will vary depending on the API's response format
        data = response.json()
        embeddings = data.get("embeddings", [])
        
        # Ensure the response format matches the expected format
        if not embeddings:
            return [[] for _ in texts]
        
        return embeddings
```

## Example: Anthropic API Client

Here's an example of a custom API client for the Anthropic API:

```python
from merit.api.client import AIAPIClient
import requests

class AnthropicClient(AIAPIClient):
    """Client for the Anthropic API."""
    
    def __init__(
        self,
        api_key=None,
        model="claude-2",
        base_url="https://api.anthropic.com",
        **kwargs
    ):
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )
        self.model = model
    
    def _get_headers(self):
        """Get headers for Anthropic API requests."""
        headers = {
            "Content-Type": "application/json",
            "X-API-Key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        return headers
    
    def generate_text(self, prompt, **kwargs):
        """
        Generate text using the Anthropic API.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            str: The generated text.
        """
        # Prepare the request
        url = f"{self.base_url}/v1/complete"
        
        # Set default parameters
        params = {
            "model": self.model,
            "prompt": f"\n\nHuman: {prompt}\n\nAssistant:",
            "max_tokens_to_sample": kwargs.get("max_tokens", 1000),
            "temperature": kwargs.get("temperature", 0.7),
            "stop_sequences": ["\n\nHuman:"]
        }
        
        # Update with any additional parameters
        for key, value in kwargs.items():
            if key not in ["max_tokens", "temperature"]:
                params[key] = value
        
        # Make the request
        try:
            response = requests.post(url, headers=self._get_headers(), json=params)
            response.raise_for_status()
            data = response.json()
            return data.get("completion", "")
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            return ""
    
    def get_embeddings(self, texts):
        """
        Get embeddings using the Anthropic API.
        
        Note: As of my knowledge cutoff, Anthropic doesn't provide a public embeddings API.
        This is a placeholder implementation.
        
        Args:
            texts: A string or list of strings to get embeddings for.
            
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Return empty embeddings since Anthropic doesn't provide embeddings
        return [[] for _ in texts]
```

## Best Practices for Custom API Clients

When creating custom API clients, follow these best practices:

### 1. Follow the Interface

Ensure your custom client follows the same interface as the base classes:

- Implement `get_embeddings()` and `generate_text()` methods
- Return data in the expected formats (see [Response Formats](./response_formats.md))
- Handle errors gracefully

### 2. Use Decorators

MERIT provides decorators to ensure consistent response formats:

```python
from merit.api.client import validate_embeddings_response, validate_text_response

class CustomAPIClient(AIAPIClient):
    @validate_embeddings_response
    def get_embeddings(self, texts):
        # Implementation...
        
    @validate_text_response
    def generate_text(self, prompt, **kwargs):
        # Implementation...
```

### 3. Handle Authentication

Implement appropriate authentication for your API:

```python
def _authenticate(self):
    """Authenticate with the API."""
    if self.api_key:
        return
    
    if self.username and self.password:
        # Implement username/password authentication
        pass
    
    raise ValueError("Authentication failed: No valid authentication method provided")
```

### 4. Log Appropriately

Use MERIT's logging system for consistent logging:

```python
from merit.core.logging import get_logger

class CustomAPIClient(AIAPIClient):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.logger = get_logger(__name__)
    
    def generate_text(self, prompt, **kwargs):
        try:
            # Implementation...
        except Exception as e:
            self.logger.error(f"Error generating text: {str(e)}")
            return ""
```

### 5. Document Your Client

Provide clear documentation for your custom client:

```python
class CustomAPIClient(AIAPIClient):
    """
    Custom client for the Example API.
    
    This client provides methods for interacting with the Example API,
    including text generation and embeddings.
    
    Args:
        api_key: The API key for authentication.
        model: The model to use for text generation.
        base_url: The base URL for the API.
        **kwargs: Additional arguments to pass to the AIAPIClient.
    """
    
    def generate_text(self, prompt, **kwargs):
        """
        Generate text using the Example API.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments to pass to the API.
                - temperature: The temperature for text generation (default: 0.7).
                - max_tokens: The maximum number of tokens to generate (default: 1000).
                
        Returns:
            str: The generated text.
            
        Raises:
            ValueError: If authentication fails.
            requests.exceptions.RequestException: If the API request fails.
        """
        # Implementation...
```

## Next Steps

Now that you know how to create custom API clients, you can:

- Learn about [response formats](./response_formats.md) to ensure your client returns data in the expected format
- Explore [authentication methods](./authentication.md) for different APIs
- Check out the [OpenAI client](./openai_client.md) for an example of a specialized client
