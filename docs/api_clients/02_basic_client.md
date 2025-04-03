# AIAPIClient Guide

## Overview
The `AIAPIClient` provides a flexible, extensible client for interacting with REST APIs. It supports multiple configuration methods, authentication, and common API operations.

## Required Methods

The `AIAPIClient` implements two required abstract methods from `BaseAPIClient`:

1. **get_embeddings()**: Generates vector embeddings for text inputs
2. **generate_text()**: Generates text based on a prompt

If you're extending `AIAPIClient`, these methods are already implemented with a generic implementation that you can override if needed.

If you need to create a client for an API that doesn't provide these specific AI capabilities:

```python
from merit.api.base import BaseAPIClient

class CustomAPIClient(BaseAPIClient):
    """A custom API client that doesn't need AI capabilities."""
    
    def get_embeddings(self, texts):
        # Must implement this abstract method
        raise NotImplementedError("This client doesn't support embeddings")
    
    def generate_text(self, prompt, **kwargs):
        # Must implement this abstract method
        raise NotImplementedError("This client doesn't support text generation")
        
    def custom_api_call(self, data):
        # Your custom API functionality
        pass
```

## Basic Usage

```python
from merit.api.client import AIAPIClient

# Create a client with direct parameters
client = AIAPIClient(
    base_url="https://api.example.com",
    api_key="your-api-key"
)

# Generate text
response = client.generate_text("Hello, world!")
print(f"Generated text: {response}")

# Get embeddings
embeddings = client.get_embeddings(["Hello, world!", "How are you?"])
print(f"Embeddings: {embeddings}")
```

## Configuration Options

### Direct Parameters
```python
client = AIAPIClient(
    base_url="https://api.example.com",
    api_key="your-api-key",
    model="gpt-4"
)
```

### Environment Variables
Create a `.env` file:
```
BASE_URL=https://api.example.com
API_KEY=your-api-key
MODEL=gpt-4
```

Then load it:
```python
client = AIAPIClient(
    env_file=".env",
    required_vars=["BASE_URL", "API_KEY"]
)
```

### Configuration Object
```python
from merit.api.client import AIAPIClient, AIAPIClientConfig

config = AIAPIClientConfig(
    base_url="https://api.example.com",
    api_key="your-api-key"
)

# Validate configuration
config.validate(required_params=["base_url", "api_key"])

# Create client with config
client = AIAPIClient(config=config)
```

## Extending the Client

Create custom clients for specific APIs:

```python
class OpenAIClient(AIAPIClient):
    """Custom client for OpenAI API."""
    
    def __init__(self, organization_id=None, **kwargs):
        super().__init__(**kwargs)
        self.organization_id = organization_id
        
    def _get_headers(self):
        headers = super()._get_headers()
        if self.organization_id:
            headers["OpenAI-Organization"] = self.organization_id
        return headers
```

## Advanced Usage

### Multiple Environment Support
```python
def get_client(environment="production"):
    env_files = {
        "production": ".env.prod",
        "staging": ".env.staging",
        "development": ".env.dev"
    }
    
    return AIAPIClient(
        env_file=env_files.get(environment, ".env"),
        required_vars=["BASE_URL", "API_KEY"]
    )
```

### Custom Parameters
The client automatically handles any additional parameters:

```python
client = AIAPIClient(
    base_url="https://api.example.com",
    custom_timeout=30,  # Custom parameter
    retry_count=3       # Custom parameter
)
```

## Authentication

The client supports multiple authentication methods:

### API Key Authentication
```python
client = AIAPIClient(
    base_url="https://api.example.com",
    api_key="your-api-key"
)
```

### Username/Password Authentication
```python
client = AIAPIClient(
    base_url="https://api.example.com",
    login_url="https://api.example.com/login",
    username="your-username",
    password="your-password"
)

# Authentication happens automatically when needed
response = client.generate_text("Hello, world!")
```

### Token Authentication
```python
client = AIAPIClient(
    base_url="https://api.example.com",
    token="your-existing-token"
)
```

## Error Handling

The client provides error handling for API requests:

```python
try:
    response = client.generate_text("Hello, world!")
except ValueError as e:
    print(f"Authentication error: {e}")
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
```

## Dynamic Configuration

The client is designed to be dynamic and flexible:

1. **Automatic Environment Variables**: Any parameter in the constructor can be loaded from environment variables
2. **Dynamic Configuration**: The `to_dict()` method automatically includes all instance attributes
3. **Flexible Validation**: The `validate()` method can validate any required parameters

## Best Practices

1. **Required Parameters**: Always specify required parameters with `required_vars` when using environment variables
2. **Error Handling**: Wrap API calls in try/except blocks to handle potential errors
3. **Configuration Validation**: Use `validate()` to ensure all required parameters are present
4. **Secure Credentials**: Use environment variables for sensitive information like API keys
5. **Extend for Specific APIs**: Create custom clients for specific APIs to encapsulate their unique behavior
