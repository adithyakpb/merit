# API Clients Overview

This guide provides an overview of API clients in MERIT. API clients are a core component of MERIT, providing a unified interface for interacting with various AI APIs and enabling seamless integration with your evaluation workflows.

## What are API Clients?

API clients in MERIT are classes that provide a standardized interface for interacting with AI APIs. They handle the complexities of API authentication, request formatting, response parsing, and error handling, allowing you to focus on your evaluation tasks rather than the details of API integration.

## Why Use MERIT API Clients?

Using MERIT's API clients offers several advantages:

- **Unified Interface**: Interact with different AI APIs using a consistent interface
- **Standardized Responses**: Get responses in a consistent format regardless of the underlying API
- **Flexible Configuration**: Configure clients with various options to suit your needs
- **Built-in Error Handling**: Handle API errors gracefully with built-in error handling
- **Extensibility**: Create custom clients for specific APIs or use cases

## API Client Hierarchy

MERIT provides a hierarchy of API client classes:

### BaseAPIClient

The `BaseAPIClient` is the foundation for all API clients in MERIT. It defines the basic interface that all clients must implement, including methods for generating text and embeddings.

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

### AIAPIClient

The `AIAPIClient` extends `BaseAPIClient` and provides a generic implementation for AI APIs. It includes methods for:

- Generating text from prompts
- Getting embeddings for text
- Making API requests with authentication
- Handling errors and retries

```python
from merit.api.client import AIAPIClient

# Create a client
client = AIAPIClient(
    base_url="https://api.example.com",
    api_key="your-api-key"
)

# Generate text
response = client.generate_text("Hello, world!")

# Get embeddings
embeddings = client.get_embeddings(["Hello, world!", "How are you?"])
```

### Specialized Clients

MERIT includes specialized clients for specific AI APIs, such as the `OpenAIClient` for OpenAI's API. These clients provide API-specific functionality while maintaining the same interface as other MERIT clients.

```python
from merit.api.client import OpenAIClient

# Create an OpenAI client
client = OpenAIClient(
    api_key="your-openai-api-key",
    model="gpt-4"
)

# Generate text
response = client.generate_text("Explain quantum computing in simple terms")
```

## Configuration Options

MERIT API clients support various configuration options:

### Direct Parameters

```python
client = AIAPIClient(
    base_url="https://api.example.com",
    api_key="your-api-key",
    model="gpt-4"
)
```

### Environment Variables

```python
# Create a .env file:
# BASE_URL=https://api.example.com
# API_KEY=your-api-key
# MODEL=gpt-4

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

## Authentication Methods

MERIT API clients support multiple authentication methods:

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
```

### Token Authentication

```python
client = AIAPIClient(
    base_url="https://api.example.com",
    token="your-existing-token"
)
```

## Response Formats

MERIT ensures consistent response formats across different API implementations:

### Text Generation

```python
# Generate text
response = client.generate_text("Hello, world!")
print(f"Generated text: {response}")
# Output: "Generated text: Hello! How can I assist you today?"
```

### Embeddings

```python
# Get embeddings
embeddings = client.get_embeddings(["Hello, world!", "How are you?"])
print(f"Number of embeddings: {len(embeddings)}")
print(f"Embedding dimensions: {len(embeddings[0])}")
# Output: "Number of embeddings: 2"
#         "Embedding dimensions: 1536"
```

## Error Handling

MERIT API clients include built-in error handling:

```python
try:
    response = client.generate_text("Hello, world!")
except ValueError as e:
    print(f"Authentication error: {e}")
except requests.exceptions.RequestException as e:
    print(f"API request failed: {e}")
```

## Next Steps

Now that you understand the basics of API clients in MERIT, you can:

- Learn how to use the [Basic Client](./basic_client.md) for general AI APIs
- Explore the [OpenAI Client](./openai_client.md) for OpenAI-specific functionality
- Discover how to create [Custom Clients](./custom_clients.md) for your specific needs
- Understand [Authentication](./authentication.md) methods for different APIs
- Learn about [Response Formats](./response_formats.md) for consistent API interactions
