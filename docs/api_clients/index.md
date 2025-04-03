# API Clients

This section covers how to use and create API clients in MERIT. API clients provide a unified interface for interacting with various AI APIs, making it easy to connect your AI systems to MERIT for evaluation.

## In This Section

- [Overview](./01_overview.md): Introduction to API clients in MERIT
- [Basic Client](./02_basic_client.md): How to use the basic `AIAPIClient`
- [OpenAI Client](./03_openai_client.md): How to use the OpenAI-specific client
- [Custom Clients](./04_custom_clients.md): How to create custom API clients
- [Response Formats](./05_response_formats.md): Understanding and working with response formats

## Why Use MERIT API Clients?

MERIT provides a unified interface for interacting with various AI APIs, offering several advantages:

- **Consistency**: Work with different APIs using a consistent interface
- **Flexibility**: Configure clients with various options
- **Extensibility**: Create custom clients for specific APIs
- **Reliability**: Built-in error handling and retry logic

## Key Components

### BaseAPIClient

The `BaseAPIClient` is the foundation for all API clients in MERIT. It defines the basic interface that all clients must implement, including methods for generating text and embeddings.

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

### OpenAIClient

The `OpenAIClient` is a specialized client for the OpenAI API. It provides OpenAI-specific functionality while maintaining the same interface as other MERIT clients.

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

### Custom Clients

You can create custom clients for specific APIs by extending `AIAPIClient` or `BaseAPIClient`. This allows you to integrate any AI API with MERIT.

```python
from merit.api.client import AIAPIClient

class CustomAPIClient(AIAPIClient):
    """Custom client for a specific API."""
    
    def __init__(self, custom_param=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
        
    def _get_headers(self):
        headers = super()._get_headers()
        headers["Custom-Header"] = self.custom_param
        return headers
```

## Next Steps

Start by reading the [Overview](./01_overview.md) to learn more about API clients in MERIT. Then, follow the [Basic Client](./02_basic_client.md) guide to get started with the `AIAPIClient`. For OpenAI-specific functionality, check out the [OpenAI Client](./03_openai_client.md) guide. If you need to create a custom client, refer to the [Custom Clients](./04_custom_clients.md) guide.
