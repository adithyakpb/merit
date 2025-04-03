# OpenAI API Client Guide

## Overview

The `OpenAIClient` provides a specialized client for interacting with OpenAI's API services, including text generation with GPT models and embeddings generation. It extends the `AIAPIClient` class and provides OpenAI-specific functionality.

## Installation Requirements

To use the OpenAI client, you need:

1. An OpenAI API key (get one at [OpenAI's website](https://platform.openai.com/api-keys))
2. The Merit package installed

## Basic Usage

```python
from merit.api.client import OpenAIClient

# Create a client with your API key
client = OpenAIClient(
    api_key="your-openai-api-key"  # Replace with your actual API key
)

# Generate text with GPT
response = client.generate_text("Explain quantum computing in simple terms")
print(response)

# Get embeddings for semantic search
embeddings = client.get_embeddings(["quantum computing", "artificial intelligence"])
```

## Configuration Options

### Direct Parameters

```python
client = OpenAIClient(
    api_key="your-openai-api-key",
    model="gpt-4",                    # Default: "gpt-3.5-turbo"
    embedding_model="text-embedding-3-small",  # Default: "text-embedding-ada-002"
    organization_id="your-org-id",    # Optional: For users with multiple organizations
    base_url="https://api.openai.com/v1"  # Default OpenAI API endpoint
)
```

### Environment Variables

Create a `.env` file:
```
OPENAI_API_KEY=your-openai-api-key
OPENAI_ORGANIZATION=your-org-id
OPENAI_MODEL=gpt-4
```

Then load it:
```python
client = OpenAIClient(env_file=".env")
```

### Configuration Object

```python
from merit.api.client import OpenAIClient, OpenAIClientConfig

# Create a configuration object
config = OpenAIClientConfig(
    api_key="your-openai-api-key",
    model="gpt-4",
    embedding_model="text-embedding-3-small"
)

# Create client with config
client = OpenAIClient(config=config)
```

## Response Formats

The `OpenAIClient` ensures consistent response formats by transforming OpenAI's API responses to match the expected formats of the `AIAPIClient` class:

### Text Generation

```python
# OpenAI's response format:
# {
#   "choices": [
#     {
#       "message": {
#         "content": "This is the generated text."
#       }
#     }
#   ]
# }

# OpenAIClient transforms this to a simple string:
# "This is the generated text."

response = client.generate_text("Write a poem about coding")
print(response)  # Just the text content
```

### Embeddings

```python
# OpenAI's response format:
# {
#   "data": [
#     {
#       "embedding": [0.1, 0.2, 0.3, ...]
#     },
#     {
#       "embedding": [0.4, 0.5, 0.6, ...]
#     }
#   ]
# }

# OpenAIClient transforms this to a list of embeddings:
# [[0.1, 0.2, 0.3, ...], [0.4, 0.5, 0.6, ...]]

embeddings = client.get_embeddings(["quantum computing", "artificial intelligence"])
print(len(embeddings))  # Number of embeddings
print(len(embeddings[0]))  # Dimensions of first embedding
```

## Advanced Features

### Multi-turn Conversations

```python
# Create a conversation with multiple messages
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello, who are you?"},
    {"role": "assistant", "content": "I'm an AI assistant. How can I help you today?"},
    {"role": "user", "content": "Tell me about machine learning."}
]

# Get the full response
response = client.create_chat_completion(messages)

# Extract the assistant's message
assistant_message = response["choices"][0]["message"]["content"]
print(assistant_message)
```

### Listing Available Models

```python
# List available models
models = client.list_models()
print(models)
```

## Error Handling

The client includes built-in error handling:

```python
try:
    response = client.generate_text("Write a story")
except Exception as e:
    print(f"Error: {str(e)}")
```

## Best Practices

1. **API Key Security**: Never hardcode your API key in your code. Use environment variables or a secure configuration system.

2. **Error Handling**: Always wrap API calls in try/except blocks to handle potential errors gracefully.

3. **Rate Limiting**: Be aware of OpenAI's rate limits and implement appropriate retry logic for production applications.

4. **Model Selection**: Choose the appropriate model for your task:
   - For text generation: GPT-4 (highest quality), GPT-3.5-Turbo (good balance of quality and cost)
   - For embeddings: text-embedding-3-small (efficient), text-embedding-3-large (highest quality)

5. **Cost Management**: Monitor your API usage to control costs, especially when using more expensive models like GPT-4.
