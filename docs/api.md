# MERIT API Reference

> **Unified LLM API Client with Enterprise-Grade Features**

The MERIT API module provides a production-ready interface for interacting with multiple LLM providers through a single, consistent API. Built for enterprise applications with automatic caching, intelligent error handling, and comprehensive monitoring.

📓 **For detailed examples and tutorials, see [examples/api_examples.ipynb](../examples/api_examples.ipynb)**

## 📚 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🏗️ Architecture](#️-architecture)
- [⚙️ Configuration Reference](#️-configuration-reference)
- [🔧 API Reference](#-api-reference)
- [🛡️ Error Handling](#️-error-handling)
- [⚡ Performance Features](#-performance-features)
- [🏭 Production Patterns](#-production-patterns)
- [🐛 Troubleshooting](#-troubleshooting)

---

## 🚀 Quick Start

```python
from merit.api import OpenAIClient, GeminiClient

# Environment-based setup (recommended)
openai_client = OpenAIClient()
gemini_client = GeminiClient()

# Generate text with any provider
result = openai_client.generate_text("What is AI?")
result = gemini_client.generate_text("What is AI?")  # Same interface

# Get embeddings with automatic caching
embeddings = openai_client.get_embeddings(["text 1", "text 2"])
```

**📓 See [examples/api_examples.ipynb](../examples/api_examples.ipynb) for complete tutorials and before/after comparisons.**

---

## 🏗️ Architecture

### Class Hierarchy

```
BaseAPIClient (Abstract)
    ↓
AIAPIClient (Generic Implementation)
    ↓
├── OpenAIClient (OpenAI-specific)
├── GeminiClient (Gemini-specific)
└── YourCustomClient (Extensible)
```

### Design Principles

1. **Unified Interface**: All providers implement the same contract
2. **Fail-Safe Design**: Graceful degradation with optional strict mode
3. **Performance First**: Automatic caching and adaptive rate limiting
4. **Production Ready**: Comprehensive error handling and monitoring
5. **Extensible**: Easy to add new providers

### Core Components

| Component | Purpose |
|-----------|---------|
| `base.py` | Abstract base classes and decorators |
| `client.py` | Generic AIAPIClient implementation |
| `openai_client.py` | OpenAI-specific client |
| `gemini_client.py` | Google Gemini client |
| `errors.py` | API-specific error classes |
| `run_config.py` | Retry, throttling, and adaptive delay |

---

## ⚙️ Configuration Reference

### Environment Variables

#### OpenAI Client
```bash
OPENAI_API_KEY=sk-proj-...           # Required
OPENAI_ORGANIZATION=org-...          # Optional
MODEL=gpt-4                          # Optional (default: gpt-3.5-turbo)
EMBEDDING_MODEL=text-embedding-ada-002  # Optional
TEMPERATURE=0.7                      # Optional
MAX_TOKENS=1000                      # Optional
STRICT=false                         # Optional (true/false)
```

#### Gemini Client
```bash
GOOGLE_API_KEY=AIzaSy...             # Required (or GEMINI_API_KEY)
GEMINI_API_KEY=AIzaSy...             # Alternative to GOOGLE_API_KEY
GENERATION_MODEL=gemini-2.0-flash-exp  # Optional
EMBEDDING_MODEL=text-embedding-004   # Optional
TEMPERATURE=0.1                      # Optional
MAX_OUTPUT_TOKENS=1024               # Optional
TOP_P=0.95                          # Optional
TOP_K=40                            # Optional
```

### Configuration Methods

| Method | Use Case | Example |
|--------|----------|---------|
| **Environment Variables** | Production (recommended) | `OpenAIClient()` |
| **Direct Parameters** | Development | `OpenAIClient(api_key="sk-...")` |
| **Configuration Objects** | Complex setups | `OpenAIClient(config=config)` |
| **Mixed Configuration** | Override specific params | `OpenAIClient(env_file=".env", model="gpt-4")` |

**📓 See [examples/api_examples.ipynb](../examples/api_examples.ipynb) for detailed configuration examples.**

---

## 🔧 API Reference

### BaseAPIClient Interface

All clients implement this interface:

```python
class BaseAPIClient(ABC):
    @abstractmethod
    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt."""
        
    @abstractmethod
    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Get embeddings for text(s)."""
        
    @property
    @abstractmethod
    def is_authenticated(self) -> bool:
        """Check if client is authenticated."""
        
    @abstractmethod
    def login(self) -> bool:
        """Authenticate with the API."""
```

### OpenAI Client Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate_text()` | `prompt: str, **kwargs` | `str` | Generate text using chat completions |
| `get_embeddings()` | `texts: Union[str, List[str]]` | `List[List[float]]` | Get embeddings (cached automatically) |
| `create_chat_completion()` | `messages: List[Dict], **kwargs` | `Dict[str, Any]` | Multi-turn conversations |
| `list_models()` | None | `List[str]` | List available models |
| `is_authenticated` | Property | `bool` | Check authentication status |

### Gemini Client Methods

| Method | Parameters | Returns | Description |
|--------|------------|---------|-------------|
| `generate_text()` | `prompt: str, **kwargs` | `str` | Generate text using Gemini models |
| `get_embeddings()` | `texts: Union[str, List[str]]` | `List[List[float]]` | Get embeddings (cached automatically) |
| `count_tokens()` | `text: str, model: str = None` | `int` | Count tokens for cost estimation |
| `list_models()` | None | `List[str]` | List available models |
| `create_chat_session()` | None | `Optional[Any]` | Create chat session (if supported) |

### Common Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `strict` | `bool` | Error handling mode | `False` |
| `temperature` | `float` | Creativity level (0.0-1.0) | `0.7` |
| `max_tokens` | `int` | Maximum response tokens | `1000` |
| `top_p` | `float` | Nucleus sampling | `1.0` |
| `top_k` | `int` | Top-k sampling (Gemini) | `40` |

---

## 🛡️ Error Handling

### Error Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Graceful** (`strict=False`) | Returns `None`/empty on failure | Production applications |
| **Strict** (`strict=True`) | Raises exceptions on failure | Development/debugging |

### Error Types

| Error Class | HTTP Status | Typical Cause |
|-------------|-------------|---------------|
| `MeritAPIAuthenticationError` | 401, 403 | Invalid API key |
| `MeritAPIRateLimitError` | 429 | Rate limit exceeded |
| `MeritAPIInvalidRequestError` | 400 | Bad request parameters |
| `MeritAPIResourceNotFoundError` | 404 | Wrong endpoint/model |
| `MeritAPIServerError` | 5xx | Provider server issues |
| `MeritAPITimeoutError` | Timeout | Network/processing timeout |
| `MeritAPIConnectionError` | Connection | Network connectivity |

### Error Handling Patterns

```python
# Method-level override
result = client.generate_text("query", strict=True)  # Override client default

# Exception handling
try:
    result = client.generate_text("query", strict=True)
except MeritAPIRateLimitError as e:
    if e.retry_after:
        time.sleep(e.retry_after)
```

**📓 See [examples/api_examples.ipynb](../examples/api_examples.ipynb) for detailed error handling scenarios.**

---

## ⚡ Performance Features

### Automatic Caching

- **What's cached**: Embeddings only (by content hash)
- **Cache hits**: 1000x+ faster, 60-80% cost savings
- **Thread-safe**: Safe for concurrent usage
- **Memory efficient**: LRU eviction policy

### Performance Benefits

| Feature | Benefit |
|---------|---------|
| **Embedding Caching** | 60-80% cost reduction, 1000x+ speed improvement |
| **Batch Processing** | Efficient handling of multiple requests |
| **Connection Reuse** | Reduced connection overhead |
| **Adaptive Rate Limiting** | Prevents 429 errors automatically |

### Cache Behavior

```python
# First call - hits API
embeddings1 = client.get_embeddings(["text"])  # ~500ms

# Second call - uses cache  
embeddings2 = client.get_embeddings(["text"])  # ~0.001ms
```

**📓 See [examples/api_examples.ipynb](../examples/api_examples.ipynb) for performance demonstrations.**

---

## 🏭 Production Patterns

### Multi-Provider Fallback

```python
def robust_generation(prompt):
    providers = [OpenAIClient(strict=False), GeminiClient(strict=False)]
    for provider in providers:
        result = provider.generate_text(prompt)
        if result and len(result.strip()) > 0:
            return result
    return "Service unavailable"
```

### Configuration Management

```python
# Environment-based configuration (recommended)
client = OpenAIClient(env_file=".env")

# Validation
config.validate(required_params=["api_key", "model"])

# Dynamic switching
clients = {
    "creative": OpenAIClient(temperature=0.9),
    "factual": OpenAIClient(temperature=0.1)
}
```

### Batch Processing

```python
# Efficient batch processing
embeddings = client.get_embeddings(documents)  # Single batch call
for doc, embedding in zip(documents, embeddings):
    # Process each document with its embedding
    pass
```

### Monitoring

```python
# Check authentication
if not client.is_authenticated:
    logger.error("Client not authenticated")

# Monitor cache performance
cache_hit_rate = cache_hits / total_calls * 100
if cache_hit_rate < 30:
    logger.warning("Low cache hit rate")
```

---

## 🐛 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| `Authentication failed` | Invalid API key | Check `OPENAI_API_KEY` or `GOOGLE_API_KEY` |
| `Model not found` | Invalid model name | Use `client.list_models()` to see available models |
| `Rate limited` | Too many requests | Enable graceful mode or implement backoff |
| `Connection timeout` | Network issues | Check internet connection, try again |
| `Empty responses` | API failure in graceful mode | Check logs, switch to strict mode for debugging |

### Debug Mode

```python
# Enable strict mode for debugging
client = OpenAIClient(strict=True)

# Check authentication
print(f"Authenticated: {client.is_authenticated}")

# List available models
models = client.list_models()
print(f"Available models: {models[:5]}")
```

### Logging

```python
import logging
logging.getLogger("merit.api").setLevel(logging.DEBUG)
```

### Environment Validation

```python
# Check required environment variables
required_vars = ["OPENAI_API_KEY", "MODEL"]
missing = [var for var in required_vars if not os.getenv(var)]
if missing:
    raise ValueError(f"Missing environment variables: {missing}")
```

---

## 📖 Additional Resources

- **📓 Tutorial Notebook**: [examples/api_examples.ipynb](../examples/api_examples.ipynb)
- **🔧 Source Code**: [merit/api/](../merit/api/)
- **🧪 Tests**: [tests/unit/api/](../tests/unit/api/)
- **📊 Examples**: [examples/](../examples/)

---

## 🎯 Key Benefits Summary

| Without MERIT | With MERIT |
|---------------|------------|
| 50+ lines of boilerplate | 2 lines of clean code |
| Manual error handling | Automatic resilience |
| No caching (expensive) | 60-80% cost savings |
| Provider-specific code | Write once, use anywhere |
| Complex configuration | Multiple config methods |
| Manual rate limiting | Adaptive throttling |

**📓 For complete examples and tutorials, see [examples/api_examples.ipynb](../examples/api_examples.ipynb)**
