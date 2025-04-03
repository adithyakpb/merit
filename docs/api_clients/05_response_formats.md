# API Client Response Format Guidelines

## Overview

This document defines the standard response formats for the `AIAPIClient` methods. When extending `AIAPIClient` or implementing a new client that inherits from it, it's crucial to maintain consistent response formats to ensure interoperability across different API implementations.

## Response Format Enforcement

The `AIAPIClient` class uses decorators to enforce consistent response formats across all implementations:

```python
@validate_embeddings_response
def get_embeddings(self, texts):
    # Implementation...

@validate_text_response
def generate_text(self, prompt, **kwargs):
    # Implementation...
```

These decorators automatically validate and correct the response format:

1. `@validate_embeddings_response`: Ensures the method returns a list of embeddings (list of lists of floats)
2. `@validate_text_response`: Ensures the method returns a string

When you extend `AIAPIClient`, your implementation will automatically inherit these decorators, ensuring that your methods return data in the expected format even if the underlying API has a different response structure.

## Standard Response Formats

### `get_embeddings(texts)`

**Input:**
- `texts`: A string or list of strings to get embeddings for.

**Output:**
- A list of embeddings, where each embedding is a list of floats.
- Example: `[[0.1, 0.2, 0.3, ...], [0.4, 0.5, 0.6, ...], ...]`
- For a single input text, still returns a list containing one embedding.

**Error Handling:**
- If the API call fails, return a list of empty lists matching the length of the input texts.
- Example: `[[], [], ...]`

### `generate_text(prompt, **kwargs)`

**Input:**
- `prompt`: The prompt to generate text from.
- `**kwargs`: Additional arguments to pass to the API.

**Output:**
- A string containing the generated text.
- Example: `"This is the generated text response."`

**Error Handling:**
- If the API call fails, return an empty string.
- Example: `""`

## Implementing Custom Clients

When implementing a custom client that extends `AIAPIClient`, you must ensure that your implementation of `get_embeddings()` and `generate_text()` returns data in the expected format, regardless of how the underlying API structures its responses.

### Example: Transforming API Responses

```python
def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
    """
    Get embeddings for the given texts.
    
    Args:
        texts: A string or list of strings to get embeddings for.
        
    Returns:
        List[List[float]]: A list of embeddings, where each embedding is a list of floats.
    """
    # Ensure texts is a list
    if isinstance(texts, str):
        texts = [texts]
    
    try:
        # Make API call to your service
        response = requests.post(
            f"{self.base_url}/your-embeddings-endpoint",
            json={"texts": texts},
            headers=self._get_headers()
        )
        
        response.raise_for_status()
        data = response.json()
        
        # Transform the API's response to match the expected format
        # Example: If your API returns {"results": {"vectors": [[...], [...]]}}
        if "results" in data and "vectors" in data["results"]:
            return data["results"]["vectors"]
        else:
            logger.error("No embeddings in response")
            return [[] for _ in texts]
    
    except Exception as e:
        logger.error(f"Failed to get embeddings: {str(e)}")
        return [[] for _ in texts]
```

## Best Practices

1. **Document Transformations**: Always document how your implementation transforms the API's native response format to match the expected format.

2. **Consistent Error Handling**: Follow the standard error handling patterns - empty lists for embeddings, empty strings for text generation.

3. **Type Annotations**: Use proper type annotations to make the expected input and output formats clear.

4. **Unit Tests**: Write unit tests that verify your client returns data in the expected format, even when the underlying API changes.

5. **Logging**: Log any transformations or format issues to help with debugging.

## Example: OpenAI Client Implementation

The `OpenAIClient` class demonstrates how to transform API-specific responses to match the standard format:

```python
def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
    """
    Get embeddings for the given texts using OpenAI's embeddings API.
    
    Args:
        texts: A string or list of strings to get embeddings for.
        
    Returns:
        List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        
    Note:
        This method transforms OpenAI's response format to match the AIAPIClient format.
        OpenAI returns: {"data": [{"embedding": [...]}, ...]}
        AIAPIClient expects: [[...], ...]
    """
    # Implementation details...
    
    # Extract embeddings from the response and format to match AIAPIClient
    if "data" in data:
        embeddings = [item["embedding"] for item in data["data"]]
        return embeddings
    else:
        return [[] for _ in texts]
```

By following these guidelines, you ensure that clients built on top of the `AIAPIClient` class will work consistently regardless of the underlying API implementation.
