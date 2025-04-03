"""
OpenAI API Client Examples

This file demonstrates various ways to use the OpenAIClient.
"""

from merit.api.client import OpenAIClient, OpenAIClientConfig
import os
from typing import List, Dict, Any


def basic_usage_example():
    """Basic usage with direct parameters."""
    print("\n=== Basic Usage Example ===\n")
    
    # Create a client with direct parameters
    client = OpenAIClient(
        api_key="your-openai-api-key",  # Replace with your actual API key
    )
    
    # Example API calls
    try:
        # Generate text
        response = client.generate_text("Explain quantum computing in simple terms")
        print(f"Generated text: {response[:100]}...")
        
        # Get embeddings
        embeddings = client.get_embeddings(["quantum computing", "artificial intelligence"])
        print(f"Number of embeddings: {len(embeddings)}")
        print(f"Embedding dimensions: {len(embeddings[0])}")
    except Exception as e:
        print(f"API call failed: {str(e)}")


def environment_variables_example():
    """Using environment variables for configuration."""
    print("\n=== Environment Variables Example ===\n")
    print("Note: This example requires environment variables or a .env file with OPENAI_API_KEY")
    
    try:
        # Load from environment variables
        client = OpenAIClient(
            env_file=".env",  # Path to your .env file
            required_vars=["OPENAI_API_KEY"]
        )
        
        print(f"Client initialized with model: {client.model}")
    except Exception as e:
        print(f"Failed to initialize from environment: {str(e)}")


def config_object_example():
    """Using a configuration object."""
    print("\n=== Configuration Object Example ===\n")
    
    # Create a configuration object
    config = OpenAIClientConfig(
        api_key="your-openai-api-key",  # Replace with your actual API key
        model="gpt-4",
        embedding_model="text-embedding-3-small"
    )
    
    try:
        # Create client with config
        client = OpenAIClient(config=config)
        print(f"Client initialized with model: {client.model}")
        print(f"Client initialized with embedding model: {client.embedding_model}")
    except ValueError as e:
        print(f"Configuration validation failed: {str(e)}")


def chat_completion_example():
    """Using the chat completion API."""
    print("\n=== Chat Completion Example ===\n")
    
    client = OpenAIClient(
        api_key="your-openai-api-key",  # Replace with your actual API key
        model="gpt-3.5-turbo"
    )
    
    try:
        # Create a multi-turn conversation
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, who are you?"},
            {"role": "assistant", "content": "I'm an AI assistant. How can I help you today?"},
            {"role": "user", "content": "Tell me about machine learning."}
        ]
        
        # Get the full response
        response = client.create_chat_completion(messages)
        
        # Print the assistant's message
        if "choices" in response and len(response["choices"]) > 0:
            print(f"Assistant: {response['choices'][0]['message']['content'][:100]}...")
        
        # You can also use generate_text for a simpler interface
        simple_response = client.generate_text("Explain what an API is")
        print(f"\nSimple response: {simple_response[:100]}...")
    except Exception as e:
        print(f"Chat completion failed: {str(e)}")


def advanced_parameters_example():
    """Using advanced parameters."""
    print("\n=== Advanced Parameters Example ===\n")
    
    client = OpenAIClient(
        api_key="your-openai-api-key",  # Replace with your actual API key
    )
    
    try:
        # Generate text with specific parameters
        response = client.generate_text(
            "Write a short poem about coding",
            temperature=0.9,  # Higher temperature for more creativity
            max_tokens=150,   # Limit response length
            top_p=0.95,       # Nucleus sampling
            frequency_penalty=0.5,  # Reduce repetition
            presence_penalty=0.5,   # Encourage new topics
        )
        print(f"Generated poem:\n{response}")
    except Exception as e:
        print(f"Advanced parameters example failed: {str(e)}")


def list_models_example():
    """List available models."""
    print("\n=== List Models Example ===\n")
    
    client = OpenAIClient(
        api_key="your-openai-api-key",  # Replace with your actual API key
    )
    
    try:
        models = client.list_models()
        print(f"Available models: {', '.join(models[:5])}...")
        print(f"Total models available: {len(models)}")
    except Exception as e:
        print(f"List models failed: {str(e)}")


def error_handling_example():
    """Error handling examples."""
    print("\n=== Error Handling Example ===\n")
    
    # Invalid API key
    try:
        client = OpenAIClient(api_key="invalid-key")
        response = client.generate_text("This will fail")
    except Exception as e:
        print(f"Authentication error handled: {type(e).__name__}")
    
    # Invalid model
    try:
        client = OpenAIClient(
            api_key="your-openai-api-key",  # Replace with your actual API key
            model="non-existent-model"
        )
        response = client.generate_text("This will fail")
    except Exception as e:
        print(f"Invalid model error handled: {type(e).__name__}")


if __name__ == "__main__":
    print("OpenAI API Client Examples")
    print("==========================")
    print("Note: Replace 'your-openai-api-key' with your actual OpenAI API key to run these examples.")
    print("You can also set the OPENAI_API_KEY environment variable instead.")
    
    # Uncomment the examples you want to run
    # basic_usage_example()
    # environment_variables_example()
    # config_object_example()
    # chat_completion_example()
    # advanced_parameters_example()
    # list_models_example()
    # error_handling_example()
    
    print("\nTo run these examples, uncomment the example function calls in the __main__ section.")
