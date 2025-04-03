"""
AIAPIClient Usage Examples

This file demonstrates various ways to use the AIAPIClient.
"""

from merit.api.client import AIAPIClient, AIAPIClientConfig


def basic_usage_example():
    """Basic usage with direct parameters."""
    print("\n=== Basic Usage Example ===\n")
    
    # Create a client with direct parameters
    client = AIAPIClient(
        base_url="https://api.example.com",
        api_key="your-api-key"
    )
    
    # Example API calls
    try:
        # Generate text
        response = client.generate_text("Hello, world!")
        print(f"Generated text: {response}")
        
        # Get embeddings
        embeddings = client.get_embeddings(["Hello, world!", "How are you?"])
        print(f"Number of embeddings: {len(embeddings)}")
    except Exception as e:
        print(f"API call failed: {str(e)}")


def environment_variables_example():
    """Using environment variables for configuration."""
    print("\n=== Environment Variables Example ===\n")
    print("Note: This example requires a .env file with BASE_URL and API_KEY")
    
    try:
        # Load from environment variables
        client = AIAPIClient(
            env_file=".env",
            required_vars=["BASE_URL", "API_KEY"]
        )
        
        print(f"Client initialized with base_url: {client.base_url}")
    except Exception as e:
        print(f"Failed to initialize from environment: {str(e)}")


def config_object_example():
    """Using a configuration object."""
    print("\n=== Configuration Object Example ===\n")
    
    # Create a configuration object
    config = AIAPIClientConfig(
        base_url="https://api.example.com",
        api_key="your-api-key",
        model="gpt-4"
    )
    
    try:
        # Validate configuration
        config.validate(required_params=["base_url", "api_key"])
        print("Configuration validated successfully")
        
        # Create client with config
        client = AIAPIClient(config=config)
        print(f"Client initialized with model: {client.model}")
    except ValueError as e:
        print(f"Configuration validation failed: {str(e)}")


def custom_client_example():
    """Creating a custom client for a specific API."""
    print("\n=== Custom Client Example ===\n")
    
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
        
        def list_models(self):
            """List available models (example custom method)."""
            return ["gpt-3.5-turbo", "gpt-4", "dall-e-3"]
    
    # Create custom client
    client = OpenAIClient(
        base_url="https://api.openai.com/v1",
        api_key="your-openai-key",
        organization_id="your-org-id"
    )
    
    # Use custom method
    models = client.list_models()
    print(f"Available models: {models}")
    
    # Check custom header
    headers = client._get_headers()
    print(f"Headers include organization ID: {'OpenAI-Organization' in headers}")


def advanced_usage_example():
    """Advanced usage patterns."""
    print("\n=== Advanced Usage Example ===\n")
    
    # Multiple environment support
    def get_client(environment="production"):
        """Get client for specific environment."""
        env_files = {
            "production": ".env.prod",
            "staging": ".env.staging",
            "development": ".env.dev"
        }
        
        print(f"Using environment: {environment}")
        # In a real scenario, you would use the actual env file
        return AIAPIClient(
            base_url=f"https://api-{environment}.example.com",
            api_key="your-api-key"
        )
    
    # Get client for specific environment
    dev_client = get_client("development")
    print(f"Development client base_url: {dev_client.base_url}")
    
    # Custom parameters
    client = AIAPIClient(
        base_url="https://api.example.com",
        custom_timeout=30,  # Custom parameter
        retry_count=3       # Custom parameter
    )
    
    print(f"Client has custom parameters: {client._additional_params}")


def authentication_examples():
    """Different authentication methods."""
    print("\n=== Authentication Examples ===\n")
    
    # API Key Authentication
    api_key_client = AIAPIClient(
        base_url="https://api.example.com",
        api_key="your-api-key"
    )
    print("API Key Authentication: Client initialized")
    
    # Username/Password Authentication
    username_client = AIAPIClient(
        base_url="https://api.example.com",
        login_url="https://api.example.com/login",
        username="your-username",
        password="your-password"
    )
    print("Username/Password Authentication: Client initialized")
    
    # Token Authentication
    token_client = AIAPIClient(
        base_url="https://api.example.com",
        token="your-existing-token"
    )
    print("Token Authentication: Client initialized")
    print(f"Client is authenticated: {token_client.is_authenticated}")


def error_handling_example():
    """Error handling examples."""
    print("\n=== Error Handling Example ===\n")
    
    client = AIAPIClient(
        base_url="https://api.example.com",
        api_key="your-api-key"
    )
    
    # Authentication error
    try:
        # Simulate missing authentication
        client._token = None
        client.api_key = None
        response = client.generate_text("Hello, world!")
    except ValueError as e:
        print(f"Authentication error handled: {e}")
    
    # API request error
    try:
        # Simulate a request error by using an invalid URL
        client.base_url = "https://invalid-url.example.com"
        response = client.generate_text("Hello, world!")
    except Exception as e:
        print(f"API request error handled: {type(e).__name__}")


if __name__ == "__main__":
    # Run all examples
    basic_usage_example()
    environment_variables_example()
    config_object_example()
    custom_client_example()
    advanced_usage_example()
    authentication_examples()
    error_handling_example()
