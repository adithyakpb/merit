
import pytest
from unittest.mock import MagicMock, patch

from openai import APIError, RateLimitError, AuthenticationError, BadRequestError
from openai.types import Completion, CompletionChoice
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import ChatCompletionMessage, Choice
from openai.types.create_embedding_response import CreateEmbeddingResponse, Usage
from openai.types.embedding import Embedding

from merit.api.openai_client import OpenAIClient, OpenAIClientConfig
from merit.api.errors import (
    MeritAPIAuthenticationError,
    MeritAPIRateLimitError,
    MeritAPIInvalidRequestError,
    MeritAPIServerError,
)

@pytest.fixture
def mock_openai_sdk_client():
    """Fixture to mock the OpenAI SDK client."""
    return MagicMock()

@pytest.fixture
def openai_client_config():
    """Fixture for a basic OpenAIClientConfig."""
    return OpenAIClientConfig(api_key="test_api_key", base_url="https://api.openai.com/v1")

@pytest.fixture
def openai_client(openai_client_config, mock_openai_sdk_client):
    """Fixture to create an OpenAIClient with a mocked SDK client."""
    with patch("merit.api.openai_client.SDKOpenAIClient", return_value=mock_openai_sdk_client):
        return OpenAIClient(openai_client_config)

class TestOpenAIClientInitialization:
    def test_successful_initialization(self, openai_client_config):
        with patch("merit.api.openai_client.SDKOpenAIClient") as mock_sdk:
            client = OpenAIClient(openai_client_config)
            mock_sdk.assert_called_once_with(
                api_key="test_api_key",
                base_url="https://api.openai.com/v1",
                max_retries=2,
                timeout=60.0,
            )
            assert isinstance(client, OpenAIClient)

    def test_azure_initialization(self):
        config = OpenAIClientConfig(
            api_key="azure_api_key",
            api_type="azure",
            api_version="2023-05-15",
            base_url="https://my-azure-endpoint.openai.azure.com/",
        )
        with patch("merit.api.openai_client.SDKAzureOpenAIClient") as mock_sdk:
            client = OpenAIClient(config)
            mock_sdk.assert_called_once_with(
                api_key="azure_api_key",
                azure_endpoint="https://my-azure-endpoint.openai.azure.com/",
                api_version="2023-05-15",
                max_retries=2,
                timeout=60.0,
            )
            assert isinstance(client, OpenAIClient)

    def test_initialization_missing_api_key(self):
        with pytest.raises(MeritAPIAuthenticationError):
            OpenAIClient(OpenAIClientConfig(api_key=None))

class TestOpenAIGenerateText:
    def test_generate_text_success(self, openai_client, mock_openai_sdk_client):
        mock_response = ChatCompletion(
            id="chatcmpl-123",
            object="chat.completion",
            created=1677652288,
            model="gpt-4o-mini",
            choices=[
                Choice(
                    index=0,
                    message=ChatCompletionMessage(
                        role="assistant",
                        content="\n\nHello there, how may I assist you today?",
                    ),
                    finish_reason="stop",
                )
            ],
        )
        mock_openai_sdk_client.chat.completions.create.return_value = mock_response

        result = openai_client.generate_text("Hello", model="gpt-4o-mini")
        assert result == "\n\nHello there, how may I assist you today?"
        mock_openai_sdk_client.chat.completions.create.assert_called_once_with(
            messages=[{'role': 'user', 'content': 'Hello'}], model='gpt-4o-mini'
        )

    @pytest.mark.parametrize(
        "sdk_error, merit_error",
        [
            (RateLimitError("Rate limited", response=MagicMock(), body=None), MeritAPIRateLimitError),
            (AuthenticationError("Invalid key", response=MagicMock(), body=None), MeritAPIAuthenticationError),
            (BadRequestError("Bad request", response=MagicMock(), body=None), MeritAPIInvalidRequestError),
            (APIError("Server error", request=MagicMock(), body=None), MeritAPIServerError),
        ],
    )
    def test_generate_text_api_errors_strict(
        self, openai_client, mock_openai_sdk_client, sdk_error, merit_error
    ):
        openai_client.config.strict = True
        mock_openai_sdk_client.chat.completions.create.side_effect = sdk_error

        with pytest.raises(merit_error):
            openai_client.generate_text("test prompt", model="gpt-4o-mini")

    def test_generate_text_graceful_mode(self, openai_client, mock_openai_sdk_client):
        openai_client.config.strict = False
        mock_openai_sdk_client.chat.completions.create.side_effect = RateLimitError(
            "Rate limited", response=MagicMock(), body=None
        )

        result = openai_client.generate_text("test prompt", model="gpt-4o-mini")
        assert result == ""

class TestOpenAIGetEmbeddings:
    def test_get_embeddings_success(self, openai_client, mock_openai_sdk_client):
        mock_response = CreateEmbeddingResponse(
            object="list",
            data=[
                Embedding(object="embedding", embedding=[0.1, 0.2, 0.3], index=0),
                Embedding(object="embedding", embedding=[0.4, 0.5, 0.6], index=1),
            ],
            model="text-embedding-ada-002",
            usage=Usage(prompt_tokens=8, total_tokens=8),
        )
        mock_openai_sdk_client.embeddings.create.return_value = mock_response

        embeddings = openai_client.get_embeddings(["text1", "text2"], model="text-embedding-ada-002")
        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_openai_sdk_client.embeddings.create.assert_called_once_with(
            input=["text1", "text2"], model="text-embedding-ada-002"
        )

    @pytest.mark.parametrize(
        "sdk_error, merit_error",
        [
            (RateLimitError("Rate limited", response=MagicMock(), body=None), MeritAPIRateLimitError),
            (AuthenticationError("Invalid key", response=MagicMock(), body=None), MeritAPIAuthenticationError),
            (BadRequestError("Bad request", response=MagicMock(), body=None), MeritAPIInvalidRequestError),
            (APIError("Server error", request=MagicMock(), body=None), MeritAPIServerError),
        ],
    )
    def test_get_embeddings_api_errors_strict(
        self, openai_client, mock_openai_sdk_client, sdk_error, merit_error
    ):
        openai_client.config.strict = True
        mock_openai_sdk_client.embeddings.create.side_effect = sdk_error

        with pytest.raises(merit_error):
            openai_client.get_embeddings(["test text"], model="text-embedding-ada-002")

    def test_get_embeddings_graceful_mode(self, openai_client, mock_openai_sdk_client):
        openai_client.config.strict = False
        mock_openai_sdk_client.embeddings.create.side_effect = RateLimitError(
            "Rate limited", response=MagicMock(), body=None
        )

        result = openai_client.get_embeddings(["test text"], model="text-embedding-ada-002")
        assert result == [[]]
