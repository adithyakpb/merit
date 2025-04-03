# Custom API Evaluation

This tutorial explains how to evaluate a custom API using MERIT. You'll learn how to create a custom API client, generate a test set, and evaluate the API's performance using various metrics.

## Prerequisites

Before you begin, make sure you have:

- Installed MERIT (see the [Installation Guide](../getting_started/installation.md))
- Access to the API you want to evaluate
- Basic understanding of how your API works (endpoints, authentication, etc.)

## Step 1: Create a Custom API Client

First, you need to create a custom API client that can communicate with your API. MERIT provides a `BaseAPIClient` class that you can extend to create your own client:

```python
from merit.api.base import BaseAPIClient
import requests

class CustomAPIClient(BaseAPIClient):
    """A custom API client for your API."""
    
    def __init__(self, api_key, api_url, model=None):
        """
        Initialize the custom API client.
        
        Args:
            api_key: The API key for authentication.
            api_url: The base URL of the API.
            model: The model to use (if applicable).
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_text(self, prompt, **kwargs):
        """
        Generate text using the API.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            str: The generated text.
        """
        # Prepare the request payload
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "model": self.model or kwargs.get("model", "default")
        }
        
        # Send the request to the API
        response = requests.post(
            f"{self.api_url}/generate",
            headers=self.headers,
            json=payload
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Return the generated text
        return result.get("text", "")
    
    def get_embeddings(self, texts, **kwargs):
        """
        Get embeddings for texts using the API.
        
        Args:
            texts: The texts to get embeddings for.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            List[List[float]]: The embeddings for the texts.
        """
        # Prepare the request payload
        payload = {
            "texts": texts,
            "model": kwargs.get("model", "embedding-model")
        }
        
        # Send the request to the API
        response = requests.post(
            f"{self.api_url}/embeddings",
            headers=self.headers,
            json=payload
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Return the embeddings
        return result.get("embeddings", [])
```

## Step 2: Test Your Custom API Client

Before using your custom API client for evaluation, it's a good idea to test it to make sure it works correctly:

```python
# Create a custom API client
client = CustomAPIClient(
    api_key="your-api-key",
    api_url="https://api.example.com",
    model="text-generation-model"
)

# Test text generation
prompt = "What is artificial intelligence?"
response = client.generate_text(prompt)
print(f"Generated text: {response}")

# Test embeddings
texts = ["What is artificial intelligence?", "What is machine learning?"]
embeddings = client.get_embeddings(texts)
print(f"Generated {len(embeddings)} embeddings with dimension {len(embeddings[0])}")
```

## Step 3: Create a Knowledge Base

Next, let's create a knowledge base with some sample documents:

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document

# Create sample documents
documents = [
    Document(
        content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
        metadata={"source": "AI Overview", "topic": "AI Basics"},
        id="doc1"
    ),
    Document(
        content="Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.",
        metadata={"source": "Machine Learning", "topic": "ML Basics"},
        id="doc2"
    ),
    Document(
        content="Deep Learning is a subset of machine learning where artificial neural networks, algorithms inspired by the human brain, learn from large amounts of data. Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
        metadata={"source": "Deep Learning", "topic": "DL Basics"},
        id="doc3"
    )
]

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=documents)

# Save the knowledge base for future use
knowledge_base.save("ai_knowledge_base.json")

print(f"Created knowledge base with {len(knowledge_base)} documents")
```

## Step 4: Generate a Test Set

Now, let's generate a test set based on our knowledge base:

```python
from merit.testset_generation import TestSetGenerator
from merit.api.client import OpenAIClient  # We'll use OpenAI for test set generation

# Load the knowledge base
knowledge_base = KnowledgeBase.load("ai_knowledge_base.json")

# Create an OpenAI client for test set generation
# (We use OpenAI here because it's good at generating test sets,
# but we'll evaluate our custom API)
openai_client = OpenAIClient(api_key="your-openai-api-key")

# Create a test set generator
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    llm_client=openai_client,
    language="en",
    agent_description="A chatbot that answers questions about artificial intelligence and related topics."
)

# Generate a test set
test_set = generator.generate(num_inputs=10)

# Save the test set for future use
test_set.save("ai_test_set.json")

# Print the generated inputs
print(f"Generated {len(test_set.inputs)} test inputs:")
for i, input_sample in enumerate(test_set.inputs):
    print(f"\nInput {i+1}: {input_sample.input}")
    print(f"Reference answer: {input_sample.reference_answer[:100]}...")  # Print the first 100 characters
```

## Step 5: Create a Simple RAG System with Your Custom API

Now, let's create a simple RAG system that uses your custom API:

```python
def custom_rag_system(query):
    """
    A simple RAG system that uses your custom API.
    
    Args:
        query: The user's query
        
    Returns:
        str: The generated answer
    """
    # Step 1: Retrieve relevant documents
    relevant_docs = knowledge_base.search(query, k=2)
    
    # Step 2: Format the retrieved documents
    context = ""
    for doc, score in relevant_docs:
        context += f"{doc.content}\n\n"
    
    # Step 3: Generate an answer using the retrieved documents
    prompt = f"""
    Answer the following question based on the provided context. If the context doesn't contain the information needed to answer this question, say "I don't have enough information to answer this question."
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    # Step 4: Generate the answer using your custom API
    answer = client.generate_text(prompt)
    
    return answer
```

Let's test our simple RAG system:

```python
# Test the RAG system with a sample query
query = "What is machine learning?"
answer = custom_rag_system(query)
print(f"Query: {query}")
print(f"Answer: {answer}")
```

## Step 6: Evaluate Your Custom API

Now, let's evaluate your custom API using MERIT:

```python
from merit.evaluation import evaluate_rag
from merit.core.models import TestSet

# Load the test set
test_set = TestSet.load("ai_test_set.json")

# Load the knowledge base
knowledge_base = KnowledgeBase.load("ai_knowledge_base.json")

# Evaluate the RAG system
report = evaluate_rag(
    answer_fn=custom_rag_system,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=openai_client,  # We use OpenAI for evaluation metrics
    metrics=["correctness", "faithfulness", "relevance", "coherence", "fluency"],
    agent_description="A chatbot that answers questions about artificial intelligence and related topics."
)

# Print the overall scores
print("Evaluation results:")
for metric, score in report.get_overall_scores().items():
    print(f"{metric}: {score:.2f}")
```

## Step 7: Analyze the Results

Finally, let's analyze the results in more detail:

```python
# Print detailed results for each input
print("\nDetailed results:")
for i, result in enumerate(report.results):
    print(f"\nInput {i+1}: {result.input}")
    print(f"Reference answer: {result.reference_answer[:100]}...")  # Print the first 100 characters
    print(f"Model answer: {result.model_answer[:100]}...")  # Print the first 100 characters
    print("Scores:")
    for metric, score in result.scores.items():
        print(f"  {metric}: {score:.2f}")
    
    # Print explanations if available
    if result.explanations:
        print("Explanations:")
        for metric, explanation in result.explanations.items():
            print(f"  {metric}: {explanation[:100]}...")  # Print the first 100 characters

# Save the report for future reference
report.save("custom_api_evaluation_report.json")
```

## Complete Example

Here's the complete example:

```python
from merit.api.base import BaseAPIClient
from merit.knowledge import KnowledgeBase
from merit.core.models import Document, TestSet
from merit.testset_generation import TestSetGenerator
from merit.api.client import OpenAIClient
from merit.evaluation import evaluate_rag
import requests

# Step 1: Create a Custom API Client
class CustomAPIClient(BaseAPIClient):
    """A custom API client for your API."""
    
    def __init__(self, api_key, api_url, model=None):
        """
        Initialize the custom API client.
        
        Args:
            api_key: The API key for authentication.
            api_url: The base URL of the API.
            model: The model to use (if applicable).
        """
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def generate_text(self, prompt, **kwargs):
        """
        Generate text using the API.
        
        Args:
            prompt: The prompt to generate text from.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            str: The generated text.
        """
        # Prepare the request payload
        payload = {
            "prompt": prompt,
            "max_tokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7),
            "model": self.model or kwargs.get("model", "default")
        }
        
        # Send the request to the API
        response = requests.post(
            f"{self.api_url}/generate",
            headers=self.headers,
            json=payload
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Return the generated text
        return result.get("text", "")
    
    def get_embeddings(self, texts, **kwargs):
        """
        Get embeddings for texts using the API.
        
        Args:
            texts: The texts to get embeddings for.
            **kwargs: Additional arguments to pass to the API.
            
        Returns:
            List[List[float]]: The embeddings for the texts.
        """
        # Prepare the request payload
        payload = {
            "texts": texts,
            "model": kwargs.get("model", "embedding-model")
        }
        
        # Send the request to the API
        response = requests.post(
            f"{self.api_url}/embeddings",
            headers=self.headers,
            json=payload
        )
        
        # Check if the request was successful
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Return the embeddings
        return result.get("embeddings", [])

# Step 2: Create a Custom API Client
client = CustomAPIClient(
    api_key="your-api-key",
    api_url="https://api.example.com",
    model="text-generation-model"
)

# Step 3: Create a Knowledge Base
documents = [
    Document(
        content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
        metadata={"source": "AI Overview", "topic": "AI Basics"},
        id="doc1"
    ),
    Document(
        content="Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
        metadata={"source": "Machine Learning", "topic": "ML Basics"},
        id="doc2"
    ),
    Document(
        content="Deep Learning is a subset of machine learning where artificial neural networks, algorithms inspired by the human brain, learn from large amounts of data.",
        metadata={"source": "Deep Learning", "topic": "DL Basics"},
        id="doc3"
    )
]
knowledge_base = KnowledgeBase(documents=documents)
knowledge_base.save("ai_knowledge_base.json")

# Step 4: Generate a Test Set
openai_client = OpenAIClient(api_key="your-openai-api-key")
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    llm_client=openai_client,
    language="en",
    agent_description="A chatbot that answers questions about artificial intelligence and related topics."
)
test_set = generator.generate(num_inputs=10)
test_set.save("ai_test_set.json")

# Step 5: Create a Simple RAG System
def custom_rag_system(query):
    # Retrieve relevant documents
    relevant_docs = knowledge_base.search(query, k=2)
    
    # Format the retrieved documents
    context = ""
    for doc, score in relevant_docs:
        context += f"{doc.content}\n\n"
    
    # Generate an answer using the retrieved documents
    prompt = f"""
    Answer the following question based on the provided context. If the context doesn't contain the information needed to answer this question, say "I don't have enough information to answer this question."
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    # Generate the answer using your custom API
    answer = client.generate_text(prompt)
    
    return answer

# Step 6: Evaluate Your Custom API
report = evaluate_rag(
    answer_fn=custom_rag_system,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=openai_client,
    metrics=["correctness", "faithfulness", "relevance", "coherence", "fluency"],
    agent_description="A chatbot that answers questions about artificial intelligence and related topics."
)

# Step 7: Analyze the Results
print("Evaluation results:")
for metric, score in report.get_overall_scores().items():
    print(f"{metric}: {score:.2f}")

# Save the report for future reference
report.save("custom_api_evaluation_report.json")
```

## Adapting to Different APIs

The example above shows how to create a custom API client for a generic API. Here are some tips for adapting it to different types of APIs:

### REST APIs

For REST APIs, you can use the `requests` library as shown in the example. Make sure to:

- Set the correct headers for authentication
- Format the request payload according to the API's requirements
- Parse the response correctly

### GraphQL APIs

For GraphQL APIs, you can use the `gql` library:

```python
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

class GraphQLAPIClient(BaseAPIClient):
    def __init__(self, api_key, api_url):
        self.transport = RequestsHTTPTransport(
            url=api_url,
            headers={"Authorization": f"Bearer {api_key}"}
        )
        self.client = Client(transport=self.transport)
    
    def generate_text(self, prompt, **kwargs):
        query = gql("""
        mutation GenerateText($prompt: String!, $maxTokens: Int, $temperature: Float) {
            generateText(prompt: $prompt, maxTokens: $maxTokens, temperature: $temperature) {
                text
            }
        }
        """)
        
        variables = {
            "prompt": prompt,
            "maxTokens": kwargs.get("max_tokens", 100),
            "temperature": kwargs.get("temperature", 0.7)
        }
        
        result = self.client.execute(query, variable_values=variables)
        return result["generateText"]["text"]
```

### gRPC APIs

For gRPC APIs, you'll need to use the appropriate gRPC client:

```python
import grpc
import your_api_pb2
import your_api_pb2_grpc

class GRPCAPIClient(BaseAPIClient):
    def __init__(self, api_key, api_url):
        self.channel = grpc.secure_channel(api_url, grpc.ssl_channel_credentials())
        self.stub = your_api_pb2_grpc.YourServiceStub(self.channel)
        self.api_key = api_key
    
    def generate_text(self, prompt, **kwargs):
        metadata = [("authorization", f"Bearer {self.api_key}")]
        
        request = your_api_pb2.GenerateTextRequest(
            prompt=prompt,
            max_tokens=kwargs.get("max_tokens", 100),
            temperature=kwargs.get("temperature", 0.7)
        )
        
        response = self.stub.GenerateText(request, metadata=metadata)
        return response.text
```

## Next Steps

Now that you've learned how to evaluate a custom API, you can:

- Explore [advanced test set generation](./advanced_testset_generation.md) techniques
- Learn how to create [custom metrics](./custom_metrics_creation.md) for specialized evaluation
- Discover how to create [custom evaluators](../customization/custom_evaluators.md) for advanced evaluation scenarios
- Try the [End-to-End Workflow](./end_to_end_workflow.md) tutorial for a more comprehensive example
