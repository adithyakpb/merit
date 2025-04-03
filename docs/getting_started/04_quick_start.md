# Quick Start Guide

This guide provides a quick introduction to MERIT with a simple end-to-end example. By following this guide, you'll learn how to:

1. Set up a knowledge base
2. Generate a test set
3. Connect to an API
4. Evaluate a RAG system
5. Interpret the results

## Prerequisites

Before you begin, make sure you have:

- Installed MERIT (see the [Installation Guide](./installation.md))
- An API key for OpenAI or another supported API provider

## Step 1: Set Up a Knowledge Base

First, let's create a simple knowledge base with a few documents:

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document

# Create some documents
documents = [
    Document(
        content="Paris is the capital of France. It is known for the Eiffel Tower, which is a famous landmark.",
        metadata={"source": "geography", "topic": "France"},
        id="doc1"
    ),
    Document(
        content="The Eiffel Tower was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.",
        metadata={"source": "history", "topic": "Eiffel Tower"},
        id="doc2"
    ),
    Document(
        content="France is a country in Western Europe. It has a population of about 67 million people.",
        metadata={"source": "geography", "topic": "France"},
        id="doc3"
    )
]

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=documents)

# Print the number of documents in the knowledge base
print(f"Knowledge base contains {len(knowledge_base)} documents")
```

## Step 2: Connect to an API

Next, let's create an API client to interact with an LLM API:

```python
from merit.api.client import OpenAIClient

# Create an OpenAI client
client = OpenAIClient(
    api_key="your-openai-api-key",  # Replace with your actual API key
    model="gpt-3.5-turbo"
)

# Test the client
response = client.generate_text("Hello, world!")
print(f"API response: {response}")
```

## Step 3: Generate a Test Set

Now, let's generate a test set from our knowledge base:

```python
from merit.testset_generation import TestSetGenerator

# Create a test set generator
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    language="en",
    agent_description="A chatbot that answers questions about geography and history."
)

# Generate a test set
test_set = generator.generate(num_inputs=5)

# Print the generated inputs
print("Generated test inputs:")
for i, input_sample in enumerate(test_set.inputs):
    print(f"{i+1}. {input_sample.input}")
    print(f"   Reference answer: {input_sample.reference_answer}")
    print()
```

## Step 4: Define an Answer Function

Next, let's define a function that simulates a RAG system by generating answers to questions:

```python
def get_answer(query):
    """
    Simulate a RAG system by generating answers to questions.
    
    In a real scenario, this function would:
    1. Search for relevant documents in the knowledge base
    2. Generate an answer based on the retrieved documents
    
    For this example, we'll use the OpenAI client directly.
    """
    # In a real RAG system, you would search for relevant documents first
    prompt = f"""
    Answer the following question based on your knowledge:
    
    Question: {query}
    
    Answer:
    """
    
    # Generate an answer
    return client.generate_text(prompt)
```

## Step 5: Evaluate the RAG System

Now, let's evaluate our simulated RAG system:

```python
from merit.evaluation import evaluate_rag

# Evaluate the RAG system
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)

# Print the overall scores
print("Evaluation results:")
for metric, score in report.get_overall_scores().items():
    print(f"{metric}: {score:.2f}")
```

## Step 6: Analyze the Results

Finally, let's analyze the results in more detail:

```python
# Print detailed results for each input
print("\nDetailed results:")
for i, result in enumerate(report.results):
    print(f"\nInput {i+1}: {result.input}")
    print(f"Reference answer: {result.reference_answer}")
    print(f"Model answer: {result.model_answer}")
    print("Scores:")
    for metric, score in result.scores.items():
        print(f"  {metric}: {score:.2f}")
    
    # Print explanations if available
    if result.explanations:
        print("Explanations:")
        for metric, explanation in result.explanations.items():
            print(f"  {metric}: {explanation}")
    
    # Print errors if available
    if result.errors:
        print("Errors:")
        for metric, errors in result.errors.items():
            print(f"  {metric}: {errors}")
```

## Complete Example

Here's the complete example:

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
from merit.api.client import OpenAIClient
from merit.testset_generation import TestSetGenerator
from merit.evaluation import evaluate_rag

# Step 1: Set up a knowledge base
documents = [
    Document(
        content="Paris is the capital of France. It is known for the Eiffel Tower, which is a famous landmark.",
        metadata={"source": "geography", "topic": "France"},
        id="doc1"
    ),
    Document(
        content="The Eiffel Tower was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.",
        metadata={"source": "history", "topic": "Eiffel Tower"},
        id="doc2"
    ),
    Document(
        content="France is a country in Western Europe. It has a population of about 67 million people.",
        metadata={"source": "geography", "topic": "France"},
        id="doc3"
    )
]
knowledge_base = KnowledgeBase(documents=documents)

# Step 2: Connect to an API
client = OpenAIClient(
    api_key="your-openai-api-key",  # Replace with your actual API key
    model="gpt-3.5-turbo"
)

# Step 3: Generate a test set
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    language="en",
    agent_description="A chatbot that answers questions about geography and history."
)
test_set = generator.generate(num_inputs=5)

# Step 4: Define an answer function
def get_answer(query):
    prompt = f"""
    Answer the following question based on your knowledge:
    
    Question: {query}
    
    Answer:
    """
    return client.generate_text(prompt)

# Step 5: Evaluate the RAG system
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)

# Step 6: Analyze the results
print("Evaluation results:")
for metric, score in report.get_overall_scores().items():
    print(f"{metric}: {score:.2f}")
```

## Next Steps

Now that you've completed this quick start guide, you can:

- Explore the [Test Set Generation](../testset_generation/index.md) guide to learn more about generating test sets
- Check out the [API Clients](../api_clients/index.md) guide to learn more about connecting to AI APIs
- Read the [Evaluation](../evaluation/index.md) guide to learn more about evaluating AI systems
- Try the [Tutorials](../tutorials/index.md) for more comprehensive examples
