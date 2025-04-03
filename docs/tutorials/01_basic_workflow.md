# Basic Workflow Tutorial

This tutorial provides a step-by-step guide for a complete end-to-end workflow for evaluating a RAG (Retrieval-Augmented Generation) system using MERIT. By following this tutorial, you'll learn how to:

1. Set up a knowledge base
2. Generate a test set
3. Create a simple RAG system
4. Evaluate the RAG system
5. Analyze the results

## Prerequisites

Before you begin, make sure you have:

- Installed MERIT (see the [Installation Guide](../getting_started/installation.md))
- An API key for OpenAI or another supported API provider

## Step 1: Set Up a Knowledge Base

First, let's create a knowledge base with some sample documents:

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document

# Create some documents about artificial intelligence
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
    ),
    Document(
        content="Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.",
        metadata={"source": "NLP", "topic": "NLP Basics"},
        id="doc4"
    ),
    Document(
        content="Computer Vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects and then react to what they 'see'.",
        metadata={"source": "Computer Vision", "topic": "CV Basics"},
        id="doc5"
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
response = client.generate_text("What is artificial intelligence?")
print(f"API response: {response[:100]}...")  # Print the first 100 characters
```

## Step 3: Generate a Test Set

Now, let's generate a test set from our knowledge base:

```python
from merit.testset_generation import TestSetGenerator

# Create a test set generator
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    language="en",
    agent_description="A chatbot that answers questions about artificial intelligence and related topics."
)

# Generate a test set
test_set = generator.generate(num_inputs=10)

# Print the generated inputs
print("Generated test inputs:")
for i, input_sample in enumerate(test_set.inputs):
    print(f"{i+1}. {input_sample.input}")
    print(f"   Reference answer: {input_sample.reference_answer[:100]}...")  # Print the first 100 characters
    print()

# Save the test set for future use
test_set.save("ai_test_set.json")
```

## Step 4: Create a Simple RAG System

Now, let's create a simple RAG system that retrieves relevant documents and generates answers:

```python
def simple_rag_system(query):
    """
    A simple RAG system that retrieves relevant documents and generates answers.
    
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
    Answer the following question based on the provided context. If the context doesn't contain the information needed to answer the question, say "I don't have enough information to answer this question."
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    # Step 4: Generate the answer
    answer = client.generate_text(prompt)
    
    return answer
```

Let's test our simple RAG system:

```python
# Test the RAG system with a sample query
query = "What is machine learning?"
answer = simple_rag_system(query)
print(f"Query: {query}")
print(f"Answer: {answer}")
```

## Step 5: Evaluate the RAG System

Now, let's evaluate our RAG system using MERIT:

```python
from merit.evaluation import evaluate_rag

# Evaluate the RAG system
report = evaluate_rag(
    answer_fn=simple_rag_system,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance", "coherence", "fluency"],
    agent_description="A chatbot that answers questions about artificial intelligence and related topics."
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
    
    # Print errors if available
    if result.errors:
        print("Errors:")
        for metric, errors in result.errors.items():
            print(f"  {metric}: {errors}")

# Save the report for future reference
import json
with open("rag_evaluation_report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

## Complete Example

Here's the complete example:

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
from merit.api.client import OpenAIClient
from merit.testset_generation import TestSetGenerator
from merit.evaluation import evaluate_rag
import json

# Step 1: Set up a knowledge base
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
    ),
    Document(
        content="Natural Language Processing (NLP) is a branch of artificial intelligence that helps computers understand, interpret and manipulate human language. NLP draws from many disciplines, including computer science and computational linguistics, in its pursuit to fill the gap between human communication and computer understanding.",
        metadata={"source": "NLP", "topic": "NLP Basics"},
        id="doc4"
    ),
    Document(
        content="Computer Vision is a field of artificial intelligence that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects and then react to what they 'see'.",
        metadata={"source": "Computer Vision", "topic": "CV Basics"},
        id="doc5"
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
    agent_description="A chatbot that answers questions about artificial intelligence and related topics."
)
test_set = generator.generate(num_inputs=10)

# Step 4: Create a simple RAG system
def simple_rag_system(query):
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
    
    # Generate the answer
    answer = client.generate_text(prompt)
    
    return answer

# Step 5: Evaluate the RAG system
report = evaluate_rag(
    answer_fn=simple_rag_system,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance", "coherence", "fluency"],
    agent_description="A chatbot that answers questions about artificial intelligence and related topics."
)

# Step 6: Analyze the results
print("Evaluation results:")
for metric, score in report.get_overall_scores().items():
    print(f"{metric}: {score:.2f}")

# Save the report for future reference
with open("rag_evaluation_report.json", "w") as f:
    json.dump(report.to_dict(), f, indent=2)
```

## Next Steps

Now that you've completed this basic workflow tutorial, you can:

- Explore more advanced test set generation techniques in the [Advanced Test Set Generation](./advanced_testset_generation.md) tutorial
- Learn how to evaluate custom APIs in the [Custom API Evaluation](./custom_api_evaluation.md) tutorial
- Discover how to create custom metrics in the [Custom Metrics Creation](./custom_metrics_creation.md) tutorial
- Try the [End-to-End Workflow](./end_to_end_workflow.md) tutorial for a more comprehensive example
