# Knowledge Base Overview

This guide provides an overview of knowledge bases in MERIT. Knowledge bases are a core component of MERIT, serving as the source of information for both test set generation and evaluation.

## What is a Knowledge Base?

In MERIT, a knowledge base is a collection of documents that contains the information your AI systems use to answer questions. It serves as the foundation for:

- **Test Set Generation**: Creating test inputs and reference answers based on document content
- **RAG Evaluation**: Evaluating how well RAG systems retrieve and use information from documents
- **Document Retrieval**: Finding relevant documents for a given query

A knowledge base in MERIT is designed to be:

- **Flexible**: Can contain any type of text documents
- **Searchable**: Supports semantic search to find relevant documents
- **Extensible**: Can be customized for specific use cases
- **Persistent**: Can be saved and loaded for reuse

## Knowledge Base Components

### Documents

Each document in a knowledge base has:

- **Content**: The text content of the document
- **Metadata**: Additional information about the document
- **ID**: A unique identifier for the document
- **Embeddings** (optional): Vector representations of the document for semantic search

```python
from merit.core.models import Document

# Create a document
document = Document(
    content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
    metadata={"source": "AI Overview", "topic": "AI Basics"},
    id="doc1"
)
```

### KnowledgeBase Class

The `KnowledgeBase` class is the main interface for working with knowledge bases in MERIT. It provides methods for:

- Creating knowledge bases from documents
- Searching for relevant documents
- Getting random documents
- Saving and loading knowledge bases

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=[
    Document(
        content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
        metadata={"source": "AI Overview", "topic": "AI Basics"},
        id="doc1"
    ),
    Document(
        content="Machine Learning is a subset of artificial intelligence that provides systems the ability to learn.",
        metadata={"source": "ML Overview", "topic": "ML Basics"},
        id="doc2"
    )
])
```

## How Knowledge Bases Work

### Document Storage

MERIT stores documents in memory for fast access. Each document is represented by a `Document` object, which includes the document's content, metadata, ID, and optional embeddings.

### Document Retrieval

MERIT uses semantic search to find relevant documents for a given query. This involves:

1. **Embedding Generation**: Converting the query and documents to vector embeddings
2. **Similarity Calculation**: Computing the similarity between the query and document embeddings
3. **Ranking**: Ranking documents by their similarity to the query

```python
# Search for relevant documents
query = "What is artificial intelligence?"
results = knowledge_base.search(query, k=2)

# Print the search results
for doc, score in results:
    print(f"Document ID: {doc.id}, Score: {score}")
    print(f"Content: {doc.content}")
    print()
```

### Document Sampling

MERIT can sample random documents from the knowledge base, which is useful for test set generation:

```python
# Get random documents
random_docs = knowledge_base.get_random_documents(n=2)
for doc in random_docs:
    print(f"Random document ID: {doc.id}")
    print(f"Content: {doc.content}")
    print()
```

## Knowledge Base Use Cases

### Test Set Generation

Knowledge bases are used in test set generation to create test inputs and reference answers:

```python
from merit.testset_generation import TestSetGenerator
from merit.api.client import OpenAIClient

# Create an API client for generation
client = OpenAIClient(api_key="your-openai-api-key")

# Create a test set generator
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    language="en",
    agent_description="A chatbot that answers questions about artificial intelligence."
)

# Generate a test set
test_set = generator.generate(num_inputs=10)
```

### RAG Evaluation

Knowledge bases are used in RAG evaluation to assess how well systems retrieve and use information:

```python
from merit.evaluation import evaluate_rag

# Define an answer function
def get_answer(query):
    # Your RAG system implementation here
    return "This is the answer to the query."

# Evaluate the RAG system
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)
```

### Custom Applications

Knowledge bases can be used in custom applications for document retrieval and information access:

```python
# Create a simple question-answering system
def answer_question(knowledge_base, client, question):
    # Search for relevant documents
    results = knowledge_base.search(question, k=2)
    
    # Extract the content of the relevant documents
    context = ""
    for doc, score in results:
        context += f"{doc.content}\n\n"
    
    # Generate an answer using the retrieved documents
    prompt = f"""
    Answer the following question based on the provided context:
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:
    """
    
    # Generate the answer
    answer = client.generate_text(prompt)
    
    return answer
```

## Best Practices for Knowledge Bases

### Document Quality

Ensure your documents contain high-quality, accurate information:

- Use well-written, factual content
- Include sufficient detail for answering questions
- Ensure the content is relevant to your use case

### Document Size

Choose an appropriate size for your documents:

- Too large: May contain too much information, making it difficult to find relevant parts
- Too small: May not contain enough context to answer questions effectively
- Recommended: 100-1000 words per document, focusing on a specific topic or concept

### Document Organization

Organize your documents effectively:

- Group related information in the same document
- Use metadata to categorize documents
- Ensure each document has a unique ID
- Consider using a hierarchical organization for complex knowledge bases

### Document Preprocessing

Preprocess your documents to ensure consistency:

- Remove irrelevant content (e.g., headers, footers, navigation elements)
- Normalize text formatting (e.g., whitespace, line breaks)
- Handle special characters and encoding issues
- Consider removing or replacing sensitive information

## Next Steps

Now that you understand the basics of knowledge bases in MERIT, you can:

- Learn how to [create knowledge bases](./creating_knowledge_bases.md) from various sources
- Explore how to [work with knowledge bases](./working_with_knowledge_bases.md) for various operations
- Discover how to create [custom knowledge bases](./custom_knowledge_bases.md) for specific use cases
