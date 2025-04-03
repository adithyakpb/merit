# Knowledge Bases

This section covers how to create and use knowledge bases in MERIT. Knowledge bases are collections of documents that serve as the source of information for both test set generation and evaluation.

## In This Section

- [Overview](./01_overview.md): Introduction to knowledge bases in MERIT
- [Creating Knowledge Bases](./02_creating_knowledge_bases.md): How to create knowledge bases from various sources
- [Working with Knowledge Bases](./03_working_with_knowledge_bases.md): How to use knowledge bases for various operations
- [Custom Knowledge Bases](./04_custom_knowledge_bases.md): How to create custom knowledge base implementations

## What is a Knowledge Base?

In MERIT, a knowledge base is a collection of documents that contains the information your AI systems use to answer questions. Each document in the knowledge base has:

- **Content**: The text content of the document
- **Metadata**: Additional information about the document
- **ID**: A unique identifier for the document
- **Embeddings** (optional): Vector representations of the document for semantic search

Knowledge bases are used in MERIT for:

- **Test Set Generation**: Creating test inputs and reference answers based on document content
- **RAG Evaluation**: Evaluating how well RAG systems retrieve and use information from documents
- **Document Retrieval**: Finding relevant documents for a given query

## Key Components

### KnowledgeBase

The `KnowledgeBase` class is the main entry point for working with knowledge bases in MERIT. It provides methods for:

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
        content="Paris is the capital of France.",
        metadata={"source": "geography", "topic": "France"},
        id="doc1"
    ),
    Document(
        content="The Eiffel Tower is located in Paris.",
        metadata={"source": "landmarks", "topic": "France"},
        id="doc2"
    )
])

# Search the knowledge base
results = knowledge_base.search("What is the capital of France?", k=1)
for doc, score in results:
    print(f"Document: {doc.content}, Score: {score}")
```

### Document

The `Document` class represents a single document in a knowledge base. It includes:

- The document content
- Metadata about the document
- An identifier for the document
- Optional embeddings for semantic search

```python
from merit.core.models import Document

# Create a document
document = Document(
    content="Paris is the capital of France.",
    metadata={"source": "geography", "topic": "France"},
    id="doc1"
)
```

## Use Cases

### Test Set Generation

Knowledge bases are used in test set generation to create test inputs and reference answers:

```python
from merit.testset_generation import TestSetGenerator

# Create a test set generator
generator = TestSetGenerator(knowledge_base=knowledge_base)

# Generate a test set
test_set = generator.generate(num_inputs=10)
```

### RAG Evaluation

Knowledge bases are used in RAG evaluation to assess how well systems retrieve and use information:

```python
from merit.evaluation import evaluate_rag

# Evaluate a RAG system
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)
```

### Document Retrieval

Knowledge bases can be used directly for document retrieval:

```python
# Search for relevant documents
query = "What is the capital of France?"
results = knowledge_base.search(query, k=2)

# Print the search results
for doc, score in results:
    print(f"Document ID: {doc.id}, Score: {score}")
    print(f"Content: {doc.content}")
    print()
```

## Next Steps

Start by reading the [Overview](./01_overview.md) to learn more about knowledge bases in MERIT. Then, follow the [Creating Knowledge Bases](./02_creating_knowledge_bases.md) guide to create your first knowledge base. For more advanced use cases, check out the [Working with Knowledge Bases](./03_working_with_knowledge_bases.md) and [Custom Knowledge Bases](./04_custom_knowledge_bases.md) guides.
