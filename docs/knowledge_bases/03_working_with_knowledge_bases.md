# Working with Knowledge Bases

This guide explains how to work with knowledge bases in MERIT for various operations. Once you've created a knowledge base, you can use it for searching, retrieving documents, generating test sets, and evaluating RAG systems.

## Basic Knowledge Base Operations

### Accessing Documents

You can access documents in a knowledge base in several ways:

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
    ),
    Document(
        content="France is a country in Western Europe.",
        metadata={"source": "geography", "topic": "France"},
        id="doc3"
    )
])

# Get the number of documents
print(f"Knowledge base contains {len(knowledge_base)} documents")

# Access a document by ID
document = knowledge_base.get_document("doc1")
print(f"Document content: {document.content}")

# Access all documents
all_documents = knowledge_base.get_all_documents()
for doc in all_documents:
    print(f"Document ID: {doc.id}, Content: {doc.content}")
```

### Searching the Knowledge Base

You can search for documents relevant to a query:

```python
# Search for documents relevant to a query
query = "What is the capital of France?"
results = knowledge_base.search(query, k=2)

# Print the search results
for doc, score in results:
    print(f"Document ID: {doc.id}, Score: {score}")
    print(f"Content: {doc.content}")
    print()
```

### Getting Random Documents

You can get random documents from the knowledge base:

```python
# Get random documents
random_docs = knowledge_base.get_random_documents(n=2)
for doc in random_docs:
    print(f"Random document ID: {doc.id}")
    print(f"Content: {doc.content}")
    print()
```

### Saving and Loading Knowledge Bases

You can save knowledge bases to files and load them later:

```python
# Save a knowledge base
knowledge_base.save("my_knowledge_base.json")

# Load a knowledge base
loaded_kb = KnowledgeBase.load("my_knowledge_base.json")
```

## Advanced Knowledge Base Operations

### Filtering Documents

You can filter documents based on their metadata:

```python
# Filter documents by topic
geography_docs = [doc for doc in knowledge_base.get_all_documents() 
                 if doc.metadata.get("topic") == "France"]

print(f"Found {len(geography_docs)} documents about France")
for doc in geography_docs:
    print(f"Document ID: {doc.id}, Content: {doc.content}")
```

### Updating Documents

You can update documents in a knowledge base:

```python
# Get a document
document = knowledge_base.get_document("doc1")

# Update the document content
document.content = "Paris is the capital and largest city of France."

# Update the document metadata
document.metadata["updated"] = "2023-01-15"

# The document is updated in the knowledge base automatically
updated_doc = knowledge_base.get_document("doc1")
print(f"Updated content: {updated_doc.content}")
print(f"Updated metadata: {updated_doc.metadata}")
```

### Adding Documents

You can add documents to an existing knowledge base:

```python
# Create a new document
new_document = Document(
    content="The Louvre is a famous museum in Paris.",
    metadata={"source": "landmarks", "topic": "France"},
    id="doc4"
)

# Add the document to the knowledge base
knowledge_base.add_document(new_document)

# Verify the document was added
print(f"Knowledge base now contains {len(knowledge_base)} documents")
added_doc = knowledge_base.get_document("doc4")
print(f"Added document: {added_doc.content}")
```

### Removing Documents

You can remove documents from a knowledge base:

```python
# Remove a document by ID
knowledge_base.remove_document("doc1")

# Verify the document was removed
print(f"Knowledge base now contains {len(knowledge_base)} documents")
try:
    removed_doc = knowledge_base.get_document("doc1")
    print("Document was not removed")
except KeyError:
    print("Document was successfully removed")
```

### Merging Knowledge Bases

You can merge multiple knowledge bases:

```python
# Create two knowledge bases
kb1 = KnowledgeBase(documents=[
    Document(
        content="Paris is the capital of France.",
        metadata={"source": "geography", "topic": "France"},
        id="doc1"
    )
])

kb2 = KnowledgeBase(documents=[
    Document(
        content="Berlin is the capital of Germany.",
        metadata={"source": "geography", "topic": "Germany"},
        id="doc2"
    )
])

# Merge the knowledge bases
merged_kb = KnowledgeBase.merge([kb1, kb2])

# Verify the merge
print(f"Merged knowledge base contains {len(merged_kb)} documents")
for doc in merged_kb.get_all_documents():
    print(f"Document ID: {doc.id}, Content: {doc.content}")
```

## Using Knowledge Bases for Test Set Generation

Knowledge bases are a key component for test set generation in MERIT:

```python
from merit.testset_generation import TestSetGenerator
from merit.api.client import OpenAIClient

# Create an API client for generation
client = OpenAIClient(api_key="your-openai-api-key")

# Create a test set generator
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    language="en",
    agent_description="A chatbot that answers questions about geography."
)

# Generate a test set
test_set = generator.generate(num_inputs=5)

# Print the generated inputs
print(f"Generated {len(test_set.inputs)} test inputs:")
for i, input_sample in enumerate(test_set.inputs):
    print(f"\nInput {i+1}: {input_sample.input}")
    print(f"Reference answer: {input_sample.reference_answer}")
    print(f"Document: {input_sample.document.content}")
```

## Using Knowledge Bases for RAG Evaluation

Knowledge bases are also used for evaluating RAG (Retrieval-Augmented Generation) systems:

```python
from merit.evaluation import evaluate_rag
from merit.core.models import TestSet

# Load or create a test set
test_set = TestSet.load("my_test_set.json")

# Define an answer function for your RAG system
def get_answer(query):
    # Your RAG system implementation here
    # This is a simple example that retrieves documents and generates an answer
    results = knowledge_base.search(query, k=2)
    
    # Extract the content of the relevant documents
    context = ""
    for doc, score in results:
        context += f"{doc.content}\n\n"
    
    # Generate an answer using the retrieved documents
    prompt = f"""
    Answer the following question based on the provided context:
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    # Generate the answer
    answer = client.generate_text(prompt)
    
    return answer

# Evaluate the RAG system
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)

# Print the evaluation results
print("Evaluation results:")
for metric, score in report.get_overall_scores().items():
    print(f"{metric}: {score:.2f}")
```

## Building a Simple RAG System

You can use a knowledge base to build a simple RAG system:

```python
def simple_rag_system(knowledge_base, client, query):
    """
    A simple RAG system that retrieves relevant documents and generates an answer.
    
    Args:
        knowledge_base: The knowledge base to search.
        client: The API client to use for generating text.
        query: The user's query.
        
    Returns:
        str: The generated answer.
    """
    # Step 1: Retrieve relevant documents
    results = knowledge_base.search(query, k=2)
    
    # Step 2: Format the retrieved documents
    context = ""
    for doc, score in results:
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

# Use the RAG system
query = "What is the capital of France?"
answer = simple_rag_system(knowledge_base, client, query)
print(f"Query: {query}")
print(f"Answer: {answer}")
```

## Advanced RAG System with Document References

You can build a more advanced RAG system that includes document references:

```python
def advanced_rag_system(knowledge_base, client, query):
    """
    An advanced RAG system that includes document references.
    
    Args:
        knowledge_base: The knowledge base to search.
        client: The API client to use for generating text.
        query: The user's query.
        
    Returns:
        dict: The generated answer and document references.
    """
    # Step 1: Retrieve relevant documents
    results = knowledge_base.search(query, k=3)
    
    # Step 2: Format the retrieved documents
    context = ""
    references = []
    for i, (doc, score) in enumerate(results):
        # Add the document to the context
        context += f"[{i+1}] {doc.content}\n\n"
        
        # Add the document to the references
        references.append({
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "relevance_score": score
        })
    
    # Step 3: Generate an answer using the retrieved documents
    prompt = f"""
    Answer the following question based on the provided context. If the context doesn't contain the information needed to answer the question, say "I don't have enough information to answer this question."
    
    Include references to the source documents in your answer using the format [1], [2], etc.
    
    Context:
    {context}
    
    Question: {query}
    
    Answer:
    """
    
    # Step 4: Generate the answer
    answer = client.generate_text(prompt)
    
    # Step 5: Return the answer and references
    return {
        "answer": answer,
        "references": references
    }

# Use the advanced RAG system
query = "What is the capital of France and what famous landmark is located there?"
result = advanced_rag_system(knowledge_base, client, query)
print(f"Query: {query}")
print(f"Answer: {result['answer']}")
print("\nReferences:")
for i, ref in enumerate(result["references"]):
    print(f"[{i+1}] {ref['content']}")
```

## Best Practices for Working with Knowledge Bases

### Efficient Searching

For efficient searching, consider these tips:

```python
# Use specific queries
specific_query = "What is the capital of France?"  # Good
vague_query = "Tell me about France"  # Too vague

# Limit the number of results
results = knowledge_base.search(query, k=3)  # Get only the top 3 results

# Use embeddings for faster search
# (Assuming documents already have embeddings)
results = knowledge_base.search(query, k=3, use_embeddings=True)
```

### Handling Large Knowledge Bases

For large knowledge bases, consider these approaches:

```python
# Sample a subset of documents for testing
sample_docs = knowledge_base.get_random_documents(n=100)
sample_kb = KnowledgeBase(documents=sample_docs)

# Use batching for processing
batch_size = 50
all_docs = knowledge_base.get_all_documents()
for i in range(0, len(all_docs), batch_size):
    batch = all_docs[i:i+batch_size]
    # Process the batch
    print(f"Processing batch {i//batch_size + 1}, documents {i+1}-{min(i+batch_size, len(all_docs))}")
```

### Maintaining Knowledge Bases

Keep your knowledge bases up to date:

```python
# Update a document with new information
doc = knowledge_base.get_document("doc1")
doc.content = "Paris is the capital of France. As of 2023, it has a population of about 2.1 million people."
doc.metadata["last_updated"] = "2023-01-15"

# Add new documents as they become available
new_doc = Document(
    content="The Louvre Museum in Paris is the world's largest art museum.",
    metadata={"source": "landmarks", "topic": "France", "added": "2023-01-15"},
    id="doc_louvre"
)
knowledge_base.add_document(new_doc)

# Remove outdated documents
knowledge_base.remove_document("outdated_doc_id")

# Save the updated knowledge base
knowledge_base.save("updated_knowledge_base.json")
```

## Next Steps

Now that you know how to work with knowledge bases, you can:

- Learn how to create [custom knowledge bases](./custom_knowledge_bases.md) for specific use cases
- Explore how to use knowledge bases for [test set generation](../testset_generation/basic_generation.md)
- Learn how to use knowledge bases for [RAG evaluation](../evaluation/rag_evaluation.md)
