# Creating Knowledge Bases

This guide explains how to create knowledge bases in MERIT from various sources. Knowledge bases are collections of documents that serve as the source of information for both test set generation and evaluation.

## Basic Knowledge Base Creation

To create a knowledge base, you need to provide data and an API client:

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
from merit.api.client import OpenAIClient
import pandas as pd

# Create an API client for embeddings and text generation
client = OpenAIClient(api_key="your-openai-api-key")

# Create a list of document dictionaries
documents = [
    {
        "content": "Paris is the capital of France. It is known for the Eiffel Tower, which is a famous landmark.",
        "metadata": {"source": "geography", "topic": "France"},
        "id": "doc1"
    },
    {
        "content": "The Eiffel Tower was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.",
        "metadata": {"source": "history", "topic": "Eiffel Tower"},
        "id": "doc2"
    },
    {
        "content": "France is a country in Western Europe. It has a population of about 67 million people.",
        "metadata": {"source": "geography", "topic": "France"},
        "id": "doc3"
    }
]

# Create a knowledge base from a list of dictionaries
knowledge_base = KnowledgeBase(data=documents, client=client)

# Print the number of documents
print(f"Knowledge base contains {len(knowledge_base)} documents")

# Access all documents
for doc in knowledge_base.documents:
    print(f"Document ID: {doc.id}, Content: {doc.content}")
```

## KnowledgeBase Constructor Parameters

The `KnowledgeBase` constructor accepts the following parameters:

```python
KnowledgeBase(
    data: Union[pd.DataFrame, List[Dict[str, Any]]],  # Required: Data to create the knowledge base from
    client: BaseAPIClient,                            # Required: API client for embeddings and text generation
    columns: Optional[Sequence[str]] = None,          # Optional: Columns to use from DataFrame
    seed: Optional[int] = None,                       # Optional: Random seed for reproducibility
    min_topic_size: Optional[int] = None,             # Optional: Minimum number of documents to form a topic
    batch_size: int = 32                              # Optional: Batch size for processing
)
```

### Data Parameter

The `data` parameter can be either:
1. A pandas DataFrame where each row represents a document
2. A list of dictionaries where each dictionary represents a document

### Client Parameter

The `client` parameter is an instance of a class that inherits from `BaseAPIClient`. This client is used for:
- Generating embeddings for documents
- Finding topics in the knowledge base
- Generating topic names

Common client implementations include:
- `OpenAIClient`: For using OpenAI's API
- `AIAPIClient`: A generic API client that can be extended

## Creating Documents

Documents are created internally by the KnowledgeBase class from the provided data. You can provide document data in several ways:

### Using Dictionaries

```python
# Create document dictionaries
document_dicts = [
    {
        "content": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
        "metadata": {"source": "AI Overview", "topic": "AI Basics"},
        "id": "doc1"
    }
]

# Create a knowledge base from dictionaries
knowledge_base = KnowledgeBase(data=document_dicts, client=client)
```

### Using a DataFrame

```python
import pandas as pd

# Create a DataFrame
df = pd.DataFrame([
    {
        "content": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
        "source": "AI Overview",
        "topic": "AI Basics",
        "id": "doc1"
    }
])

# Create a knowledge base from a DataFrame
knowledge_base = KnowledgeBase(data=df, client=client)
```

### Auto-generating IDs

If you don't provide an ID for a document in your data, MERIT will generate one automatically based on the row index:

```python
# Create document dictionaries without IDs
document_dicts = [
    {
        "content": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
        "metadata": {"source": "AI Overview", "topic": "AI Basics"}
    }
]

# Create a knowledge base
knowledge_base = KnowledgeBase(data=document_dicts, client=client)

# The ID will be automatically generated
for doc in knowledge_base.documents:
    print(f"Generated ID: {doc.id}")
```

## Creating Knowledge Bases from Various Sources

### From Text Files

```python
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient
import os

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Directory containing text files
text_dir = "path/to/text/files"

# Create document dictionaries from text files
documents = []
for filename in os.listdir(text_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(text_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Create a document dictionary
        document = {
            "content": content,
            "metadata": {"source": "text_file", "filename": filename},
            "id": filename
        }
        
        documents.append(document)

# Create a knowledge base
knowledge_base = KnowledgeBase(data=documents, client=client)
```

### From CSV Files

```python
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient
import pandas as pd

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Load documents from a CSV file
df = pd.read_csv("documents.csv")

# Create a knowledge base directly from the DataFrame
# Specify which columns to use for content if needed
knowledge_base = KnowledgeBase(data=df, client=client, columns=["content"])
```

### From JSON Files

```python
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient
import json

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Load documents from a JSON file
with open("documents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Create a knowledge base from the JSON data
knowledge_base = KnowledgeBase(data=data["documents"], client=client)
```

### From Web Pages

```python
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient
import requests
from bs4 import BeautifulSoup

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# URLs to scrape
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
]

# Create document dictionaries from web pages
documents = []
for i, url in enumerate(urls):
    # Fetch the web page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract the main content (this will vary depending on the website)
    main_content = soup.find("main") or soup.find("article") or soup.find("body")
    
    # Extract the text content
    content = main_content.get_text(separator="\n", strip=True)
    
    # Create a document dictionary
    document = {
        "content": content,
        "metadata": {"source": "web", "url": url},
        "id": f"web_{i+1}"
    }
    
    documents.append(document)

# Create a knowledge base
knowledge_base = KnowledgeBase(data=documents, client=client)
```

### From PDF Files

```python
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient
import os
import PyPDF2

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Directory containing PDF files
pdf_dir = "path/to/pdf/files"

# Create document dictionaries from PDF files
documents = []
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_dir, filename)
        
        # Extract text from the PDF
        with open(file_path, "rb") as f:
            pdf_reader = PyPDF2.PdfReader(f)
            content = ""
            for page_num in range(len(pdf_reader.pages)):
                content += pdf_reader.pages[page_num].extract_text() + "\n"
        
        # Create a document dictionary
        document = {
            "content": content,
            "metadata": {"source": "pdf", "filename": filename},
            "id": filename
        }
        
        documents.append(document)

# Create a knowledge base
knowledge_base = KnowledgeBase(data=documents, client=client)
```

## Chunking Documents

For large documents, it's often useful to chunk them into smaller pieces:

```python
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

def chunk_text(text, chunk_size=1000, overlap=100):
    """
    Chunk text into smaller pieces.
    
    Args:
        text: The text to chunk.
        chunk_size: The maximum size of each chunk.
        overlap: The overlap between chunks.
        
    Returns:
        List[str]: The chunked text.
    """
    chunks = []
    start = 0
    
    while start < len(text):
        # Find the end of the chunk
        end = min(start + chunk_size, len(text))
        
        # If we're not at the end of the text, try to find a good breaking point
        if end < len(text):
            # Try to find a period, question mark, or exclamation point
            for i in range(end, max(start, end - 200), -1):
                if text[i] in ['.', '?', '!', '\n']:
                    end = i + 1
                    break
        
        # Add the chunk to the list
        chunks.append(text[start:end])
        
        # Move the start position, accounting for overlap
        start = end - overlap
    
    return chunks

# Large document
large_text = """
[... large amount of text ...]
"""

# Chunk the document
chunks = chunk_text(large_text)

# Create document dictionaries from the chunks
documents = []
for i, chunk in enumerate(chunks):
    document = {
        "content": chunk,
        "metadata": {"source": "large_document", "chunk": i},
        "id": f"chunk_{i}"
    }
    
    documents.append(document)

# Create a knowledge base
knowledge_base = KnowledgeBase(data=documents, client=client)
```

## Working with Embeddings

The KnowledgeBase class automatically generates embeddings when needed for search operations:

```python
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Create a knowledge base
documents = [
    {
        "content": "Paris is the capital of France.",
        "metadata": {"source": "geography", "topic": "France"},
        "id": "doc1"
    },
    {
        "content": "The Eiffel Tower is located in Paris.",
        "metadata": {"source": "landmarks", "topic": "France"},
        "id": "doc2"
    }
]

knowledge_base = KnowledgeBase(data=documents, client=client)

# Search for documents (this will automatically generate embeddings if needed)
results = knowledge_base.search("What is the capital of France?", k=1)
for doc, score in results:
    print(f"Document: {doc.content}")
    print(f"Relevance score: {score}")
```

You can also access the embeddings directly:

```python
# Access embeddings (this will generate them if they don't exist yet)
embeddings = knowledge_base.embeddings
print(f"Embeddings shape: {embeddings.shape}")
```

## Accessing Documents

You can access documents in a knowledge base in several ways:

```python
# Get all documents as a list
all_docs = knowledge_base.documents

# Get a document by ID
doc = knowledge_base.get_document("doc1")
print(f"Document content: {doc.content}")

# Get documents by topic
topic_docs = knowledge_base.get_documents_by_topic(topic_id=1)
print(f"Found {len(topic_docs)} documents in topic 1")

# Get a random document
random_doc = knowledge_base.get_random_document()
print(f"Random document: {random_doc.content}")

# Get multiple random documents
random_docs = knowledge_base.get_random_documents(n=2)
for doc in random_docs:
    print(f"Random document: {doc.content}")
```

## Best Practices for Creating Knowledge Bases

### Document Quality

Ensure your documents contain high-quality, accurate information:

```python
# Good document
good_document = {
    "content": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
    "metadata": {"source": "AI Overview", "quality": "high"},
    "id": "good_doc"
}

# Poor document (too short, lacks detail)
poor_document = {
    "content": "AI is smart computers.",
    "metadata": {"source": "AI Overview", "quality": "low"},
    "id": "poor_doc"
}
```

### Document Size

Choose an appropriate size for your documents:

```python
# Too large (contains too much information)
too_large_document = {
    "content": "[... very long text covering multiple topics ...]",
    "metadata": {"size": "too_large"},
    "id": "large_doc"
}

# Too small (not enough context)
too_small_document = {
    "content": "AI is artificial intelligence.",
    "metadata": {"size": "too_small"},
    "id": "small_doc"
}

# Just right (focused on a specific topic with sufficient detail)
good_size_document = {
    "content": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
    "metadata": {"size": "good"},
    "id": "good_doc"
}
```

### Document Organization

Organize your documents effectively:

```python
# Group related information in the same document
france_document = {
    "content": "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower.",
    "metadata": {"topic": "France"},
    "id": "france_doc"
}

# Use metadata to categorize documents
document_with_metadata = {
    "content": "...",
    "metadata": {
        "source": "Wikipedia",
        "topic": "Artificial Intelligence",
        "subtopic": "Machine Learning",
        "date": "2023-01-15",
        "author": "John Doe",
        "language": "en",
        "difficulty": "intermediate",
        "tags": ["AI", "ML", "technology"]
    },
    "id": "doc1"
}
```

### Document Preprocessing

Preprocess your documents to ensure consistency:

```python
import re

def preprocess_document(content):
    """Preprocess document content."""
    # Remove extra whitespace
    content = re.sub(r'\s+', ' ', content).strip()
    
    # Remove URLs
    content = re.sub(r'https?://\S+', '', content)
    
    # Remove special characters
    content = re.sub(r'[^\w\s.,;:!?()[\]{}"\'-]', '', content)
    
    return content

# Preprocess document dictionaries
preprocessed_documents = []
for doc in raw_documents:
    preprocessed_content = preprocess_document(doc["content"])
    
    # Create a preprocessed document dictionary
    preprocessed_doc = {
        "content": preprocessed_content,
        "metadata": doc["metadata"],
        "id": doc["id"]
    }
    
    preprocessed_documents.append(preprocessed_doc)

# Create a knowledge base with preprocessed documents
knowledge_base = KnowledgeBase(data=preprocessed_documents, client=client)
```

## Next Steps

Now that you know how to create knowledge bases, you can:

- Learn how to [work with knowledge bases](./working_with_knowledge_bases.md) for various operations
- Discover how to create [custom knowledge bases](./custom_knowledge_bases.md) for specific use cases
- Explore how to use knowledge bases for [test set generation](../testset_generation/basic_generation.md)
- Learn how to use knowledge bases for [RAG evaluation](../evaluation/rag_evaluation.md)
