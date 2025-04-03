# Creating Knowledge Bases

This guide explains how to create knowledge bases in MERIT from various sources. Knowledge bases are collections of documents that serve as the source of information for both test set generation and evaluation.

## Basic Knowledge Base Creation

The simplest way to create a knowledge base is to provide a list of documents:

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document

# Create documents
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

# Print the number of documents
print(f"Knowledge base contains {len(knowledge_base)} documents")
```

## Creating Documents

Documents can be created in several ways:

### Using the Document Class

```python
from merit.core.models import Document

# Create a document
document = Document(
    content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
    metadata={"source": "AI Overview", "topic": "AI Basics"},
    id="doc1"
)
```

### Using Dictionaries

```python
# Create a document as a dictionary
document_dict = {
    "content": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
    "metadata": {"source": "AI Overview", "topic": "AI Basics"},
    "id": "doc1"
}

# Create a knowledge base from dictionaries
knowledge_base = KnowledgeBase(documents=[document_dict])
```

### Auto-generating IDs

If you don't provide an ID for a document, MERIT will generate one automatically:

```python
# Create a document without an ID
document = Document(
    content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.",
    metadata={"source": "AI Overview", "topic": "AI Basics"}
)

# The ID will be automatically generated
print(f"Generated ID: {document.id}")
```

## Creating Knowledge Bases from Various Sources

### From Text Files

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
import os

# Directory containing text files
text_dir = "path/to/text/files"

# Create documents from text files
documents = []
for filename in os.listdir(text_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(text_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Create a document
        document = Document(
            content=content,
            metadata={"source": "text_file", "filename": filename},
            id=filename
        )
        
        documents.append(document)

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=documents)
```

### From CSV Files

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
import pandas as pd

# Load documents from a CSV file
df = pd.read_csv("documents.csv")

# Create documents from the DataFrame
documents = []
for _, row in df.iterrows():
    document = Document(
        content=row["content"],
        metadata={"source": row["source"], "topic": row["topic"]},
        id=row["id"]
    )
    
    documents.append(document)

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=documents)
```

### From JSON Files

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
import json

# Load documents from a JSON file
with open("documents.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Create documents from the JSON data
documents = []
for item in data["documents"]:
    document = Document(
        content=item["content"],
        metadata=item["metadata"],
        id=item["id"]
    )
    
    documents.append(document)

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=documents)
```

### From Web Pages
<!-- TODO we should make this easier by overloading the constructor -->
```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
import requests
from bs4 import BeautifulSoup

# URLs to scrape
urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/page3"
]

# Create documents from web pages
documents = []
for i, url in enumerate(urls):
    # Fetch the web page
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Extract the main content (this will vary depending on the website)
    main_content = soup.find("main") or soup.find("article") or soup.find("body")
    
    # Extract the text content
    content = main_content.get_text(separator="\n", strip=True)
    
    # Create a document
    document = Document(
        content=content,
        metadata={"source": "web", "url": url},
        id=f"web_{i+1}"
    )
    
    documents.append(document)

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=documents)
```

### From PDF Files

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
import os
import PyPDF2

# Directory containing PDF files
pdf_dir = "path/to/pdf/files"

# Create documents from PDF files
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
        
        # Create a document
        document = Document(
            content=content,
            metadata={"source": "pdf", "filename": filename},
            id=filename
        )
        
        documents.append(document)

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=documents)
```

## Chunking Documents

For large documents, it's often useful to chunk them into smaller pieces:

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document

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

# Create documents from the chunks
documents = []
for i, chunk in enumerate(chunks):
    document = Document(
        content=chunk,
        metadata={"source": "large_document", "chunk": i},
        id=f"chunk_{i}"
    )
    
    documents.append(document)

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=documents)
```

## Adding Embeddings

You can add embeddings to documents for more efficient search:

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
from merit.api.client import OpenAIClient

# Create an API client for embeddings
client = OpenAIClient(api_key="your-openai-api-key")

# Create documents
documents = [
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
]

# Generate embeddings for the documents
for document in documents:
    document.embeddings = client.get_embeddings(document.content)[0]

# Create a knowledge base with embeddings
knowledge_base = KnowledgeBase(documents=documents)
```

## Saving and Loading Knowledge Bases

You can save knowledge bases to files and load them later:

```python
# Save a knowledge base
knowledge_base.save("my_knowledge_base.json")

# Load a knowledge base
loaded_kb = KnowledgeBase.load("my_knowledge_base.json")
```

## Best Practices for Creating Knowledge Bases

### Document Quality

Ensure your documents contain high-quality, accurate information:

```python
# Good document
good_document = Document(
    content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
    metadata={"source": "AI Overview", "quality": "high"},
    id="good_doc"
)

# Poor document (too short, lacks detail)
poor_document = Document(
    content="AI is smart computers.",
    metadata={"source": "AI Overview", "quality": "low"},
    id="poor_doc"
)
```

### Document Size

Choose an appropriate size for your documents:

```python
# Too large (contains too much information)
too_large_document = Document(
    content="[... very long text covering multiple topics ...]",
    metadata={"size": "too_large"},
    id="large_doc"
)

# Too small (not enough context)
too_small_document = Document(
    content="AI is artificial intelligence.",
    metadata={"size": "too_small"},
    id="small_doc"
)

# Just right (focused on a specific topic with sufficient detail)
good_size_document = Document(
    content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
    metadata={"size": "good"},
    id="good_doc"
)
```

### Document Organization

Organize your documents effectively:

```python
# Group related information in the same document
france_document = Document(
    content="France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower.",
    metadata={"topic": "France"},
    id="france_doc"
)

# Use metadata to categorize documents
document_with_metadata = Document(
    content="...",
    metadata={
        "source": "Wikipedia",
        "topic": "Artificial Intelligence",
        "subtopic": "Machine Learning",
        "date": "2023-01-15",
        "author": "John Doe",
        "language": "en",
        "difficulty": "intermediate",
        "tags": ["AI", "ML", "technology"]
    },
    id="doc1"
)
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

# Preprocess documents
preprocessed_documents = []
for doc in raw_documents:
    preprocessed_content = preprocess_document(doc.content)
    
    # Create a preprocessed document
    preprocessed_doc = Document(
        content=preprocessed_content,
        metadata=doc.metadata,
        id=doc.id
    )
    
    preprocessed_documents.append(preprocessed_doc)

# Create a knowledge base with preprocessed documents
knowledge_base = KnowledgeBase(documents=preprocessed_documents)
```

## Next Steps

Now that you know how to create knowledge bases, you can:

- Learn how to [work with knowledge bases](./working_with_knowledge_bases.md) for various operations
- Discover how to create [custom knowledge bases](./custom_knowledge_bases.md) for specific use cases
- Explore how to use knowledge bases for [test set generation](../testset_generation/basic_generation.md)
- Learn how to use knowledge bases for [RAG evaluation](../evaluation/rag_evaluation.md)
