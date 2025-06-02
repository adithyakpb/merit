# MongoDB Metrics Calculator

This tool calculates MERIT metrics for data stored in MongoDB collections and generates comprehensive HTML reports with visualizations.

## Features

- **Configurable Field Mapping**: Python-based configuration for flexible field mapping
- **Multiple LLM Clients**: Support for OpenAI (including Azure OpenAI) and Google Gemini
- **Five Core Metrics**: Correctness, Faithfulness, Relevance, Coherence, and Fluency
- **Interactive Reports**: Rich HTML reports with charts, filtering, and dark/light themes
- **Batch Processing**: Efficient processing of large datasets
- **MongoDB Integration**: Uses MERIT's MongoDB storage module
- **Validation**: Configuration and data validation before processing

## Files

1. **`mongodb_metrics_config.py`** - Configuration file (edit this for your setup)
2. **`calculate_mongodb_metrics.py`** - Main script to run the calculator
3. **`README_MongoDB_Metrics.md`** - This documentation file

## Quick Start

### 1. Configure Your Setup

Edit `mongodb_metrics_config.py` to match your MongoDB schema and requirements:

```python
# Configure your LLM client
LLM_CLIENT = OpenAIClient(
    api_key="your-api-key-here",
    model="gpt-4"
)

# Configure MongoDB connection
MONGODB_CONFIG = {
    "connection_string": "mongodb://localhost:27017/",
    "database": "your_database_name",
}

# Configure collection and field mapping
COLLECTION_CONFIG = {
    "collection_name": "your_collection_name",
    "field_mapping": {
        "user_input": "user_query",  # Map to your field names
        "bot_response": "llm_response",
        "context_documents": extract_context_documents,  # Use functions for complex mapping
        # ... more mappings
    }
}
```

### 2. Run the Calculator

```bash
# Basic usage
python calculate_mongodb_metrics.py

# Process only 100 documents (for testing)
python calculate_mongodb_metrics.py --limit 100

# Calculate specific metrics only
python calculate_mongodb_metrics.py --metrics correctness,faithfulness,relevance

# Dry run to validate configuration
python calculate_mongodb_metrics.py --dry-run

# Custom output filename
python calculate_mongodb_metrics.py --output my_analysis_report.html
```

### 3. View Results

The script generates an interactive HTML report that you can open in your browser. The report includes:

- **Metrics Summary**: Overview of all calculated metrics
- **Interactive Charts**: Distribution and comparison visualizations
- **Knowledge Base Visualization**: Document clustering by topics
- **Detailed Results**: Expandable cards for each processed document
- **Filtering & Search**: Find specific results quickly

## Configuration Guide

### LLM Client Configuration

#### OpenAI (including Azure OpenAI)
```python
from merit.api.openai_client import OpenAIClient

# Standard OpenAI
LLM_CLIENT = OpenAIClient(
    api_key="your-api-key-here",
    model="gpt-4"
)

# Azure OpenAI
LLM_CLIENT = OpenAIClient(
    api_key="your-azure-api-key",
    base_url="https://your-endpoint.openai.azure.com/",
    api_version="2024-02-15-preview",
    model="gpt-4"
)
```

#### Google Gemini
```python
from merit.api.gemini_client import GeminiClient

LLM_CLIENT = GeminiClient(
    api_key="your-gemini-api-key",
    model="gemini-pro"
)
```

### Field Mapping Examples

#### Simple Field Mapping
```python
"field_mapping": {
    "user_input": "user_message",        # Direct field access
    "bot_response": "assistant_reply",   # Direct field access
    "timestamp": "created_at",           # Direct field access
}
```

#### Nested Field Access
```python
"field_mapping": {
    "user_input": "message.content",     # Dot notation for nested fields
    "bot_response": "response.text",     # Nested field access
}
```

#### Function-Based Mapping
```python
def extract_context_documents(doc):
    """Extract context documents from various possible structures."""
    contexts = []
    
    # Try different field names
    for field in ["retrieved_contexts", "documents", "context"]:
        if field in doc:
            field_value = doc[field]
            
            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, str):
                        contexts.append(item)
                    elif isinstance(item, dict) and "content" in item:
                        contexts.append(item["content"])
            break
    
    return contexts

"field_mapping": {
    "context_documents": extract_context_documents,  # Custom function
    "prompt": lambda doc: f"{doc.get('system_prompt', '')}\n{doc.get('user_input', '')}",  # Lambda
}
```

### MongoDB Configuration

#### Basic Connection
```python
MONGODB_CONFIG = {
    "connection_string": "mongodb://localhost:27017/",
    "database": "your_database_name",
}
```

#### With Authentication
```python
MONGODB_CONFIG = {
    "connection_string": "mongodb://localhost:27017/",
    "database": "your_database_name",
    "authentication": {
        "username": "your_username",
        "password": "your_password",
        "auth_source": "admin"
    }
}
```

#### Advanced Connection Options
```python
MONGODB_CONFIG = {
    "connection_string": "mongodb://localhost:27017/",
    "database": "your_database_name",
    "connection_options": {
        "maxPoolSize": 50,
        "minPoolSize": 5,
        "serverSelectionTimeoutMS": 5000,
        "connectTimeoutMS": 10000,
        "socketTimeoutMS": 30000
    }
}
```

### Collection Configuration

#### Basic Collection Setup
```python
COLLECTION_CONFIG = {
    "collection_name": "chat_logs",
    "field_mapping": {
        # Your field mappings here
    }
}
```

#### With Filtering and Sorting
```python
COLLECTION_CONFIG = {
    "collection_name": "chat_logs",
    "field_mapping": {
        # Your field mappings here
    },
    "query_filter": {
        "created_at": {"$gte": "2024-01-01T00:00:00Z"},  # Only recent documents
        "status": "completed"  # Only completed interactions
    },
    "sort_order": [("created_at", -1)],  # Process newest first
    "limit": 1000  # Limit for testing
}
```

## Command Line Options

```bash
python calculate_mongodb_metrics.py [OPTIONS]

Options:
  --limit INTEGER          Maximum number of documents to process
  --metrics TEXT           Comma-separated list of metrics (correctness,faithfulness,relevance,coherence,fluency)
  --output, -o TEXT        Output filename for the report
  --dry-run               Validate configuration without calculating metrics
  --no-save               Do not save results to MongoDB
  --no-report             Do not generate HTML report
  --verbose, -v           Enable verbose logging
  --help                  Show this message and exit
```

## Examples

### Example 1: Basic Analysis
```bash
# Analyze all documents with default settings
python calculate_mongodb_metrics.py
```

### Example 2: Quick Test
```bash
# Test with first 50 documents, only correctness and relevance
python calculate_mongodb_metrics.py --limit 50 --metrics correctness,relevance --verbose
```

### Example 3: Production Analysis
```bash
# Full analysis with custom output
python calculate_mongodb_metrics.py --output production_analysis_2024.html
```

### Example 4: Validation Only
```bash
# Validate configuration without processing
python calculate_mongodb_metrics.py --dry-run --verbose
```

## Output

### MongoDB Results
Results are saved to a new collection: `{original_collection_name}_metrics`

Example result document:
```json
{
  "document_id": "507f1f77bcf86cd799439011",
  "timestamp": "2024-01-15T10:30:00Z",
  "metrics": {
    "correctness": {
      "value": 0.85,
      "explanation": "The answer is mostly correct...",
      "method": "llm_evaluation",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    "faithfulness": {
      "value": 0.92,
      "explanation": "The answer stays faithful to the source...",
      "method": "llm_evaluation",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  },
  "extracted_data": {
    "user_input": "What is machine learning?",
    "bot_response": "Machine learning is a subset of AI...",
    "context_documents_count": 3
  },
  "metadata": {
    "model_name": "gpt-4",
    "temperature": 0.7
  }
}
```

### HTML Report
Interactive HTML report with:
- **Summary Dashboard**: Key metrics overview
- **Charts & Visualizations**: Distribution and comparison charts
- **Document Analysis**: Detailed view of each processed document
- **Filtering & Search**: Interactive filtering by metrics, scores, and content
- **Dark/Light Theme**: Toggle between themes
- **Knowledge Base Visualization**: Topic clustering if document content is available

## Troubleshooting

### Common Issues

1. **Import Error**: Make sure `mongodb_metrics_config.py` is in the same directory as the script
2. **MongoDB Connection**: Check your connection string and database name
3. **Collection Not Found**: Verify the collection name in your configuration
4. **Field Mapping Errors**: Use `--dry-run` to validate your field mappings
5. **LLM Client Issues**: Test your API key and endpoint configuration

### Debugging

Use verbose mode for detailed logging:
```bash
python calculate_mongodb_metrics.py --dry-run --verbose
```

### Validation

Always test with a small subset first:
```bash
python calculate_mongodb_metrics.py --limit 10 --dry-run
```

## Advanced Usage

### Custom Metrics Selection
```bash
# Only calculate specific metrics
python calculate_mongodb_metrics.py --metrics correctness,faithfulness
```

### Batch Processing
The script automatically processes documents in batches (configurable in `BATCH_SIZE`). For very large datasets, consider:

1. Using the `--limit` option to process in chunks
2. Adjusting `BATCH_SIZE` in the configuration
3. Using MongoDB query filters to process specific date ranges

### Integration with Existing Workflows
The script can be integrated into automated workflows:

```bash
#!/bin/bash
# Daily metrics calculation
python calculate_mongodb_metrics.py \
  --output "daily_report_$(date +%Y%m%d).html" \
  --metrics correctness,faithfulness,relevance \
  --no-save  # Don't save to avoid duplicates
```

## Requirements

- Python 3.7+
- pymongo
- MERIT package with all dependencies
- Access to MongoDB instance
- LLM API credentials (OpenAI or Gemini)

## Support

For issues or questions:
1. Check the verbose logs with `--verbose`
2. Validate configuration with `--dry-run`
3. Test with a small dataset using `--limit`
4. Review the field mapping functions in the configuration file
