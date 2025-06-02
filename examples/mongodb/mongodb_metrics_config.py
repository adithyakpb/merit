"""
MongoDB Metrics Calculator Configuration

This configuration file defines how to connect to MongoDB, map document fields
to MERIT models, and configure which metrics to calculate.

Edit this file to match your MongoDB schema and requirements.
"""

from merit.api.openai_client import OpenAIClient
from merit.api.gemini_client import GeminiClient
from merit.core.models import Input, Response

# =============================================================================
# LLM CLIENT CONFIGURATION
# =============================================================================

# Configure your LLM client for metrics that require LLM evaluation
# Uncomment and configure one of the following:

# Option 1: OpenAI (including Azure OpenAI)
LLM_CLIENT = OpenAIClient(
    api_key="your-api-key-here",  # Or set OPENAI_API_KEY environment variable
    # base_url="https://your-azure-endpoint.openai.azure.com/",  # For Azure OpenAI
    # api_version="2024-02-15-preview",  # For Azure OpenAI
    model="gpt-4"
)

# Option 2: Google Gemini
# LLM_CLIENT = GeminiClient(
#     api_key="your-gemini-api-key-here",  # Or set GEMINI_API_KEY environment variable
#     model="gemini-pro"
# )

# Option 3: No LLM (only for metrics that don't require LLM evaluation)
# LLM_CLIENT = None

# =============================================================================
# MONGODB CONNECTION CONFIGURATION
# =============================================================================

MONGODB_CONFIG = {
    "connection_string": "mongodb://localhost:27017/",
    "database": "your_database_name",
    "connection_options": {
        "maxPoolSize": 50,
        "minPoolSize": 5,
        "serverSelectionTimeoutMS": 5000,
        "connectTimeoutMS": 10000,
        "socketTimeoutMS": 30000
    },
    # Uncomment for authentication:
    # "authentication": {
    #     "username": "your_username",
    #     "password": "your_password",
    #     "auth_source": "admin"
    # }
}

# =============================================================================
# METRICS CONFIGURATION
# =============================================================================

# List of metrics to calculate (configurable)
# Available metrics: "correctness", "faithfulness", "relevance", "coherence", "fluency"
ENABLED_METRICS = [
    "correctness",
    "faithfulness", 
    "relevance",
    "coherence",
    "fluency"
]

# Batch size for processing documents
BATCH_SIZE = 50

# =============================================================================
# FIELD MAPPING CONFIGURATION
# =============================================================================

def extract_context_documents(doc):
    """
    Extract context documents from a MongoDB document.
    
    Example: If your documents have retrieved contexts in different formats:
    - List of strings: doc["retrieved_contexts"]
    - List of dicts with 'content' field: [{"content": "text", "score": 0.9}, ...]
    - Nested structure: doc["retrieval"]["documents"]
    
    Args:
        doc: MongoDB document
        
    Returns:
        List of context document strings
    """
    # Example implementation - modify based on your schema
    contexts = []
    
    # Try different possible field names and structures
    for field_name in ["retrieved_contexts", "context_documents", "documents", "contexts"]:
        if field_name in doc:
            field_value = doc[field_name]
            
            if isinstance(field_value, list):
                for item in field_value:
                    if isinstance(item, str):
                        contexts.append(item)
                    elif isinstance(item, dict) and "content" in item:
                        contexts.append(item["content"])
                    elif isinstance(item, dict) and "text" in item:
                        contexts.append(item["text"])
            elif isinstance(field_value, str):
                contexts.append(field_value)
            break
    
    return contexts

def extract_full_prompt(doc):
    """
    Extract or construct the full prompt used for the LLM.
    
    Args:
        doc: MongoDB document
        
    Returns:
        Full prompt string
    """
    # Example: Construct prompt from multiple fields
    system_prompt = doc.get("system_prompt", "")
    user_input = doc.get("user_input", doc.get("query", ""))
    
    # If you have the full prompt stored directly:
    if "full_prompt" in doc:
        return doc["full_prompt"]
    
    # If you need to construct it:
    if system_prompt and user_input:
        return f"{system_prompt}\n\nUser: {user_input}"
    
    return user_input

# =============================================================================
# COLLECTION CONFIGURATIONS
# =============================================================================

# Configuration for your MongoDB collection
# Modify the collection name and field mappings to match your schema
COLLECTION_CONFIG = {
    "collection_name": "your_collection_name",  # Change this to your actual collection name
    
    "field_mapping": {
        # MERIT field -> MongoDB field or function
        "user_input": "user_query",  # Direct field mapping
        "prompt": extract_full_prompt,  # Function to extract/construct prompt
        "bot_response": "llm_response",  # Direct field mapping
        "context_documents": extract_context_documents,  # Function to extract contexts
        "reference_answer": "ground_truth",  # Optional: reference answer if available
        "session_id": "conversation_id",  # For grouping related interactions
        "timestamp": "created_at",  # For filtering by date
        
        # Additional metadata fields you want to preserve in results
        "model_name": "model",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "user_id": "user_id",
    },
    
    # Optional: MongoDB query to filter documents
    # Leave empty {} to process all documents
    "query_filter": {
        # Example filters:
        # "created_at": {"$gte": "2024-01-01T00:00:00Z"},  # Only recent documents
        # "model": "gpt-4",  # Only specific model
        # "status": "completed"  # Only completed interactions
    },
    
    # Optional: Sort order for processing documents
    "sort_order": [("created_at", -1)],  # Process newest first
    
    # Optional: Limit number of documents to process (for testing)
    "limit": None,  # Set to a number like 100 for testing
}

# =============================================================================
# RESULTS CONFIGURATION
# =============================================================================

# Where to store the calculated metrics
RESULTS_CONFIG = {
    "collection_suffix": "_metrics",  # Results stored in "{collection_name}_metrics"
    "include_original_doc": False,  # Whether to include the original document in results
    "include_metadata": True,  # Whether to include additional metadata
    "overwrite_existing": False,  # Whether to recalculate metrics for existing documents
}

# =============================================================================
# REPORT CONFIGURATION
# =============================================================================

# Configuration for the generated report
REPORT_CONFIG = {
    "generate_report": True,  # Whether to generate an HTML report
    "report_title": "MongoDB Metrics Analysis Report",
    "report_filename": "metrics_report.html",  # Output filename
    "include_charts": True,  # Whether to include charts in the report
    "include_summary_stats": True,  # Whether to include summary statistics
    "group_by_fields": ["model_name"],  # Fields to group metrics by in the report
}

# =============================================================================
# EXAMPLE ALTERNATIVE CONFIGURATIONS
# =============================================================================

# Example configuration for a different collection structure
ALTERNATIVE_CONFIG_EXAMPLE = {
    "collection_name": "chat_logs",
    "field_mapping": {
        "user_input": "message.content",  # Nested field access
        "prompt": lambda doc: doc.get("system_message", "") + "\n" + doc.get("message", {}).get("content", ""),
        "bot_response": "response.text",
        "context_documents": lambda doc: [chunk["text"] for chunk in doc.get("rag_context", {}).get("chunks", [])],
        "timestamp": "timestamp",
        "session_id": "session_id",
    },
    "query_filter": {"response.status": "success"},
    "sort_order": [("timestamp", 1)],
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_document(doc, field_mapping):
    """
    Validate that a document has the required fields for metric calculation.
    
    Args:
        doc: MongoDB document
        field_mapping: Field mapping configuration
        
    Returns:
        bool: True if document is valid for processing
    """
    required_fields = ["user_input", "bot_response"]
    
    for merit_field in required_fields:
        mapping = field_mapping.get(merit_field)
        
        if isinstance(mapping, str):
            # Direct field mapping
            if "." in mapping:
                # Nested field access
                keys = mapping.split(".")
                value = doc
                for key in keys:
                    if isinstance(value, dict) and key in value:
                        value = value[key]
                    else:
                        return False
            else:
                # Simple field access
                if mapping not in doc:
                    return False
        elif callable(mapping):
            # Function mapping - try to call it
            try:
                result = mapping(doc)
                if not result:
                    return False
            except Exception:
                return False
    
    return True

def extract_field_value(doc, mapping):
    """
    Extract a field value from a document using the mapping configuration.
    
    Args:
        doc: MongoDB document
        mapping: Field mapping (string or callable)
        
    Returns:
        Extracted value
    """
    if isinstance(mapping, str):
        # Direct field mapping
        if "." in mapping:
            # Nested field access
            keys = mapping.split(".")
            value = doc
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            return value
        else:
            # Simple field access
            return doc.get(mapping)
    elif callable(mapping):
        # Function mapping
        try:
            return mapping(doc)
        except Exception as e:
            print(f"Error extracting field with function: {e}")
            return None
    else:
        return mapping
