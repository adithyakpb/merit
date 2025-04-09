"""
MERIT Knowledge Base Error Classes

This module defines specific knowledge base-related error classes for the MERIT system.
"""

from typing import Dict, Any, Optional
from merit.core.errors import MeritKBError


class KBDocumentNotFoundError(MeritKBError):
    """Raised when a document is not found in the knowledge base."""
    def __init__(
        self, 
        doc_id: Optional[str] = None, 
        message: Optional[str] = None, 
        code: str = "001", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = f"Document{' with ID ' + str(doc_id) if doc_id else ''} not found in the knowledge base."
        help_text = (
            "Please verify the document ID is correct. "
            "Ensure the document has been correctly added to the knowledge base. "
            "Check that you're querying the correct knowledge base instance."
        )
        if doc_id and not details:
            details = {"document_id": doc_id}
        elif doc_id and details:
            details["document_id"] = doc_id
            
        super().__init__(message or default_message, code, details, help_text)


class KBInvalidEmbeddingError(MeritKBError):
    """Raised when there's an issue with document embeddings."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "002", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Invalid or missing document embeddings."
        help_text = (
            "Documents require valid embeddings for semantic search operations. "
            "Ensure the embedding generation process completed successfully. "
            "Try regenerating embeddings for the affected documents. "
            "Check that your embedding model is properly configured."
        )
        super().__init__(message or default_message, code, details, help_text)


class KBSearchError(MeritKBError):
    """Raised when a knowledge base search operation fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "003", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to perform search on the knowledge base."
        help_text = (
            "Ensure the knowledge base has been correctly initialized with documents. "
            "Verify that document embeddings have been generated. "
            "Check the search query for any formatting issues or invalid characters. "
            "If using a custom search function, verify it's implemented correctly."
        )
        super().__init__(message or default_message, code, details, help_text)


class KBInitializationError(MeritKBError):
    """Raised when knowledge base initialization fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "004", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to initialize knowledge base."
        help_text = (
            "Check that all required parameters for knowledge base initialization are provided. "
            "Ensure all referenced files or resources exist and are accessible. "
            "Verify that any external embedding services are available. "
            "Check for disk space or permission issues if storing embeddings locally."
        )
        super().__init__(message or default_message, code, details, help_text)


class KBUpdateError(MeritKBError):
    """Raised when a knowledge base update operation fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "005", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to update knowledge base."
        help_text = (
            "Ensure the document format is valid for the knowledge base. "
            "Check for any duplicate document IDs or conflicting metadata. "
            "Verify that you have permission to modify the knowledge base. "
            "If using document versioning, ensure version numbers are consistent."
        )
        super().__init__(message or default_message, code, details, help_text)


class KBSerializationError(MeritKBError):
    """Raised when serialization or deserialization of the knowledge base fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "006", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to serialize or deserialize knowledge base."
        help_text = (
            "Check the format of the serialized knowledge base file. "
            "Ensure the file is not corrupted or incomplete. "
            "Verify that the knowledge base version is compatible with the current code. "
            "Check for disk space or permission issues when saving or loading."
        )
        super().__init__(message or default_message, code, details, help_text)
