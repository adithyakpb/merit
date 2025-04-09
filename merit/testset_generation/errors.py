"""
MERIT TestSet Generation Error Classes

This module defines specific test set generation-related error classes for the MERIT system.
"""

from typing import Dict, Any, Optional
from merit.core.errors import MeritTestSetError


class TestSetGenerationError(MeritTestSetError):
    """Raised when test set generation fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "001", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to generate test set."
        help_text = (
            "Test set generation encountered an error. "
            "Ensure your knowledge base contains valid documents. "
            "Check that the LLM client is correctly configured. "
            "Try generating a smaller test set or using simpler parameters."
        )
        super().__init__(message or default_message, code, details, help_text)


class TestSetInputValidationError(MeritTestSetError):
    """Raised when test set inputs fail validation."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "002", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Test set inputs failed validation."
        help_text = (
            "The generated or provided test inputs did not pass validation. "
            "Check for missing required fields or invalid formats. "
            "Ensure example inputs follow the expected structure. "
            "Verify that all required metadata is present for each input."
        )
        super().__init__(message or default_message, code, details, help_text)


class TestSetSerializationError(MeritTestSetError):
    """Raised when serialization or deserialization of a test set fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "003", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to serialize or deserialize test set."
        help_text = (
            "An error occurred while saving or loading the test set. "
            "Check the format of the test set file. "
            "Ensure the file is not corrupted or incomplete. "
            "Verify that you have the necessary permissions to read from or write to the file."
        )
        super().__init__(message or default_message, code, details, help_text)


class ExampleGuidedGenerationError(MeritTestSetError):
    """Raised when example-guided generation fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "004", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to perform example-guided test set generation."
        help_text = (
            "Example-guided generation encountered an error. "
            "Ensure your example inputs are valid and well-formed. "
            "Check that the examples are relevant to your knowledge base content. "
            "Verify that the LLM client used for generation is correctly configured."
        )
        super().__init__(message or default_message, code, details, help_text)


class ReferenceAnswerGenerationError(MeritTestSetError):
    """Raised when generating reference answers fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "005", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to generate reference answers for test inputs."
        help_text = (
            "An error occurred while generating reference answers. "
            "Check that the document content is valid and relevant to the test inputs. "
            "Ensure the LLM client used for answer generation is correctly configured. "
            "Verify that the input format matches what the answer generator expects."
        )
        super().__init__(message or default_message, code, details, help_text)


class TestSetAnalysisError(MeritTestSetError):
    """Raised when analyzing test sets or examples fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "006", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to analyze test set or examples."
        help_text = (
            "An error occurred during test set or example analysis. "
            "Check that the inputs are valid and well-formed. "
            "Ensure the LLM client used for analysis is correctly configured. "
            "Consider simplifying the analysis or breaking it into smaller parts."
        )
        super().__init__(message or default_message, code, details, help_text)


class DocumentRelevanceError(MeritTestSetError):
    """Raised when document relevance checks fail."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "007", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to determine document relevance for test inputs."
        help_text = (
            "An error occurred while checking document relevance for test inputs. "
            "Ensure the knowledge base is properly initialized and accessible. "
            "Check that the search functionality is working correctly. "
            "Consider relaxing relevance criteria if no relevant documents are found."
        )
        super().__init__(message or default_message, code, details, help_text)
