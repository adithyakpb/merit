"""
MERIT Error System

This module defines the complete error hierarchy for the MERIT system,
including base error classes and specific core error implementations.
Module-specific error classes are defined in their respective module directories.
"""

import datetime
from typing import Dict, Any, Optional


class MeritBaseError(Exception):
    """Base exception class for all MERIT errors."""
    def __init__(
        self, 
        message: str, 
        code: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None, 
        help_text: Optional[str] = None
    ):
        """
        Initialize the base error.
        
        Args:
            message: The error message
            code: The error code
            details: Additional error details
            help_text: Helpful troubleshooting text
        """
        self.message = message
        self.code = code
        self.details = details or {}
        self.timestamp = datetime.datetime.now().isoformat()
        self.help_text = help_text
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """Format the error as a string with helpful information."""
        parts = [self.message]
        if self.code:
            parts.append(f"(Error Code: M{self.code})")
        if self.help_text:
            parts.append(f"\n\nTroubleshooting: {self.help_text}")
        return " ".join(parts)


# API Related Errors
class MeritAPIError(MeritBaseError):
    """Base exception class for MERIT API-related errors."""
    ERROR_PREFIX = "API"
    
    def __init__(
        self, 
        message: str, 
        code: Optional[str] = None, 
        request_id: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None, 
        help_text: Optional[str] = None
    ):
        """
        Initialize the API error.
        
        Args:
            message: The error message
            code: The error code
            request_id: The API request ID
            details: Additional error details
            help_text: Helpful troubleshooting text
        """
        self.request_id = request_id
        # Prepend API to the error code
        full_code = f"{self.ERROR_PREFIX}-{code}" if code else None
        super().__init__(message, full_code, details, help_text)
    
    def __str__(self) -> str:
        """Format the API error as a string with request ID if available."""
        base_str = super().__str__()
        if self.request_id:
            return f"{base_str} (Request ID: {self.request_id})"
        return base_str


# Core Related Errors
class MeritCoreError(MeritBaseError):
    """Base exception class for MERIT core functionality errors."""
    ERROR_PREFIX = "CORE"
    
    def __init__(
        self, 
        message: str, 
        code: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None, 
        help_text: Optional[str] = None
    ):
        """
        Initialize the core error.
        
        Args:
            message: The error message
            code: The error code
            details: Additional error details
            help_text: Helpful troubleshooting text
        """
        full_code = f"{self.ERROR_PREFIX}-{code}" if code else None
        super().__init__(message, full_code, details, help_text)


# Evaluation Related Errors
class MeritEvaluationError(MeritBaseError):
    """Base exception class for MERIT evaluation errors."""
    ERROR_PREFIX = "EVAL"
    
    def __init__(
        self, 
        message: str, 
        code: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None, 
        help_text: Optional[str] = None
    ):
        """
        Initialize the evaluation error.
        
        Args:
            message: The error message
            code: The error code
            details: Additional error details
            help_text: Helpful troubleshooting text
        """
        full_code = f"{self.ERROR_PREFIX}-{code}" if code else None
        super().__init__(message, full_code, details, help_text)


# Knowledge Base Related Errors
class MeritKBError(MeritBaseError):
    """Base exception class for MERIT knowledge base errors."""
    ERROR_PREFIX = "KB"
    
    def __init__(
        self, 
        message: str, 
        code: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None, 
        help_text: Optional[str] = None
    ):
        """
        Initialize the knowledge base error.
        
        Args:
            message: The error message
            code: The error code
            details: Additional error details
            help_text: Helpful troubleshooting text
        """
        full_code = f"{self.ERROR_PREFIX}-{code}" if code else None
        super().__init__(message, full_code, details, help_text)


# TestSet Generation Related Errors
class MeritTestSetError(MeritBaseError):
    """Base exception class for MERIT test set generation errors."""
    ERROR_PREFIX = "TEST"
    
    def __init__(
        self, 
        message: str, 
        code: Optional[str] = None, 
        details: Optional[Dict[str, Any]] = None, 
        help_text: Optional[str] = None
    ):
        """
        Initialize the test set error.
        
        Args:
            message: The error message
            code: The error code
            details: Additional error details
            help_text: Helpful troubleshooting text
        """
        full_code = f"{self.ERROR_PREFIX}-{code}" if code else None
        super().__init__(message, full_code, details, help_text)


#
# Core-specific error implementations
#

class ConfigurationError(MeritCoreError):
    """Raised when there's an issue with configuration."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "001", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Invalid or missing configuration."
        help_text = (
            "The operation could not be completed due to configuration issues. "
            "Check for missing required configuration parameters. "
            "Verify that all configuration values are of the correct type and format. "
            "Ensure that any referenced files or resources exist and are accessible."
        )
        super().__init__(message or default_message, code, details, help_text)


class CacheError(MeritCoreError):
    """Raised when there's an issue with caching."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "002", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Cache operation failed."
        help_text = (
            "An error occurred during a cache operation. "
            "Check for issues with cache storage (disk space, permissions). "
            "Verify that the cached data format is valid and consistent. "
            "Consider clearing the cache and regenerating cache entries if needed."
        )
        super().__init__(message or default_message, code, details, help_text)


class ValidationError(MeritCoreError):
    """Raised when validation fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "003", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Validation failed."
        help_text = (
            "The data or parameters did not pass validation. "
            "Check for missing required fields or parameters. "
            "Verify that all values are of the correct type and format. "
            "Ensure that any constraints or requirements are satisfied."
        )
        super().__init__(message or default_message, code, details, help_text)


class ResourceError(MeritCoreError):
    """Raised when there's an issue with system resources."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "004", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Resource error."
        help_text = (
            "An error occurred related to system resources. "
            "Check for insufficient disk space, memory, or CPU resources. "
            "Verify that all required external resources are available. "
            "Consider closing other applications or freeing up system resources."
        )
        super().__init__(message or default_message, code, details, help_text)


class SerializationError(MeritCoreError):
    """Raised when serialization or deserialization fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "005", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Serialization or deserialization failed."
        help_text = (
            "An error occurred during data serialization or deserialization. "
            "Check the format of the data being serialized or deserialized. "
            "Ensure that the data structure matches the expected format. "
            "Verify that all required fields are present and correctly formatted."
        )
        super().__init__(message or default_message, code, details, help_text)


class FileOperationError(MeritCoreError):
    """Raised when a file operation fails."""
    def __init__(
        self, 
        file_path: Optional[str] = None,
        message: Optional[str] = None, 
        code: str = "006", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = f"File operation failed{' for ' + file_path if file_path else ''}."
        help_text = (
            "An error occurred during a file operation. "
            "Check for file permissions, existence, or disk space issues. "
            "Verify that the file path is correct and accessible. "
            "Ensure that any required directories exist and are writable."
        )
        if file_path and not details:
            details = {"file_path": file_path}
        elif file_path and details:
            details["file_path"] = file_path
            
        super().__init__(message or default_message, code, details, help_text)


class EnvironmentError(MeritCoreError):
    """Raised when there's an issue with the environment configuration."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "007", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Environment configuration error."
        help_text = (
            "An error occurred related to the environment configuration. "
            "Check that all required environment variables are set correctly. "
            "Verify that any .env files are properly formatted and accessible. "
            "Ensure that the runtime environment meets all requirements."
        )
        super().__init__(message or default_message, code, details, help_text)


class DependencyError(MeritCoreError):
    """Raised when there's an issue with a dependency."""
    def __init__(
        self, 
        dependency: Optional[str] = None,
        message: Optional[str] = None, 
        code: str = "008", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = f"Dependency error{' for ' + dependency if dependency else ''}."
        help_text = (
            "An error occurred related to a required dependency. "
            "Check that all required dependencies are installed and available. "
            "Verify that dependency versions are compatible with the current version of MERIT. "
            "Consider updating dependencies or installing missing ones."
        )
        if dependency and not details:
            details = {"dependency": dependency}
        elif dependency and details:
            details["dependency"] = dependency
            
        super().__init__(message or default_message, code, details, help_text)
