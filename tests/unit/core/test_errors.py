"""
Tests for the MERIT error handling system.

This module tests the error classes defined in merit.core.errors.
"""

import pytest
import re
from datetime import datetime

from merit.core.errors import (
    MeritBaseError,
    MeritAPIError,
    MeritCoreError,
    MeritEvaluationError,
    MeritKBError,
    MeritTestSetError,
    ConfigurationError,
    ValidationError,
    FileOperationError
)


class TestErrorHierarchy:
    """Tests for the error hierarchy and inheritance."""
    
    def test_base_error_inheritance(self):
        """Test that base errors inherit from Exception."""
        error = MeritBaseError("Test error")
        assert isinstance(error, Exception)
    
    def test_module_error_inheritance(self):
        """Test that module-specific errors inherit from MeritBaseError."""
        api_error = MeritAPIError("API error")
        core_error = MeritCoreError("Core error")
        eval_error = MeritEvaluationError("Evaluation error")
        kb_error = MeritKBError("Knowledge base error")
        test_error = MeritTestSetError("Test set error")
        
        assert isinstance(api_error, MeritBaseError)
        assert isinstance(core_error, MeritBaseError)
        assert isinstance(eval_error, MeritBaseError)
        assert isinstance(kb_error, MeritBaseError)
        assert isinstance(test_error, MeritBaseError)
    
    def test_specific_error_inheritance(self):
        """Test that specific errors inherit from their module errors."""
        config_error = ConfigurationError("Config error")
        validation_error = ValidationError("Validation error")
        
        assert isinstance(config_error, MeritCoreError)
        assert isinstance(validation_error, MeritCoreError)
        assert isinstance(config_error, MeritBaseError)


class TestErrorFormatting:
    """Tests for error string formatting."""
    
    def test_base_error_formatting(self):
        """Test that base errors format correctly."""
        error = MeritBaseError("Test error")
        assert str(error) == "Test error"
    
    def test_error_with_code(self):
        """Test that errors with codes include the code in their string representation."""
        error = MeritBaseError("Test error", code="123")
        assert "Test error" in str(error)
        assert "M123" in str(error)
    
    def test_error_with_help_text(self):
        """Test that errors with help text include the help text in their string representation."""
        error = MeritBaseError("Test error", help_text="Try this to fix it")
        assert "Test error" in str(error)
        assert "Troubleshooting: Try this to fix it" in str(error)
    
    def test_api_error_with_request_id(self):
        """Test that API errors include request IDs in their string representation."""
        error = MeritAPIError("API error", request_id="req-12345")
        assert "API error" in str(error)
        assert "Request ID: req-12345" in str(error)
    
    def test_error_with_all_fields(self):
        """Test that errors with all fields format correctly."""
        error = MeritAPIError(
            "API error", 
            code="001", 
            request_id="req-12345",
            help_text="Check your API key"
        )
        error_str = str(error)
        
        assert "API error" in error_str
        assert "MAPI-001" in error_str
        assert "Request ID: req-12345" in error_str
        assert "Troubleshooting: Check your API key" in error_str


class TestErrorDetails:
    """Tests for error details and metadata."""
    
    def test_error_timestamp(self):
        """Test that errors include timestamps."""
        error = MeritBaseError("Test error")
        
        # Check that the timestamp exists and is a valid ISO format
        assert hasattr(error, "timestamp")
        
        # Parse the timestamp to ensure it's valid
        try:
            datetime.fromisoformat(error.timestamp)
        except ValueError:
            pytest.fail("Error timestamp is not a valid ISO format")
    
    def test_error_details(self):
        """Test that error details are correctly stored."""
        details = {"file": "test.json", "line": 42}
        error = MeritBaseError("Test error", details=details)
        
        assert error.details == details
        assert error.details["file"] == "test.json"
        assert error.details["line"] == 42
    
    def test_file_path_in_file_operation_error(self):
        """Test that FileOperationError correctly includes the file path."""
        error = FileOperationError(file_path="/path/to/file.txt")
        
        assert "file.txt" in str(error)
        assert error.details["file_path"] == "/path/to/file.txt"


class TestErrorUsability:
    """Tests for error usability and help text."""
    
    def test_configuration_error_help_text(self):
        """Test that ConfigurationError provides useful help text."""
        error = ConfigurationError()
        
        # Help text should mention configuration parameters
        help_text = str(error)
        assert "missing required configuration parameters" in help_text.lower()
    
    def test_validation_error_help_text(self):
        """Test that ValidationError provides useful help text."""
        error = ValidationError()
        
        # Help text should mention validation
        help_text = str(error)
        assert "missing required fields" in help_text.lower()


class TestErrorPropagation:
    """Tests for error propagation and chaining."""
    
    def test_error_chaining(self):
        """Test that errors can be chained using from_exception."""
        try:
            try:
                # Original error
                raise ValueError("Original error")
            except ValueError as e:
                # Wrap in a MERIT error
                raise ConfigurationError("Configuration failed") from e
        except ConfigurationError as e:
            # Get the original cause
            assert isinstance(e.__cause__, ValueError)
            assert str(e.__cause__) == "Original error"
