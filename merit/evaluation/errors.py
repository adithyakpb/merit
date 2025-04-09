"""
MERIT Evaluation Error Classes

This module defines specific evaluation-related error classes for the MERIT system.
"""

from typing import Dict, Any, Optional
from merit.core.errors import MeritEvaluationError


class EvaluationMetricError(MeritEvaluationError):
    """Raised when a metric calculation fails."""
    def __init__(
        self, 
        metric_name: Optional[str] = None, 
        message: Optional[str] = None, 
        code: str = "001", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = f"Failed to calculate {metric_name or 'metric'} during evaluation."
        help_text = (
            "This typically occurs when the metric can't process the inputs or outputs correctly. "
            "Verify that your model outputs are in the expected format for this metric. "
            "Check for any missing required fields in your test samples. "
            "Consider enabling debug logging for more detailed error information."
        )
        if metric_name and not details:
            details = {"metric_name": metric_name}
        elif metric_name and details:
            details["metric_name"] = metric_name
            
        super().__init__(message or default_message, code, details, help_text)


class EvaluationLLMError(MeritEvaluationError):
    """Raised when the evaluation LLM fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "002", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Evaluation LLM failed to process the request."
        help_text = (
            "The LLM used for evaluation encountered an error. "
            "Check your LLM client configuration and API credentials. "
            "Verify that the evaluation prompts are properly formatted. "
            "Consider switching to a different LLM provider or model if the issue persists."
        )
        super().__init__(message or default_message, code, details, help_text)


class EvaluationInputError(MeritEvaluationError):
    """Raised when there's an issue with evaluation inputs."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "003", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Invalid or missing evaluation inputs."
        help_text = (
            "The evaluation could not be completed due to issues with the input data. "
            "Ensure your test set contains valid inputs and reference answers. "
            "Check that all required fields are present in your test samples. "
            "Verify that the format of inputs matches what the evaluator expects."
        )
        super().__init__(message or default_message, code, details, help_text)


class EvaluationConfigError(MeritEvaluationError):
    """Raised when there's an issue with the evaluation configuration."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "004", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Invalid evaluation configuration."
        help_text = (
            "The evaluation configuration is invalid or incomplete. "
            "Ensure all required parameters for the evaluator are provided. "
            "Check that the specified metrics are valid and available. "
            "Verify that the evaluation parameters are compatible with each other."
        )
        super().__init__(message or default_message, code, details, help_text)


class EvaluationReportError(MeritEvaluationError):
    """Raised when there's an issue with generating or processing an evaluation report."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "005", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to generate or process evaluation report."
        help_text = (
            "An error occurred while generating or processing the evaluation report. "
            "Check that all evaluation results are valid and complete. "
            "Ensure the report template is valid and accessible. "
            "Verify that you have permission to save the report if writing to a file."
        )
        super().__init__(message or default_message, code, details, help_text)


class MetricImplementationError(MeritEvaluationError):
    """Raised when there's an issue with a metric implementation."""
    def __init__(
        self, 
        metric_name: Optional[str] = None, 
        message: Optional[str] = None, 
        code: str = "006", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = f"Implementation error in {metric_name or 'metric'}."
        help_text = (
            "The metric implementation has an error or is incomplete. "
            "Check the metric implementation for any bugs or missing functionality. "
            "Ensure the metric is compatible with the current version of MERIT. "
            "Consider updating the metric implementation or using a different metric."
        )
        if metric_name and not details:
            details = {"metric_name": metric_name}
        elif metric_name and details:
            details["metric_name"] = metric_name
            
        super().__init__(message or default_message, code, details, help_text)


class ResultParsingError(MeritEvaluationError):
    """Raised when parsing evaluation results fails."""
    def __init__(
        self, 
        message: Optional[str] = None, 
        code: str = "007", 
        details: Optional[Dict[str, Any]] = None
    ):
        default_message = "Failed to parse evaluation results."
        help_text = (
            "An error occurred while parsing the evaluation results. "
            "This typically happens when the LLM output doesn't match the expected format. "
            "Check the LLM output format and ensure it matches what the parser expects. "
            "Consider updating your prompts to ensure the LLM generates properly formatted output."
        )
        super().__init__(message or default_message, code, details, help_text)
