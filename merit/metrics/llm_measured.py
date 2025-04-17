"""
MERIT Classification Metrics

This module provides metrics for evaluating classification-based systems.
"""

from abc import ABCMeta
from datetime import datetime
from .base import BaseMetric, MetricContext, MetricCategory
from ..core.logging import get_logger
from ..core.models import TestItem  
from ..core.prompts import Prompt
from ..api.client import AIAPIClient

logger = get_logger(__name__)

class LLMMeasuredBaseMetric(BaseMetric, metaclass=ABCMeta):
    """
    Base class for metrics that are calculated using an LLM.
    
    This class provides a standardized way to implement metrics that use an LLM
    to evaluate responses. It handles prompt management and provides a consistent
    response format.
    
    Standard Output Format:
    ----------------------
    {
        "value": float,                # The primary metric value (required)
        "explanation": str,            # Explanation of the score (optional)
        "metric_name": str,            # Name of the metric (added automatically)
        "timestamp": str,              # ISO timestamp (added automatically)
        "raw_llm_response": str,       # The raw response from the LLM (included if needed)
        "custom_measurements": dict    # Any additional data (optional)
    }
    """
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY
    
    # Default value to use if the score field is not found
    default_score = 0.5
    
    # Store the custom name if provided
    _custom_name = None
    
    @property
    def name(self):
        """
        Get the name of the metric.
        
        If a custom name was provided in the constructor, that name is used.
        Otherwise, if the class has a name attribute, that is used.
        If neither is available, the class name is used.
        
        Returns:
            str: The name of the metric
        """
        if self._custom_name:
            return self._custom_name
        
        # Check if the class has a name attribute
        class_name = getattr(self.__class__, 'name', None)
        if class_name:
            return class_name
            
        # Use the class name as a fallback
        return self.__class__.__name__.replace('Metric', '')
    
    @name.setter
    def name(self, value):
        """
        Set a custom name for the metric.
        
        Args:
            value: The custom name
        """
        self._custom_name = value
    
    # No separate score_field property needed - we'll use the name property directly
    
    def __init__(self, prompt: "Prompt" = None, llm_client: AIAPIClient = None, include_raw_response: bool = False):
        """
        Initialize the LLM-measured metric.
        
        Args:
            prompt: The prompt to use for the evaluation
            llm_client: The LLM client to use for generating text
            include_raw_response: Whether to include the raw LLM response in the output
        """
        self.prompt = prompt
        self._llm_client = llm_client
        self._include_raw_response = include_raw_response
    
    def __call__(self, test_item: TestItem, response, client_llm_callable: callable = None, prompt: "Prompt" = None):
        """
        Calculate the metric using an LLM call.
        
        Args:
            test_item: The test item to evaluate
            response: The evaluated system's response
            client_llm_callable: Optional callable to override the stored LLM client
            prompt: Optional prompt to override the stored prompt
            
        Returns:
            dict: A dictionary with the metric value and additional information
        """
        # Use provided callable or stored client
        llm_callable = client_llm_callable or (self._llm_client.generate_text if self._llm_client else None)
        if not llm_callable:
            raise ValueError("No LLM client provided for metric calculation")
            
        # Use provided prompt or stored prompt
        used_prompt = prompt or self.prompt
        if not used_prompt:
            raise ValueError("No prompt provided for metric calculation")
            
        # Format the prompt with standard variables
        formatted_prompt = self._format_prompt(used_prompt, test_item, response)
        
        # Call the LLM
        llm_response = llm_callable(formatted_prompt)
        
        # Process the LLM response
        result = self._process_llm_response(llm_response)
        
        # Always include the raw LLM response for custom measurements
        if "raw_llm_response" not in result:
            result["raw_llm_response"] = llm_response
        
        # Extract any LLM-provided custom measurements
        llm_custom_measurements = result.pop("custom_measurements", None)
        
        # Get additional custom measurements from the implementation
        impl_custom_measurements = self._get_custom_measurements(
            llm_response=llm_response,
            test_item=test_item, 
            response=response, 
            result=result,
            llm_custom_measurements=llm_custom_measurements
        )
        
        # Combine LLM and implementation custom measurements
        combined_measurements = {}
        if llm_custom_measurements:
            combined_measurements.update(llm_custom_measurements)
        if impl_custom_measurements:
            combined_measurements.update(impl_custom_measurements)
            
        # Add combined custom measurements if any exist
        if combined_measurements:
            result["custom_measurements"] = combined_measurements
            
        # Add standard metadata
        result["metric_name"] = self.name
        result["timestamp"] = datetime.now().isoformat()
        
        # Remove raw LLM response if not needed for the final output
        if not self._include_raw_response:
            result.pop("raw_llm_response", None)
            
        return result
    
    def _format_prompt(self, prompt, test_item, response):
        """
        Format the prompt with test item and response data.
        
        Override this method to customize how the prompt is formatted.
        
        Args:
            prompt: The prompt template
            test_item: The test item
            response: The response
            
        Returns:
            str: The formatted prompt
        """
        # Get response text
        response_text = response.content if hasattr(response, "content") else str(response)
        
        # Format with standard variables
        try:
            return prompt.safe_format(
                input=test_item.input if hasattr(test_item, "input") else "",
                response=response_text,
                model_answer=response_text  # For backward compatibility
            )
        except Exception as e:
            logger.warning(f"Error formatting prompt: {e}")
            # Fallback to basic formatting
            return prompt.template.format(response=response_text)
    
    def _process_llm_response(self, llm_response):
        """
        Process the LLM response to extract the metric value.
        
        Override this method to customize how the LLM response is processed.
        The method automatically looks for a field named "{name.lower()}"
        in the JSON response (e.g., "correctness" for a metric named "Correctness").
        
        Args:
            llm_response: The raw response from the LLM
            test_item: The test item
            response: The response
            
        Returns:
            dict: A dictionary with at least a "value" key
        """
        # Default implementation tries to parse as a number or JSON
        try:
            # Try parsing as a float first
            value = float(llm_response.strip())
            return {"value": value}
        except ValueError:
            pass
            
        # Try parsing as JSON
        try:
            from ..core.utils import parse_json
            result = parse_json(llm_response, return_type="object")
            
            # Convert to dict if it's not already
            if not isinstance(result, dict):
                result = {k: getattr(result, k) for k in dir(result) 
                         if not k.startswith('_') and not callable(getattr(result, k))}
            
            # Construct the expected field name from the metric name (e.g., "correctness" for "Correctness" metric)
            expected_field = self.name.lower()
            if expected_field in result:
                score = float(result.get(expected_field, self.default_score))
                result["value"] = score
            else:
                # If the expected field is not found, use the default score
                result["value"] = self.default_score
                logger.warning(f"Expected field '{expected_field}' not found in LLM response")
                
            return result
        except Exception as err:
            # If all else fails, return a default value and the raw response
            logger.error(f"Error processing LLM response: {str(err)}")
            return {
                "value": self.default_score, 
                "explanation": f"Error processing response: {str(err)}", 
                "raw_llm_response": llm_response
            }
    
    def _get_custom_measurements(self, llm_response, test_item, response, result, llm_custom_measurements=None):
        """
        Get custom measurements based on the LLM response.
        
        Override this method to add custom measurements to the result.
        
        Args:
            llm_response: The raw response from the LLM
            test_item: The test item
            response: The response
            result: The processed result dictionary
            llm_custom_measurements: Custom measurements provided by the LLM, if any
            
        Returns:
            dict or None: Custom measurements to add to the result
        """
        return None
