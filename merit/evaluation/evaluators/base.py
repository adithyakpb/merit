"""
MERIT Evaluator Classes

This module provides base classes for evaluators used in evaluating Large Language Model powered systems.
"""

import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Callable, Sequence, TypeVar
from collections import defaultdict

from ...core.prompts import Prompt
from ...core.logging import get_logger
from ...core.models import EvaluationResult, EvaluationReport, TestSet, TestItem, Response
from ...metrics.base import BaseMetric

logger = get_logger(__name__)

# Type variables for better type hinting
System = TypeVar('System')
Metric = TypeVar('Metric')

class BaseEvaluator(ABC):
    """
    Base class for all evaluators.
    
    This class defines the interface and common functionality for evaluating
    systems using a set of metrics on a test set. Subclasses must implement
    the abstract methods to provide specific evaluation logic.
    """
    
    def __init__(
        self, 
        test_set: TestSet,
        metrics: Sequence[BaseMetric]
    ):
        """
        Initialize the evaluator.
        
        Args:
            metrics: List of metrics to evaluate. If None, no metrics will be used.
            test_set: Optional test set to evaluate on. Can be provided later.
        """
        self.metrics = metrics or []
        self.test_set = test_set
        
    def evaluate(self, system: Union[Callable, Dict[str, List]], test_set: Optional[TestSet] = None, metrics: Optional[Sequence[BaseMetric]] = None) -> EvaluationReport:
        """
        Evaluate the system on a test set.
        
        This method orchestrates the evaluation process by:
        1. Generating responses from the system for each TestItem in the given TestSet
           (or using pre-generated responses if system is a dictionary)
        2. Evaluating each response on the metrics
        3. Aggregating the results into an evaluation report
        
        The system can be:
        1. A callable that generates responses for inputs
        2. A dictionary with keys 'inputs', 'responses'
           where each value is a list of the same length
        
        Args:
            system: The system being evaluated
            test_set: Optional test set to evaluate on. If None, uses self.test_set
            metrics: Optional metrics to use. If None, uses self.metrics
            
        Returns:
            EvaluationReport: The evaluation report containing all results
            
        Raises:
            ValueError: If no test set is provided and self.test_set is None
        """
        # Use provided test set or fall back to self.test_set
        test_set = test_set or self.test_set
        if test_set is None:
            raise ValueError("No test set provided for evaluation")
        
        # Use provided metrics or fall back to self.metrics
        metrics = metrics or self.metrics
        
        # Handle dictionary-based system (pre-generated responses)
        if isinstance(system, dict):
            return self._evaluate_with_pregenerated(system, test_set, metrics)
        
        # Handle callable system (generate responses on the fly)
        return self._evaluate_with_callable(system, test_set, metrics)
    
    def _evaluate_with_pregenerated(self, system_dict: Dict[str, List], test_set: TestSet, metrics: Sequence[BaseMetric]) -> EvaluationReport:
        """
        Evaluate using pre-generated responses from a dictionary.
        
        Args:
            system_dict: Dictionary with keys 'inputs' and 'responses'
            test_set: The test set
            metrics: The metrics to use
            
        Returns:
            EvaluationReport: The evaluation report
        """
        # Validate the system dictionary
        if 'inputs' not in system_dict or 'responses' not in system_dict:
            raise ValueError("System dictionary must contain 'inputs' and 'responses' keys")
        
        inputs = system_dict['inputs']
        responses = system_dict['responses']
        
        if len(inputs) != len(responses):
            raise ValueError("Inputs and responses must have the same length")
        
        # Initialize results
        results = []
        
        # Evaluate each input-response pair
        for i, (input_item, response) in enumerate(zip(inputs, responses)):
            try:
                # Find the corresponding test item
                test_item = self._find_test_item(input_item, test_set)
                
                # Normalize the response
                normalized_response = self._normalize_response(response)
                
                # Evaluate with metrics
                result = self._evaluate_sample(test_item, normalized_response)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating item {i}: {str(e)}")
        
        # Create and return the report
        return self._create_evaluation_report(results)
    
    @abstractmethod
    def _evaluate_with_callable(self, system: Callable, test_set: TestSet, metrics: Sequence[BaseMetric]) -> EvaluationReport:
        """
        Evaluate using a callable system that generates responses.
        
        Args:
            system: The callable system
            test_set: The test set
            metrics: The metrics to use
            
        Returns:
            EvaluationReport: The evaluation report
        """
        raise NotImplementedError
    
    def _evaluate_batch(
        self, 
        test_items: Sequence[TestItem], 
        system: Callable
    ) -> List[EvaluationResult]:
        """
        Evaluate a batch of test items.
        
        This method:
        1. Generates responses for each test item using the system
        2. Evaluates each response using the _evaluate_sample method
        3. Collects and returns the results
        
        Args:
            test_items: The test items to evaluate
            system: The system to evaluate
            
        Returns:
            List[EvaluationResult]: The evaluation results for each test item
        """
        results = []
        
        for item in test_items:
            try:
                # Generate response from the system
                response = self._generate_response(system, item)
                
                # Evaluate the response
                result = self._evaluate_sample(item, response)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating test item {getattr(item, 'id', 'unknown')}: {str(e)}")
                # Optionally add a failed result to maintain the count
                # results.append(self._create_error_result(item, str(e)))
        
        return results
    
    @abstractmethod
    def _evaluate_sample(self, test_item: TestItem, response: Response) -> EvaluationResult:
        """
        Evaluate a single sample.
        
        This method applies all metrics to a single test item and its response,
        collecting the results into an EvaluationResult object.
        
        Args:
            test_item: The test item to evaluate
            response: The system's response to the test item
            
        Returns:
            EvaluationResult: The evaluation result for this sample
        """
        raise NotImplementedError
    
    def _find_test_item(self, input_item: Union[str, Dict, TestItem], test_set: TestSet) -> TestItem:
        """
        Find a test item in the test set that matches the given input.
        
        Args:
            input_item: The input to match (string, dict, or TestItem)
            test_set: The test set to search in
            
        Returns:
            TestItem: The matching test item
            
        Raises:
            ValueError: If no matching test item is found
        """
        # If input_item is already a TestItem, return it
        if isinstance(input_item, TestItem):
            return input_item
        
        # Get the input pnt to match
        input_content = None
        if isinstance(input_item, str):
            input_content = input_item
        elif isinstance(input_item, dict) and 'content' in input_item:
            input_content = input_item['content']
        elif hasattr(input_item, 'content'):
            input_content = input_item.content
        
        # If we have input content, try to find a matching test item
        if input_content:
            for item in test_set.inputs:
                item_content = item.input.content if hasattr(item.input, 'content') else str(item.input)
                if item_content == input_content:
                    return item
        
        # If we have an ID, try to find a matching test item
        input_id = None
        if isinstance(input_item, dict) and 'id' in input_item:
            input_id = input_item['id']
        elif hasattr(input_item, 'id'):
            input_id = input_item.id
        
        if input_id:
            for item in test_set.inputs:
                if getattr(item, 'id', None) == input_id:
                    return item
        
        # If we couldn't find a matching test item, raise an error
        raise ValueError(f"Could not find a matching test item for input: {input_item}")
    
    def _generate_response(self, system: Callable, test_item: TestItem) -> Response:
        """
        Generate a response from the system for a test item.
        
        This method handles calling the system with the appropriate arguments
        and converting the result to a Response object.
        
        Args:
            system: The system to generate a response from
            test_item: The test item to generate a response for
            
        Returns:
            Response: The system's response
            
        Raises:
            ValueError: If the system returns an invalid response type
        """
        # Extract the input from the test item
        input_content = test_item.input.content if hasattr(test_item.input, 'content') else str(test_item.input)
        
        # Call the system with the input
        raw_response = system(input_content)
        
        # Convert the response to a Response object if needed
        return self._normalize_response(raw_response)
    
    def _normalize_response(self, response: Any) -> Response:
        """
        Normalize a response to a Response object.
        
        Args:
            response: The response to normalize
            
        Returns:
            Response: The normalized response
            
        Raises:
            ValueError: If the response cannot be normalized
        """
        if isinstance(response, Response):
            return response
        
        if isinstance(response, str):
            return Response(content=response)
        
        if isinstance(response, dict) and 'content' in response:
            return Response(**response)
        
        raise ValueError(f"Cannot normalize response of type {type(response)}. Expected Response, str, or dict with 'content' key.")
    
    def _create_evaluation_report(
        self, 
        results: List[EvaluationResult], 
        metadata: Optional[Dict[str, Any]] = None
    ) -> EvaluationReport:
        """
        Create an evaluation report from a list of results.
        
        This method aggregates the results and calculates summary statistics
        for each metric.
        
        Args:
            results: The evaluation results
            metadata: Optional metadata to include in the report
            
        Returns:
            EvaluationReport: The evaluation report
        """
        # Collect metric names
        metric_names = [m.name for m in self.metrics] if hasattr(self.metrics[0], 'name') else []
        
        # Calculate summary statistics
        summary = self._calculate_summary_statistics(results, metric_names)
        
        # Create and return the report
        return EvaluationReport(
            results=results,
            summary=summary,
            metrics=metric_names,
            metadata=metadata or {}
        )
    
    def _calculate_summary_statistics(
        self, 
        results: List[EvaluationResult], 
        metric_names: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate summary statistics for each metric.
        
        Args:
            results: The evaluation results
            metric_names: The names of the metrics
            
        Returns:
            Dict[str, Dict[str, float]]: Summary statistics for each metric
        """
        summary = {}
        
        for metric_name in metric_names:
            # Collect all values for this metric
            values = []
            for result in results:
                if hasattr(result, 'metrics'):
                    # Find the metric result with the matching name
                    for metric_result in result.metrics:
                        if metric_name in metric_result:
                            values.append(metric_result[metric_name])
                            break
            
            # Calculate statistics if we have values
            if values:
                summary[metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values)
                }
        
        return summary
    
    def _create_error_result(self, test_item: TestItem, error_message: str) -> EvaluationResult:
        """
        Create an evaluation result for a failed evaluation.
        
        Args:
            test_item: The test item that failed
            error_message: The error message
            
        Returns:
            EvaluationResult: The error result
        """
        return EvaluationResult(
            input=getattr(test_item, 'input', None),
            response=None,
            reference=getattr(test_item, 'reference_answer', None),
            metadata={
                "status": "error",
                "error": error_message
            }
        )
