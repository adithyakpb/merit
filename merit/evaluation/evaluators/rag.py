"""
MERIT RAG Evaluator

This module provides evaluator classes for RAG (Retrieval-Augmented Generation) systems.
"""

import json
from typing import Dict, Any, List, Optional, Union, Callable, Sequence
from inspect import signature
from ...metrics.base import BaseMetric
from ...metrics.rag import CorrectnessMetric, FaithfulnessMetric, RelevanceMetric, CoherenceMetric, FluencyMetric
from .base import BaseEvaluator, EvaluationReport, EvaluationResult
from ...core.models import TestSet, TestItem, Response
from ...knowledge import KnowledgeBase
from ...core.logging import get_logger

logger = get_logger(__name__)

# Constants
ANSWER_FN_HISTORY_PARAM = "history"
class RAGEvaluator(BaseEvaluator):
    """
    Evaluator for RAG systems.
    
    This class evaluates a RAG system using various metrics.
    """
    
    def __init__(
        self,
        test_set: TestSet,
        metrics: Sequence[BaseMetric],
        knowledge_base: KnowledgeBase,
        llm_client=None,
        agent_description: str = "This agent is a chatbot that answers input from users.",
        context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            test_set: The test set to evaluate on
            metrics: List of metrics to evaluate
            knowledge_base: The knowledge base
            llm_client: The LLM client
            agent_description: Description of the agent
            context: Additional context for evaluation
        """
        # Initialize metrics if they are provided as strings
        initialized_metrics = self._initialize_metrics(metrics) if any(isinstance(m, str) for m in metrics) else metrics
        
        # Call the parent constructor with the required parameters
        super().__init__(test_set, initialized_metrics)
        
        self.knowledge_base = knowledge_base
        self.llm_client = llm_client
        self.agent_description = agent_description
        self.context = context or {}
    
    def _initialize_metrics(self, metrics):
        """
        Initialize metrics from strings or instances.
        
        Args:
            metrics: List of metrics to initialize
        """
        initialized_metrics = []
        for metric in metrics:
            if isinstance(metric, str):
                # Convert string to metric instance
                if metric == "correctness":
                    initialized_metrics.append(CorrectnessMetric(
                        llm_client=self.llm_client, 
                        agent_description=self.agent_description
                    ))
                elif metric == "faithfulness":
                    initialized_metrics.append(FaithfulnessMetric(
                        llm_client=self.llm_client
                    ))
                elif metric == "relevance":
                    initialized_metrics.append(RelevanceMetric(
                        llm_client=self.llm_client
                    ))
                elif metric == "coherence":
                    initialized_metrics.append(CoherenceMetric(
                        llm_client=self.llm_client
                    ))
                elif metric == "fluency":
                    initialized_metrics.append(FluencyMetric(
                        llm_client=self.llm_client
                    ))
                else:
                    logger.warning(f"Unknown metric: {metric}")
            else:
                # Assume it's already a metric instance
                initialized_metrics.append(metric)
        
        self.metrics = initialized_metrics
    
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
        # Get inputs from test set
        inputs = self._get_inputs_from_testset()
        
        if not inputs:
            logger.warning("No inputs found in test set")
            return EvaluationReport(
                results=[],
                summary={},
                metadata={"metrics": [m.name for m in metrics]}
            )
        
        # Check if system accepts history parameter
        needs_history = (
            len(signature(system).parameters) > 1 and 
            ANSWER_FN_HISTORY_PARAM in signature(system).parameters
        )
        
        # Initialize results
        results = []
        
        # Evaluate each input
        for sample in inputs:
            try:
                # Prepare kwargs for system
                kwargs = {}
                if needs_history and hasattr(sample, 'conversation_history'):
                    kwargs[ANSWER_FN_HISTORY_PARAM] = sample.conversation_history
                
                # Generate answer and context
                system_output = system(sample.input, **kwargs)
                
                # Extract response and context
                response = self._extract_response_and_context(system_output)
                
                # Evaluate with metrics
                result = self._evaluate_sample(sample, response)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating input {sample.id}: {str(e)}")
        
        # Create report
        return EvaluationReport(
            results=results,
            summary={},
            metadata={
                "num_inputs": len(inputs),
                "num_evaluated": len(results),
                "agent_description": self.agent_description,
                "knowledge_base_id": getattr(self.knowledge_base, 'id', None),
                "knowledge_base_name": getattr(self.knowledge_base, 'name', None),
                "testset_id": getattr(test_set, 'id', None),
                "testset_name": getattr(test_set, 'name', None),
                "metrics": [m.name for m in metrics]
            }
        )
    
    def _extract_response_and_context(self, system_output):
        """
        Extract response and context from the system output.
        
        The system output can be:
        1. A Response object
        2. A string (which will be converted to a Response object)
        3. A tuple of (response, context) where response is a Response object or string
        4. A dict with 'response' and 'context' keys
        
        Args:
            system_output: The output from the system
            
        Returns:
            Response: A Response object with context in documents field
        """
        response = None
        context = {}
        
        # Case 1 & 2: Response object or string
        if isinstance(system_output, (str, Response)):
            response = self._cast_to_agent_answer(system_output)
        
        # Case 3: Tuple of (response, context)
        elif isinstance(system_output, tuple) and len(system_output) == 2:
            response_content, context = system_output
            response = self._cast_to_agent_answer(response_content)
        
        # Case 4: Dict with 'response' and 'context' keys
        elif isinstance(system_output, dict) and 'response' in system_output:
            response_content = system_output['response']
            context = system_output.get('context', {})
            response = self._cast_to_agent_answer(response_content)
        
        # Default case: Try to cast to Response
        else:
            try:
                response = self._cast_to_agent_answer(system_output)
            except ValueError:
                raise ValueError(f"Cannot extract response from system output of type {type(system_output)}")
        
        # Store context in response.documents
        if context and response:
            response.documents = context if isinstance(context, list) else [context]
        
        return response
    
    def _evaluate_sample(self, test_item, response):
        """
        Evaluate a single sample with all metrics.
        
        Args:
            test_item: The test item to evaluate
            response: The model's response (with context in documents field)
            
        Returns:
            EvaluationResult: The evaluation result
        """
        # Create evaluation result
        result = EvaluationResult(
            input=test_item.input,
            response=response,
            reference=test_item.reference_answer,
            metadata={
                "document_id": test_item.document.id,
                "document_content": test_item.document.content,
                **test_item.metadata
            }
        )
        
        # Apply each metric and store the complete results
        for metric in self.metrics:
            try:
                # Pass response to the metric (context is in response.documents)
                metric_result = metric(test_item, response)
                
                # Store the complete metric result
                result.metrics.append(metric_result)
            except Exception as e:
                logger.error(f"Error applying metric {metric.name}: {str(e)}")
        
        return result
    
    def _get_inputs_from_testset(self):
        """
        Get inputs from the test set.
        
        Returns:
            List[TestItem]: The inputs
        """
        for attr in ['inputs', 'samples', 'inputs']:
            if hasattr(self.test_set, attr):
                return getattr(self.test_set, attr)
        return []
    
    def _cast_to_agent_answer(self, answer):
        """
        Cast an answer to an Response object.
        
        Args:
            answer: The answer to cast
            
        Returns:
            Response: The cast answer
        """
        if isinstance(answer, Response):
            return answer
        
        if isinstance(answer, str):
            return Response(content=answer)
        
        raise ValueError(f"The answer function must return a string or an Response object. Got {type(answer)} instead.")

def evaluate_rag(
    answer_fn: Union[Callable, Sequence[Union[Response, str]]],
    testset: Optional[TestSet] = None,
    knowledge_base: Optional[KnowledgeBase] = None,
    llm_client = None,
    agent_description: str = "This agent is a chatbot that answers input from users.",
    metrics: Optional[Sequence[Union[str, Callable]]] = None
) -> EvaluationReport:
    """
    Evaluate a RAG system.
    
    Args:
        answer_fn: A function that takes a input and optional history and returns an answer,
                  or a list of answers
        testset: The test set to evaluate on
        knowledge_base: The knowledge base to use for evaluation
        llm_client: The LLM client to use for evaluation
        agent_description: Description of the agent
        metrics: List of metrics to evaluate
        
    Returns:
        EvaluationReport: The evaluation report
    """
    # Validate inputs
    if testset is None and knowledge_base is None:
        raise ValueError("At least one of testset or knowledge base must be provided to the evaluate function.")
    
    if testset is None and not isinstance(answer_fn, Sequence):
        raise ValueError(
            "If the testset is not provided, the answer_fn must be a list of answers to ensure the matching between inputs and answers."
        )
    
    # Check basic types in case the user passed the params in the wrong order
    if knowledge_base is not None and not isinstance(knowledge_base, KnowledgeBase):
        raise ValueError(
            f"knowledge_base must be a KnowledgeBase object (got {type(knowledge_base)} instead). Are you sure you passed the parameters in the right order?"
        )
    
    if testset is not None and not isinstance(testset, TestSet):
        raise ValueError(
            f"testset must be a TestSet object (got {type(testset)} instead). Are you sure you passed the parameters in the right order?"
        )
    
    # Generate testset if not provided
    if testset is None:
        from ...testset_generation import generate_testset
        testset = generate_testset(knowledge_base)
    
    # Use default metrics if none are specified
    if metrics is None:
        metrics = ["correctness", "relevance", "faithfulness", "coherence", "fluency"]
    
    # If answer_fn is a sequence, convert it to a function
    if isinstance(answer_fn, Sequence):
        answers = [_cast_to_agent_answer(ans) for ans in answer_fn]
        answer_fn = lambda q, **kwargs: answers[0]  # Just return the first answer for now
    
    # Create evaluator
    evaluator = RAGEvaluator(
        test_set=testset,
        metrics=metrics,
        knowledge_base=knowledge_base,
        llm_client=llm_client,
        agent_description=agent_description
    )
    
    # Evaluate
    return evaluator.evaluate(answer_fn)

def _cast_to_agent_answer(answer):
    """
    Cast an answer to an Response object.
    
    Args:
        answer: The answer to cast
        
    Returns:
        Response: The cast answer
    """
    if isinstance(answer, Response):
        return answer
    
    if isinstance(answer, str):
        return Response(content=answer)
    
    raise ValueError(f"The answer function must return a string or an Response object. Got {type(answer)} instead.")
