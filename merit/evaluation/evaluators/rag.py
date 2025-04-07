"""
MERIT RAG Evaluator

This module provides evaluator classes for RAG (Retrieval-Augmented Generation) systems.
"""

import json
from typing import Dict, Any, List, Optional, Union, Callable, Sequence
from inspect import signature
from ..metrics.rag import CorrectnessMetric, FaithfulnessMetric, RelevanceMetric, CoherenceMetric, FluencyMetric
from .base import BaseEvaluator, EvaluationReport, EvaluationResult
from ...core.models import  TestSet, TestItem
from ...knowledge import KnowledgeBase
from ...core.logging import get_logger

logger = get_logger(__name__)

# Constants
ANSWER_FN_HISTORY_PARAM = "history"

class Response:
    """
    A class representing a response from the evaluated system.
    
    Attributes:
        content: The answer text
        documents: The documents used to generate the answer
        metadata: Additional metadata
    """
    
    def __init__(self, content, documents=None, metadata=None):
        """
        Initialize the agent answer.
        
        Args:
            content: The answer text
            documents: The documents used to generate the answer
            metadata: Additional metadata
        """
        self.content = content
        self.documents = documents
        self.metadata = metadata or {}

class RAGEvaluator(BaseEvaluator):
    """
    Evaluator for RAG systems.
    
    This class evaluates a RAG system using various metrics.
    """
    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        testset: TestSet,
        llm_client=None,
        agent_description: str = "This agent is a chatbot that answers input from users.",
        metrics: Optional[Sequence[Union[str, Callable]]] = None
    ):
        """
        Initialize the RAG evaluator.
        
        Args:
            knowledge_base: The knowledge base
            testset: The test set
            llm_client: The LLM client
            agent_description: Description of the agent
            metrics: List of metrics to evaluate
        """
        super().__init__(metrics)
        self.knowledge_base = knowledge_base
        self.testset = testset
        self.llm_client = llm_client
        self.agent_description = agent_description
        
        # Initialize metrics if provided
        if metrics:
            self._initialize_metrics(metrics)
    
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
    
    def evaluate(self, answer_fn):
        """
        Evaluate the RAG system.
        
        Args:
            answer_fn: A function that takes a input and optional history and returns an answer
            
        Returns:
            EvaluationReport: The evaluation report
        """
        # Get inputs from testset
        inputs = self._get_inputs_from_testset()
        
        if not inputs:
            logger.warning("No inputs found in testset")
            return EvaluationReport(results=[], metrics=[m.name for m in self.metrics])
        
        # Check if answer_fn accepts history parameter
        needs_history = (
            len(signature(answer_fn).parameters) > 1 and 
            ANSWER_FN_HISTORY_PARAM in signature(answer_fn).parameters
        )
        
        # Initialize results
        results = []
        
        # Evaluate each input
        for sample in inputs:
            try:
                # Prepare kwargs for answer_fn
                kwargs = {}
                if needs_history and hasattr(sample, 'conversation_history'):
                    kwargs[ANSWER_FN_HISTORY_PARAM] = sample.conversation_history
                
                # Generate answer
                answer = answer_fn(sample.input, **kwargs)
                
                # Cast answer to Response if needed
                answer = self._cast_to_agent_answer(answer)
                
                # Evaluate with metrics
                result = self._evaluate_sample(sample, answer)
                results.append(result)
            except Exception as e:
                logger.error(f"Error evaluating input {sample.id}: {str(e)}")
        
        # Create report
        return EvaluationReport(
            results=results,
            metrics=[m.name for m in self.metrics],
            metadata={
                "num_inputs": len(inputs),
                "num_evaluated": len(results),
                "agent_description": self.agent_description,
                "knowledge_base_id": getattr(self.knowledge_base, 'id', None),
                "knowledge_base_name": getattr(self.knowledge_base, 'name', None),
                "testset_id": getattr(self.testset, 'id', None),
                "testset_name": getattr(self.testset, 'name', None),
            }
        )
    
    def _evaluate_sample(self, sample, answer):
        """
        Evaluate a single sample with all metrics.
        
        Args:
            sample: The input sample
            answer: The model's answer
            
        Returns:
            EvaluationResult: The evaluation result
        """
        # Create evaluation result
        result = EvaluationResult(
            input_id=sample.id,
            input=sample.input,
            reference_answer=sample.reference_answer,
            model_answer=answer.content,
            document_id=sample.document.id,
            document_content=sample.document.content,
            metadata=sample.metadata
        )
        
        # Apply each metric
        for metric in self.metrics:
            try:
                metric_result = metric(sample, answer)
                for name, value in metric_result.items():
                    if name == metric.name:
                        result.scores[name] = value
                    elif name.endswith("_explanation"):
                        result.explanations[name.replace("_explanation", "")] = value
                    elif name.endswith("_reason"):
                        # Add this condition to also handle _reason fields
                        result.explanations[name.replace("_reason", "")] = value
                    elif name.endswith("_errors"):
                        result.errors[name.replace("_errors", "")] = value
                    elif name.endswith("_hallucinations"):
                        result.hallucinations = value
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
            if hasattr(self.testset, attr):
                return getattr(self.testset, attr)
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
        knowledge_base=knowledge_base,
        testset=testset,
        llm_client=llm_client,
        agent_description=agent_description,
        metrics=metrics
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
