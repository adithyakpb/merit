"""
MERIT RAG Metrics

This module provides metrics for evaluating RAG (Retrieval-Augmented Generation) systems.
"""

from .prompts import (
    CORRECTNESS_EVALUATION_PROMPT,
    FAITHFULNESS_EVALUATION_PROMPT,
    RELEVANCE_EVALUATION_PROMPT,
    COHERENCE_EVALUATION_PROMPT,
    FLUENCY_EVALUATION_PROMPT
)
from .base import BaseMetric, MetricContext, MetricCategory, register_metric
from ..core.utils import parse_json
from ..core.logging import get_logger

logger = get_logger(__name__)

class RAGMetric(BaseMetric):
    """Base class for RAG metrics."""
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY
    
    def __call__(self, input_sample, answer) -> dict:
        """
        Calculate the metric for a input-answer pair.
        
        Args:
            input_sample: The input sample
            answer: The model's answer
            
        Returns:
            dict: A dictionary with the metric name and value
        """
        raise NotImplementedError

class CorrectnessMetric(RAGMetric):
    """Metric for evaluating the correctness of an answer."""
    name = "Correctness"
    description = "Measures how accurate and correct the answer is"
    greater_is_better = True
    
    def __init__(self, llm_client=None, agent_description=None):
        """
        Initialize the correctness metric.
        
        Args:
            llm_client: The LLM client
            agent_description: Description of the agent
        """
        self._llm_client = llm_client
        self.agent_description = agent_description or "This agent is a chatbot that answers input from users."
    
    def __call__(self, input_sample, answer) -> dict:
        """
        Evaluate the correctness of an answer.
        
        Args:
            input_sample: The input sample
            answer: The model's answer (Response object or string)
            
        Returns:
            dict: A dictionary with the correctness evaluation
        """
        llm_client = self._llm_client
        
        try:
            # Get answer text and document content
            answer_text = answer.content if hasattr(answer, "content") else answer
            
            # Try to get document content from different sources
            document_content = ""
            if hasattr(input_sample, "document") and hasattr(input_sample.document, "content"):
                document_content = input_sample.document.content
            elif hasattr(input_sample, "document_content"):
                document_content = input_sample.document_content
            elif isinstance(input_sample, dict) and "document_content" in input_sample:
                document_content = input_sample["document_content"]
            
            # Use the correctness evaluation prompt
            prompt = CORRECTNESS_EVALUATION_PROMPT.safe_format(
                document_content=document_content,
                input=input_sample.input,
                reference_answer=input_sample.reference_answer,
                model_answer=answer_text
            )
            
            # Get evaluation from LLM
            response = llm_client.generate_text(prompt)
            
            # Parse response
            logger.info(f"Raw LLM response for correctness: {response}")
            json_output = parse_json(response, return_type="object")
            logger.info(f"Parsed JSON for correctness: {json_output}")
            
            # Extract score and explanation
            score = float(json_output.get("correctness_score", 0.0))
            explanation = json_output.get("explanation", "")
            errors = json_output.get("errors", [])
            
            return {
                "value": score,
                "explanation": explanation,
                "errors": errors
            }
        
        except Exception as err:
            logger.error(f"Error while evaluating correctness: {str(err)}")
            return {"value": 0.0, "explanation": f"Error: {str(err)}"}
    
class FaithfulnessMetric(RAGMetric):
    """
    Metric for evaluating the faithfulness of an answer to the retrieved documents.
    
    This metric measures how well the answer sticks to the information in the documents.
    """
    name = "Faithfulness"
    description = "Measures how faithful the answer is to the retrieved documents"
    greater_is_better = True
    
    def __init__(self, llm_client=None):
        """
        Initialize the faithfulness metric.
        
        Args:
            llm_client: The LLM client
        """
        self._llm_client = llm_client
    
    def __call__(self, input_sample, answer):
        """
        Evaluate the faithfulness of an answer.
        
        Args:
            input_sample: The input sample
            answer: The model's answer (Response object or string)
            
        Returns:
            dict: A dictionary with the faithfulness score
        """
        llm_client = self._llm_client 
        
        # Get answer text and documents
        answer_text = answer.content if hasattr(answer, "content") else answer
        
        # Try to get documents from different sources
        documents = []
        if hasattr(answer, "documents") and answer.documents:
            documents = answer.documents
        elif hasattr(input_sample, "document") and hasattr(input_sample.document, "content"):
            documents = [input_sample.document.content]
        elif hasattr(input_sample, "document_content"):
            documents = [input_sample.document_content]
        elif isinstance(input_sample, dict) and "document_content" in input_sample:
            documents = [input_sample["document_content"]]
        
        if not documents:
            logger.warning(f"No documents provided for faithfulness evaluation of input: {input_sample.input}")
            return {"value": 0.5}  # Default score when no documents are available
        
        try:
            # Use the faithfulness evaluation prompt
            prompt = FAITHFULNESS_EVALUATION_PROMPT.safe_format(
                document_content=documents[0],  # Use the first document for now
                model_answer=answer_text
            )
            
            # Get evaluation from LLM
            response = llm_client.generate_text(prompt)
            
            # Parse response
            evaluation = parse_json(response, return_type="object")
            
            # Extract score and explanation
            score = float(evaluation.get("faithfulness_score", 0.5))
            explanation = evaluation.get("explanation", "")
            hallucinations = evaluation.get("hallucinations", [])
            
            return {
                "value": score,
                "explanation": explanation,
                "hallucinations": hallucinations
            }
        
        except Exception as err:
            logger.error(f"Error while evaluating faithfulness: {str(err)}")
            return {"value": 0.5}  # Default score on error

class RelevanceMetric(RAGMetric):
    """
    Metric for evaluating the relevance of an answer to the input.
    
    This metric measures how well the answer addresses the input.
    """
    name = "Relevance"
    description = "Measures how relevant the answer is to the input"
    greater_is_better = True
    
    def __init__(self, llm_client=None):
        """
        Initialize the relevance metric.
        
        Args:
            llm_client: The LLM client
        """
        self._llm_client = llm_client
    
    def __call__(self, input_sample, answer) -> dict:
        """
        Evaluate the relevance of an answer.
        
        Args:
            input_sample: The input sample
            answer: The model's answer (Response object or string)
            
        Returns:
            dict: A dictionary with the relevance score
        """
        llm_client = self._llm_client 
        
        # Get answer text
        answer_text = answer.content if hasattr(answer, "content") else answer
        
        try:
            # Use the relevance evaluation prompt
            prompt = RELEVANCE_EVALUATION_PROMPT.safe_format(
                input=input_sample.input,
                model_answer=answer_text
            )
            
            # Get evaluation from LLM
            response = llm_client.generate_text(prompt)
            
            # Parse response
            evaluation = parse_json(response, return_type="object")
            
            # Extract score and explanation
            score = float(evaluation.get("relevance_score", 0.5))
            explanation = evaluation.get("explanation", "")
            
            return {
                "value": score,
                "explanation": explanation
            }
        
        except Exception as err:
            logger.error(f"Error while evaluating relevance: {str(err)}")
            return {"value": 0.5}  # Default score on error

class CoherenceMetric(RAGMetric):
    """
    Metric for evaluating the coherence of an answer.
    
    This metric measures how well-structured and logical the answer is.
    """
    name = "Coherence"
    description = "Measures how coherent, well-structured, and logical the answer is"
    greater_is_better = True
    
    def __init__(self, llm_client=None):
        """
        Initialize the coherence metric.
        
        Args:
            llm_client: The LLM client
        """
        self._llm_client = llm_client
    
    def __call__(self, input_sample, answer):
        """
        Evaluate the coherence of an answer.
        
        Args:
            input_sample: The input sample
            answer: The model's answer (Response object or string)
            
        Returns:
            dict: A dictionary with the coherence score
        """
        llm_client = self._llm_client 
        
        # Get answer text
        answer_text = answer.content if hasattr(answer, "content") else answer
        
        try:
            # Use the coherence evaluation prompt
            prompt = COHERENCE_EVALUATION_PROMPT.safe_format(
                model_answer=answer_text
            )
            
            # Get evaluation from LLM
            response = llm_client.generate_text(prompt)
            
            # Parse response
            evaluation = parse_json(response, return_type="object")
            
            # Extract score and explanation
            score = float(evaluation.get("coherence_score", 0.5))
            explanation = evaluation.get("explanation", "")
            
            return {
                "value": score,
                "explanation": explanation
            }
        
        except Exception as err:
            logger.error(f"Error while evaluating coherence: {str(err)}")
            return {"value": 0.5}  # Default score on error

class FluencyMetric(RAGMetric):
    """
    Metric for evaluating the fluency of an answer.
    
    This metric measures how grammatically correct and natural the answer is.
    """
    name = "Fluency"
    description = "Measures how grammatically correct and natural the answer is"
    greater_is_better = True
    
    def __init__(self, llm_client=None):
        """
        Initialize the fluency metric.
        
        Args:
            llm_client: The LLM client
        """
        self._llm_client = llm_client
    
    def __call__(self, input_sample, answer):
        """
        Evaluate the fluency of an answer.
        
        Args:
            input_sample: The input sample
            answer: The model's answer (Response object or string)
            
        Returns:
            dict: A dictionary with the fluency score
        """
        llm_client = self._llm_client
        
        # Get answer text
        answer_text = answer.content if hasattr(answer, "content") else answer
        
        try:
            # Use the fluency evaluation prompt
            prompt = FLUENCY_EVALUATION_PROMPT.safe_format(
                model_answer=answer_text
            )
            
            # Get evaluation from LLM
            response = llm_client.generate_text(prompt)
            
            # Parse response
            evaluation = parse_json(response, return_type="object")
            
            # Extract score, explanation, and errors
            score = float(evaluation.get("fluency_score", 0.5))
            explanation = evaluation.get("explanation", "")
            errors = evaluation.get("errors", [])
            
            return {
                "value": score,
                "explanation": explanation,
                "errors": errors
            }
        
        except Exception as err:
            logger.error(f"Error while evaluating fluency: {str(err)}")
            return {"value": 0.5}  # Default score on error
    
# Register metrics
register_metric(CorrectnessMetric)
register_metric(FaithfulnessMetric)
register_metric(RelevanceMetric)
register_metric(CoherenceMetric)
register_metric(FluencyMetric)
