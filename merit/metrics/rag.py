"""
MERIT RAG Metrics

This module provides metrics for evaluating RAG (Retrieval-Augmented Generation) systems.
"""

from datetime import datetime
from .prompts import (
    CORRECTNESS_EVALUATION_PROMPT,
    FAITHFULNESS_EVALUATION_PROMPT,
    RELEVANCE_EVALUATION_PROMPT,
    COHERENCE_EVALUATION_PROMPT,
    FLUENCY_EVALUATION_PROMPT,
    CONTEXT_PRECISION_WITH_REFERENCE_PROMPT,
    CONTEXT_PRECISION_WITHOUT_REFERENCE_PROMPT
)
from .base import MetricContext, MetricCategory, register_metric
from ..core.logging import get_logger

logger = get_logger(__name__)

from .llm_measured import LLMMeasuredBaseMetric

class CorrectnessMetric(LLMMeasuredBaseMetric):
    """Metric for evaluating the correctness of an answer."""
    name = "Correctness"
    description = "Measures how accurate and correct the answer is"
    greater_is_better = True
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY
    
    def __init__(self, llm_client=None, agent_description=None, include_raw_response=False):
        """
        Initialize the correctness metric.
        
        Args:
            llm_client: The LLM client
            agent_description: Description of the agent
            include_raw_response: Whether to include the raw LLM response in the output
        """
        self.agent_description = agent_description or "This agent is a chatbot that answers input from users."
        super().__init__(prompt=CORRECTNESS_EVALUATION_PROMPT, llm_client=llm_client, include_raw_response=include_raw_response)
    
    def _format_prompt(self, prompt, test_item, response):
        """
        Format the prompt with test item and response data.
        
        Args:
            prompt: The prompt template
            test_item: The test item
            response: The response
            
        Returns:
            str: The formatted prompt
        """
        # Get answer text
        answer_text = response.content if hasattr(response, "content") else str(response)
        
        # Try to get document content from different sources
        document_content = ""
        if hasattr(test_item, "document") and hasattr(test_item.document, "content"):
            document_content = test_item.document.content
        elif hasattr(test_item, "document_content"):
            document_content = test_item.document_content
        elif isinstance(test_item, dict) and "document_content" in test_item:
            document_content = test_item["document_content"]
        
        # Format the prompt
        try:
            return prompt.safe_format(
                document_content=document_content,
                input=test_item.input if hasattr(test_item, "input") else "",
                reference_answer=test_item.reference_answer if hasattr(test_item, "reference_answer") else "",
                model_answer=answer_text
            )
        except Exception as e:
            logger.warning(f"Error formatting correctness prompt: {e}")
            return str(prompt)
    
    
    def _get_custom_measurements(self, llm_response, test_item, response, result, llm_custom_measurements=None):
        """
        Get custom measurements based on the LLM response.
        
        Args:
            llm_response: The raw response from the LLM
            test_item: The test item
            response: The response
            result: The processed result dictionary
            llm_custom_measurements: Custom measurements provided by the LLM, if any
            
        Returns:
            dict or None: Custom measurements to add to the result
        """
        # Extract errors if present in the LLM response
        try:
            from ..core.utils import parse_json
            json_output = parse_json(llm_response, return_type="object")
            errors = json_output.get("errors", [])
            if errors:
                return {"errors": errors}
        except Exception:
            pass
        return None
    
class FaithfulnessMetric(LLMMeasuredBaseMetric):
    """
    Metric for evaluating the faithfulness of an answer to the retrieved documents.
    
    This metric measures how well the answer sticks to the information in the documents.
    """
    name = "Faithfulness"
    description = "Measures how faithful the answer is to the retrieved documents"
    greater_is_better = True
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY
    
    def __init__(self, llm_client=None, include_raw_response=False):
        """
        Initialize the faithfulness metric.
        
        Args:
            llm_client: The LLM client
            include_raw_response: Whether to include the raw LLM response in the output
        """
        super().__init__(prompt=FAITHFULNESS_EVALUATION_PROMPT, llm_client=llm_client, include_raw_response=include_raw_response)
    
    def _format_prompt(self, prompt, test_item, response):
        """
        Format the prompt with test item and response data.
        
        Args:
            prompt: The prompt template
            test_item: The test item
            response: The response
            
        Returns:
            str: The formatted prompt
        """
        # Get answer text
        answer_text = response.content if hasattr(response, "content") else str(response)
        
        # Try to get documents from different sources
        documents = []
        if hasattr(response, "documents") and response.documents:
            documents = response.documents
        elif hasattr(test_item, "document") and hasattr(test_item.document, "content"):
            documents = [test_item.document.content]
        elif hasattr(test_item, "document_content"):
            documents = [test_item.document_content]
        elif isinstance(test_item, dict) and "document_content" in test_item:
            documents = [test_item["document_content"]]
        
        if not documents:
            logger.warning(f"No documents provided for faithfulness evaluation")
            return prompt.safe_format(document_content="", model_answer=answer_text)
        
        # Format the prompt
        try:
            return prompt.safe_format(
                document_content=documents[0],  # Use the first document for now
                model_answer=answer_text
            )
        except Exception as e:
            logger.warning(f"Error formatting faithfulness prompt: {e}")
            return str(prompt)
    
    
    def _get_custom_measurements(self, llm_response, test_item, response, result, llm_custom_measurements=None):
        """
        Get custom measurements based on the LLM response.
        
        Args:
            llm_response: The raw response from the LLM
            test_item: The test item
            response: The response
            result: The processed result dictionary
            llm_custom_measurements: Custom measurements provided by the LLM, if any
            
        Returns:
            dict or None: Custom measurements to add to the result
        """
        # Extract hallucinations if present in the LLM response
        try:
            from ..core.utils import parse_json
            json_output = parse_json(llm_response, return_type="object")
            hallucinations = json_output.get("hallucinations", [])
            if hallucinations:
                return {"hallucinations": hallucinations}
        except Exception:
            pass
        return None

class RelevanceMetric(LLMMeasuredBaseMetric):
    """
    Metric for evaluating the relevance of an answer to the input.
    
    This metric measures how well the answer addresses the input.
    """
    name = "Relevance"
    description = "Measures how relevant the answer is to the input"
    greater_is_better = True
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY
    
    def __init__(self, llm_client=None, include_raw_response=False):
        """
        Initialize the relevance metric.
        
        Args:
            llm_client: The LLM client
            include_raw_response: Whether to include the raw LLM response in the output
        """
        super().__init__(prompt=RELEVANCE_EVALUATION_PROMPT, llm_client=llm_client, include_raw_response=include_raw_response)
    
    def _format_prompt(self, prompt, test_item, response):
        """
        Format the prompt with test item and response data.
        
        Args:
            prompt: The prompt template
            test_item: The test item
            response: The response
            
        Returns:
            str: The formatted prompt
        """
        # Get answer text
        answer_text = response.content if hasattr(response, "content") else str(response)
        
        # Format the prompt
        try:
            return prompt.safe_format(
                input=test_item.input if hasattr(test_item, "input") else "",
                model_answer=answer_text
            )
        except Exception as e:
            logger.warning(f"Error formatting relevance prompt: {e}")
            return str(prompt)
    

class CoherenceMetric(LLMMeasuredBaseMetric):
    """
    Metric for evaluating the coherence of an answer.
    
    This metric measures how well-structured and logical the answer is.
    """
    name = "Coherence"
    description = "Measures how coherent, well-structured, and logical the answer is"
    greater_is_better = True
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY
    
    def __init__(self, llm_client=None, include_raw_response=False):
        """
        Initialize the coherence metric.
        
        Args:
            llm_client: The LLM client
            include_raw_response: Whether to include the raw LLM response in the output
        """
        super().__init__(prompt=COHERENCE_EVALUATION_PROMPT, llm_client=llm_client, include_raw_response=include_raw_response)
    
    def _format_prompt(self, prompt, test_item, response):
        """
        Format the prompt with test item and response data.
        
        Args:
            prompt: The prompt template
            test_item: The test item
            response: The response
            
        Returns:
            str: The formatted prompt
        """
        # Get answer text
        answer_text = response.content if hasattr(response, "content") else str(response)
        
        # Format the prompt
        try:
            return prompt.safe_format(
                model_answer=answer_text
            )
        except Exception as e:
            logger.warning(f"Error formatting coherence prompt: {e}")
            return str(prompt)
    

class FluencyMetric(LLMMeasuredBaseMetric):
    """
    Metric for evaluating the fluency of an answer.
    
    This metric measures how grammatically correct and natural the answer is.
    """
    name = "Fluency"
    description = "Measures how grammatically correct and natural the answer is"
    greater_is_better = True
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY
    
    def __init__(self, llm_client=None, include_raw_response=False):
        """
        Initialize the fluency metric.
        
        Args:
            llm_client: The LLM client
            include_raw_response: Whether to include the raw LLM response in the output
        """
        super().__init__(prompt=FLUENCY_EVALUATION_PROMPT, llm_client=llm_client, include_raw_response=include_raw_response)
    
    def _format_prompt(self, prompt, test_item, response):
        """
        Format the prompt with test item and response data.
        
        Args:
            prompt: The prompt template
            test_item: The test item
            response: The response
            
        Returns:
            str: The formatted prompt
        """
        # Get answer text
        answer_text = response.content if hasattr(response, "content") else str(response)
        
        # Format the prompt
        try:
            return prompt.safe_format(
                model_answer=answer_text
            )
        except Exception as e:
            logger.warning(f"Error formatting fluency prompt: {e}")
            return str(prompt)
    
    
    def _get_custom_measurements(self, llm_response, test_item, response, result, llm_custom_measurements=None):
        """
        Get custom measurements based on the LLM response.
        
        Args:
            llm_response: The raw response from the LLM
            test_item: The test item
            response: The response
            result: The processed result dictionary
            llm_custom_measurements: Custom measurements provided by the LLM, if any
            
        Returns:
            dict or None: Custom measurements to add to the result
        """
        # Extract errors if present in the LLM response
        try:
            from ..core.utils import parse_json
            json_output = parse_json(llm_response, return_type="object")
            errors = json_output.get("errors", [])
            if errors:
                return {"errors": errors}
        except Exception:
            pass
        return None
    
class ContextPrecisionMetric(LLMMeasuredBaseMetric):
    """
    Metric for evaluating the precision of retrieved contexts.
    
    This metric measures the proportion of relevant chunks in the retrieved contexts.
    It can operate in different modes:
    1. LLM-based with reference answer
    2. LLM-based without reference (comparing to response)
    3. Non-LLM-based with reference contexts (using similarity measures)
    
    The mode is determined by the parameters provided during initialization and call.
    """
    name = "ContextPrecision"
    description = "Measures the precision of retrieved contexts"
    greater_is_better = True
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY
    
    def __init__(
        self,
        llm_client=None,
        use_llm=True,
        similarity_threshold=0.7,
        similarity_measure="cosine",
        include_raw_response=False,
        prompt=None
    ):
        """
        Initialize the context precision metric.
        
        Args:
            llm_client: The LLM client to use for evaluation
            use_llm: Whether to use LLM for relevance determination
            similarity_threshold: Threshold for non-LLM similarity
            similarity_measure: Similarity measure to use for non-LLM comparison
            include_raw_response: Whether to include raw LLM response
            prompt: Custom prompt to use for LLM evaluation
        """
        # Initialize with default prompt (will be selected in _format_prompt based on parameters)
        super().__init__(prompt=prompt, llm_client=llm_client, include_raw_response=include_raw_response)
        
        self.use_llm = use_llm
        self.similarity_threshold = similarity_threshold
        self.similarity_measure = similarity_measure
    
    def __call__(self, test_item, response, client_llm_callable=None, prompt=None):
        """
        Calculate the context precision.
        
        Args:
            test_item: The test item containing input and reference
            response: The response from the system
            client_llm_callable: Optional callable to override the stored LLM client
            prompt: Optional prompt to override the stored prompt
            
        Returns:
            Dict: The metric result
        """
        # Determine if we have reference answer or contexts
        has_reference_answer = (hasattr(test_item, "reference_answer") and test_item.reference_answer is not None)
        has_reference_contexts = (hasattr(test_item, "reference_contexts") and test_item.reference_contexts is not None)
        
        # Get retrieved contexts
        retrieved_contexts = []
        if hasattr(response, "documents") and response.documents:
            retrieved_contexts = response.documents
        elif hasattr(response, "contexts") and response.contexts:
            retrieved_contexts = response.contexts
        
        if not retrieved_contexts:
            logger.warning("No retrieved contexts found for context precision evaluation")
            return {
                "value": 0.0,
                "explanation": "No retrieved contexts found",
                "metric_name": self.name,
                "timestamp": datetime.now().isoformat()
            }
        
        # Choose evaluation method based on parameters and available data
        if self.use_llm:
            # Use LLM-based evaluation
            if has_reference_answer:
                return self._evaluate_with_llm(test_item, response, retrieved_contexts, 
                                              use_reference=True, client_llm_callable=client_llm_callable, prompt=prompt)
            else:
                return self._evaluate_with_llm(test_item, response, retrieved_contexts, 
                                              use_reference=False, client_llm_callable=client_llm_callable, prompt=prompt)
        else:
            # Use non-LLM evaluation
            if has_reference_contexts:
                return self._evaluate_with_similarity(test_item, response, retrieved_contexts)
            else:
                logger.warning("Non-LLM evaluation requires reference contexts")
                return {
                    "value": 0.0,
                    "explanation": "Non-LLM evaluation requires reference contexts",
                    "metric_name": self.name,
                    "timestamp": datetime.now().isoformat()
                }
    
    def _evaluate_with_llm(self, test_item, response, retrieved_contexts, use_reference=True, client_llm_callable=None, prompt=None):
        """
        Evaluate context precision using LLM.
        
        Args:
            test_item: The test item
            response: The response
            retrieved_contexts: The retrieved contexts
            use_reference: Whether to use reference answer
            client_llm_callable: Optional callable to override the stored LLM client
            prompt: Optional prompt to override the stored prompt
            
        Returns:
            Dict: The metric result
        """
        # Use provided callable or stored client
        llm_callable = client_llm_callable or (self._llm_client.generate_text if self._llm_client else None)
        if not llm_callable:
            raise ValueError("No LLM client provided for metric calculation")
        
        # Select appropriate prompt
        used_prompt = prompt or self.prompt
        if used_prompt is None:
            if use_reference:
                used_prompt = CONTEXT_PRECISION_WITH_REFERENCE_PROMPT
            else:
                used_prompt = CONTEXT_PRECISION_WITHOUT_REFERENCE_PROMPT
        
        # Evaluate each context
        relevance_scores = []
        relevant_contexts = []
        irrelevant_contexts = []
        explanations = []
        
        for i, context in enumerate(retrieved_contexts):
            # Format prompt for this context
            formatted_prompt = self._format_context_prompt(
                used_prompt, 
                test_item, 
                response, 
                context, 
                use_reference
            )
            
            # Call LLM
            llm_response = llm_callable(formatted_prompt)
            
            # Process response
            try:
                from ..core.utils import parse_json
                result = parse_json(llm_response, return_type="object")
                
                # Extract relevance information
                is_relevant = result.get("is_relevant", False)
                relevance_score = float(result.get("relevance_score", 0.0))
                explanation = result.get("explanation", "")
                
                relevance_scores.append(relevance_score)
                explanations.append(f"Context {i+1}: {explanation}")
                
                if is_relevant:
                    relevant_contexts.append(context)
                else:
                    irrelevant_contexts.append(context)
                
            except Exception as e:
                logger.error(f"Error processing LLM response for context {i+1}: {str(e)}")
                relevance_scores.append(0.0)
                explanations.append(f"Context {i+1}: Error processing LLM response")
        
        # Calculate overall precision
        if not relevance_scores:
            precision = 0.0
        else:
            precision = sum(relevance_scores) / len(relevance_scores)
        
        # Create result
        result = {
            "value": precision,
            "explanation": "\n".join(explanations),
            "relevant_contexts_count": len(relevant_contexts),
            "irrelevant_contexts_count": len(irrelevant_contexts),
            "context_scores": relevance_scores,
            "metric_name": self.name,
            "timestamp": datetime.now().isoformat()
        }
        
        if self._include_raw_response:
            result["raw_llm_response"] = llm_response
        
        return result
    
    def _evaluate_with_similarity(self, test_item, response, retrieved_contexts):
        """
        Evaluate context precision using similarity measures.
        
        Args:
            test_item: The test item
            response: The response
            retrieved_contexts: The retrieved contexts
            
        Returns:
            Dict: The metric result
        """
        # Get reference contexts
        reference_contexts = []
        if hasattr(test_item, "reference_contexts") and test_item.reference_contexts:
            reference_contexts = test_item.reference_contexts
        else:
            logger.warning("No reference contexts found for non-LLM context precision evaluation")
            return {
                "value": 0.0,
                "explanation": "No reference contexts found",
                "metric_name": self.name,
                "timestamp": datetime.now().isoformat()
            }
        
        # Calculate similarity for each retrieved context
        relevance_scores = []
        relevant_contexts = []
        irrelevant_contexts = []
        explanations = []
        
        for i, retrieved_context in enumerate(retrieved_contexts):
            # Find best matching reference context
            best_similarity = 0.0
            best_reference = None
            
            for ref_context in reference_contexts:
                similarity = self._calculate_similarity(retrieved_context, ref_context)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_reference = ref_context
            
            # Determine if relevant based on similarity threshold
            is_relevant = best_similarity >= self.similarity_threshold
            
            relevance_scores.append(best_similarity)
            explanation = f"Context {i+1}: Similarity {best_similarity:.2f} (threshold: {self.similarity_threshold})"
            explanations.append(explanation)
            
            if is_relevant:
                relevant_contexts.append(retrieved_context)
            else:
                irrelevant_contexts.append(retrieved_context)
        
        # Calculate overall precision
        if not relevance_scores:
            precision = 0.0
        else:
            precision = sum(relevance_scores) / len(relevance_scores)
        
        # Create result
        result = {
            "value": precision,
            "explanation": "\n".join(explanations),
            "relevant_contexts_count": len(relevant_contexts),
            "irrelevant_contexts_count": len(irrelevant_contexts),
            "context_scores": relevance_scores,
            "metric_name": self.name,
            "timestamp": datetime.now().isoformat()
        }
        
        return result
    
    def _format_context_prompt(self, prompt, test_item, response, context, use_reference):
        """
        Format the prompt for a specific context.
        
        Args:
            prompt: The prompt template
            test_item: The test item
            response: The response
            context: The context to evaluate
            use_reference: Whether to use reference answer
            
        Returns:
            str: The formatted prompt
        """
        # Get input text
        input_text = ""
        if hasattr(test_item, "input"):
            if hasattr(test_item.input, "content"):
                input_text = test_item.input.content
            else:
                input_text = str(test_item.input)
        
        # Get response text
        response_text = ""
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)
        
        # Get reference answer text
        reference_text = ""
        if hasattr(test_item, "reference_answer"):
            if hasattr(test_item.reference_answer, "content"):
                reference_text = test_item.reference_answer.content
            else:
                reference_text = str(test_item.reference_answer)
        
        # Get context text
        context_text = context
        if hasattr(context, "content"):
            context_text = context.content
        
        # Format the prompt
        try:
            if use_reference:
                return prompt.safe_format(
                    user_input=input_text,
                    reference_answer=reference_text,
                    retrieved_context=context_text
                )
            else:
                return prompt.safe_format(
                    user_input=input_text,
                    system_response=response_text,
                    retrieved_context=context_text
                )
        except Exception as e:
            logger.warning(f"Error formatting context precision prompt: {e}")
            return str(prompt)
    
    def _calculate_similarity(self, text1, text2):
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            float: Similarity score
        """
        # Extract text content if needed
        if hasattr(text1, "content"):
            text1 = text1.content
        if hasattr(text2, "content"):
            text2 = text2.content
        
        # Convert to string if needed
        text1 = str(text1)
        text2 = str(text2)
        
        # Use appropriate similarity measure
        if self.similarity_measure == "cosine":
            from ..core.utils import cosine_similarity
            
            # If we have embeddings from the client, use them
            if self._llm_client and hasattr(self._llm_client, "get_embeddings"):
                try:
                    emb1 = self._llm_client.get_embeddings(text1)[0]
                    emb2 = self._llm_client.get_embeddings(text2)[0]
                    return cosine_similarity(emb1, emb2)
                except Exception as e:
                    logger.warning(f"Error getting embeddings: {e}")
            
            # Fallback to simple token overlap
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            
            return len(intersection) / len(union)
        
        elif self.similarity_measure == "jaccard":
            # Simple Jaccard similarity
            tokens1 = set(text1.lower().split())
            tokens2 = set(text2.lower().split())
            
            if not tokens1 or not tokens2:
                return 0.0
            
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            
            return len(intersection) / len(union)
        
        else:
            logger.warning(f"Unknown similarity measure: {self.similarity_measure}")
            return 0.0

# Register metrics
register_metric(CorrectnessMetric)
register_metric(FaithfulnessMetric)
register_metric(RelevanceMetric)
register_metric(CoherenceMetric)
register_metric(FluencyMetric)
register_metric(ContextPrecisionMetric)
