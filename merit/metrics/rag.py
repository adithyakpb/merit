"""
MERIT RAG Metrics

This module provides metrics for evaluating RAG (Retrieval-Augmented Generation) systems.
"""

from datetime import datetime
from .prompts import (
    CORRECTNESS_EVALUATION_PROMPT,
    FAITHFULNESS_EVALUATION_PROMPT, # Original prompt, will be unused by modified FaithfulnessMetric
    RELEVANCE_EVALUATION_PROMPT,    # Original prompt, will be unused by modified RelevanceMetric
    COHERENCE_EVALUATION_PROMPT,
    FLUENCY_EVALUATION_PROMPT,
    CONTEXT_PRECISION_WITH_REFERENCE_PROMPT,
    CONTEXT_PRECISION_WITHOUT_REFERENCE_PROMPT,
    CONTEXT_RECALL_PROMPT, # New
    CLAIM_EXTRACTION_PROMPT, # New
    CLAIM_VERIFICATION_PROMPT, # New
    RESPONSE_QUESTION_GENERATION_PROMPT # New
)
from ..core.utils import parse_json, cosine_similarity # Ensure these are available
from .base import MetricContext, MetricCategory, register_metric
from ..core.logging import get_logger

logger = get_logger(__name__)

from .llm_measured import LLMMeasuredBaseMetric
from ..core.models import Input, Response
from ..core.prompts import Prompt
from ..monitoring.models import LLMRequest, LLMResponse

class CorrectnessMetric(LLMMeasuredBaseMetric):
    """Metric for evaluating the correctness of an answer in both monitoring and evaluation contexts."""
    name = "Correctness"
    description = "Measures how accurate and correct the answer is"
    greater_is_better = True
    context = MetricContext.BOTH  # Works in both contexts
    category = MetricCategory.QUALITY
    
    # Model-based requirements
    monitoring_requires = {
        "request": LLMRequest,
        "response": LLMResponse
    }
    evaluation_requires = {
        "input": Input,
        "response": Response
    }
    
    def __init__(self, mode=None, llm_client=None, agent_description=None, include_raw_response=False):
        """
        Initialize the correctness metric.
        
        Args:
            mode: The context mode for this metric instance
            llm_client: The LLM client
            agent_description: Description of the agent
            include_raw_response: Whether to include the raw LLM response in the output
        """
        super().__init__(prompt=CORRECTNESS_EVALUATION_PROMPT, llm_client=llm_client, include_raw_response=include_raw_response, mode=mode)
        self.agent_description = agent_description or "This agent is a chatbot that answers input from users."
    
    def calculate_monitoring(self, request: LLMRequest, response: LLMResponse) -> dict:
        """
        Calculate correctness in monitoring context.
        
        Args:
            request: LLMRequest object
            response: LLMResponse object
            
        Returns:
            Dict containing metric result
        """
        if not self.llm_client:
            raise ValueError("LLM client required for correctness monitoring")
        
        # LLM-based evaluation for monitoring
        prompt = CORRECTNESS_EVALUATION_PROMPT.safe_format(
            document_content=getattr(response, 'context', ""),  # Context if available
            input=request.input.content,
            reference_answer="",  # No reference in monitoring
            model_answer=response.completion
        )
        
        try:
            llm_response = self.llm_client.generate_text(prompt)
            from ..core.utils import parse_json
            result = parse_json(llm_response, return_type="object")
            
            score = float(result.get("score", 0.0))
            explanation = result.get("explanation", "LLM evaluation")
            
            metric_result = {
                "value": score,
                "explanation": explanation,
                "method": "llm_evaluation",
                "timestamp": datetime.now().isoformat()
            }
            
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            
            return metric_result
            
        except Exception as e:
            logger.error(f"Error in LLM-based correctness evaluation: {e}")
            return {
                "value": 0.0,
                "explanation": f"Error in evaluation: {str(e)}",
                "method": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_evaluation(self, input_obj: Input, response: Response, reference: Response, llm_client=None) -> dict:
        """
        Calculate correctness in evaluation context.
        
        Args:
            input_obj: Input object
            response_obj: Response object
            llm_client: Optional LLM client for evaluation
            
        Returns:
            Dict containing metric result
        """
        # Use provided client or stored client
        client = llm_client or self._llm_client
        
        if not client:
            #raise a MERIT error. do this every where
            raise ValueError("LLM client required for correctness evaluation")
        
        # Format prompt for evaluation context
        prompt = CORRECTNESS_EVALUATION_PROMPT.safe_format(
            document_content=str(response.documents),  
            input=input_obj.content,
            reference_answer=reference.content if reference else "",
            model_answer=response.content
        )
        
        try:
            llm_response = client.generate_text(prompt)
            from ..core.utils import parse_json
            result = parse_json(llm_response, return_type="object")
            
            score = float(result.get("correctness", 0.0)) # Changed here
            explanation = result.get("explanation", "LLM evaluation")
            
            metric_result = {
                "value": score,
                "explanation": explanation,
                "method": "llm_evaluation",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}#TODO get some metadata if needed
            }
            
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            
            return metric_result
            
        except Exception as e:
            logger.error(f"Error in correctness evaluation: {e}")
            return {
                "value": 0.0,
                "explanation": f"Error in evaluation: {str(e)}",
                "method": "error",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            }
    
    def _format_prompt(self, prompt: Prompt, test_item, response):
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
    
    
class FaithfulnessMetric(LLMMeasuredBaseMetric):
    # ... (name, description, __init__ etc. remain the same)

    def _calculate_evaluation_primary(self, input_obj: Input, response: Response, client) -> dict: # Changed response_obj to response
        model_answer = response.content # Changed from response_obj
        if not model_answer.strip():
            logger.warning(f"{self.name} (Primary): Model answer is empty.")
            raise ValueError("Model answer is empty for primary calculation")

        if not response.documents: # Changed from response_obj
            logger.warning(f"{self.name} (Primary): No documents for context.")
            raise ValueError("No context documents for primary calculation")
        
        retrieved_context_str = "\n\n---\n\n".join([doc.content for doc in response.documents if hasattr(doc, 'content') and doc.content]) # Changed from response_obj
        if not retrieved_context_str.strip():
            logger.warning(f"{self.name} (Primary): Context documents are empty.")
            raise ValueError("Context documents are empty for primary calculation")

        # ... (rest of the claims extraction logic remains the same, using model_answer and retrieved_context_str)
        claims = []
        raw_claim_extraction_response = None
        try:
            formatted_extraction_prompt = self.claim_extraction_prompt_template.safe_format(model_answer=model_answer)
            raw_claim_extraction_response = client.generate_text(formatted_extraction_prompt)
            extracted_data = parse_json(raw_claim_extraction_response, return_type="object")
            claims = extracted_data.get("claims", [])
            if not isinstance(claims, list):
                logger.warning(f"{self.name} (Primary): Claim extraction did not return a list. Found: {type(claims)}. Treating as no claims.")
                claims = []
        except Exception as e:
            logger.error(f"Error in {self.name} (Primary) claim extraction: {e}", exc_info=True)
            raise RuntimeError(f"Claim extraction failed: {str(e)}")

        if not claims:
            logger.info(f"{self.name} (Primary): No claims extracted. Faithfulness score is 1.0 by default.")
            return {"value": 1.0, "explanation": "No claims identified in the response.", "method": "claim_extraction_primary", "timestamp": datetime.now().isoformat(), "supported_claims": 0, "total_claims": 0}

        total_claims = len(claims)
        supported_claims = 0
        claim_verifications = []

        for i, claim_text in enumerate(claims):
            try:
                formatted_verification_prompt = self.claim_verification_prompt_template.safe_format(retrieved_context=retrieved_context_str, claim=claim_text)
                raw_verification_response = client.generate_text(formatted_verification_prompt)
                verification_result = parse_json(raw_verification_response, return_type="object")
                is_supported = verification_result.get("is_supported", False)
                verification_explanation = verification_result.get("explanation", "No explanation provided.")
                claim_verifications.append({"claim": claim_text, "is_supported": is_supported, "explanation": verification_explanation, "raw_llm_response": raw_verification_response if self._include_raw_response else None})
                if is_supported:
                    supported_claims += 1
            except Exception as e:
                logger.error(f"Error verifying claim '{claim_text}' (Primary): {e}", exc_info=True)
                claim_verifications.append({"claim": claim_text, "is_supported": False, "explanation": f"Error during verification: {str(e)}", "raw_llm_response": None})
        
        faithfulness_score = (supported_claims / total_claims) if total_claims > 0 else 1.0
        final_explanation = f"Total claims: {total_claims}. Supported claims: {supported_claims}."
        
        metric_result = {
            "value": faithfulness_score,
            "explanation": final_explanation,
            "method": "claim_verification_llm_primary",
            "timestamp": datetime.now().isoformat(),
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "claim_details": claim_verifications
        }
        if self._include_raw_response:
            metric_result["raw_claim_extraction_llm_response"] = raw_claim_extraction_response
        return metric_result

    def _calculate_evaluation_fallback(self, input_obj: Input, response: Response, client) -> dict: # Changed response_obj to response
        logger.warning(f"{self.name}: Primary calculation failed. Using fallback prompt-based estimation.")
        if not response.documents: # Changed from response_obj
            return {"value": 0.0, "explanation": "Fallback: No context documents provided.", "method": "llm_evaluation_fallback", "timestamp": datetime.now().isoformat()}
        
        retrieved_context_str = "\n\n---\n\n".join([doc.content for doc in response.documents if hasattr(doc, 'content') and doc.content]) # Changed from response_obj
        if not retrieved_context_str.strip():
            return {"value": 0.0, "explanation": "Fallback: Context documents are empty.", "method": "llm_evaluation_fallback", "timestamp": datetime.now().isoformat()}

        prompt_text = FAITHFULNESS_EVALUATION_PROMPT.safe_format(
            document_content=retrieved_context_str, 
            model_answer=response.content # Changed from response_obj
        )
        try:
            llm_response_text = client.generate_text(prompt_text)
            result = parse_json(llm_response_text, return_type="object")
            score = float(result.get("faithfulness", result.get("score", 0.0)))
            explanation = result.get("explanation", "LLM fallback evaluation.")
            metric_result = {"value": score, "explanation": explanation, "method": "llm_evaluation_fallback", "timestamp": datetime.now().isoformat()}
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response_text
            return metric_result
        except Exception as e:
            logger.error(f"Error in {self.name} fallback evaluation: {e}", exc_info=True)
            return {"value": 0.0, "explanation": f"Error during fallback LLM evaluation: {str(e)}", "method": "error_fallback", "timestamp": datetime.now().isoformat()}

    def calculate_evaluation(self, input_obj: Input, response: Response, reference: Response = None, llm_client=None) -> dict: # Changed response_obj to response
        client = llm_client or self._llm_client
        if not client:
            raise ValueError(f"LLM client required for {self.name} evaluation")
        try:
            return self._calculate_evaluation_primary(input_obj, response, client) # Changed response_obj to response
        except Exception as e:
            logger.error(f"{self.name}: Primary calculation failed with error: {str(e)}. Attempting fallback.", exc_info=True)
            return self._calculate_evaluation_fallback(input_obj, response, client) # Changed response_obj to response

    # ... (existing class attributes like name, description, etc. remain the same)
    # Ensure FAITHFULNESS_EVALUATION_PROMPT (the original one) is imported at the top of the file.
    # from .prompts import FAITHFULNESS_EVALUATION_PROMPT

    # __init__ remains the same, using CLAIM_EXTRACTION_PROMPT and CLAIM_VERIFICATION_PROMPT for the primary method
    # def __init__(self, mode=None, llm_client=None, include_raw_response=False):
    #     super().__init__(prompt=None, llm_client=llm_client, include_raw_response=include_raw_response, mode=mode)
    #     self.claim_extraction_prompt_template = CLAIM_EXTRACTION_PROMPT
    #     self.claim_verification_prompt_template = CLAIM_VERIFICATION_PROMPT

    # calculate_monitoring can remain as a placeholder or be simplified further
    # def calculate_monitoring(self, request: LLMRequest, response: LLMResponse) -> dict:
    #     # ... (existing simplified monitoring logic or placeholder)

    def _calculate_evaluation_primary(self, input_obj: Input, response_obj: Response, client) -> dict:
        # This method contains the original claims-based logic
        model_answer = response_obj.content
        if not model_answer.strip():
            logger.warning(f"{self.name} (Primary): Model answer is empty. Returning score 0.")
            # For primary failure, we might want to raise an exception to trigger fallback or return a specific error structure
            raise ValueError("Model answer is empty for primary calculation")

        if not response_obj.documents:
            logger.warning(f"{self.name} (Primary): No documents for context. Returning score 0.")
            raise ValueError("No context documents for primary calculation")
        
        retrieved_context_str = "\n\n---\n\n".join([doc.content for doc in response_obj.documents if hasattr(doc, 'content') and doc.content])
        if not retrieved_context_str.strip():
            logger.warning(f"{self.name} (Primary): Context documents are empty. Returning score 0.")
            raise ValueError("Context documents are empty for primary calculation")

        claims = []
        raw_claim_extraction_response = None
        try:
            formatted_extraction_prompt = self.claim_extraction_prompt_template.safe_format(model_answer=model_answer)
            raw_claim_extraction_response = client.generate_text(formatted_extraction_prompt)
            extracted_data = parse_json(raw_claim_extraction_response, return_type="object")
            claims = extracted_data.get("claims", [])
            if not isinstance(claims, list):
                logger.warning(f"{self.name} (Primary): Claim extraction did not return a list. Found: {type(claims)}. Treating as no claims.")
                claims = []
        except Exception as e:
            logger.error(f"Error in {self.name} (Primary) claim extraction: {e}", exc_info=True)
            raise RuntimeError(f"Claim extraction failed: {str(e)}") # Propagate to trigger fallback

        if not claims:
            logger.info(f"{self.name} (Primary): No claims extracted. Faithfulness score is 1.0 by default.")
            return {"value": 1.0, "explanation": "No claims identified in the response.", "method": "claim_extraction_primary", "timestamp": datetime.now().isoformat(), "supported_claims": 0, "total_claims": 0}

        total_claims = len(claims)
        supported_claims = 0
        claim_verifications = []

        for i, claim_text in enumerate(claims):
            try:
                formatted_verification_prompt = self.claim_verification_prompt_template.safe_format(retrieved_context=retrieved_context_str, claim=claim_text)
                raw_verification_response = client.generate_text(formatted_verification_prompt)
                verification_result = parse_json(raw_verification_response, return_type="object")
                is_supported = verification_result.get("is_supported", False)
                verification_explanation = verification_result.get("explanation", "No explanation provided.")
                claim_verifications.append({"claim": claim_text, "is_supported": is_supported, "explanation": verification_explanation, "raw_llm_response": raw_verification_response if self._include_raw_response else None})
                if is_supported:
                    supported_claims += 1
            except Exception as e:
                logger.error(f"Error verifying claim '{claim_text}' (Primary): {e}", exc_info=True)
                # Decide if one claim verification error fails the whole primary method or just marks this claim as unverified
                claim_verifications.append({"claim": claim_text, "is_supported": False, "explanation": f"Error during verification: {str(e)}", "raw_llm_response": None})
        
        faithfulness_score = (supported_claims / total_claims) if total_claims > 0 else 1.0
        final_explanation = f"Total claims: {total_claims}. Supported claims: {supported_claims}."
        
        metric_result = {
            "value": faithfulness_score,
            "explanation": final_explanation,
            "method": "claim_verification_llm_primary",
            "timestamp": datetime.now().isoformat(),
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "claim_details": claim_verifications
        }
        if self._include_raw_response:
            metric_result["raw_claim_extraction_llm_response"] = raw_claim_extraction_response
        return metric_result

    def _calculate_evaluation_fallback(self, input_obj: Input, response_obj: Response, client) -> dict:
        logger.warning(f"{self.name}: Primary calculation failed. Using fallback prompt-based estimation.")
        if not response_obj.documents:
            return {"value": 0.0, "explanation": "Fallback: No context documents provided.", "method": "llm_evaluation_fallback", "timestamp": datetime.now().isoformat()}
        
        retrieved_context_str = "\n\n---\n\n".join([doc.content for doc in response_obj.documents if hasattr(doc, 'content') and doc.content])
        if not retrieved_context_str.strip():
            return {"value": 0.0, "explanation": "Fallback: Context documents are empty.", "method": "llm_evaluation_fallback", "timestamp": datetime.now().isoformat()}

        # Use the original FAITHFULNESS_EVALUATION_PROMPT for fallback
        # Ensure FAITHFULNESS_EVALUATION_PROMPT is imported from .prompts
        prompt_text = FAITHFULNESS_EVALUATION_PROMPT.safe_format(
            document_content=retrieved_context_str, 
            model_answer=response_obj.content
        )
        try:
            llm_response = client.generate_text(prompt_text)
            result = parse_json(llm_response, return_type="object")
            score = float(result.get("faithfulness", result.get("score", 0.0))) # Check for 'faithfulness' or 'score'
            explanation = result.get("explanation", "LLM fallback evaluation.")
            metric_result = {"value": score, "explanation": explanation, "method": "llm_evaluation_fallback", "timestamp": datetime.now().isoformat()}
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            return metric_result
        except Exception as e:
            logger.error(f"Error in {self.name} fallback evaluation: {e}", exc_info=True)
            return {"value": 0.0, "explanation": f"Error during fallback LLM evaluation: {str(e)}", "method": "error_fallback", "timestamp": datetime.now().isoformat()}

    def calculate_evaluation(self, input_obj: Input, response_obj: Response, reference: Response = None, llm_client=None) -> dict:
        client = llm_client or self._llm_client
        if not client:
            raise ValueError(f"LLM client required for {self.name} evaluation")
        try:
            return self._calculate_evaluation_primary(input_obj, response_obj, client)
        except Exception as e:
            logger.error(f"{self.name}: Primary calculation failed with error: {str(e)}. Attempting fallback.", exc_info=True)
            return self._calculate_evaluation_fallback(input_obj, response_obj, client)

    """
    Metric for evaluating the faithfulness of an answer to the retrieved documents.
    This version uses a claims-based approach: extracts claims from the answer,
    then verifies each claim against the provided context.
    """
    name = "Faithfulness"
    description = "Measures how factually consistent a response is with the retrieved context, based on claim verification."
    greater_is_better = True
    context = MetricContext.EVALUATION # Primarily for evaluation due to complexity
    category = MetricCategory.QUALITY
    
    evaluation_requires = {
        "input": Input, # Though not directly used in this version's prompt, kept for consistency
        "response": Response # Expects response.content and response.documents
    }
    
    def __init__(self, mode=None, llm_client=None, include_raw_response=False):
        super().__init__(prompt=None, llm_client=llm_client, include_raw_response=include_raw_response, mode=mode)
        self.claim_extraction_prompt_template = CLAIM_EXTRACTION_PROMPT
        self.claim_verification_prompt_template = CLAIM_VERIFICATION_PROMPT

    def calculate_monitoring(self, request: LLMRequest, response: LLMResponse) -> dict:
        logger.warning(f"{self.name} claims-based logic is designed for evaluation. Monitoring may not be fully supported with this detailed approach.")
        return {
            "value": None,
            "explanation": "Claims-based faithfulness monitoring not fully implemented with this detailed logic.",
            "method": "placeholder",
            "timestamp": datetime.now().isoformat()
        }

    def calculate_evaluation(self, input_obj: Input, response_obj: Response, reference: Response = None, llm_client=None) -> dict:
        client = llm_client or self._llm_client
        if not client:
            raise ValueError(f"LLM client required for {self.name} evaluation")

        model_answer = response_obj.content
        if not model_answer.strip():
            logger.warning(f"{self.name}: Model answer is empty. Returning score 0.")
            return {"value": 0.0, "explanation": "Model answer is empty.", "method": "direct_check", "timestamp": datetime.now().isoformat(), "supported_claims": 0, "total_claims": 0}

        if not response_obj.documents:
            logger.warning(f"{self.name}: No documents found in response_obj.documents for context. Returning score 0.")
            return {"value": 0.0, "explanation": "No context (documents) provided.", "method": "direct_check", "timestamp": datetime.now().isoformat(), "supported_claims": 0, "total_claims": 0}
        
        retrieved_context_str = "\n\n---\n\n".join([doc.content for doc in response_obj.documents if hasattr(doc, 'content') and doc.content])
        if not retrieved_context_str.strip():
            logger.warning(f"{self.name}: Context documents are empty. Returning score 0.")
            return {"value": 0.0, "explanation": "Provided context documents are empty.", "method": "direct_check", "timestamp": datetime.now().isoformat(), "supported_claims": 0, "total_claims": 0}

        claims = []
        raw_claim_extraction_response = None
        try:
            formatted_extraction_prompt = self.claim_extraction_prompt_template.safe_format(model_answer=model_answer)
            raw_claim_extraction_response = client.generate_text(formatted_extraction_prompt)
            extracted_data = parse_json(raw_claim_extraction_response, return_type="object")
            claims = extracted_data.get("claims", [])
            if not isinstance(claims, list):
                logger.warning(f"{self.name}: Claim extraction did not return a list. Found: {type(claims)}. Treating as no claims.")
                claims = []
        except Exception as e:
            logger.error(f"Error in {self.name} claim extraction: {e}", exc_info=True)
            return {"value": 0.0, "explanation": f"Error during claim extraction: {str(e)}", "method": "error", "timestamp": datetime.now().isoformat(), "supported_claims": 0, "total_claims": 0}

        if not claims:
            logger.info(f"{self.name}: No claims extracted. Faithfulness score is 1.0 by default (vacuously true).")
            return {"value": 1.0, "explanation": "No claims identified in the response.", "method": "claim_extraction", "timestamp": datetime.now().isoformat(), "supported_claims": 0, "total_claims": 0}

        total_claims = len(claims)
        supported_claims = 0
        claim_verifications = []

        for i, claim_text in enumerate(claims):
            try:
                formatted_verification_prompt = self.claim_verification_prompt_template.safe_format(retrieved_context=retrieved_context_str, claim=claim_text)
                raw_verification_response = client.generate_text(formatted_verification_prompt)
                verification_result = parse_json(raw_verification_response, return_type="object")
                is_supported = verification_result.get("is_supported", False)
                verification_explanation = verification_result.get("explanation", "No explanation provided.")
                
                claim_verifications.append({
                    "claim": claim_text,
                    "is_supported": is_supported,
                    "explanation": verification_explanation,
                    "raw_llm_response": raw_verification_response if self._include_raw_response else None
                })
                if is_supported:
                    supported_claims += 1
            except Exception as e:
                logger.error(f"Error verifying claim '{claim_text}': {e}", exc_info=True)
                claim_verifications.append({"claim": claim_text, "is_supported": False, "explanation": f"Error during verification: {str(e)}", "raw_llm_response": None})
        
        faithfulness_score = (supported_claims / total_claims) if total_claims > 0 else 1.0
        final_explanation = f"Total claims: {total_claims}. Supported claims: {supported_claims}."
        
        metric_result = {
            "value": faithfulness_score,
            "explanation": final_explanation,
            "method": "claim_verification_llm",
            "timestamp": datetime.now().isoformat(),
            "total_claims": total_claims,
            "supported_claims": supported_claims,
            "claim_details": claim_verifications
        }
        if self._include_raw_response:
            metric_result["raw_claim_extraction_llm_response"] = raw_claim_extraction_response
        return metric_result

    """
    Metric for evaluating the faithfulness of an answer to the retrieved documents in both monitoring and evaluation contexts.
    
    This metric measures how well the answer sticks to the information in the documents.
    """
    name = "Faithfulness"
    description = "Measures how faithful the answer is to the retrieved documents"
    greater_is_better = True
    context = MetricContext.BOTH  # Works in both contexts
    category = MetricCategory.QUALITY
    
    # Model-based requirements
    monitoring_requires = {
        "request": LLMRequest,
        "response": LLMResponse
    }
    evaluation_requires = {
        "input": Input,
        "response": Response
    }
    
    def __init__(self, mode=None, llm_client=None, include_raw_response=False):
        """
        Initialize the faithfulness metric.
        
        Args:
            mode: The context mode for this metric instance
            llm_client: The LLM client
            include_raw_response: Whether to include the raw LLM response in the output
        """
        super().__init__(prompt=FAITHFULNESS_EVALUATION_PROMPT, llm_client=llm_client, include_raw_response=include_raw_response, mode=mode)
    
    def calculate_monitoring(self, request: LLMRequest, response: LLMResponse) -> dict:
        """
        Calculate faithfulness in monitoring context.
        
        Args:
            request: LLMRequest object
            response: LLMResponse object
            
        Returns:
            Dict containing metric result
        """
        if not self.llm_client:
            raise ValueError("LLM client required for faithfulness monitoring")
        
        # Get documents from response context
        documents = getattr(response, 'context', "") or str(getattr(response, 'documents', ""))
        
        # LLM-based evaluation for monitoring
        prompt = FAITHFULNESS_EVALUATION_PROMPT.safe_format(
            document_content=documents,
            model_answer=response.completion
        )
        
        try:
            llm_response = self.llm_client.generate_text(prompt)
            from ..core.utils import parse_json
            result = parse_json(llm_response, return_type="object")
            
            score = float(result.get("score", 0.0))
            explanation = result.get("explanation", "LLM evaluation")
            
            metric_result = {
                "value": score,
                "explanation": explanation,
                "method": "llm_evaluation",
                "timestamp": datetime.now().isoformat()
            }
            
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            
            return metric_result
            
        except Exception as e:
            logger.error(f"Error in LLM-based faithfulness evaluation: {e}")
            return {
                "value": 0.0,
                "explanation": f"Error in evaluation: {str(e)}",
                "method": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_evaluation(self, input_obj: Input, response: Response, reference: Response, llm_client=None) -> dict:
        """
        Calculate faithfulness in evaluation context.
        
        Args:
            input_obj: Input object
            response: Response object
            reference: Reference response object
            llm_client: Optional LLM client for evaluation
            
        Returns:
            Dict containing metric result
        """
        # Use provided client or stored client
        client = llm_client or self._llm_client
        
        if not client:
            raise ValueError("LLM client required for faithfulness evaluation")
        
        # Format prompt for evaluation context
        prompt = FAITHFULNESS_EVALUATION_PROMPT.safe_format(
            document_content=str(response.documents) if response.documents else "",
            model_answer=response.content
        )
        
        try:
            llm_response = client.generate_text(prompt)
            from ..core.utils import parse_json
            result = parse_json(llm_response, return_type="object")
            
            score = float(result.get("faithfulness", 0.0)) # Changed here
            explanation = result.get("explanation", "LLM evaluation")
            
            metric_result = {
                "value": score,
                "explanation": explanation,
                "method": "llm_evaluation",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            }
            
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            
            return metric_result
            
        except Exception as e:
            logger.error(f"Error in faithfulness evaluation: {e}")
            return {
                "value": 0.0,
                "explanation": f"Error in evaluation: {str(e)}",
                "method": "error",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            }
    

class RelevanceMetric(LLMMeasuredBaseMetric):
    # ... (name, description, __init__ etc. remain the same)

    def _calculate_evaluation_primary(self, input_obj: Input, response: Response, client) -> dict: # Changed response_obj to response
        if not self.cosine_similarity_func:
            raise ImportError(f"{self.name} (Primary): cosine_similarity function not available.")
        if not hasattr(client, 'get_embeddings') or not callable(client.get_embeddings):
            raise AttributeError(f"{self.name} (Primary): llm_client does not have 'get_embeddings' method.")

        user_input_text = input_obj.content
        model_answer_text = response.content # Changed from response_obj

        if not user_input_text.strip() or not model_answer_text.strip():
            logger.warning(f"{self.name} (Primary): User input or model answer is empty.")
            raise ValueError("User input or model answer is empty for primary calculation")

        # ... (rest of the question generation and embedding logic remains the same, using model_answer_text)
        generated_questions = []
        raw_question_gen_response = None
        try:
            formatted_qg_prompt = self.question_generation_prompt_template.safe_format(
                model_answer=model_answer_text, 
                num_questions=self.num_generated_questions
            )
            raw_question_gen_response = client.generate_text(formatted_qg_prompt)
            qg_data = parse_json(raw_question_gen_response, return_type="object")
            generated_questions = qg_data.get("generated_questions", [])
            if not isinstance(generated_questions, list) or not all(isinstance(q, str) for q in generated_questions):
                logger.warning(f"{self.name} (Primary): Question generation did not return list of strings. Found: {generated_questions}.")
                generated_questions = []
        except Exception as e:
            logger.error(f"Error in {self.name} (Primary) question generation: {e}", exc_info=True)
            raise RuntimeError(f"Question generation failed: {str(e)}")

        if not generated_questions:
            logger.warning(f"{self.name} (Primary): No questions generated. Cannot calculate relevancy.")
            raise ValueError("No questions generated from model answer for primary calculation")

        try:
            user_input_embedding = client.get_embeddings([user_input_text])[0]
            generated_question_embeddings = client.get_embeddings(generated_questions)
        except Exception as e:
            logger.error(f"Error getting embeddings in {self.name} (Primary): {e}", exc_info=True)
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

        if not user_input_embedding or len(generated_question_embeddings) != len(generated_questions) or not all(emb for emb in generated_question_embeddings):
            logger.warning(f"{self.name} (Primary): Failed to get all embeddings.")
            raise ValueError("Failed to generate all necessary embeddings for primary calculation")

        similarity_scores = []
        for q_emb in generated_question_embeddings:
            try:
                score = self.cosine_similarity_func(user_input_embedding, q_emb)
                similarity_scores.append(score)
            except Exception as e:
                logger.error(f"Error calculating cosine similarity in {self.name} (Primary): {e}", exc_info=True)
                similarity_scores.append(0.0)
        
        final_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        explanation = f"Average cosine similarity over {len(similarity_scores)} generated questions."
        
        metric_result = {
            "value": float(final_score),
            "explanation": explanation,
            "method": "embedding_similarity_primary",
            "timestamp": datetime.now().isoformat(),
            "individual_scores": similarity_scores,
            "generated_questions": generated_questions
        }
        if self._include_raw_response:
            metric_result["raw_question_generation_llm_response"] = raw_question_gen_response
        return metric_result

    def _calculate_evaluation_fallback(self, input_obj: Input, response: Response, client) -> dict: # Changed response_obj to response
        logger.warning(f"{self.name}: Primary calculation failed. Using fallback prompt-based estimation.")
        prompt_text = RELEVANCE_EVALUATION_PROMPT.safe_format(
            input=input_obj.content,
            model_answer=response.content # Changed from response_obj
        )
        try:
            llm_response_text = client.generate_text(prompt_text)
            result = parse_json(llm_response_text, return_type="object")
            score = float(result.get("relevance", result.get("score", 0.0)))
            explanation = result.get("explanation", "LLM fallback evaluation.")
            metric_result = {"value": score, "explanation": explanation, "method": "llm_evaluation_fallback", "timestamp": datetime.now().isoformat()}
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response_text
            return metric_result
        except Exception as e:
            logger.error(f"Error in {self.name} fallback evaluation: {e}", exc_info=True)
            return {"value": 0.0, "explanation": f"Error during fallback LLM evaluation: {str(e)}", "method": "error_fallback", "timestamp": datetime.now().isoformat()}

    def calculate_evaluation(self, input_obj: Input, response: Response, reference: Response = None, llm_client=None) -> dict: # Changed response_obj to response
        client = llm_client or self._llm_client
        if not client:
            raise ValueError(f"LLM client required for {self.name} evaluation")
        try:
            return self._calculate_evaluation_primary(input_obj, response, client) # Changed response_obj to response
        except Exception as e:
            logger.error(f"{self.name}: Primary calculation failed with error: {str(e)}. Attempting fallback.", exc_info=True)
            return self._calculate_evaluation_fallback(input_obj, response, client) # Changed response_obj to response

    # ... (existing class attributes like name, description, etc. remain the same)
    # Ensure RELEVANCE_EVALUATION_PROMPT (the original one) is imported at the top of the file.
    # from .prompts import RELEVANCE_EVALUATION_PROMPT

    # __init__ remains the same, using RESPONSE_QUESTION_GENERATION_PROMPT for the primary method
    # def __init__(self, mode=None, llm_client=None, include_raw_response=False, num_generated_questions: int = 3):
    #    # ... (init logic for primary method)

    # calculate_monitoring can remain as a placeholder
    # def calculate_monitoring(self, request: LLMRequest, response: LLMResponse) -> dict:
    #    # ... (existing simplified monitoring logic or placeholder)

    def _calculate_evaluation_primary(self, input_obj: Input, response_obj: Response, client) -> dict:
        # This method contains the original question-generation & embedding logic
        if not self.cosine_similarity_func:
            raise ImportError(f"{self.name} (Primary): cosine_similarity function not available.")
        if not hasattr(client, 'get_embeddings') or not callable(client.get_embeddings):
            raise AttributeError(f"{self.name} (Primary): llm_client does not have 'get_embeddings' method.")

        user_input_text = input_obj.content
        model_answer_text = response_obj.content

        if not user_input_text.strip() or not model_answer_text.strip():
            logger.warning(f"{self.name} (Primary): User input or model answer is empty.")
            raise ValueError("User input or model answer is empty for primary calculation")

        generated_questions = []
        raw_question_gen_response = None
        try:
            formatted_qg_prompt = self.question_generation_prompt_template.safe_format(
                model_answer=model_answer_text, 
                num_questions=self.num_generated_questions
            )
            raw_question_gen_response = client.generate_text(formatted_qg_prompt)
            qg_data = parse_json(raw_question_gen_response, return_type="object")
            generated_questions = qg_data.get("generated_questions", [])
            if not isinstance(generated_questions, list) or not all(isinstance(q, str) for q in generated_questions):
                logger.warning(f"{self.name} (Primary): Question generation did not return list of strings. Found: {generated_questions}.")
                generated_questions = []
        except Exception as e:
            logger.error(f"Error in {self.name} (Primary) question generation: {e}", exc_info=True)
            raise RuntimeError(f"Question generation failed: {str(e)}")

        if not generated_questions:
            logger.warning(f"{self.name} (Primary): No questions generated. Cannot calculate relevancy.")
            # This case might not be an error per se, but an inability to proceed with this method.
            # Depending on desired behavior, could return a specific score (e.g., 0) or raise to fallback.
            raise ValueError("No questions generated from model answer for primary calculation")

        try:
            user_input_embedding = client.get_embeddings([user_input_text])[0]
            generated_question_embeddings = client.get_embeddings(generated_questions)
        except Exception as e:
            logger.error(f"Error getting embeddings in {self.name} (Primary): {e}", exc_info=True)
            raise RuntimeError(f"Embedding generation failed: {str(e)}")

        if not user_input_embedding or len(generated_question_embeddings) != len(generated_questions) or not all(emb for emb in generated_question_embeddings):
            logger.warning(f"{self.name} (Primary): Failed to get all embeddings.")
            raise ValueError("Failed to generate all necessary embeddings for primary calculation")

        similarity_scores = []
        for q_emb in generated_question_embeddings:
            try:
                score = self.cosine_similarity_func(user_input_embedding, q_emb)
                similarity_scores.append(score)
            except Exception as e:
                logger.error(f"Error calculating cosine similarity in {self.name} (Primary): {e}", exc_info=True)
                similarity_scores.append(0.0) # Or raise an error to fail the primary method
        
        final_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        explanation = f"Average cosine similarity over {len(similarity_scores)} generated questions."
        
        metric_result = {
            "value": float(final_score),
            "explanation": explanation,
            "method": "embedding_similarity_primary",
            "timestamp": datetime.now().isoformat(),
            "individual_scores": similarity_scores,
            "generated_questions": generated_questions
        }
        if self._include_raw_response:
            metric_result["raw_question_generation_llm_response"] = raw_question_gen_response
        return metric_result

    def _calculate_evaluation_fallback(self, input_obj: Input, response_obj: Response, client) -> dict:
        logger.warning(f"{self.name}: Primary calculation failed. Using fallback prompt-based estimation.")
        # Use the original RELEVANCE_EVALUATION_PROMPT for fallback
        # Ensure RELEVANCE_EVALUATION_PROMPT is imported from .prompts
        prompt_text = RELEVANCE_EVALUATION_PROMPT.safe_format(
            input=input_obj.content,
            model_answer=response_obj.content
        )
        try:
            llm_response = client.generate_text(prompt_text)
            result = parse_json(llm_response, return_type="object")
            score = float(result.get("relevance", result.get("score", 0.0))) # Check for 'relevance' or 'score'
            explanation = result.get("explanation", "LLM fallback evaluation.")
            metric_result = {"value": score, "explanation": explanation, "method": "llm_evaluation_fallback", "timestamp": datetime.now().isoformat()}
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            return metric_result
        except Exception as e:
            logger.error(f"Error in {self.name} fallback evaluation: {e}", exc_info=True)
            return {"value": 0.0, "explanation": f"Error during fallback LLM evaluation: {str(e)}", "method": "error_fallback", "timestamp": datetime.now().isoformat()}

    def calculate_evaluation(self, input_obj: Input, response_obj: Response, reference: Response = None, llm_client=None) -> dict:
        client = llm_client or self._llm_client
        if not client:
            raise ValueError(f"LLM client required for {self.name} evaluation")
        try:
            return self._calculate_evaluation_primary(input_obj, response_obj, client)
        except Exception as e:
            logger.error(f"{self.name}: Primary calculation failed with error: {str(e)}. Attempting fallback.", exc_info=True)
            return self._calculate_evaluation_fallback(input_obj, response_obj, client)

    """
    Metric for evaluating the relevance of an answer to the input.
    This version uses question generation from the response and compares embeddings with the user input.
    """
    name = "Relevance"
    description = "Measures how relevant a response is to the user input by generating questions from the response and comparing their embeddings to the user input embedding."
    greater_is_better = True
    context = MetricContext.EVALUATION # Primarily for evaluation due to complexity
    category = MetricCategory.QUALITY

    evaluation_requires = {
        "input": Input,
        "response": Response
    }

    def __init__(self, mode=None, llm_client=None, include_raw_response=False, num_generated_questions: int = 3):
        super().__init__(prompt=None, llm_client=llm_client, include_raw_response=include_raw_response, mode=mode)
        self.question_generation_prompt_template = RESPONSE_QUESTION_GENERATION_PROMPT
        self.num_generated_questions = num_generated_questions
        try:
            self.cosine_similarity_func = cosine_similarity
        except AttributeError:
            logger.error(f"{self.name}: cosine_similarity not found (likely import issue from ..core.utils). Embedding-based calculations will fail.")
            self.cosine_similarity_func = None

    def calculate_monitoring(self, request: LLMRequest, response: LLMResponse) -> dict:
        logger.warning(f"{self.name} embedding-based logic is designed for evaluation. Monitoring may not be fully supported.")
        return {
            "value": None,
            "explanation": "Embedding-based relevance monitoring not fully implemented.",
            "method": "placeholder",
            "timestamp": datetime.now().isoformat()
        }

    def calculate_evaluation(self, input_obj: Input, response_obj: Response, reference: Response = None, llm_client=None) -> dict:
        client = llm_client or self._llm_client
        if not client:
            raise ValueError(f"LLM client required for {self.name} evaluation")
        if not self.cosine_similarity_func:
            raise ImportError(f"{self.name}: cosine_similarity function not available.")
        if not hasattr(client, 'get_embeddings') or not callable(client.get_embeddings):
            raise AttributeError(f"{self.name}: llm_client does not have 'get_embeddings' method.")

        user_input_text = input_obj.content
        model_answer_text = response_obj.content

        if not user_input_text.strip() or not model_answer_text.strip():
            logger.warning(f"{self.name}: User input or model answer is empty. Returning score 0.")
            return {"value": 0.0, "explanation": "User input or model answer is empty.", "method": "direct_check", "timestamp": datetime.now().isoformat()}

        generated_questions = []
        raw_question_gen_response = None
        try:
            formatted_qg_prompt = self.question_generation_prompt_template.safe_format(
                model_answer=model_answer_text, 
                num_questions=self.num_generated_questions
            )
            raw_question_gen_response = client.generate_text(formatted_qg_prompt)
            qg_data = parse_json(raw_question_gen_response, return_type="object")
            generated_questions = qg_data.get("generated_questions", [])
            if not isinstance(generated_questions, list) or not all(isinstance(q, str) for q in generated_questions):
                logger.warning(f"{self.name}: Question generation did not return list of strings. Found: {generated_questions}.")
                generated_questions = []
        except Exception as e:
            logger.error(f"Error in {self.name} question generation: {e}", exc_info=True)
            return {"value": 0.0, "explanation": f"Error during question generation: {str(e)}", "method": "error", "timestamp": datetime.now().isoformat()}

        if not generated_questions:
            logger.warning(f"{self.name}: No questions generated. Cannot calculate relevancy. Score 0.")
            return {"value": 0.0, "explanation": "No questions generated from model answer.", "method": "question_generation", "timestamp": datetime.now().isoformat()}

        try:
            user_input_embedding = client.get_embeddings([user_input_text])[0]
            generated_question_embeddings = client.get_embeddings(generated_questions)
        except Exception as e:
            logger.error(f"Error getting embeddings in {self.name}: {e}", exc_info=True)
            return {"value": 0.0, "explanation": f"Error during embedding generation: {str(e)}", "method": "error", "timestamp": datetime.now().isoformat()}

        if not user_input_embedding or len(generated_question_embeddings) != len(generated_questions) or not all(emb for emb in generated_question_embeddings):
            logger.warning(f"{self.name}: Failed to get all embeddings. Cannot calculate relevancy.")
            return {"value": 0.0, "explanation": "Failed to generate all necessary embeddings.", "method": "embedding_generation", "timestamp": datetime.now().isoformat()}

        similarity_scores = []
        for q_emb in generated_question_embeddings:
            try:
                score = self.cosine_similarity_func(user_input_embedding, q_emb)
                similarity_scores.append(score)
            except Exception as e:
                logger.error(f"Error calculating cosine similarity in {self.name}: {e}", exc_info=True)
                similarity_scores.append(0.0)
        
        final_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0
        explanation = f"Average cosine similarity over {len(similarity_scores)} generated questions."
        
        metric_result = {
            "value": float(final_score),
            "explanation": explanation,
            "method": "embedding_similarity",
            "timestamp": datetime.now().isoformat(),
            "individual_scores": similarity_scores,
            "generated_questions": generated_questions
        }
        if self._include_raw_response:
            metric_result["raw_question_generation_llm_response"] = raw_question_gen_response
        return metric_result

    """
    Metric for evaluating the relevance of an answer to the input in both monitoring and evaluation contexts.
    
    This metric measures how well the answer addresses the input.
    """
    name = "Relevance"
    description = "Measures how relevant the answer is to the input"
    greater_is_better = True
    context = MetricContext.BOTH  # Works in both contexts
    category = MetricCategory.QUALITY
    
    # Model-based requirements
    monitoring_requires = {
        "request": LLMRequest,
        "response": LLMResponse
    }
    evaluation_requires = {
        "input": Input,
        "response": Response
    }
    
    def __init__(self, mode=None, llm_client=None, include_raw_response=False):
        """
        Initialize the relevance metric.
        
        Args:
            mode: The context mode for this metric instance
            llm_client: The LLM client
            include_raw_response: Whether to include the raw LLM response in the output
        """
        super().__init__(prompt=RELEVANCE_EVALUATION_PROMPT, llm_client=llm_client, include_raw_response=include_raw_response, mode=mode)
    
    def calculate_monitoring(self, request: LLMRequest, response: LLMResponse) -> dict:
        """
        Calculate relevance in monitoring context.
        
        Args:
            request: LLMRequest object
            response: LLMResponse object
            
        Returns:
            Dict containing metric result
        """
        if not self.llm_client:
            raise ValueError("LLM client required for relevance monitoring")
        
        # LLM-based evaluation for monitoring
        prompt = RELEVANCE_EVALUATION_PROMPT.safe_format(
            input=request.input.content,
            model_answer=response.completion
        )
        
        try:
            llm_response = self.llm_client.generate_text(prompt)
            from ..core.utils import parse_json
            result = parse_json(llm_response, return_type="object")
            
            score = float(result.get("score", 0.0))
            explanation = result.get("explanation", "LLM evaluation")
            
            metric_result = {
                "value": score,
                "explanation": explanation,
                "method": "llm_evaluation",
                "timestamp": datetime.now().isoformat()
            }
            
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            
            return metric_result
            
        except Exception as e:
            logger.error(f"Error in LLM-based relevance evaluation: {e}")
            return {
                "value": 0.0,
                "explanation": f"Error in evaluation: {str(e)}",
                "method": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_evaluation(self, input_obj: Input, response: Response, reference: Response, llm_client=None) -> dict:
        """
        Calculate relevance in evaluation context.
        
        Args:
            input_obj: Input object
            response: Response object
            reference: Reference response object
            llm_client: Optional LLM client for evaluation
            
        Returns:
            Dict containing metric result
        """
        # Use provided client or stored client
        client = llm_client or self._llm_client
        
        if not client:
            raise ValueError("LLM client required for relevance evaluation")
        
        # Format prompt for evaluation context
        prompt = RELEVANCE_EVALUATION_PROMPT.safe_format(
            input=input_obj.content,
            model_answer=response.content
        )
        
        try:
            llm_response = client.generate_text(prompt)
            from ..core.utils import parse_json
            result = parse_json(llm_response, return_type="object")
            
            score = float(result.get("relevance", 0.0)) # Changed here
            explanation = result.get("explanation", "LLM evaluation")
            
            metric_result = {
                "value": score,
                "explanation": explanation,
                "method": "llm_evaluation",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            }
            
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            
            return metric_result
            
        except Exception as e:
            logger.error(f"Error in relevance evaluation: {e}")
            return {
                "value": 0.0,
                "explanation": f"Error in evaluation: {str(e)}",
                "method": "error",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            }
    

class CoherenceMetric(LLMMeasuredBaseMetric):
    """
    Metric for evaluating the coherence of an answer in both monitoring and evaluation contexts.
    
    This metric measures how well-structured and logical the answer is.
    """
    name = "Coherence"
    description = "Measures how coherent, well-structured, and logical the answer is"
    greater_is_better = True
    context = MetricContext.BOTH  # Works in both contexts
    category = MetricCategory.QUALITY
    
    # Model-based requirements
    monitoring_requires = {
        "request": LLMRequest,
        "response": LLMResponse
    }
    evaluation_requires = {
        "input": Input,
        "response": Response
    }
    
    def __init__(self, mode=None, llm_client=None, include_raw_response=False):
        """
        Initialize the coherence metric.
        
        Args:
            mode: The context mode for this metric instance
            llm_client: The LLM client
            include_raw_response: Whether to include the raw LLM response in the output
        """
        super().__init__(prompt=COHERENCE_EVALUATION_PROMPT, llm_client=llm_client, include_raw_response=include_raw_response, mode=mode)
    
    def calculate_monitoring(self, request: LLMRequest, response: LLMResponse) -> dict:
        """
        Calculate coherence in monitoring context.
        
        Args:
            request: LLMRequest object
            response: LLMResponse object
            
        Returns:
            Dict containing metric result
        """
        if not self.llm_client:
            raise ValueError("LLM client required for coherence monitoring")
        
        # LLM-based evaluation for monitoring
        prompt = COHERENCE_EVALUATION_PROMPT.safe_format(
            model_answer=response.completion
        )
        
        try:
            llm_response = self.llm_client.generate_text(prompt)
            from ..core.utils import parse_json
            result = parse_json(llm_response, return_type="object")
            
            score = float(result.get("score", 0.0))
            explanation = result.get("explanation", "LLM evaluation")
            
            metric_result = {
                "value": score,
                "explanation": explanation,
                "method": "llm_evaluation",
                "timestamp": datetime.now().isoformat()
            }
            
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            
            return metric_result
            
        except Exception as e:
            logger.error(f"Error in LLM-based coherence evaluation: {e}")
            return {
                "value": 0.0,
                "explanation": f"Error in evaluation: {str(e)}",
                "method": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_evaluation(self, input_obj: Input, response: Response, reference: Response, llm_client=None) -> dict:
        """
        Calculate coherence in evaluation context.
        
        Args:
            input_obj: Input object
            response: Response object
            reference: Reference response object
            llm_client: Optional LLM client for evaluation
            
        Returns:
            Dict containing metric result
        """
        # Use provided client or stored client
        client = llm_client or self._llm_client
        
        if not client:
            raise ValueError("LLM client required for coherence evaluation")
        
        # Format prompt for evaluation context
        prompt = COHERENCE_EVALUATION_PROMPT.safe_format(
            model_answer=response.content
        )
        
        try:
            llm_response = client.generate_text(prompt)
            from ..core.utils import parse_json
            result = parse_json(llm_response, return_type="object")
            
            score = float(result.get("coherence", 0.0)) # Changed here
            explanation = result.get("explanation", "LLM evaluation")
            
            metric_result = {
                "value": score,
                "explanation": explanation,
                "method": "llm_evaluation",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            }
            
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            
            return metric_result
            
        except Exception as e:
            logger.error(f"Error in coherence evaluation: {e}")
            return {
                "value": 0.0,
                "explanation": f"Error in evaluation: {str(e)}",
                "method": "error",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            }
    

class FluencyMetric(LLMMeasuredBaseMetric):
    """
    Metric for evaluating the fluency of an answer in both monitoring and evaluation contexts.
    
    This metric measures how grammatically correct and natural the answer is.
    """
    name = "Fluency"
    description = "Measures how grammatically correct and natural the answer is"
    greater_is_better = True
    context = MetricContext.BOTH  # Works in both contexts
    category = MetricCategory.QUALITY
    
    # Model-based requirements
    monitoring_requires = {
        "request": LLMRequest,
        "response": LLMResponse
    }
    evaluation_requires = {
        "input": Input,
        "response": Response
    }
    
    def __init__(self, mode=None, llm_client=None, include_raw_response=False):
        """
        Initialize the fluency metric.
        
        Args:
            mode: The context mode for this metric instance
            llm_client: The LLM client
            include_raw_response: Whether to include the raw LLM response in the output
        """
        super().__init__(prompt=FLUENCY_EVALUATION_PROMPT, llm_client=llm_client, include_raw_response=include_raw_response, mode=mode)
    
    def calculate_monitoring(self, request: LLMRequest, response: LLMResponse) -> dict:
        """
        Calculate fluency in monitoring context.
        
        Args:
            request: LLMRequest object
            response: LLMResponse object
            
        Returns:
            Dict containing metric result
        """
        if not self.llm_client:
            raise ValueError("LLM client required for fluency monitoring")
        
        # LLM-based evaluation for monitoring
        prompt = FLUENCY_EVALUATION_PROMPT.safe_format(
            model_answer=response.completion
        )
        
        try:
            llm_response = self.llm_client.generate_text(prompt)
            from ..core.utils import parse_json
            result = parse_json(llm_response, return_type="object")
            
            score = float(result.get("score", 0.0))
            explanation = result.get("explanation", "LLM evaluation")
            
            metric_result = {
                "value": score,
                "explanation": explanation,
                "method": "llm_evaluation",
                "timestamp": datetime.now().isoformat()
            }
            
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            
            return metric_result
            
        except Exception as e:
            logger.error(f"Error in LLM-based fluency evaluation: {e}")
            return {
                "value": 0.0,
                "explanation": f"Error in evaluation: {str(e)}",
                "method": "error",
                "timestamp": datetime.now().isoformat()
            }
    
    def calculate_evaluation(self, input_obj: Input, response: Response, reference: Response, llm_client=None) -> dict:
        """
        Calculate fluency in evaluation context.
        
        Args:
            input_obj: Input object
            response: Response object
            reference: Reference response object
            llm_client: Optional LLM client for evaluation
            
        Returns:
            Dict containing metric result
        """
        # Use provided client or stored client
        client = llm_client or self._llm_client
        
        if not client:
            raise ValueError("LLM client required for fluency evaluation")
        
        # Format prompt for evaluation context
        prompt = FLUENCY_EVALUATION_PROMPT.safe_format(
            model_answer=response.content
        )
        
        try:
            llm_response = client.generate_text(prompt)
            from ..core.utils import parse_json
            result = parse_json(llm_response, return_type="object")
            
            score = float(result.get("fluency", 0.0)) # Changed here
            explanation = result.get("explanation", "LLM evaluation")
            
            metric_result = {
                "value": score,
                "explanation": explanation,
                "method": "llm_evaluation",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            }
            
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_response
            
            return metric_result
            
        except Exception as e:
            logger.error(f"Error in fluency evaluation: {e}")
            return {
                "value": 0.0,
                "explanation": f"Error in evaluation: {str(e)}",
                "method": "error",
                "timestamp": datetime.now().isoformat(),
                "metadata": {}
            }
    
    
class ContextPrecisionMetric(LLMMeasuredBaseMetric):
    """
    Metric for evaluating the precision of retrieved contexts in both monitoring and evaluation contexts.
    
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
    context = MetricContext.BOTH  # Works in both contexts
    category = MetricCategory.QUALITY
    
    # Model-based requirements
    monitoring_requires = {
        "request": LLMRequest,
        "response": LLMResponse
    }
    evaluation_requires = {
        "input": Input,
        "response": Response
    }
    
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
    
    def _format_prompt_evaluation(self, input_obj: Input, response: Response, reference: Response = None) -> str:
           """
           Format the prompt for evaluation, specific to ContextPrecisionMetric.
           """
           if not self.prompt: # self.prompt should have been set in __init__
               raise ValueError("No prompt template provided for ContextPrecisionMetric")

           # ContextPrecisionMetric's prompt expects user_input, system_response, and retrieved_context
           # Ensure all documents in response.documents are concatenated for the context.
           if not response.documents:
               logger.warning(f"{self.name}: No documents found in response.documents for prompt formatting.")
               retrieved_context_str = "No context provided."
           else:
               retrieved_context_str = "\\n\\n---\\n\\n".join(
                   [doc.content for doc in response.documents if hasattr(doc, 'content') and doc.content]
               )
               if not retrieved_context_str.strip():
                   retrieved_context_str = "Provided context documents are empty."
           
           # The prompt CONTEXT_PRECISION_WITHOUT_REFERENCE_PROMPT uses {user_input}, {system_response}, {retrieved_context}
           # These are NOT standard placeholders that safe_format in LLMMeasuredBaseMetric would fill.
           # So, we directly format the template string here.
           
           prompt_template_str = str(self.prompt) # Get the template string from the Prompt object

           # Manually replace placeholders if they are simple like {placeholder}
           # This is a basic replacement; for more complex Prompt objects, you might need a different approach.
           try:
               formatted_prompt = prompt_template_str.format(
                   user_input=input_obj.content,
                   system_response=response.content,
                   retrieved_context=retrieved_context_str
               )
           except KeyError as e:
               logger.error(f"Failed to format prompt for {self.name}. Missing key: {e}. Prompt template: {prompt_template_str[:500]}...")
               # Fallback or raise error if formatting fails
               # For now, let's try to proceed with a partially formatted prompt or a generic error message
               # This part might need more robust error handling depending on Prompt object structure
               raise ValueError(f"Failed to format prompt for {self.name} due to missing key: {e}")

           return formatted_prompt
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

# Context Recall Metric (New)
class ContextRecallMetric(LLMMeasuredBaseMetric):
    name = "ContextRecall"
    description = "Measures how well the retrieved context covers the information needed to answer the query and support the model's answer."
    greater_is_better = True
    context = MetricContext.EVALUATION
    category = MetricCategory.QUALITY

    evaluation_requires = {
        "input": Input,
        "response": Response # Expects response.documents to contain the context
    }

    def __init__(self, mode=None, llm_client=None, include_raw_response=False):
        super().__init__(prompt=CONTEXT_RECALL_PROMPT, llm_client=llm_client, include_raw_response=include_raw_response, mode=mode)

    def calculate_evaluation(self, input_obj: Input, response: Response, reference: Response = None, llm_client=None) -> dict:
        client = llm_client or self._llm_client # Ensure this uses _llm_client
        if not client:
            raise ValueError(f"{self.name} evaluation requires an LLM client")

        if not response.documents:
            logger.warning(f"{self.name}: No documents found in response.documents. Returning score 0.")
            return {
                "value": 0.0,
                "explanation": "No context (documents) provided in the response object.",
                "method": "direct_check",
                "timestamp": datetime.now().isoformat()
            }
        
        retrieved_context_str = "\n\n---\n\n".join([doc.content for doc in response.documents if hasattr(doc, 'content') and doc.content])
        if not retrieved_context_str.strip():
            logger.warning(f"{self.name}: Context documents are empty. Returning score 0.")
            return {
                "value": 0.0,
                "explanation": "Provided context documents are empty.",
                "method": "direct_check",
                "timestamp": datetime.now().isoformat()
            }

        # Ensure self.prompt is the Prompt object
        current_prompt_obj = self.prompt
        if not isinstance(current_prompt_obj, Prompt):
            logger.error(f"{self.name}: self.prompt is not a Prompt object. It is {type(current_prompt_obj)}. Attempting to use global CONTEXT_RECALL_PROMPT.")
            if isinstance(CONTEXT_RECALL_PROMPT, Prompt):
                 current_prompt_obj = CONTEXT_RECALL_PROMPT
            else:
                # This is a deeper issue if CONTEXT_RECALL_PROMPT itself is not a Prompt object
                raise ValueError(f"{self.name}: CONTEXT_RECALL_PROMPT is not a valid Prompt object either.") 

        formatted_prompt = current_prompt_obj.safe_format(
            user_input=input_obj.content,
            model_answer=response.content,
            retrieved_context=retrieved_context_str
        )

        try:
            llm_eval_response = client.generate_text(formatted_prompt)
            result = parse_json(llm_eval_response, return_type="object")
            
            score = float(result.get("score", 0.0))
            explanation = result.get("explanation", "LLM evaluation failed to provide explanation.")
            
            metric_result = {
                "value": score,
                "explanation": explanation,
                "method": "llm_evaluation",
                "timestamp": datetime.now().isoformat()
            }
            
            if self._include_raw_response:
                metric_result["raw_llm_response"] = llm_eval_response
            
            return metric_result
            
        except Exception as e:
            logger.error(f"Error in {self.name} evaluation: {e}", exc_info=True)
            return {
                "value": 0.0,
                "explanation": f"Error during LLM evaluation: {str(e)}",
                "method": "error",
                "timestamp": datetime.now().isoformat()
            }

register_metric(ContextRecallMetric)
register_metric(ContextPrecisionMetric)
