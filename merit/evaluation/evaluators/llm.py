from .base import BaseEvaluator
from merit.core.prompts import Prompt
from merit.core.logging import get_logger
from merit.core.models import EvaluationResult, EvaluationReport
from merit.core.utils import parse_json

logger = get_logger(__name__)

class LLMEvaluator(BaseEvaluator):
    """
    LLM-based evaluator with custom prompt.
    
    This class uses an LLM to evaluate model outputs based on a provided prompt.
    """
    
    def __init__(
        self,
        prompt: Prompt,
        llm_client=None,
        prefix_messages=None,
        llm_temperature=0,
        llm_seed=42,
        llm_output_format="json_object",
        metrics=None
    ):
        """
        Initialize the LLM evaluator.
        
        Args:
            prompt: The evaluation prompt template
            llm_client: The LLM client
            prefix_messages: Prefix messages for the LLM
            llm_temperature: Temperature for LLM generation
            llm_seed: Seed for LLM generation
            llm_output_format: Output format for LLM generation
            metrics: List of metrics to evaluate
        """
        super().__init__(None, metrics)
        self.prompt = prompt
        self.llm_client = llm_client
        self.prefix_messages = prefix_messages or []
        self.llm_temperature = llm_temperature
        self.llm_seed = llm_seed
        self.llm_output_format = llm_output_format
    
    def evaluate(self, model, dataset) -> EvaluationReport:
        """
        Evaluate the model on the dataset.
        
        Args:
            model: The model to evaluate
            dataset: The dataset to evaluate on
            
        Returns:
            EvaluationReport: The evaluation report
        """
        # Get model outputs
        model_outputs = model.predict(dataset).prediction
        
        # Initialize results
        results = []
        errors = []
        
        # Evaluate each sample
        for (row_id, row), model_output in zip(dataset.df.iterrows(), model_outputs):
            # Extract input variables
            input_vars = {k: v for k, v in row.items() if k in model.feature_names}
            if len(input_vars) == 1:
                input_vars = list(input_vars.values())[0]
            
            # Extract metadata
            input_meta = {k: v for k, v in row.items() if k not in model.feature_names}
            input_meta["__sample_id"] = row_id
            
            # Create conversation
            conversation = [{"role": "user", "content": input_vars}, {"role": "agent", "content": model_output}]
            
            # Create sample
            sample = {
                "conversation": conversation,
                "meta": input_meta
            }
            
            # Evaluate sample
            try:
                eval_result = self._evaluate_sample(model, sample)
                results.append(eval_result)
            except Exception as err:
                logger.error(f"Error evaluating sample: {str(err)}")
                errors.append({"error": str(err), "sample": sample})
        
        # Create report
        return self._create_report(results, errors)
    
    def _evaluate_sample(self, model, sample) -> EvaluationResult:
        """
        Evaluate a single sample.
        
        Args:
            model: The model
            sample: The sample
            
        Returns:
            EvaluationResult: The evaluation result
        """
        # Format messages
        messages = self._format_messages(model, sample["conversation"], meta=sample.get("meta"))
        
        # Get evaluation from LLM
        response = self.llm_client.generate_chat_response(
            messages,
            temperature=self.llm_temperature,
            seed=self.llm_seed,
            format=self.llm_output_format
        )
        
        # Parse evaluation
        eval_result = self._parse_evaluation_output(response)
        
        # Create evaluation result
        result = EvaluationResult(
            input=sample["conversation"][0]["content"],
            response=sample["conversation"][1]["content"],
            metadata=sample.get("meta", {})
        )
        
        # Add the complete evaluation result to metrics
        result.metrics.append(eval_result)
        
        return result
    # NOTE what is this
    # NOTE we should try doing this as Input-Response Pairs  
    def _format_messages(self, model, conversation, meta=None):
        """
        Format a conversation into messages for LLM processing.
        
        This method takes a conversation (a list of message dictionaries with 'role' and 'content'),
        formats it into a human-readable string representation, and then incorporates it into
        the prompt template along with model information and metadata. The result is a list of
        message dictionaries ready to be sent to the LLM.
        
        Args:
            model: The model being evaluated
            conversation: List of message dictionaries, each containing 'role' and 'content' keys
            meta: Optional dictionary of additional metadata to include in the prompt
            
        Returns:
            list: Formatted messages ready for LLM submission, consisting of any prefix messages
                  followed by a user message containing the formatted prompt
        """
        # Format conversation into a string
        formatted_conversation = ""
        #NOTE what is this 
        for message in conversation:
            role = message.get("role", "").lower()
            content = message.get("content", "")
            
            if role == "user":
                formatted_conversation += f"User: {content}\n"
            elif role in ["assistant", "agent"]:
                formatted_conversation += f"Assistant: {content}\n"
            else:
                formatted_conversation += f"{role.capitalize()}: {content}\n"
        
        formatted_conversation = formatted_conversation.strip()
        
        # Format prompt with the conversation string
        formatted_prompt = self.prompt.format(
            model=model,
            conversation=formatted_conversation,
            meta=meta or {}
        )
        
        # Create and return messages
        return self.prefix_messages + [{"role": "user", "content": formatted_prompt}]
    
    def _parse_evaluation_output(self, response):
        """
        Parse the evaluation output.
        
        Args:
            response: The LLM response
            
        Returns:
            dict: The parsed evaluation
        """
        try:
            content = response.get("content", "{}")
            return parse_json(content)
        except Exception as err:
            logger.error(f"Error parsing evaluation output: {str(err)}")
            return {"error": str(err)}
    
    def _create_report(self, results, errors=None) -> EvaluationReport:
        """
        Create an evaluation report.
        
        Args:
            results: The evaluation results
            errors: Any errors that occurred during evaluation
            
        Returns:
            EvaluationReport: The evaluation report
        """
        # Get metric names
        metric_names = set()
        for result in results:
            for metric_result in result.metrics:
                metric_names.update(metric_result.keys())
        
        # Create report
        report = EvaluationReport(
            results=results,
            metrics=list(metric_names),
            metadata={
                "num_samples": len(results),
                "num_errors": len(errors or []),
                "errors": errors or []
            }
        )
        
        return report
