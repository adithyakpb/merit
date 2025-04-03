# Custom Evaluators

This guide explains how to create custom evaluators in MERIT. Custom evaluators allow you to implement specialized evaluation logic for your AI systems, tailored to your specific use cases and requirements.

## Why Create Custom Evaluators?

You might want to create custom evaluators for several reasons:

- **Domain-Specific Evaluation**: Implement evaluation logic specific to your domain (e.g., medical, legal, financial)
- **Custom Evaluation Workflows**: Create evaluation workflows that differ from the standard ones
- **Specialized Analysis**: Perform specialized analysis of evaluation results
- **Integration with External Systems**: Integrate evaluation with your existing systems
- **Novel Evaluation Approaches**: Experiment with new evaluation methodologies

## Evaluator Types in MERIT

MERIT includes several built-in evaluator types:

- **RAGEvaluator**: For evaluating RAG (Retrieval-Augmented Generation) systems
- **LLMEvaluator**: For evaluating general LLM outputs
- **ClassificationEvaluator**: For evaluating classification tasks

All of these evaluators extend the `BaseEvaluator` class, which provides the foundation for creating custom evaluators.

## Creating a Basic Custom Evaluator

To create a custom evaluator, you need to extend the `BaseEvaluator` class and implement the required methods:

```python
from merit.evaluation.evaluators.base import BaseEvaluator
from merit.core.models import EvaluationReport, EvaluationResult
from typing import List, Dict, Any, Callable

class CustomEvaluator(BaseEvaluator):
    """A custom evaluator for specialized evaluation."""
    
    def __init__(self, metrics=None, **kwargs):
        """
        Initialize the custom evaluator.
        
        Args:
            metrics: The metrics to use for evaluation.
            **kwargs: Additional arguments for the evaluator.
        """
        super().__init__(metrics)
        self._kwargs = kwargs
    
    def evaluate(self, model, dataset) -> EvaluationReport:
        """
        Evaluate a model on a dataset.
        
        Args:
            model: The model to evaluate.
            dataset: The dataset to evaluate on.
            
        Returns:
            EvaluationReport: The evaluation report.
        """
        # Implement your custom evaluation logic here
        results = []
        
        # Evaluate each sample in the dataset
        for i, sample in enumerate(dataset.df.itertuples()):
            # Get the input
            input_text = getattr(sample, model.feature_names[0])
            
            # Get the model's prediction
            prediction = model.predict(dataset.df.iloc[i:i+1]).prediction[0]
            
            # Evaluate the prediction
            result = self._evaluate_sample(input_text, prediction)
            
            # Add the result to the list
            results.append(result)
        
        # Create an evaluation report
        report = EvaluationReport(
            results=results,
            metrics=[metric.name for metric in self.metrics],
            metadata={"evaluator": self.__class__.__name__}
        )
        
        return report
    
    def _evaluate_sample(self, input_sample, answer) -> EvaluationResult:
        """
        Evaluate a single sample.
        
        Args:
            input_sample: The input sample.
            answer: The model's answer.
            
        Returns:
            EvaluationResult: The evaluation result.
        """
        # Apply each metric to the sample
        scores = {}
        explanations = {}
        errors = {}
        
        for metric in self.metrics:
            try:
                # Apply the metric
                result = metric(input_sample, answer)
                
                # Extract the score and explanation
                for key, value in result.items():
                    if key == metric.name:
                        scores[key] = value
                    elif key.endswith("_explanation"):
                        explanations[key] = value
                    elif key.endswith("_errors"):
                        errors[key] = value
            except Exception as e:
                # Handle metric errors
                errors[f"{metric.name}_error"] = str(e)
        
        # Create an evaluation result
        result = EvaluationResult(
            input=input_sample,
            model_answer=answer,
            scores=scores,
            explanations=explanations,
            errors=errors
        )
        
        return result
```

## Creating a Domain-Specific Evaluator

Here's an example of a domain-specific evaluator for medical question answering:

```python
from merit.evaluation.evaluators.base import BaseEvaluator
from merit.core.models import EvaluationReport, EvaluationResult
from merit.evaluation.metrics.base import BaseMetric
from typing import List, Dict, Any, Callable

class MedicalEvaluator(BaseEvaluator):
    """An evaluator for medical question answering systems."""
    
    def __init__(self, metrics=None, medical_ontology=None, **kwargs):
        """
        Initialize the medical evaluator.
        
        Args:
            metrics: The metrics to use for evaluation.
            medical_ontology: A medical ontology for evaluation.
            **kwargs: Additional arguments for the evaluator.
        """
        super().__init__(metrics)
        self.medical_ontology = medical_ontology or {}
        self._kwargs = kwargs
    
    def evaluate(self, model, dataset) -> EvaluationReport:
        """
        Evaluate a medical question answering system.
        
        Args:
            model: The model to evaluate.
            dataset: The dataset to evaluate on.
            
        Returns:
            EvaluationReport: The evaluation report.
        """
        results = []
        
        # Evaluate each sample in the dataset
        for i, sample in enumerate(dataset.df.itertuples()):
            # Get the input
            input_text = getattr(sample, model.feature_names[0])
            
            # Get the model's prediction
            prediction = model.predict(dataset.df.iloc[i:i+1]).prediction[0]
            
            # Evaluate the prediction
            result = self._evaluate_sample(input_text, prediction)
            
            # Add medical-specific evaluation
            medical_scores, medical_explanations = self._evaluate_medical_aspects(input_text, prediction)
            
            # Update the result with medical-specific evaluation
            result.scores.update(medical_scores)
            result.explanations.update(medical_explanations)
            
            # Add the result to the list
            results.append(result)
        
        # Create an evaluation report
        report = EvaluationReport(
            results=results,
            metrics=[metric.name for metric in self.metrics] + list(medical_scores.keys()),
            metadata={
                "evaluator": self.__class__.__name__,
                "domain": "medical"
            }
        )
        
        return report
    
    def _evaluate_medical_aspects(self, input_text, answer) -> tuple:
        """
        Evaluate medical-specific aspects of the answer.
        
        Args:
            input_text: The input text.
            answer: The model's answer.
            
        Returns:
            tuple: A tuple containing medical scores and explanations.
        """
        # Implement medical-specific evaluation logic here
        scores = {}
        explanations = {}
        
        # Example: Check for medical terminology
        medical_terms = self._extract_medical_terms(answer)
        terminology_score = len(medical_terms) / 10  # Simplified scoring
        scores["medical_terminology"] = min(1.0, terminology_score)
        explanations["medical_terminology_explanation"] = f"The answer contains {len(medical_terms)} medical terms."
        
        # Example: Check for medical accuracy
        accuracy_score = self._check_medical_accuracy(answer)
        scores["medical_accuracy"] = accuracy_score
        explanations["medical_accuracy_explanation"] = f"The answer has a medical accuracy score of {accuracy_score:.2f}."
        
        # Example: Check for completeness
        completeness_score = self._check_medical_completeness(input_text, answer)
        scores["medical_completeness"] = completeness_score
        explanations["medical_completeness_explanation"] = f"The answer has a medical completeness score of {completeness_score:.2f}."
        
        return scores, explanations
    
    def _extract_medical_terms(self, text) -> List[str]:
        """
        Extract medical terms from text.
        
        Args:
            text: The text to extract terms from.
            
        Returns:
            List[str]: The extracted medical terms.
        """
        # Implement medical term extraction logic here
        # This is a simplified example
        medical_terms = []
        
        for term in self.medical_ontology.keys():
            if term.lower() in text.lower():
                medical_terms.append(term)
        
        return medical_terms
    
    def _check_medical_accuracy(self, text) -> float:
        """
        Check the medical accuracy of text.
        
        Args:
            text: The text to check.
            
        Returns:
            float: The medical accuracy score.
        """
        # Implement medical accuracy checking logic here
        # This is a simplified example
        import random
        return random.random()
    
    def _check_medical_completeness(self, query, answer) -> float:
        """
        Check the medical completeness of an answer.
        
        Args:
            query: The query.
            answer: The answer to check.
            
        Returns:
            float: The medical completeness score.
        """
        # Implement medical completeness checking logic here
        # This is a simplified example
        import random
        return random.random()
```

## Creating a Custom RAG Evaluator

Here's an example of a custom RAG evaluator that extends the built-in `RAGEvaluator`:

```python
from merit.evaluation.evaluators.rag import RAGEvaluator
from merit.core.models import EvaluationReport, EvaluationResult
from typing import List, Dict, Any, Callable

class CustomRAGEvaluator(RAGEvaluator):
    """A custom RAG evaluator with additional analysis."""
    
    def __init__(self, knowledge_base, testset, llm_client=None, metrics=None, agent_description=None, **kwargs):
        """
        Initialize the custom RAG evaluator.
        
        Args:
            knowledge_base: The knowledge base to use for evaluation.
            testset: The test set to evaluate on.
            llm_client: The LLM client to use for evaluation.
            metrics: The metrics to use for evaluation.
            agent_description: A description of the agent being evaluated.
            **kwargs: Additional arguments for the evaluator.
        """
        super().__init__(knowledge_base, testset, llm_client, metrics, agent_description)
        self._kwargs = kwargs
    
    def evaluate(self, answer_fn) -> EvaluationReport:
        """
        Evaluate a RAG system.
        
        Args:
            answer_fn: A function that takes a query and returns an answer.
            
        Returns:
            EvaluationReport: The evaluation report.
        """
        # Use the parent class to evaluate the RAG system
        report = super().evaluate(answer_fn)
        
        # Add custom analysis to the report
        self._add_custom_analysis(report)
        
        return report
    
    def _add_custom_analysis(self, report) -> None:
        """
        Add custom analysis to the evaluation report.
        
        Args:
            report: The evaluation report to add analysis to.
        """
        # Implement custom analysis logic here
        # This is a simplified example
        
        # Calculate the percentage of answers with hallucinations
        hallucination_count = 0
        for result in report.results:
            if "faithfulness" in result.scores and result.scores["faithfulness"] < 0.5:
                hallucination_count += 1
        
        hallucination_percentage = hallucination_count / len(report.results) * 100
        
        # Add the analysis to the report metadata
        report.metadata["hallucination_percentage"] = hallucination_percentage
        
        # Calculate the average score for each metric
        metric_averages = {}
        for metric in report.metrics:
            scores = [result.scores.get(metric, 0) for result in report.results]
            metric_averages[f"{metric}_average"] = sum(scores) / len(scores)
        
        # Add the metric averages to the report metadata
        report.metadata.update(metric_averages)
        
        # Identify the best and worst performing samples
        for metric in report.metrics:
            # Sort results by score
            sorted_results = sorted(report.results, key=lambda r: r.scores.get(metric, 0))
            
            # Get the worst performing samples
            worst_samples = sorted_results[:3]
            worst_sample_ids = [i for i, r in enumerate(report.results) if r in worst_samples]
            
            # Get the best performing samples
            best_samples = sorted_results[-3:]
            best_sample_ids = [i for i, r in enumerate(report.results) if r in best_samples]
            
            # Add the best and worst sample IDs to the report metadata
            report.metadata[f"{metric}_worst_samples"] = worst_sample_ids
            report.metadata[f"{metric}_best_samples"] = best_sample_ids
```

## Creating a Custom LLM Evaluator

Here's an example of a custom LLM evaluator that extends the built-in `LLMEvaluator`:

```python
from merit.evaluation.evaluators.llm import LLMEvaluator
from merit.core.models import EvaluationReport, EvaluationResult
from merit.core.prompts import Prompt
from typing import List, Dict, Any, Callable

class CustomLLMEvaluator(LLMEvaluator):
    """A custom LLM evaluator with additional features."""
    
    def __init__(self, prompt=None, llm_client=None, llm_temperature=0, llm_seed=None, llm_output_format="json_object", **kwargs):
        """
        Initialize the custom LLM evaluator.
        
        Args:
            prompt: The prompt to use for evaluation.
            llm_client: The LLM client to use for evaluation.
            llm_temperature: The temperature to use for the LLM.
            llm_seed: The seed to use for the LLM.
            llm_output_format: The output format for the LLM.
            **kwargs: Additional arguments for the evaluator.
        """
        # Create a custom prompt if not provided
        if prompt is None:
            prompt = Prompt(
                """
                Evaluate the following response to the given prompt:
                
                Prompt: {conversation[0]['content']}
                
                Response: {conversation[1]['content']}
                
                Please rate the response on a scale of 1-10 for the following criteria:
                - Accuracy: How factually correct is the response?
                - Relevance: How well does the response address the prompt?
                - Coherence: How logically structured and consistent is the response?
                - Fluency: How grammatically correct and natural is the response?
                - Helpfulness: How helpful is the response for the user's needs?
                - Creativity: How original and creative is the response?
                - Conciseness: How concise and to-the-point is the response?
                
                For each criterion, provide a score and a brief explanation.
                
                Return your evaluation in the following JSON format:
                {
                    "accuracy": {"score": 0, "explanation": ""},
                    "relevance": {"score": 0, "explanation": ""},
                    "coherence": {"score": 0, "explanation": ""},
                    "fluency": {"score": 0, "explanation": ""},
                    "helpfulness": {"score": 0, "explanation": ""},
                    "creativity": {"score": 0, "explanation": ""},
                    "conciseness": {"score": 0, "explanation": ""}
                }
                """
            )
        
        super().__init__(prompt, llm_client, llm_temperature, llm_seed, llm_output_format)
        self._kwargs = kwargs
    
    def evaluate(self, model, dataset) -> EvaluationReport:
        """
        Evaluate an LLM.
        
        Args:
            model: The model to evaluate.
            dataset: The dataset to evaluate on.
            
        Returns:
            EvaluationReport: The evaluation report.
        """
        # Use the parent class to evaluate the LLM
        report = super().evaluate(model, dataset)
        
        # Add custom analysis to the report
        self._add_custom_analysis(report)
        
        return report
    
    def _add_custom_analysis(self, report) -> None:
        """
        Add custom analysis to the evaluation report.
        
        Args:
            report: The evaluation report to add analysis to.
        """
        # Implement custom analysis logic here
        # This is a simplified example
        
        # Calculate the average score for each criterion
        criterion_averages = {}
        criteria = ["accuracy", "relevance", "coherence", "fluency", "helpfulness", "creativity", "conciseness"]
        
        for criterion in criteria:
            scores = [result.scores.get(criterion, 0) for result in report.results]
            criterion_averages[f"{criterion}_average"] = sum(scores) / len(scores)
        
        # Add the criterion averages to the report metadata
        report.metadata.update(criterion_averages)
        
        # Calculate the overall quality score
        overall_scores = []
        for result in report.results:
            criterion_scores = [result.scores.get(criterion, 0) for criterion in criteria]
            overall_score = sum(criterion_scores) / len(criterion_scores)
            overall_scores.append(overall_score)
            
            # Add the overall score to the result
            result.scores["overall_quality"] = overall_score
        
        # Add the overall quality average to the report metadata
        report.metadata["overall_quality_average"] = sum(overall_scores) / len(overall_scores)
        
        # Add the overall quality metric to the report metrics
        report.metrics.append("overall_quality")
```

## Using Custom Evaluators

Once you've created a custom evaluator, you can use it to evaluate your AI systems:

```python
from merit.core.models import TestSet
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient
from merit.evaluation.metrics.rag import CorrectnessMetric, FaithfulnessMetric, RelevanceMetric

# Create a knowledge base and test set
knowledge_base = KnowledgeBase.load("my_knowledge_base.json")
test_set = TestSet.load("my_test_set.json")

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Create metrics
metrics = [
    CorrectnessMetric(llm_client=client),
    FaithfulnessMetric(llm_client=client),
    RelevanceMetric(llm_client=client)
]

# Create a custom evaluator
evaluator = CustomRAGEvaluator(
    knowledge_base=knowledge_base,
    testset=test_set,
    llm_client=client,
    metrics=metrics,
    agent_description="A chatbot that answers questions based on a knowledge base."
)

# Define an answer function
def get_answer(query):
    # Your RAG system implementation here
    return "This is the answer to the query."

# Evaluate the RAG system
report = evaluator.evaluate(get_answer)

# Print the evaluation results
print("Evaluation results:")
for metric, score in report.get_overall_scores().items():
    print(f"{metric}: {score:.2f}")

# Print custom analysis
print("\nCustom analysis:")
for key, value in report.metadata.items():
    if key not in ["evaluator", "domain"]:
        print(f"{key}: {value}")
```

## Best Practices for Custom Evaluators

When creating custom evaluators, follow these best practices:

### 1. Extend the Appropriate Base Class

Choose the appropriate base class to extend based on your evaluation needs:

- `BaseEvaluator`: For completely custom evaluators
- `RAGEvaluator`: For customizing RAG evaluation
- `LLMEvaluator`: For customizing LLM evaluation
- `ClassificationEvaluator`: For customizing classification evaluation

```python
# For a completely custom evaluator
class MyCustomEvaluator(BaseEvaluator):
    # Implementation...

# For a custom RAG evaluator
class MyCustomRAGEvaluator(RAGEvaluator):
    # Implementation...

# For a custom LLM evaluator
class MyCustomLLMEvaluator(LLMEvaluator):
    # Implementation...
```

### 2. Implement Required Methods

Ensure you implement all required methods for your evaluator:

```python
def evaluate(self, model, dataset) -> EvaluationReport:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate.
        dataset: The dataset to evaluate on.
        
    Returns:
        EvaluationReport: The evaluation report.
    """
    # Implementation...

def _evaluate_sample(self, input_sample, answer) -> EvaluationResult:
    """
    Evaluate a single sample.
    
    Args:
        input_sample: The input sample.
        answer: The model's answer.
        
    Returns:
        EvaluationResult: The evaluation result.
    """
    # Implementation...
```

### 3. Handle Errors Gracefully

Handle errors gracefully to ensure the evaluation process doesn't fail:

```python
def _evaluate_sample(self, input_sample, answer) -> EvaluationResult:
    # Apply each metric to the sample
    scores = {}
    explanations = {}
    errors = {}
    
    for metric in self.metrics:
        try:
            # Apply the metric
            result = metric(input_sample, answer)
            
            # Extract the score and explanation
            for key, value in result.items():
                if key == metric.name:
                    scores[key] = value
                elif key.endswith("_explanation"):
                    explanations[key] = value
        except Exception as e:
            # Handle metric errors
            errors[f"{metric.name}_error"] = str(e)
    
    # Create an evaluation result
    result = EvaluationResult(
        input=input_sample,
        model_answer=answer,
        scores=scores,
        explanations=explanations,
        errors=errors
    )
    
    return result
```

### 4. Add Useful Metadata

Add useful metadata to your evaluation reports:

```python
def evaluate(self, model, dataset) -> EvaluationReport:
    # Evaluate each sample...
    
    # Create an evaluation report with metadata
    report = EvaluationReport(
        results=results,
        metrics=[metric.name for metric in self.metrics],
        metadata={
            "evaluator": self.__class__.__name__,
            "dataset_size": len(dataset.df),
            "evaluation_time": datetime.now().isoformat(),
            "model_name": getattr(model, "name", "unknown")
        }
    )
    
    return report
```

### 5. Document Your Evaluator

Document your evaluator thoroughly:

```python
class CustomEvaluator(BaseEvaluator):
    """
    A custom evaluator for specialized evaluation.
    
    This evaluator extends the BaseEvaluator to provide specialized evaluation
    for a specific use case. It adds custom analysis and additional metrics
    to the evaluation process.
    
    Args:
        metrics: The metrics to use for evaluation.
        custom_param: A custom parameter for the evaluator.
        **kwargs: Additional arguments for the evaluator.
    """
    
    def __init__(self, metrics=None, custom_param=None, **kwargs):
        """
        Initialize the custom evaluator.
        
        Args:
            metrics: The metrics to use for evaluation.
            custom_param: A custom parameter for the evaluator.
            **kwargs: Additional arguments for the evaluator.
        """
        super().__init__(metrics)
        self.custom_param = custom_param
        self._kwargs = kwargs
    
    def evaluate(self, model, dataset) -> EvaluationReport:
        """
        Evaluate a model on a dataset.
        
        This method evaluates the model on the dataset using the specified metrics
        and adds custom analysis to the evaluation report.
        
        Args:
            model: The model to evaluate.
            dataset: The dataset to evaluate on.
            
        Returns:
            EvaluationReport: The evaluation report.
        """
        # Implementation...
```

## Next Steps

Now that you know how to create custom evaluators, you can:

- Learn how to create [custom metrics](./custom_metrics.md) for specialized evaluation
- Explore how to define [custom prompts](./custom_prompts.md) for evaluation
- Discover how to create [custom knowledge bases](../knowledge_bases/custom_knowledge_bases.md) for evaluation
- Learn how to create [custom report templates](./report_templates.md) for presenting evaluation results
