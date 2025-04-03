# Custom Metrics

This guide explains how to create custom evaluation metrics in MERIT. Custom metrics allow you to evaluate specific aspects of AI system performance that are important for your use case.

## Why Create Custom Metrics?

You might want to create custom metrics for several reasons:

- **Domain-Specific Evaluation**: Evaluate aspects specific to your domain (e.g., medical accuracy, legal compliance)
- **Task-Specific Criteria**: Measure performance on specific tasks (e.g., code generation, data extraction)
- **Custom Quality Standards**: Implement your organization's quality standards
- **Novel Evaluation Approaches**: Experiment with new evaluation methodologies
- **Specialized Analysis**: Perform specialized analysis of AI system outputs

## Metric Types in MERIT

MERIT supports several types of metrics:

- **RAG Metrics**: For evaluating RAG (Retrieval-Augmented Generation) systems
- **Classification Metrics**: For evaluating classification tasks
- **LLM-based Metrics**: For evaluating general LLM outputs
- **Custom Metrics**: For specialized evaluation needs

## Creating a Basic Custom Metric

To create a custom metric, you need to extend the appropriate base metric class and implement the required methods:

```python
from merit.evaluation.metrics.base import BaseMetric
from typing import Dict, Any

class CustomMetric(BaseMetric):
    """A custom metric for evaluating a specific aspect of AI system performance."""
    
    def __init__(self, name="custom_metric", **kwargs):
        """
        Initialize the custom metric.
        
        Args:
            name: The name of the metric.
            **kwargs: Additional arguments for the metric.
        """
        self.name = name
        self._kwargs = kwargs
    
    def __call__(self, input_sample, answer) -> Dict[str, Any]:
        """
        Calculate the metric for a given input and answer.
        
        Args:
            input_sample: The input sample (question/prompt).
            answer: The answer to evaluate.
            
        Returns:
            Dict[str, Any]: A dictionary containing the metric score and explanation.
        """
        # Calculate the metric score
        score = self._calculate_score(input_sample, answer)
        
        # Generate an explanation for the score
        explanation = self._generate_explanation(input_sample, answer, score)
        
        # Return the score and explanation
        return {
            self.name: score,
            f"{self.name}_explanation": explanation
        }
    
    def _calculate_score(self, input_sample, answer) -> float:
        """
        Calculate the metric score.
        
        Args:
            input_sample: The input sample (question/prompt).
            answer: The answer to evaluate.
            
        Returns:
            float: The metric score.
        """
        # Implement your custom scoring logic here
        # This is a simple example that returns a random score
        import random
        return random.random()
    
    def _generate_explanation(self, input_sample, answer, score) -> str:
        """
        Generate an explanation for the score.
        
        Args:
            input_sample: The input sample (question/prompt).
            answer: The answer to evaluate.
            score: The metric score.
            
        Returns:
            str: The explanation for the score.
        """
        # Implement your custom explanation logic here
        # This is a simple example that returns a generic explanation
        if score >= 0.8:
            return "The answer is excellent."
        elif score >= 0.6:
            return "The answer is good but could be improved."
        elif score >= 0.4:
            return "The answer is acceptable but has significant issues."
        else:
            return "The answer is poor and needs substantial improvement."
```

## Creating a RAG Metric

For evaluating RAG systems, you can extend the `RAGMetric` class:

```python
from merit.evaluation.metrics.rag import RAGMetric
from typing import Dict, Any, List, Tuple
from merit.core.models import Document

class CustomRAGMetric(RAGMetric):
    """A custom metric for evaluating a specific aspect of RAG system performance."""
    
    def __init__(self, name="custom_rag_metric", llm_client=None, **kwargs):
        """
        Initialize the custom RAG metric.
        
        Args:
            name: The name of the metric.
            llm_client: The LLM client to use for evaluation.
            **kwargs: Additional arguments for the metric.
        """
        self.name = name
        self._llm_client = llm_client
        self._kwargs = kwargs
    
    def __call__(self, input_sample, answer) -> Dict[str, Any]:
        """
        Calculate the metric for a given input and answer.
        
        Args:
            input_sample: The input sample (question/prompt).
            answer: The answer to evaluate.
            
        Returns:
            Dict[str, Any]: A dictionary containing the metric score and explanation.
        """
        # Get the reference answer and document
        reference_answer = input_sample.reference_answer
        document = input_sample.document
        
        # Get the retrieved documents if available
        retrieved_docs = []
        if hasattr(answer, "documents") and answer.documents:
            retrieved_docs = answer.documents
        
        # Get the answer text
        answer_text = answer.message if hasattr(answer, "message") else answer
        
        # Calculate the metric score
        score = self._calculate_score(input_sample.input, answer_text, reference_answer, document, retrieved_docs)
        
        # Generate an explanation for the score
        explanation = self._generate_explanation(input_sample.input, answer_text, reference_answer, document, retrieved_docs, score)
        
        # Return the score and explanation
        return {
            self.name: score,
            f"{self.name}_explanation": explanation
        }
    
    def _calculate_score(self, query, answer, reference_answer, document, retrieved_docs) -> float:
        """
        Calculate the metric score.
        
        Args:
            query: The query (question/prompt).
            answer: The answer to evaluate.
            reference_answer: The reference answer.
            document: The source document.
            retrieved_docs: The retrieved documents.
            
        Returns:
            float: The metric score.
        """
        # Implement your custom scoring logic here
        # This is a simple example that calculates a score based on document overlap
        if not retrieved_docs:
            return 0.0
        
        # Check if the source document is in the retrieved documents
        source_doc_id = document.id
        retrieved_doc_ids = [doc.id for doc in retrieved_docs]
        
        if source_doc_id in retrieved_doc_ids:
            return 1.0
        else:
            return 0.0
    
    def _generate_explanation(self, query, answer, reference_answer, document, retrieved_docs, score) -> str:
        """
        Generate an explanation for the score.
        
        Args:
            query: The query (question/prompt).
            answer: The answer to evaluate.
            reference_answer: The reference answer.
            document: The source document.
            retrieved_docs: The retrieved documents.
            score: The metric score.
            
        Returns:
            str: The explanation for the score.
        """
        # Implement your custom explanation logic here
        if score == 1.0:
            return "The system correctly retrieved the source document."
        else:
            return "The system failed to retrieve the source document."
```

## Creating an LLM-Based Metric

For more complex evaluation, you can use an LLM to evaluate the outputs:

```python
from merit.evaluation.metrics.base import BaseMetric
from merit.core.prompts import Prompt
from typing import Dict, Any

class LLMBasedMetric(BaseMetric):
    """A metric that uses an LLM to evaluate outputs."""
    
    def __init__(self, name="llm_based_metric", llm_client=None, prompt=None, **kwargs):
        """
        Initialize the LLM-based metric.
        
        Args:
            name: The name of the metric.
            llm_client: The LLM client to use for evaluation.
            prompt: The prompt to use for evaluation.
            **kwargs: Additional arguments for the metric.
        """
        self.name = name
        self._llm_client = llm_client
        self._prompt = prompt or Prompt(
            """
            Evaluate the following answer to the given question:
            
            Question: {question}
            
            Answer: {answer}
            
            Rate the answer on a scale of 0 to 10 for the following criteria:
            - Accuracy: How factually correct is the answer?
            - Completeness: How complete is the answer?
            - Clarity: How clear and understandable is the answer?
            
            For each criterion, provide a score and a brief explanation.
            
            Return your evaluation in the following JSON format:
            {
                "accuracy": {"score": 0, "explanation": ""},
                "completeness": {"score": 0, "explanation": ""},
                "clarity": {"score": 0, "explanation": ""}
            }
            """
        )
        self._kwargs = kwargs
    
    def __call__(self, input_sample, answer) -> Dict[str, Any]:
        """
        Calculate the metric for a given input and answer.
        
        Args:
            input_sample: The input sample (question/prompt).
            answer: The answer to evaluate.
            
        Returns:
            Dict[str, Any]: A dictionary containing the metric scores and explanations.
        """
        if not self._llm_client:
            raise ValueError("LLM client is required for LLM-based metrics")
        
        # Get the answer text
        answer_text = answer.message if hasattr(answer, "message") else answer
        
        # Format the prompt
        prompt = self._prompt.format(
            question=input_sample.input,
            answer=answer_text
        )
        
        # Generate the evaluation
        evaluation = self._llm_client.generate_text(prompt)
        
        # Parse the evaluation
        try:
            import json
            result = json.loads(evaluation)
            
            # Extract the scores and explanations
            metrics = {}
            for criterion, data in result.items():
                score = data.get("score", 0)
                explanation = data.get("explanation", "")
                
                # Normalize the score to a 0-1 range
                normalized_score = score / 10.0
                
                # Add the score and explanation to the metrics
                metrics[criterion] = normalized_score
                metrics[f"{criterion}_explanation"] = explanation
            
            return metrics
        except Exception as e:
            # If parsing fails, return a default result
            return {
                self.name: 0.0,
                f"{self.name}_explanation": f"Failed to parse evaluation: {str(e)}"
            }
```

## Example: Domain-Specific Metric

Here's an example of a domain-specific metric for evaluating medical answers:

```python
from merit.evaluation.metrics.base import BaseMetric
from typing import Dict, Any, List

class MedicalAccuracyMetric(BaseMetric):
    """A metric for evaluating the medical accuracy of answers."""
    
    def __init__(self, name="medical_accuracy", medical_terms=None, **kwargs):
        """
        Initialize the medical accuracy metric.
        
        Args:
            name: The name of the metric.
            medical_terms: A dictionary of medical terms and their definitions.
            **kwargs: Additional arguments for the metric.
        """
        self.name = name
        self.medical_terms = medical_terms or {
            "diabetes": "A metabolic disease that causes high blood sugar",
            "hypertension": "High blood pressure",
            "asthma": "A condition that affects the airways in the lungs",
            "arthritis": "Inflammation of one or more joints",
            "cancer": "A disease in which abnormal cells divide uncontrollably"
        }
        self._kwargs = kwargs
    
    def __call__(self, input_sample, answer) -> Dict[str, Any]:
        """
        Calculate the metric for a given input and answer.
        
        Args:
            input_sample: The input sample (question/prompt).
            answer: The answer to evaluate.
            
        Returns:
            Dict[str, Any]: A dictionary containing the metric score and explanation.
        """
        # Get the answer text
        answer_text = answer.message if hasattr(answer, "message") else answer
        
        # Calculate the metric score
        score, errors = self._calculate_score(input_sample.input, answer_text)
        
        # Generate an explanation for the score
        explanation = self._generate_explanation(input_sample.input, answer_text, score, errors)
        
        # Return the score, explanation, and errors
        return {
            self.name: score,
            f"{self.name}_explanation": explanation,
            f"{self.name}_errors": errors
        }
    
    def _calculate_score(self, query, answer) -> tuple:
        """
        Calculate the medical accuracy score.
        
        Args:
            query: The query (question/prompt).
            answer: The answer to evaluate.
            
        Returns:
            tuple: A tuple containing the score and a list of errors.
        """
        # Implement medical accuracy scoring logic here
        # This is a simple example that checks for correct usage of medical terms
        errors = []
        correct_terms = 0
        total_terms = 0
        
        # Check for medical terms in the answer
        for term, definition in self.medical_terms.items():
            if term.lower() in answer.lower():
                total_terms += 1
                
                # Check if the term is used correctly (simplified example)
                if "not " + term.lower() not in answer.lower():
                    correct_terms += 1
                else:
                    errors.append(f"Incorrect usage of term '{term}'")
        
        # Calculate the score
        score = correct_terms / max(total_terms, 1)
        
        return score, errors
    
    def _generate_explanation(self, query, answer, score, errors) -> str:
        """
        Generate an explanation for the score.
        
        Args:
            query: The query (question/prompt).
            answer: The answer to evaluate.
            score: The metric score.
            errors: A list of errors.
            
        Returns:
            str: The explanation for the score.
        """
        # Implement explanation generation logic here
        if score == 1.0:
            return "The answer uses medical terms correctly."
        elif score >= 0.7:
            return f"The answer uses most medical terms correctly, but has some issues: {', '.join(errors)}"
        elif score >= 0.4:
            return f"The answer has significant issues with medical terminology: {', '.join(errors)}"
        else:
            return f"The answer has major issues with medical terminology: {', '.join(errors)}"
```

## Example: Task-Specific Metric

Here's an example of a task-specific metric for evaluating code generation:

```python
from merit.evaluation.metrics.base import BaseMetric
from typing import Dict, Any, List

class CodeQualityMetric(BaseMetric):
    """A metric for evaluating the quality of generated code."""
    
    def __init__(self, name="code_quality", **kwargs):
        """
        Initialize the code quality metric.
        
        Args:
            name: The name of the metric.
            **kwargs: Additional arguments for the metric.
        """
        self.name = name
        self._kwargs = kwargs
    
    def __call__(self, input_sample, answer) -> Dict[str, Any]:
        """
        Calculate the metric for a given input and answer.
        
        Args:
            input_sample: The input sample (question/prompt).
            answer: The answer to evaluate.
            
        Returns:
            Dict[str, Any]: A dictionary containing the metric scores and explanations.
        """
        # Get the answer text
        answer_text = answer.message if hasattr(answer, "message") else answer
        
        # Extract code from the answer
        code = self._extract_code(answer_text)
        
        # Calculate the metrics
        syntax_score, syntax_errors = self._check_syntax(code)
        style_score, style_issues = self._check_style(code)
        efficiency_score, efficiency_issues = self._check_efficiency(code)
        
        # Calculate the overall score
        overall_score = (syntax_score + style_score + efficiency_score) / 3
        
        # Generate explanations
        syntax_explanation = self._generate_syntax_explanation(syntax_score, syntax_errors)
        style_explanation = self._generate_style_explanation(style_score, style_issues)
        efficiency_explanation = self._generate_efficiency_explanation(efficiency_score, efficiency_issues)
        
        # Return the scores and explanations
        return {
            f"{self.name}_syntax": syntax_score,
            f"{self.name}_syntax_explanation": syntax_explanation,
            f"{self.name}_style": style_score,
            f"{self.name}_style_explanation": style_explanation,
            f"{self.name}_efficiency": efficiency_score,
            f"{self.name}_efficiency_explanation": efficiency_explanation,
            self.name: overall_score
        }
    
    def _extract_code(self, text) -> str:
        """
        Extract code from text.
        
        Args:
            text: The text to extract code from.
            
        Returns:
            str: The extracted code.
        """
        # Implement code extraction logic here
        # This is a simple example that extracts code between triple backticks
        import re
        code_blocks = re.findall(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
        
        if code_blocks:
            return code_blocks[0]
        else:
            return text
    
    def _check_syntax(self, code) -> tuple:
        """
        Check the syntax of the code.
        
        Args:
            code: The code to check.
            
        Returns:
            tuple: A tuple containing the syntax score and a list of syntax errors.
        """
        # Implement syntax checking logic here
        # This is a simple example that uses Python's ast module
        import ast
        
        try:
            ast.parse(code)
            return 1.0, []
        except SyntaxError as e:
            return 0.0, [str(e)]
    
    def _check_style(self, code) -> tuple:
        """
        Check the style of the code.
        
        Args:
            code: The code to check.
            
        Returns:
            tuple: A tuple containing the style score and a list of style issues.
        """
        # Implement style checking logic here
        # This is a simple example that checks for PEP 8 compliance
        issues = []
        
        # Check line length
        lines = code.split("\n")
        for i, line in enumerate(lines):
            if len(line) > 79:
                issues.append(f"Line {i+1} is too long ({len(line)} > 79 characters)")
        
        # Calculate the score
        if not issues:
            return 1.0, []
        else:
            return max(0.0, 1.0 - len(issues) / 10), issues
    
    def _check_efficiency(self, code) -> tuple:
        """
        Check the efficiency of the code.
        
        Args:
            code: The code to check.
            
        Returns:
            tuple: A tuple containing the efficiency score and a list of efficiency issues.
        """
        # Implement efficiency checking logic here
        # This is a simple example that checks for common inefficiencies
        issues = []
        
        # Check for inefficient list comprehensions
        if "for i in range(len(" in code:
            issues.append("Inefficient list iteration using range(len())")
        
        # Check for inefficient string concatenation
        if "+= " in code and "string" in code.lower():
            issues.append("Potential inefficient string concatenation")
        
        # Calculate the score
        if not issues:
            return 1.0, []
        else:
            return max(0.0, 1.0 - len(issues) / 5), issues
    
    def _generate_syntax_explanation(self, score, errors) -> str:
        """
        Generate an explanation for the syntax score.
        
        Args:
            score: The syntax score.
            errors: A list of syntax errors.
            
        Returns:
            str: The explanation for the syntax score.
        """
        if score == 1.0:
            return "The code has no syntax errors."
        else:
            return f"The code has syntax errors: {', '.join(errors)}"
    
    def _generate_style_explanation(self, score, issues) -> str:
        """
        Generate an explanation for the style score.
        
        Args:
            score: The style score.
            issues: A list of style issues.
            
        Returns:
            str: The explanation for the style score.
        """
        if score == 1.0:
            return "The code follows good style practices."
        else:
            return f"The code has style issues: {', '.join(issues)}"
    
    def _generate_efficiency_explanation(self, score, issues) -> str:
        """
        Generate an explanation for the efficiency score.
        
        Args:
            score: The efficiency score.
            issues: A list of efficiency issues.
            
        Returns:
            str: The explanation for the efficiency score.
        """
        if score == 1.0:
            return "The code is efficient."
        else:
            return f"The code has efficiency issues: {', '.join(issues)}"
```

## Using Custom Metrics

Once you've created a custom metric, you can use it in your evaluation:

```python
from merit.evaluation import evaluate_rag
from merit.core.models import TestSet
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient

# Create a custom metric
custom_metric = CustomMetric(name="my_custom_metric")

# Create a medical accuracy metric
medical_metric = MedicalAccuracyMetric(name="medical_accuracy")

# Create a code quality metric
code_metric = CodeQualityMetric(name="code_quality")

# Load a test set and knowledge base
test_set = TestSet.load("my_test_set.json")
knowledge_base = KnowledgeBase.load("my_knowledge_base.json")

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Define an answer function
def get_answer(query):
    # Your RAG system implementation here
    return "This is the answer to the query."

# Evaluate the RAG system with custom metrics
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", custom_metric, medical_metric]
)

# Print the evaluation results
print("Evaluation results:")
for metric, score in report.get_overall_scores().items():
    print(f"{metric}: {score:.2f}")

# Print custom metric results for each sample
for i, result in enumerate(report.results):
    print(f"\nSample {i+1}:")
    print(f"Query: {result.input}")
    print(f"Answer: {result.model_answer}")
    print(f"Custom metric score: {result.scores.get('my_custom_metric', 0):.2f}")
    print(f"Custom metric explanation: {result.explanations.get('my_custom_metric_explanation', '')}")
```

## Best Practices for Custom Metrics

When creating custom metrics, follow these best practices:

### 1. Define Clear Criteria

Define clear criteria for what your metric measures:

```python
class ClarityMetric(BaseMetric):
    """
    A metric for evaluating the clarity of answers.
    
    Clarity is defined as:
    1. The answer is easy to understand
    2. The answer uses simple language
    3. The answer is well-structured
    4. The answer avoids jargon and technical terms
    5. The answer provides clear explanations
    """
```

### 2. Return Standardized Results

Ensure your metric returns results in a standardized format:

```python
def __call__(self, input_sample, answer):
    # Calculate the score
    score = self._calculate_score(input_sample, answer)
    
    # Generate an explanation
    explanation = self._generate_explanation(input_sample, answer, score)
    
    # Return the results in a standardized format
    return {
        self.name: score,  # Score should be a float between 0 and 1
        f"{self.name}_explanation": explanation  # Explanation should be a string
    }
```

### 3. Handle Edge Cases

Handle edge cases gracefully:

```python
def _calculate_score(self, input_sample, answer):
    # Handle empty answers
    if not answer:
        return 0.0
    
    # Handle very short answers
    if len(answer) < 10:
        return 0.1
    
    # Handle normal cases
    # ...
```

### 4. Provide Detailed Explanations

Provide detailed explanations for your metric scores:

```python
def _generate_explanation(self, input_sample, answer, score):
    # Provide a detailed explanation
    strengths = []
    weaknesses = []
    
    # Identify strengths
    if "clearly explains" in answer.lower():
        strengths.append("The answer clearly explains the concept")
    
    # Identify weaknesses
    if len(answer.split()) > 100:
        weaknesses.append("The answer is too verbose")
    
    # Generate the explanation
    explanation = "Strengths: " + (", ".join(strengths) if strengths else "None") + "\n"
    explanation += "Weaknesses: " + (", ".join(weaknesses) if weaknesses else "None")
    
    return explanation
```

### 5. Test Thoroughly

Test your metrics thoroughly with various inputs:

```python
def test_metric():
    # Create the metric
    metric = CustomMetric()
    
    # Test with a good answer
    good_input = "What is artificial intelligence?"
    good_answer = "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions."
    good_result = metric(good_input, good_answer)
    print(f"Good answer score: {good_result[metric.name]:.2f}")
    print(f"Good answer explanation: {good_result[f'{metric.name}_explanation']}")
    
    # Test with a poor answer
    poor_input = "What is artificial intelligence?"
    poor_answer = "AI is smart computers."
    poor_result = metric(poor_input, poor_answer)
    print(f"Poor answer score: {poor_result[metric.name]:.2f}")
    print(f"Poor answer explanation: {poor_result[f'{metric.name}_explanation']}")
    
    # Test with an empty answer
    empty_input = "What is artificial intelligence?"
    empty_answer = ""
    empty_result = metric(empty_input, empty_answer)
    print(f"Empty answer score: {empty_result[metric.name]:.2f}")
    print(f"Empty answer explanation: {empty_result[f'{metric.name}_explanation']}")
```

## Next Steps

Now that you know how to create custom metrics, you can:

- Learn how to create [custom evaluators](./custom_evaluators.md) for specialized evaluation tasks
- Explore how to define [custom prompts](./custom_prompts.md) for evaluation
- Discover how to create [custom knowledge bases](../knowledge_bases/custom_knowledge_bases.md) for evaluation
- Learn how to create [custom report templates](./report_templates.md) for presenting evaluation results
