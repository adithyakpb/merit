# Evaluation Overview

This guide provides an overview of evaluation in MERIT. Evaluation is a critical part of developing and deploying AI systems, as it helps you understand their performance, identify areas for improvement, and ensure they meet your requirements.

## What is Evaluation in MERIT?

Evaluation in MERIT is the process of assessing the performance of AI systems using various metrics and methodologies. MERIT provides a comprehensive framework for evaluating different types of AI systems, including:

- **RAG (Retrieval-Augmented Generation) Systems**: Systems that retrieve information from a knowledge base to generate answers
- **LLM (Large Language Model) Systems**: Systems that generate text based on prompts. Any system that takes in text as input and gives out text as output can be evaluated using MERIT.
- **Classification Systems**: Systems that classify inputs into predefined categories

## Why Evaluate AI Systems?

Evaluating AI systems is essential for several reasons:

- **Measure Performance**: Quantify how well your AI system performs against defined metrics
- **Identify Issues**: Find areas where your system needs improvement, such as hallucinations or factual errors
- **Compare Approaches**: Benchmark different models, configurations, or approaches
- **Track Progress**: Monitor how your system evolves over time as you make improvements
- **Build Confidence**: Ensure your system meets quality standards before deployment
- **Understand Limitations**: Identify the boundaries and limitations of your system

## Evaluation Components in MERIT

MERIT's evaluation framework consists of several key components:

### Evaluators

Evaluators are classes that orchestrate the evaluation process. They apply metrics to test inputs, collect results, and generate reports. MERIT provides several evaluator classes:

- `RAGEvaluator`: For evaluating RAG systems
- `LLMEvaluator`: For evaluating general LLM outputs
- `ClassificationEvaluator`: For evaluating classification tasks

```python
from merit.evaluation.evaluators.rag import RAGEvaluator

# Create an evaluator
evaluator = RAGEvaluator(
    knowledge_base=knowledge_base,
    testset=test_set,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)

# Evaluate a system
report = evaluator.evaluate(answer_fn)
```

### Metrics

Metrics are measures of specific aspects of AI system performance. MERIT includes a variety of metrics for different evaluation tasks:

#### RAG Metrics

- `CorrectnessMetric`: Evaluates the factual correctness of answers
- `FaithfulnessMetric`: Evaluates how well answers stick to the source documents
- `RelevanceMetric`: Evaluates how well answers address the input
- `CoherenceMetric`: Evaluates the logical structure and flow of answers
- `FluencyMetric`: Evaluates the grammatical correctness and readability of answers

#### Classification Metrics

- `AccuracyMetric`: Measures the proportion of correct predictions
- `PrecisionMetric`: Measures the proportion of true positives among positive predictions
- `RecallMetric`: Measures the proportion of true positives among actual positives
- `F1Metric`: Harmonic mean of precision and recall

```python
from merit.evaluation.metrics.rag import CorrectnessMetric

# Create a metric
metric = CorrectnessMetric(
    llm_client=client,
    agent_description="A chatbot that answers questions about geography."
)

# Apply the metric to a sample
result = metric(input_sample, answer)
print(f"Correctness score: {result['correctness']}")
```

### Evaluation Results

Evaluation results in MERIT are structured to provide detailed insights into AI system performance:

- **Overall Scores**: Aggregate scores for each metric
- **Per-Sample Scores**: Individual scores for each test input
- **Explanations**: Detailed explanations of why certain scores were assigned
- **Errors**: Identified errors in the AI system's outputs
- **Hallucinations**: Identified hallucinations or fabricated information
- **Metadata**: Additional information about the evaluation

```python
# Get overall scores
overall_scores = report.get_overall_scores()
print(f"Overall scores: {overall_scores}")

# Get per-sample scores
for result in report.results:
    print(f"Input: {result.input}")
    print(f"Scores: {result.scores}")
    print(f"Explanations: {result.explanations}")
```

## Evaluation Workflows

MERIT supports several evaluation workflows:

### RAG Evaluation

```python
from merit.evaluation import evaluate_rag

# Define an answer function
def get_answer(query):
    # Your RAG system implementation here
    return "This is the answer to the query."

# Evaluate the RAG system
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)
```

### LLM Evaluation

```python
from merit.evaluation.evaluators.llm import LLMEvaluator
from merit.core.prompts import Prompt

# Create an LLM evaluator
evaluator = LLMEvaluator(
    prompt=Prompt("Evaluate the following response: {response}"),
    llm_client=client
)

# Evaluate an LLM
report = evaluator.evaluate(model, dataset)
```

### Classification Evaluation

```python
from merit.evaluation.evaluators.classification import ClassificationEvaluator
from merit.evaluation.metrics.classification import AccuracyMetric, PrecisionMetric

# Create a classification evaluator
evaluator = ClassificationEvaluator(
    metrics=[AccuracyMetric(), PrecisionMetric()]
)

# Evaluate a classification model
report = evaluator.evaluate(model, dataset)
```

## Customizing Evaluation

MERIT allows you to customize the evaluation process in various ways:

- **Custom Metrics**: Create your own metrics to measure specific aspects of performance
- **Custom Evaluators**: Create your own evaluators for specialized evaluation tasks
- **Custom Prompts**: Define custom prompts for LLM-based evaluation
- **Evaluation Parameters**: Configure evaluation parameters to suit your needs

```python
from merit.evaluation.metrics.base import BaseMetric

class CustomMetric(BaseMetric):
    """Custom metric for specialized evaluation."""
    
    def __init__(self, name="custom_metric", **kwargs):
        self.name = name
        self._kwargs = kwargs
    
    def __call__(self, input_sample, answer):
        # Implement metric calculation
        score = self._calculate_score(input_sample, answer)
        explanation = self._generate_explanation(input_sample, answer, score)
        
        return {
            self.name: score,
            f"{self.name}_explanation": explanation
        }
```

## Next Steps

Now that you understand the basics of evaluation in MERIT, you can:

- Learn how to [evaluate RAG systems](./rag_evaluation.md) with MERIT
- Explore [LLM evaluation](./llm_evaluation.md) for general LLM outputs
- Discover [classification metrics](./classification_metrics.md) for structured outputs
- Create [custom metrics](./custom_metrics.md) for specialized evaluation tasks
- Generate and interpret [evaluation reports](./evaluation_reports.md)
