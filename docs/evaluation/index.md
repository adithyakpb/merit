# Evaluation

This section covers how to evaluate AI systems using MERIT. Evaluation is a critical part of developing and deploying AI systems, as it helps you understand their performance, identify areas for improvement, and ensure they meet your requirements.

## In This Section

- [Overview](./01_overview.md): Introduction to evaluation in MERIT
- [RAG Evaluation](./02_rag_evaluation.md): How to evaluate RAG (Retrieval-Augmented Generation) systems
- [LLM Evaluation](./03_llm_evaluation.md): How to evaluate general LLM outputs
- [Classification Metrics](../customization/02_custom_metrics.md): How to use classification metrics
- [Custom Metrics](../customization/02_custom_metrics.md): How to create custom evaluation metrics

## Why Evaluate AI Systems?

Evaluation helps you:

- **Measure Performance**: Quantify how well your AI system performs
- **Identify Issues**: Find areas where your system needs improvement
- **Compare Approaches**: Benchmark different models or configurations
- **Track Progress**: Monitor how your system evolves over time
- **Build Confidence**: Ensure your system meets quality standards before deployment

## Key Components

### Evaluators

MERIT provides several evaluator classes for different types of AI systems:

- `RAGEvaluator`: For evaluating RAG (Retrieval-Augmented Generation) systems
- `LLMEvaluator`: For evaluating general LLM outputs
- `ClassificationEvaluator`: For evaluating classification tasks

### Metrics

MERIT includes a variety of metrics for evaluating different aspects of AI system performance:

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

### Evaluation Reports

MERIT generates detailed evaluation reports that include:

- Overall scores for each metric
- Per-sample scores and explanations
- Identified issues and errors
- Metadata about the evaluation

## Basic Usage

Here's a simple example of evaluating a RAG system:

```python
from merit.evaluation import evaluate_rag
from merit.knowledge import KnowledgeBase
from merit.core.models import TestSet
from merit.api.client import OpenAIClient

# Create a client for evaluation
client = OpenAIClient(api_key="your-openai-api-key")

# Load a knowledge base and test set
knowledge_base = KnowledgeBase.load("my_knowledge_base.json")
test_set = TestSet.load("my_test_set.json")

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

# Print the report
print(f"Overall scores: {report.get_overall_scores()}")
```

## Next Steps

Start by reading the [Overview](./01_overview.md) to learn more about evaluation in MERIT. Then, depending on your use case, check out the specific guides for [RAG Evaluation](./02_rag_evaluation.md), [LLM Evaluation](./03_llm_evaluation.md), or [Classification Metrics](../customization/02_custom_metrics.md). For advanced use cases, refer to the [Custom Metrics](../customization/02_custom_metrics.md) guide.
