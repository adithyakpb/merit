# RAG Evaluation

This guide explains how to evaluate RAG (Retrieval-Augmented Generation) systems using MERIT. RAG systems combine retrieval of relevant documents with generation of answers, and MERIT provides specialized tools for evaluating their performance.

## What is RAG Evaluation?

RAG evaluation assesses how well a RAG system:

1. **Retrieves** relevant information from a knowledge base
2. **Generates** accurate and helpful answers based on the retrieved information
3. **Avoids** hallucinations and factual errors
4. **Addresses** the user's query effectively

MERIT provides a comprehensive framework for evaluating RAG systems across multiple dimensions, including correctness, faithfulness, relevance, coherence, and fluency.

## RAG Evaluation Metrics

MERIT includes several metrics specifically designed for evaluating RAG systems:

### Correctness

The `CorrectnessMetric` evaluates the factual correctness of answers by comparing them to reference answers. It identifies factual errors and assesses how well the answer aligns with the ground truth.

### Faithfulness

The `FaithfulnessMetric` evaluates how well answers stick to the source documents. It identifies hallucinations (information not present in the source documents) and assesses whether the answer is grounded in the provided information.

### Relevance

The `RelevanceMetric` evaluates how well answers address the input query. It assesses whether the answer provides the information the user is looking for and stays on topic.

### Coherence

The `CoherenceMetric` evaluates the logical structure and flow of answers. It assesses whether the answer is well-organized, logically consistent, and easy to follow.

### Fluency

The `FluencyMetric` evaluates the grammatical correctness and readability of answers. It assesses whether the answer is well-written, free of grammatical errors, and uses natural language.

## Basic RAG Evaluation

Here's a simple example of how to evaluate a RAG system using MERIT:

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
    # This function should take a query and return an answer
    return "This is the answer to the query."

# Evaluate the RAG system
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)

# Print the overall scores
print("Evaluation results:")
for metric, score in report.get_overall_scores().items():
    print(f"{metric}: {score:.2f}")
```

## Using the RAGEvaluator Class

For more control over the evaluation process, you can use the `RAGEvaluator` class directly:

```python
from merit.evaluation.evaluators.rag import RAGEvaluator
from merit.evaluation.metrics.rag import CorrectnessMetric, FaithfulnessMetric, RelevanceMetric

# Create metrics
metrics = [
    CorrectnessMetric(llm_client=client),
    FaithfulnessMetric(llm_client=client),
    RelevanceMetric(llm_client=client)
]

# Create an evaluator
evaluator = RAGEvaluator(
    knowledge_base=knowledge_base,
    testset=test_set,
    llm_client=client,
    metrics=metrics,
    agent_description="A chatbot that answers questions based on a knowledge base."
)

# Evaluate the RAG system
report = evaluator.evaluate(get_answer)

# Print the overall scores
print("Evaluation results:")
for metric, score in report.get_overall_scores().items():
    print(f"{metric}: {score:.2f}")
```

## Returning Document Information

For more accurate evaluation, your answer function can return both the answer text and the documents used to generate it:

```python
from merit.base import Response

def get_answer_with_docs(query):
    # Retrieve relevant documents
    relevant_docs = knowledge_base.search(query, k=2)
    
    # Generate an answer
    answer = "This is the answer based on the retrieved documents."
    
    # Return both the answer and the documents
    return Response(
        content=answer,
        documents=[doc for doc, _ in relevant_docs],
        metadata={"query": query}
    )

# Evaluate the RAG system
report = evaluate_rag(
    answer_fn=get_answer_with_docs,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)
```

## Analyzing Evaluation Results

MERIT provides detailed evaluation results that you can analyze to understand your RAG system's performance:

```python
# Get overall scores
overall_scores = report.get_overall_scores()
print(f"Overall scores: {overall_scores}")

# Get per-sample scores
for result in report.results:
    print(f"\nInput: {result.input}")
    print(f"Reference answer: {result.reference_answer}")
    print(f"Model answer: {result.model_answer}")
    print("Scores:")
    for metric, score in result.scores.items():
        print(f"  {metric}: {score:.2f}")
    
    # Print explanations if available
    if result.explanations:
        print("Explanations:")
        for metric, explanation in result.explanations.items():
            print(f"  {metric}: {explanation}")
    
    # Print errors if available
    if result.errors:
        print("Errors:")
        for metric, errors in result.errors.items():
            print(f"  {metric}: {errors}")
    
    # Print hallucinations if available
    if hasattr(result, "hallucinations") and result.hallucinations:
        print("Hallucinations:")
        for hallucination in result.hallucinations:
            print(f"  {hallucination}")
```

## Customizing RAG Evaluation

MERIT allows you to customize the RAG evaluation process in various ways:

### Custom Metrics

You can create custom metrics for evaluating specific aspects of your RAG system:

```python
from merit.evaluation.metrics.rag import RAGMetric

class CustomRAGMetric(RAGMetric):
    """Custom metric for evaluating a specific aspect of RAG systems."""
    
    def __init__(self, name="custom_metric", llm_client=None):
        self.name = name
        self._llm_client = llm_client
    
    def __call__(self, input_sample, answer):
        # Implement your custom metric logic here
        score = 0.8  # Example score
        explanation = "This is a custom explanation."
        
        return {
            self.name: score,
            f"{self.name}_explanation": explanation
        }

# Use the custom metric
metrics = ["correctness", "faithfulness", CustomRAGMetric(llm_client=client)]
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=metrics
)
```

### Custom Agent Description

You can provide a custom description of your RAG system to help the evaluation metrics understand its purpose:

```python
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"],
    agent_description="A medical chatbot that answers health-related questions based on a knowledge base of medical literature."
)
```

## Best Practices for RAG Evaluation

When evaluating RAG systems with MERIT, follow these best practices:

### 1. Use Representative Test Sets

Create test sets that cover a wide range of queries and scenarios that your RAG system will encounter in production:

```python
from merit.testset_generation import TestSetGenerator

# Generate a representative test set
generator = TestSetGenerator(knowledge_base=knowledge_base)
test_set = generator.generate(num_inputs=100)
```

### 2. Include Multiple Metrics

Evaluate your RAG system across multiple dimensions to get a comprehensive understanding of its performance:

```python
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance", "coherence", "fluency"]
)
```

### 3. Return Document Information

Have your answer function return both the answer text and the documents used to generate it for more accurate evaluation:

```python
def get_answer(query):
    # Retrieve relevant documents
    relevant_docs = knowledge_base.search(query, k=2)
    
    # Generate an answer
    answer = "This is the answer based on the retrieved documents."
    
    # Return both the answer and the documents
    return Response(
        content=answer,
        documents=[doc for doc, _ in relevant_docs]
    )
```

### 4. Analyze Detailed Results

Look beyond the overall scores to understand specific strengths and weaknesses:

```python
# Analyze results for specific metrics
correctness_scores = [result.scores.get("correctness", 0) for result in report.results]
faithfulness_scores = [result.scores.get("faithfulness", 0) for result in report.results]

print(f"Average correctness: {sum(correctness_scores) / len(correctness_scores):.2f}")
print(f"Average faithfulness: {sum(faithfulness_scores) / len(faithfulness_scores):.2f}")

# Identify low-scoring samples
low_scoring_samples = [result for result in report.results if result.scores.get("correctness", 0) < 0.5]
print(f"Number of low-scoring samples: {len(low_scoring_samples)}")
```

### 5. Save and Compare Reports

Save evaluation reports to track progress over time:

```python
# Save the report
report.save("rag_evaluation_report.json")

# Load a previous report for comparison
from merit.core.models import EvaluationReport
previous_report = EvaluationReport.load("previous_rag_evaluation_report.json")

# Compare overall scores
for metric in report.metrics:
    current_score = report.get_overall_scores().get(metric, 0)
    previous_score = previous_report.get_overall_scores().get(metric, 0)
    difference = current_score - previous_score
    print(f"{metric}: {current_score:.2f} ({'+' if difference >= 0 else ''}{difference:.2f})")
```

## Next Steps

Now that you understand how to evaluate RAG systems with MERIT, you can:

- Learn about [LLM evaluation](./llm_evaluation.md) for general LLM outputs
- Explore [classification metrics](./classification_metrics.md) for structured outputs
- Create [custom metrics](./custom_metrics.md) for specialized evaluation tasks
- Generate and interpret [evaluation reports](./evaluation_reports.md)
