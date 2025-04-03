# Tutorials

This section provides step-by-step tutorials for common workflows and use cases in MERIT. These tutorials will help you understand how to use MERIT effectively for evaluating and testing your AI systems.

## In This Section

- [Basic Workflow](./01_basic_workflow.md): A complete end-to-end workflow for evaluating a RAG system
- [Custom API Evaluation](./02_custom_api_evaluation.md): How to evaluate a custom API using MERIT
- [Advanced Test Set Generation](./03_advanced_testset_generation.md): Advanced techniques for generating test sets
- [Custom Metrics Creation](../customization/02_custom_metrics.md): How to create and use custom evaluation metrics

## Getting Started

If you're new to MERIT, we recommend starting with the [Basic Workflow](./01_basic_workflow.md) tutorial. This tutorial will guide you through a complete end-to-end workflow for evaluating a RAG system, including:

1. Setting up a knowledge base
2. Generating a test set
3. Connecting to an API
4. Evaluating the system
5. Interpreting the results

## Tutorial Structure

Each tutorial follows a consistent structure:

1. **Introduction**: What you'll learn and why it's useful
2. **Prerequisites**: What you need to know and have installed
3. **Step-by-Step Guide**: Detailed instructions with code examples
4. **Explanation**: Why each step works the way it does
5. **Next Steps**: Where to go from here

## Complete Examples

Here's a preview of what you'll find in the tutorials:

### Basic RAG Evaluation Workflow

```python
from merit.knowledge import KnowledgeBase
from merit.testset_generation import TestSetGenerator
from merit.evaluation import evaluate_rag
from merit.api.client import OpenAIClient

# Step 1: Create a knowledge base
documents = [
    {"content": "Paris is the capital of France.", "id": "doc1"},
    {"content": "The Eiffel Tower is located in Paris.", "id": "doc2"},
    {"content": "France is a country in Western Europe.", "id": "doc3"},
]
knowledge_base = KnowledgeBase(documents=documents)

# Step 2: Generate a test set
generator = TestSetGenerator(knowledge_base=knowledge_base)
test_set = generator.generate(num_inputs=10)

# Step 3: Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Step 4: Define an answer function
def get_answer(query):
    # In a real scenario, this would call your RAG system
    return client.generate_text(f"Answer this question: {query}")

# Step 5: Evaluate the RAG system
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)

# Step 6: Print the results
print(f"Overall scores: {report.get_overall_scores()}")
```

## Next Steps

Start by reading the [Basic Workflow](./01_basic_workflow.md) tutorial to get a complete understanding of how to use MERIT for evaluating AI systems. Then, depending on your needs, check out the other tutorials for more advanced use cases and techniques.
