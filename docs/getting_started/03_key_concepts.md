# Key Concepts in MERIT

This guide explains the core concepts and terminology used in MERIT. Understanding these concepts will help you use MERIT effectively for evaluating and testing your AI systems.

## Core Components

### Knowledge Base

A **Knowledge Base** is a collection of documents that serves as the source of information for RAG (Retrieval-Augmented Generation) systems. In MERIT, a knowledge base is represented by the `KnowledgeBase` class, which provides methods for:

- Storing and retrieving documents
- Searching for relevant documents
- Managing document metadata
- Generating embeddings for semantic search

```python
from merit.knowledge import KnowledgeBase

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=[
    {"content": "Paris is the capital of France.", "id": "doc1"},
    {"content": "The Eiffel Tower is located in Paris.", "id": "doc2"},
])
```

### Document

A **Document** represents a single piece of content in a knowledge base. In MERIT, a document is represented by the `Document` class, which includes:

- The document content
- Metadata about the document
- An identifier for the document
- Optional embeddings for semantic search

```python
from merit.core.models import Document

# Create a document
document = Document(
    content="Paris is the capital of France.",
    metadata={"source": "geography", "date": "2023-01-01"},
    id="doc1"
)
```

### Test Set

A **Test Set** is a collection of test inputs used to evaluate an AI system. In MERIT, a test set is represented by the `TestSet` class, which includes:

- A list of test inputs
- Metadata about the test set

```python
from merit.core.models import TestSet

# Create a test set
test_set = TestSet(
    inputs=[test_input1, test_input2, test_input3],
    metadata={"name": "Geography Test Set", "version": "1.0"}
)
```

### Test Input

A **Test Input** represents a single test case in a test set. In MERIT, a test input is represented by the `TestItem` class, which includes:

- The input text (e.g., a question)
- A reference answer
- The document the input is based on
- Metadata about the input

```python
from merit.core.models import TestItem

# Create a test input
test_input = TestItem(
    input="What is the capital of France?",
    reference_answer="The capital of France is Paris.",
    document=document,
    metadata={"difficulty": "easy", "category": "geography"}
)
```

### API Client

An **API Client** provides a unified interface for interacting with AI APIs. In MERIT, API clients are represented by classes that extend `BaseAPIClient` or `AIAPIClient`, which provide methods for:

- Generating text from prompts
- Getting embeddings for text
- Making API requests with authentication
- Handling errors and retries

```python
from merit.api.client import AIAPIClient

# Create an API client
client = AIAPIClient(
    base_url="https://api.example.com",
    api_key="your-api-key"
)
```

### Evaluator

An **Evaluator** assesses the performance of an AI system using various metrics. In MERIT, evaluators are represented by classes that extend `BaseEvaluator`, such as `RAGEvaluator` and `LLMEvaluator`, which provide methods for:

- Evaluating AI systems against test sets
- Calculating metric scores
- Generating evaluation reports

```python
from merit.evaluation.evaluators.rag import RAGEvaluator

# Create an evaluator
evaluator = RAGEvaluator(
    knowledge_base=knowledge_base,
    testset=test_set,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)
```

### Metric

A **Metric** measures a specific aspect of AI system performance. In MERIT, metrics are represented by classes that extend `BaseMetric`, such as `CorrectnessMetric` and `FaithfulnessMetric`, which provide methods for:

- Calculating scores for specific aspects of performance
- Generating explanations for scores
- Identifying issues and errors

```python
from merit.evaluation.metrics.rag import CorrectnessMetric

# Create a metric
metric = CorrectnessMetric(
    llm_client=client,
    agent_description="A chatbot that answers questions about geography."
)
```

### Evaluation Report

An **Evaluation Report** summarizes the results of an evaluation. In MERIT, evaluation reports are represented by the `EvaluationReport` class, which includes:

- Overall scores for each metric
- Per-sample scores and explanations
- Identified issues and errors
- Metadata about the evaluation

```python
# Get an evaluation report
report = evaluator.evaluate(answer_fn)

# Print overall scores
print(f"Overall scores: {report.get_overall_scores()}")
```

## Key Processes

### Test Set Generation

**Test Set Generation** is the process of creating test sets for evaluating AI systems. In MERIT, test set generation is performed by the `TestSetGenerator` class, which provides methods for:

- Generating test sets from a knowledge base
- Creating example-guided test sets
- Customizing test set generation

```python
from merit.testset_generation import TestSetGenerator

# Create a test set generator
generator = TestSetGenerator(knowledge_base=knowledge_base)

# Generate a test set
test_set = generator.generate(num_inputs=50)
```

### RAG Evaluation

**RAG Evaluation** is the process of evaluating RAG (Retrieval-Augmented Generation) systems. In MERIT, RAG evaluation is performed by the `RAGEvaluator` class or the `evaluate_rag` function, which:

- Evaluates how well a RAG system answers questions based on a knowledge base
- Measures correctness, faithfulness, relevance, and other aspects of performance
- Identifies hallucinations, factual errors, and other issues

```python
from merit.evaluation import evaluate_rag

# Evaluate a RAG system
report = evaluate_rag(
    answer_fn=get_answer,
    testset=test_set,
    knowledge_base=knowledge_base,
    llm_client=client,
    metrics=["correctness", "faithfulness", "relevance"]
)
```

### LLM Evaluation

**LLM Evaluation** is the process of evaluating general LLM outputs. In MERIT, LLM evaluation is performed by the `LLMEvaluator` class, which:

- Evaluates LLM outputs using custom prompts
- Measures various aspects of output quality
- Provides flexible evaluation options

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

## Next Steps

Now that you understand the key concepts in MERIT, you can:

- Follow the [Quick Start](./quick_start.md) guide to get started with MERIT
- Explore the [Test Set Generation](../testset_generation/index.md) guide to learn how to create test sets
- Check out the [API Clients](../api_clients/index.md) guide to learn how to connect to AI APIs
- Read the [Evaluation](../evaluation/index.md) guide to learn how to evaluate AI systems
