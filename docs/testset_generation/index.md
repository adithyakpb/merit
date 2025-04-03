# Test Set Generation

This section covers how to generate test sets for evaluating AI systems using MERIT. Test sets are collections of inputs, reference answers, and associated documents that are used to evaluate the performance of AI systems, particularly RAG (Retrieval-Augmented Generation) systems.

## In This Section

- [Overview](./01_overview.md): Introduction to test set generation in MERIT
- [Basic Generation](./02_basic_generation.md): How to generate test sets from a knowledge base
- [Example-Guided Generation](./03_example_guided_generation.md): How to use example inputs to guide test set generation
- [Customizing Generation](./04_customizing_generation.md): How to customize the test set generation process
- [Working with Knowledge Bases](../knowledge_bases/03_working_with_knowledge_bases.md): How to create and use knowledge bases for test set generation

## Why Generate Test Sets?

Evaluating AI systems requires a comprehensive set of test cases that cover various scenarios and edge cases. MERIT provides tools to generate test sets that are:

- **Representative**: Cover a wide range of inputs and scenarios
- **Realistic**: Based on real-world documents and queries
- **Customizable**: Tailored to your specific use case
- **Reproducible**: Can be saved, loaded, and shared

## Key Components

### TestSetGenerator

The `TestSetGenerator` class is the main entry point for generating test sets. It provides methods for generating test sets from a knowledge base, with or without example inputs.

```python
from merit.testset_generation import TestSetGenerator
from merit.knowledge import KnowledgeBase

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=[...])

# Create a test set generator
generator = TestSetGenerator(knowledge_base=knowledge_base)

# Generate a test set
test_set = generator.generate(num_inputs=50)
```

### Example-Guided Generation

MERIT supports example-guided test set generation, where you provide example inputs to guide the generation process. This is useful when you want to generate test sets that are similar to existing inputs.

```python
# Example inputs
example_inputs = [
    "What is the capital of France?",
    "Who wrote the novel 'Pride and Prejudice'?",
    "Explain the concept of quantum computing."
]

# Generate a test set based on example inputs
test_set = generator.generate(
    num_inputs=50,
    example_inputs=example_inputs
)
```

### TestSet

The `TestSet` class represents a collection of test inputs, each with an input, reference answer, and associated document. Test sets can be saved to and loaded from files.

```python
# Save a test set
test_set.save("my_test_set.json")

# Load a test set
from merit.core.models import TestSet
loaded_test_set = TestSet.load("my_test_set.json")
```

## Next Steps

Start by reading the [Overview](./01_overview.md) to learn more about test set generation in MERIT. Then, follow the [Basic Generation](./02_basic_generation.md) guide to generate your first test set. For more advanced use cases, check out the [Example-Guided Generation](./03_example_guided_generation.md) and [Customizing Generation](./04_customizing_generation.md) guides.
