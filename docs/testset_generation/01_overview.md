# Test Set Generation Overview

This guide provides an overview of test set generation in MERIT. Test set generation is a critical part of evaluating AI systems, as it helps you create comprehensive and representative test cases for assessing your system's performance.

## What is Test Set Generation?

Test set generation is the process of creating a collection of test inputs, reference answers, and associated documents that can be used to evaluate an AI system. In MERIT, test sets are particularly useful for evaluating RAG (Retrieval-Augmented Generation) systems, which retrieve information from a knowledge base to generate answers to questions.

## Why Generate Test Sets?

Manually creating test sets can be time-consuming and may not cover all the edge cases and scenarios you want to test. MERIT's test set generation capabilities help you:

- **Save Time**: Automatically generate test inputs and reference answers
- **Ensure Coverage**: Create test sets that cover a wide range of topics and scenarios
- **Maintain Consistency**: Generate reference answers that are consistent with your knowledge base
- **Customize Tests**: Tailor test sets to your specific use case and requirements
- **Scale Evaluation**: Generate large test sets for comprehensive evaluation

## Test Set Generation Approaches

MERIT supports several approaches to test set generation:

### Standard Generation

The standard approach generates test inputs and reference answers directly from your knowledge base. This is useful when you want to create a general-purpose test set that covers the breadth of your knowledge base.

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

The example-guided approach uses example inputs to guide the generation process. This is useful when you want to create test sets that are similar to existing inputs or follow a specific pattern.

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

### Customized Generation

MERIT allows you to customize the test set generation process in various ways, such as:

- Specifying the language of the generated inputs
- Providing a description of the agent being evaluated
- Removing similar examples to ensure diversity
- Setting a similarity threshold for example filtering
- Skipping relevance checks for faster generation

```python
# Customized test set generation
test_set = generator.generate(
    num_inputs=50,
    example_inputs=example_inputs,
    remove_similar_examples=True,
    similarity_threshold=0.85,
    skip_relevance_check=False
)
```

## Test Set Structure

A test set in MERIT consists of:

- **Test Inputs**: The questions or prompts to be answered
- **Reference Answers**: The expected answers to the inputs
- **Documents**: The source documents the inputs are based on
- **Metadata**: Additional information about the test set and inputs

```python
# Accessing test set components
for input_sample in test_set.inputs:
    print(f"Input: {input_sample.input}")
    print(f"Reference Answer: {input_sample.reference_answer}")
    print(f"Document: {input_sample.document.content}")
    print(f"Metadata: {input_sample.metadata}")
    print()
```

## Saving and Loading Test Sets

Test sets can be saved to and loaded from files, making it easy to reuse them for multiple evaluations or share them with others.

```python
# Save a test set
test_set.save("my_test_set.json")

# Load a test set
from merit.core.models import TestSet
loaded_test_set = TestSet.load("my_test_set.json")
```

## Next Steps

Now that you understand the basics of test set generation in MERIT, you can:

- Learn how to [generate test sets from a knowledge base](./basic_generation.md)
- Explore [example-guided test set generation](./example_guided_generation.md)
- Discover how to [customize the test set generation process](./customizing_generation.md)
- Understand how to [work with knowledge bases](./working_with_knowledgebase.md)
