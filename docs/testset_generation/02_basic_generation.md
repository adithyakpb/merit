# Basic Test Set Generation

This guide explains how to generate test sets from a knowledge base using MERIT. Test set generation is a critical part of evaluating AI systems, as it provides the inputs and reference answers needed for assessment.

## What is Test Set Generation?

Test set generation is the process of creating a collection of test inputs, reference answers, and associated documents that can be used to evaluate an AI system. In MERIT, test sets are particularly useful for evaluating RAG (Retrieval-Augmented Generation) systems, which retrieve information from a knowledge base to generate answers to questions.

## Basic Test Set Generation

Here's a simple example of how to generate a test set from a knowledge base:

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
from merit.testset_generation import TestSetGenerator
from merit.api.client import OpenAIClient

# Step 1: Create a knowledge base
documents = [
    Document(
        content="Paris is the capital of France. It is known for the Eiffel Tower, which is a famous landmark.",
        metadata={"source": "geography", "topic": "France"},
        id="doc1"
    ),
    Document(
        content="The Eiffel Tower was constructed from 1887 to 1889 as the entrance to the 1889 World's Fair.",
        metadata={"source": "history", "topic": "Eiffel Tower"},
        id="doc2"
    ),
    Document(
        content="France is a country in Western Europe. It has a population of about 67 million people.",
        metadata={"source": "geography", "topic": "France"},
        id="doc3"
    )
]
knowledge_base = KnowledgeBase(documents=documents)

# Step 2: Create an API client for generation
client = OpenAIClient(api_key="your-openai-api-key")

# Step 3: Create a test set generator
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    language="en",
    agent_description="A chatbot that answers questions about geography and history."
)

# Step 4: Generate a test set
test_set = generator.generate(num_inputs=10)

# Step 5: Print the generated inputs
print(f"Generated {len(test_set.inputs)} test inputs:")
for i, input_sample in enumerate(test_set.inputs):
    print(f"\nInput {i+1}: {input_sample.input}")
    print(f"Reference answer: {input_sample.reference_answer}")
    print(f"Document: {input_sample.document.content}")

# Step 6: Save the test set for future use
test_set.save("geography_test_set.json")
```

## Understanding the Generation Process

When you call `generator.generate()`, MERIT performs the following steps:

1. **Document Selection**: Selects a subset of documents from the knowledge base
2. **Input Generation**: Generates inputs (questions) based on each document
3. **Reference Answer Generation**: Generates reference answers for each input
4. **Test Set Creation**: Creates a test set with the inputs, reference answers, and documents

## Customizing Basic Generation

You can customize the basic generation process in several ways:

### Number of Inputs

You can specify the number of inputs to generate:

```python
# Generate 50 inputs
test_set = generator.generate(num_inputs=50)
```

### Language

You can specify the language for the generated inputs and reference answers:

```python
# Create a generator for Spanish
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    language="es",
    agent_description="Un chatbot que responde preguntas sobre geografía e historia."
)
```

### Agent Description

You can provide a description of the agent being evaluated, which helps guide the generation process:

```python
# Create a generator with a specific agent description
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    language="en",
    agent_description="A medical chatbot that answers health-related questions based on medical literature."
)
```

### Batch Size

You can specify the batch size for processing documents during generation:

```python
# Create a generator with a custom batch size
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    language="en",
    agent_description="A chatbot that answers questions about geography and history.",
    batch_size=16
)
```

## Working with Large Knowledge Bases

When working with large knowledge bases, MERIT automatically handles the complexity:

```python
# Create a large knowledge base
large_kb = KnowledgeBase(documents=large_document_list)

# Create a generator
generator = TestSetGenerator(knowledge_base=large_kb)

# Generate a test set
# MERIT will sample a subset of documents to ensure efficient generation
test_set = generator.generate(num_inputs=100)
```

## Saving and Loading Test Sets

You can save test sets to files and load them later:

```python
# Save a test set
test_set.save("my_test_set.json")

# Load a test set
from merit.core.models import TestSet
loaded_test_set = TestSet.load("my_test_set.json")
```

## Analyzing Generated Test Sets

You can analyze the generated test sets to understand their characteristics:

```python
# Print test set metadata
print(f"Test set metadata: {test_set.metadata}")

# Count inputs per document
doc_counts = {}
for input_sample in test_set.inputs:
    doc_id = input_sample.document.id
    if doc_id not in doc_counts:
        doc_counts[doc_id] = 0
    doc_counts[doc_id] += 1

print("Inputs per document:")
for doc_id, count in doc_counts.items():
    print(f"  Document {doc_id}: {count} inputs")

# Analyze input types
input_types = {}
for input_sample in test_set.inputs:
    # Simple heuristic: categorize by first word
    first_word = input_sample.input.split()[0].lower()
    if first_word not in input_types:
        input_types[first_word] = 0
    input_types[first_word] += 1

print("Input types:")
for word, count in sorted(input_types.items(), key=lambda x: x[1], reverse=True):
    print(f"  {word}: {count} inputs")
```

## Best Practices for Basic Generation

When generating test sets with MERIT, follow these best practices:

### 1. Use Representative Documents

Ensure your knowledge base contains documents that are representative of the content your RAG system will encounter in production:

```python
# Create a knowledge base with diverse documents
knowledge_base = KnowledgeBase(documents=[
    # Technical documents
    Document(content="...", metadata={"category": "technical"}),
    # Non-technical documents
    Document(content="...", metadata={"category": "non-technical"}),
    # FAQ documents
    Document(content="...", metadata={"category": "faq"}),
    # Procedural documents
    Document(content="...", metadata={"category": "procedure"})
])
```

### 2. Generate Sufficient Inputs

Generate enough inputs to cover a wide range of scenarios:

```python
# Generate a large test set
test_set = generator.generate(num_inputs=200)
```

### 3. Use Appropriate Agent Descriptions

Provide an agent description that matches the intended use case of your RAG system:

```python
# For a customer support chatbot
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    agent_description="A customer support chatbot that helps users troubleshoot product issues."
)

# For a medical assistant
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    agent_description="A medical assistant that provides information about symptoms, treatments, and medications."
)
```

### 4. Save Test Sets for Reproducibility

Save your test sets to ensure reproducible evaluations:

```python
# Save the test set with a descriptive name
test_set.save("product_support_test_set_v1.json")
```

### 5. Validate Generated Inputs

Review a sample of the generated inputs to ensure they make sense:

```python
# Print a sample of inputs for manual review
import random
sample_inputs = random.sample(test_set.inputs, min(10, len(test_set.inputs)))
for i, input_sample in enumerate(sample_inputs):
    print(f"\nSample {i+1}:")
    print(f"Input: {input_sample.input}")
    print(f"Reference answer: {input_sample.reference_answer}")
    print(f"Document: {input_sample.document.content[:100]}...")  # Print the first 100 characters
```

## Next Steps

Now that you understand basic test set generation, you can:

- Learn about [example-guided generation](./example_guided_generation.md) to create test sets based on example inputs
- Explore how to [customize the generation process](./customizing_generation.md) for your specific needs
- Understand how to [work with knowledge bases](./working_with_knowledgebase.md) for test set generation
