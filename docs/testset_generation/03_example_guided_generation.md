# Example-Guided Test Set Generation

This guide explains how to use example inputs to guide test set generation in MERIT. Example-guided generation allows you to create test sets that follow specific patterns or styles based on example inputs you provide.

## What is Example-Guided Generation?

Example-guided generation is a process where you provide example inputs (questions or prompts) to guide the generation of new test inputs. MERIT analyzes the examples to understand their patterns, styles, and characteristics, then generates new inputs that follow similar patterns but are based on your knowledge base documents.

This approach is particularly useful when you want to:

- Create test sets that match the style of real user queries
- Focus on specific types of questions or topics
- Ensure consistency with existing test sets
- Generate inputs that follow specific patterns or formats

## Basic Example-Guided Generation

Here's a simple example of how to use example-guided generation:

```python
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
from merit.testset_generation import TestSetGenerator
from merit.api.client import OpenAIClient

# Step 1: Create a knowledge base
documents = [
    Document(
        content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving.",
        metadata={"source": "AI Overview", "topic": "AI Basics"},
        id="doc1"
    ),
    Document(
        content="Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.",
        metadata={"source": "Machine Learning", "topic": "ML Basics"},
        id="doc2"
    ),
    Document(
        content="Deep Learning is a subset of machine learning where artificial neural networks, algorithms inspired by the human brain, learn from large amounts of data. Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
        metadata={"source": "Deep Learning", "topic": "DL Basics"},
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
    agent_description="A chatbot that answers questions about artificial intelligence and related topics."
)

# Step 4: Define example inputs
example_inputs = [
    "Can you explain what AI is in simple terms?",
    "How does machine learning work?",
    "What's the difference between AI and machine learning?",
    "Could you provide a brief overview of deep learning?",
    "What are some real-world applications of AI?"
]

# Step 5: Generate a test set based on the examples
test_set = generator.generate(
    num_inputs=10,
    example_inputs=example_inputs
)

# Step 6: Print the generated inputs
print(f"Generated {len(test_set.inputs)} test inputs:")
for i, input_sample in enumerate(test_set.inputs):
    print(f"\nInput {i+1}: {input_sample.input}")
    print(f"Reference answer: {input_sample.reference_answer}")
    print(f"Document: {input_sample.document.content[:100]}...")  # Print the first 100 characters

# Step 7: Save the test set for future use
test_set.save("ai_test_set.json")
```

## How Example-Guided Generation Works

When you provide example inputs to the `generate()` method, MERIT performs the following steps:

1. **Example Analysis**: Analyzes the example inputs to understand their patterns, styles, and characteristics
2. **Document Selection**: Selects documents from the knowledge base that are relevant to the examples
3. **Input Generation**: Generates new inputs that follow similar patterns to the examples but are based on the selected documents
4. **Reference Answer Generation**: Generates reference answers for each input
5. **Test Set Creation**: Creates a test set with the inputs, reference answers, and documents

## Example Input Formats

MERIT supports several formats for providing example inputs:

### Simple List of Strings

```python
example_inputs = [
    "What is artificial intelligence?",
    "How does machine learning work?",
    "What's the difference between deep learning and machine learning?"
]
```

### List of Dictionaries

```python
example_inputs = [
    {"input": "What is artificial intelligence?"},
    {"input": "How does machine learning work?"},
    {"input": "What's the difference between deep learning and machine learning?"}
]
```

### ExampleItem Objects

```python
from merit.core.models import ExampleItem

example_inputs = [
    ExampleItem(input="What is artificial intelligence?"),
    ExampleItem(input="How does machine learning work?"),
    ExampleItem(input="What's the difference between deep learning and machine learning?")
]
```

### ExampleSet Object

```python
from merit.core.models import ExampleSet, ExampleItem

example_set = ExampleSet(inputs=[
    ExampleItem(input="What is artificial intelligence?"),
    ExampleItem(input="How does machine learning work?"),
    ExampleItem(input="What's the difference between deep learning and machine learning?")
])
```

### From a File

```python
from merit.core.models import ExampleSet

# Load examples from a JSON file
example_set = ExampleSet.load("my_examples.json")
```

## Advanced Example-Guided Generation

You can customize the example-guided generation process in several ways:

### Removing Similar Examples

You can remove similar examples to ensure diversity:

```python
test_set = generator.generate(
    num_inputs=10,
    example_inputs=example_inputs,
    remove_similar_examples=True,
    similarity_threshold=0.85
)
```

### Skipping Relevance Check

You can skip the relevance check to speed up generation:

```python
test_set = generator.generate(
    num_inputs=10,
    example_inputs=example_inputs,
    skip_relevance_check=True
)
```

### Using Examples with Reference Answers

You can provide examples that include reference answers:

```python
from merit.core.models import ExampleItem

example_inputs = [
    ExampleItem(
        input="What is artificial intelligence?",
        reference_answer="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions."
    ),
    ExampleItem(
        input="How does machine learning work?",
        reference_answer="Machine Learning works by using algorithms to parse data, learn from that data, and then make predictions or decisions based on what it has learned."
    )
]
```

### Using Examples with Responses

You can provide examples that include model responses:

```python
from merit.core.models import ExampleItem

example_inputs = [
    ExampleItem(
        input="What is artificial intelligence?",
        response="Artificial Intelligence (AI) is a field of computer science focused on creating systems capable of performing tasks that typically require human intelligence."
    ),
    ExampleItem(
        input="How does machine learning work?",
        response="Machine learning works by using algorithms to analyze data, identify patterns, and make decisions with minimal human intervention."
    )
]
```

### Using Examples with Feedback

You can provide examples that include feedback on the responses:

```python
from merit.core.models import ExampleItem

example_inputs = [
    ExampleItem(
        input="What is artificial intelligence?",
        response="Artificial Intelligence (AI) is a field of computer science focused on creating systems capable of performing tasks that typically require human intelligence.",
        feedback={"accuracy": 9, "clarity": 8, "completeness": 7}
    ),
    ExampleItem(
        input="How does machine learning work?",
        response="Machine learning works by using algorithms to analyze data, identify patterns, and make decisions with minimal human intervention.",
        feedback={"accuracy": 8, "clarity": 9, "completeness": 7}
    )
]
```

## Understanding Example Analysis

When you provide example inputs, MERIT analyzes them to understand their characteristics. You can access this analysis in the test set metadata:

```python
# Generate a test set with example inputs
test_set = generator.generate(
    num_inputs=10,
    example_inputs=example_inputs
)

# Access the example analysis
example_analysis = test_set.metadata.get("example_inputs", {}).get("style_analysis", {})
print(f"Example analysis: {example_analysis}")
```

The analysis includes information such as:

- **Question types**: The types of questions in the examples (e.g., what, how, why)
- **Complexity**: The complexity level of the examples
- **Style**: The writing style of the examples
- **Tone**: The tone of the examples (e.g., formal, informal, technical)
- **Length**: The average length of the examples

## Best Practices for Example-Guided Generation

When using example-guided generation with MERIT, follow these best practices:

### 1. Provide Diverse Examples

Include a variety of examples to ensure the generated inputs cover different aspects of your knowledge base:

```python
example_inputs = [
    # What questions
    "What is artificial intelligence?",
    "What are the main applications of machine learning?",
    
    # How questions
    "How does deep learning work?",
    "How can businesses benefit from AI?",
    
    # Why questions
    "Why is AI important for the future?",
    "Why do we need machine learning?",
    
    # Comparison questions
    "What's the difference between AI and machine learning?",
    "How does deep learning compare to traditional machine learning?",
    
    # Application questions
    "Can you give examples of AI in healthcare?",
    "What are some real-world applications of deep learning?"
]
```

### 2. Use Real User Queries

When possible, use real user queries as examples to ensure the generated inputs match the patterns of actual users:

```python
# Load real user queries from a log file
with open("user_queries.txt", "r") as f:
    real_user_queries = [line.strip() for line in f if line.strip()]

# Use the real queries as examples
test_set = generator.generate(
    num_inputs=50,
    example_inputs=real_user_queries
)
```

### 3. Balance Example Complexity

Include examples with different complexity levels to ensure the generated inputs cover a range of difficulty:

```python
example_inputs = [
    # Simple questions
    "What is AI?",
    "How does machine learning work?",
    
    # Moderate complexity
    "What's the difference between supervised and unsupervised learning?",
    "How do neural networks process information?",
    
    # Complex questions
    "Can you explain the backpropagation algorithm in neural networks?",
    "What are the ethical implications of using AI in autonomous decision-making systems?"
]
```

### 4. Include Domain-Specific Examples

If your knowledge base covers specific domains, include examples that are relevant to those domains:

```python
# For a medical knowledge base
medical_examples = [
    "What are the symptoms of diabetes?",
    "How is hypertension diagnosed?",
    "What treatments are available for arthritis?"
]

# For a technical knowledge base
technical_examples = [
    "How do I configure a firewall?",
    "What is the difference between HTTP and HTTPS?",
    "How can I optimize database performance?"
]
```

### 5. Review and Refine

Review the generated inputs and refine your examples if needed:

```python
# Generate a test set
test_set = generator.generate(
    num_inputs=10,
    example_inputs=example_inputs
)

# Review the generated inputs
for i, input_sample in enumerate(test_set.inputs):
    print(f"\nInput {i+1}: {input_sample.input}")

# If the generated inputs don't match your expectations,
# refine your examples and try again
refined_examples = [
    # More specific examples
    "What are the key components of a neural network?",
    "How does backpropagation work in deep learning?",
    "What are the advantages of using convolutional neural networks for image processing?"
]

# Generate a new test set with refined examples
refined_test_set = generator.generate(
    num_inputs=10,
    example_inputs=refined_examples
)
```

## Next Steps

Now that you understand example-guided test set generation, you can:

- Learn about [basic generation](./basic_generation.md) for simpler test set creation
- Explore how to [customize the generation process](./customizing_generation.md) for your specific needs
- Understand how to [work with knowledge bases](./working_with_knowledgebase.md) for test set generation
