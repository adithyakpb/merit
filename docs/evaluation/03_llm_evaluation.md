# LLM Evaluation

This guide explains how to evaluate general LLM (Large Language Model) outputs using MERIT. Unlike RAG evaluation, which focuses on retrieval-augmented generation systems, LLM evaluation assesses the quality of text generated by language models without necessarily involving retrieval from a knowledge base.

## What is LLM Evaluation?

LLM evaluation assesses the quality of text generated by language models across various dimensions, such as:

- **Accuracy**: How factually correct is the generated text?
- **Relevance**: How well does the generated text address the prompt?
- **Coherence**: How logically structured and consistent is the generated text?
- **Fluency**: How grammatically correct and natural is the generated text?
- **Creativity**: How original and creative is the generated text?
- **Helpfulness**: How helpful is the generated text for the user's needs?

MERIT provides a flexible framework for evaluating LLM outputs using custom prompts and metrics.

## The LLMEvaluator Class

The `LLMEvaluator` class is the main tool for evaluating LLM outputs in MERIT. It uses an LLM (typically a more capable one) to evaluate the outputs of another LLM.

```python
from merit.evaluation.evaluators.llm import LLMEvaluator
from merit.core.prompts import Prompt
from merit.api.client import OpenAIClient

# Create a client for evaluation
client = OpenAIClient(api_key="your-openai-api-key", model="gpt-4")

# Create an LLM evaluator with a custom prompt
evaluator = LLMEvaluator(
    prompt=Prompt("""
    Evaluate the following response to the given prompt:
    
    Prompt: {conversation[0]['content']}
    
    Response: {conversation[1]['content']}
    
    Please rate the response on a scale of 1-10 for the following criteria:
    - Accuracy: How factually correct is the response?
    - Relevance: How well does the response address the prompt?
    - Coherence: How logically structured and consistent is the response?
    - Fluency: How grammatically correct and natural is the response?
    - Helpfulness: How helpful is the response for the user's needs?
    
    For each criterion, provide a score and a brief explanation.
    
    Return your evaluation in the following JSON format:
    {
        "accuracy": {"score": 0, "explanation": ""},
        "relevance": {"score": 0, "explanation": ""},
        "coherence": {"score": 0, "explanation": ""},
        "fluency": {"score": 0, "explanation": ""},
        "helpfulness": {"score": 0, "explanation": ""}
    }
    """),
    llm_client=client,
    llm_temperature=0,
    llm_seed=42,
    llm_output_format="json_object"
)
```

## Basic LLM Evaluation

Here's a simple example of how to evaluate an LLM output using MERIT:

```python
# Define a model to evaluate
class SimpleModel:
    def __init__(self, client):
        self.client = client
        self.feature_names = ["prompt"]
    
    def predict(self, dataset):
        prompts = dataset.df["prompt"].tolist()
        responses = [self.client.generate_text(prompt) for prompt in prompts]
        return type('obj', (object,), {'prediction': responses})

# Create a dataset
import pandas as pd
dataset = pd.DataFrame({
    "prompt": [
        "Explain the concept of machine learning to a 10-year-old.",
        "What are the main causes of climate change?",
        "Write a short poem about the ocean."
    ]
})
dataset = type('obj', (object,), {'df': dataset})

# Create a model to evaluate
model = SimpleModel(client)

# Evaluate the model
report = evaluator.evaluate(model, dataset)

# Print the results
print("Evaluation results:")
for result in report.results:
    print(f"\nPrompt: {result.input}")
    print(f"Response: {result.model_answer}")
    print("Scores:")
    for metric, score in result.scores.items():
        print(f"  {metric}: {score:.2f}")
    
    # Print explanations if available
    if result.explanations:
        print("Explanations:")
        for metric, explanation in result.explanations.items():
            print(f"  {metric}: {explanation}")
```

## Custom Evaluation Prompts

One of the key features of the `LLMEvaluator` is the ability to use custom prompts for evaluation. This allows you to tailor the evaluation to your specific needs:

```python
# Create an evaluator for creative writing
creative_evaluator = LLMEvaluator(
    prompt=Prompt("""
    Evaluate the following creative writing response to the given prompt:
    
    Prompt: {conversation[0]['content']}
    
    Response: {conversation[1]['content']}
    
    Please rate the response on a scale of 1-10 for the following criteria:
    - Creativity: How original and imaginative is the response?
    - Engagement: How engaging and captivating is the response?
    - Style: How effective is the writing style?
    - Imagery: How vivid and evocative are the descriptions?
    - Overall Quality: What is the overall quality of the writing?
    
    For each criterion, provide a score and a brief explanation.
    
    Return your evaluation in the following JSON format:
    {
        "creativity": {"score": 0, "explanation": ""},
        "engagement": {"score": 0, "explanation": ""},
        "style": {"score": 0, "explanation": ""},
        "imagery": {"score": 0, "explanation": ""},
        "overall_quality": {"score": 0, "explanation": ""}
    }
    """),
    llm_client=client,
    llm_temperature=0,
    llm_seed=42,
    llm_output_format="json_object"
)

# Create an evaluator for factual accuracy
factual_evaluator = LLMEvaluator(
    prompt=Prompt("""
    Evaluate the factual accuracy of the following response to the given prompt:
    
    Prompt: {conversation[0]['content']}
    
    Response: {conversation[1]['content']}
    
    Please identify any factual errors or inaccuracies in the response.
    For each error, provide:
    1. The incorrect statement
    2. Why it's incorrect
    3. The correct information
    
    Then, rate the overall factual accuracy on a scale of 1-10, where:
    - 1-3: Contains major factual errors that significantly mislead the reader
    - 4-6: Contains some factual errors or inaccuracies
    - 7-9: Contains minor inaccuracies or omissions
    - 10: Completely factually accurate
    
    Return your evaluation in the following JSON format:
    {
        "errors": [
            {"statement": "", "explanation": "", "correction": ""}
        ],
        "accuracy_score": 0,
        "explanation": ""
    }
    """),
    llm_client=client,
    llm_temperature=0,
    llm_seed=42,
    llm_output_format="json_object"
)
```

## Evaluating Conversations

You can also evaluate multi-turn conversations:

```python
# Create a conversation dataset
conversations = [
    [
        {"role": "user", "content": "What is artificial intelligence?"},
        {"role": "assistant", "content": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions. The term may also be applied to any machine that exhibits traits associated with a human mind such as learning and problem-solving."}
    ],
    [
        {"role": "user", "content": "How does a neural network work?"},
        {"role": "assistant", "content": "A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. It consists of input layers, hidden layers, and output layers, with each layer containing nodes or 'neurons' that process information."}
    ]
]

# Create a dataset
conversation_dataset = pd.DataFrame({
    "conversation": conversations
})
conversation_dataset = type('obj', (object,), {'df': conversation_dataset})

# Create a conversation evaluator
conversation_evaluator = LLMEvaluator(
    prompt=Prompt("""
    Evaluate the following conversation between a user and an assistant:
    
    {conversation}
    
    Please rate the assistant's response on a scale of 1-10 for the following criteria:
    - Helpfulness: How helpful is the response for the user's query?
    - Accuracy: How factually correct is the response?
    - Clarity: How clear and understandable is the response?
    
    For each criterion, provide a score and a brief explanation.
    
    Return your evaluation in the following JSON format:
    {
        "helpfulness": {"score": 0, "explanation": ""},
        "accuracy": {"score": 0, "explanation": ""},
        "clarity": {"score": 0, "explanation": ""}
    }
    """),
    llm_client=client,
    llm_temperature=0,
    llm_seed=42,
    llm_output_format="json_object"
)

# Define a model that returns the assistant's response
class ConversationModel:
    def __init__(self):
        self.feature_names = ["conversation"]
    
    def predict(self, dataset):
        # Extract the assistant's response from each conversation
        responses = [conv[-1]["content"] for conv in dataset.df["conversation"]]
        return type('obj', (object,), {'prediction': responses})

# Create a model to evaluate
conversation_model = ConversationModel()

# Evaluate the model
report = conversation_evaluator.evaluate(conversation_model, conversation_dataset)
```

## Analyzing Evaluation Results

MERIT provides detailed evaluation results that you can analyze to understand your LLM's performance:

```python
# Get overall scores
overall_scores = {}
for result in report.results:
    for metric, score in result.scores.items():
        if metric not in overall_scores:
            overall_scores[metric] = []
        overall_scores[metric].append(score)

print("Overall scores:")
for metric, scores in overall_scores.items():
    avg_score = sum(scores) / len(scores)
    print(f"  {metric}: {avg_score:.2f}")

# Identify strengths and weaknesses
strengths = {}
weaknesses = {}
for result in report.results:
    for metric, score in result.scores.items():
        if score >= 8.0:
            if metric not in strengths:
                strengths[metric] = 0
            strengths[metric] += 1
        elif score <= 5.0:
            if metric not in weaknesses:
                weaknesses[metric] = 0
            weaknesses[metric] += 1

print("\nStrengths:")
for metric, count in strengths.items():
    print(f"  {metric}: {count} high-scoring responses")

print("\nWeaknesses:")
for metric, count in weaknesses.items():
    print(f"  {metric}: {count} low-scoring responses")
```

## Best Practices for LLM Evaluation

When evaluating LLMs with MERIT, follow these best practices:

### 1. Use a More Capable Evaluator

For best results, use a more capable LLM (like GPT-4) to evaluate the outputs of less capable LLMs:

```python
evaluator = LLMEvaluator(
    prompt=Prompt("..."),
    llm_client=OpenAIClient(api_key="your-openai-api-key", model="gpt-4"),
    llm_temperature=0
)
```

### 2. Design Clear Evaluation Prompts

Create clear and specific evaluation prompts that define the criteria and scoring system:

```python
prompt = Prompt("""
Evaluate the following response to the given prompt:

Prompt: {conversation[0]['content']}

Response: {conversation[1]['content']}

Please rate the response on a scale of 1-10 for the following criteria:
- Criterion 1: [Clear definition of what this criterion means]
- Criterion 2: [Clear definition of what this criterion means]

For each criterion, provide a score and a brief explanation.

Return your evaluation in the following JSON format:
{
    "criterion_1": {"score": 0, "explanation": ""},
    "criterion_2": {"score": 0, "explanation": ""}
}
""")
```

### 3. Use Consistent Parameters

Use consistent parameters (like temperature and seed) for reproducible evaluations:

```python
evaluator = LLMEvaluator(
    prompt=Prompt("..."),
    llm_client=client,
    llm_temperature=0,
    llm_seed=42,
    llm_output_format="json_object"
)
```

### 4. Evaluate Across Multiple Dimensions

Assess LLM outputs across multiple dimensions to get a comprehensive understanding of performance:

```python
prompt = Prompt("""
Evaluate the following response on multiple dimensions:
- Dimension 1: [Description]
- Dimension 2: [Description]
- Dimension 3: [Description]
...
""")
```

### 5. Include Explanations

Request explanations for scores to understand the reasoning behind the evaluation:

```python
prompt = Prompt("""
For each criterion, provide:
1. A score from 1-10
2. A detailed explanation of why you assigned that score
3. Specific examples from the response that support your evaluation
""")
```

## Next Steps

Now that you understand how to evaluate LLM outputs with MERIT, you can:

- Learn about [RAG evaluation](./rag_evaluation.md) for retrieval-augmented generation systems
- Explore [classification metrics](./classification_metrics.md) for structured outputs
- Create [custom metrics](./custom_metrics.md) for specialized evaluation tasks
- Generate and interpret [evaluation reports](./evaluation_reports.md)
