# Custom Prompts

This guide explains how to create and use custom prompts in MERIT. Custom prompts allow you to tailor the behavior of LLM-based components, such as evaluators, generators, and metrics, to your specific needs.

## Why Create Custom Prompts?

You might want to create custom prompts for several reasons:

- **Specialized Evaluation**: Create prompts that evaluate specific aspects of AI system outputs
- **Domain-Specific Generation**: Generate test inputs tailored to your domain
- **Custom Instructions**: Provide specific instructions to LLMs for various tasks
- **Consistent Formatting**: Ensure consistent formatting of outputs
- **Improved Performance**: Optimize prompts for better performance on specific tasks

## Prompt Types in MERIT

MERIT uses prompts in various components:

- **Evaluation Prompts**: Used by evaluators to assess AI system outputs
- **Generation Prompts**: Used by generators to create test inputs and reference answers
- **Metric Prompts**: Used by metrics to calculate scores
- **Instruction Prompts**: Used to provide instructions to LLMs for various tasks

## The Prompt Class

MERIT provides a `Prompt` class for creating and managing prompts:

```python
from merit.core.prompts import Prompt

# Create a simple prompt
prompt = Prompt("Generate a response to the following question: {question}")

# Format the prompt with variables
formatted_prompt = prompt.format(question="What is artificial intelligence?")
print(formatted_prompt)
# Output: Generate a response to the following question: What is artificial intelligence?
```

## Creating Basic Prompts

You can create basic prompts using the `Prompt` class:

```python
from merit.core.prompts import Prompt

# Create a prompt for generating questions
question_prompt = Prompt(
    """
    Generate {num_questions} diverse and interesting questions about the following topic:
    
    Topic: {topic}
    
    The questions should be varied in complexity and cover different aspects of the topic.
    
    Questions:
    """
)

# Format the prompt with variables
formatted_prompt = question_prompt.format(
    num_questions=5,
    topic="Artificial Intelligence"
)

print(formatted_prompt)
```

## Creating Evaluation Prompts

You can create custom prompts for evaluation:

```python
from merit.core.prompts import Prompt

# Create a prompt for evaluating factual accuracy
accuracy_prompt = Prompt(
    """
    Evaluate the factual accuracy of the following response to the given question:
    
    Question: {question}
    
    Response: {response}
    
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
    """
)
```

## Creating Generation Prompts

You can create custom prompts for test set generation:

```python
from merit.core.prompts import Prompt

# Create a prompt for generating test inputs
input_generation_prompt = Prompt(
    """
    Generate {num_inputs} diverse and realistic questions about the following document:
    
    {document_content}
    
    The questions should be varied in complexity and cover different aspects of the document.
    
    Questions:
    """
)

# Create a prompt for generating reference answers
reference_generation_prompt = Prompt(
    """
    Answer the following question based on the provided document:
    
    Document:
    {document_content}
    
    Question:
    {input}
    
    Provide a comprehensive and accurate answer:
    """
)
```

## Using Custom Prompts with Evaluators

You can use custom prompts with evaluators:

```python
from merit.evaluation.evaluators.llm import LLMEvaluator
from merit.core.prompts import Prompt
from merit.api.client import OpenAIClient

# Create a custom evaluation prompt
evaluation_prompt = Prompt(
    """
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
    """
)

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Create an evaluator with the custom prompt
evaluator = LLMEvaluator(
    prompt=evaluation_prompt,
    llm_client=client,
    llm_temperature=0,
    llm_seed=42,
    llm_output_format="json_object"
)
```

## Using Custom Prompts with Generators

You can use custom prompts with test set generators:

```python
from merit.testset_generation import TestSetGenerator
from merit.core.prompts import Prompt
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=[...])

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Create custom generation prompts
input_prompt = Prompt(
    """
    Generate {num_inputs} diverse and realistic questions about the following document:
    
    {document_content}
    
    The questions should be varied in complexity and cover different aspects of the document.
    
    Questions:
    """
)

reference_prompt = Prompt(
    """
    Answer the following question based on the provided document:
    
    Document:
    {document_content}
    
    Question:
    {input}
    
    Provide a comprehensive and accurate answer:
    """
)

# Create a test set generator with custom prompts
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    llm_client=client,
    input_generation_prompt=input_prompt,
    reference_generation_prompt=reference_prompt
)

# Generate a test set
test_set = generator.generate(num_inputs=10)
```

## Using Custom Prompts with Metrics

You can use custom prompts with metrics:

```python
from merit.evaluation.metrics.base import BaseMetric
from merit.core.prompts import Prompt
from merit.api.client import OpenAIClient
from typing import Dict, Any

class CustomPromptMetric(BaseMetric):
    """A metric that uses a custom prompt for evaluation."""
    
    def __init__(self, name="custom_prompt_metric", llm_client=None, prompt=None, **kwargs):
        """
        Initialize the custom prompt metric.
        
        Args:
            name: The name of the metric.
            llm_client: The LLM client to use for evaluation.
            prompt: The prompt to use for evaluation.
            **kwargs: Additional arguments for the metric.
        """
        self.name = name
        self._llm_client = llm_client
        self._prompt = prompt or Prompt(
            """
            Evaluate the following answer to the given question:
            
            Question: {question}
            
            Answer: {answer}
            
            Rate the answer on a scale of 0 to 1 for the following criteria:
            - Accuracy: How factually correct is the answer?
            - Completeness: How complete is the answer?
            - Clarity: How clear and understandable is the answer?
            
            For each criterion, provide a score and a brief explanation.
            
            Return your evaluation in the following JSON format:
            {
                "accuracy": {"score": 0.0, "explanation": ""},
                "completeness": {"score": 0.0, "explanation": ""},
                "clarity": {"score": 0.0, "explanation": ""}
            }
            """
        )
        self._kwargs = kwargs
    
    def __call__(self, input_sample, answer) -> Dict[str, Any]:
        """
        Calculate the metric for a given input and answer.
        
        Args:
            input_sample: The input sample (question/prompt).
            answer: The answer to evaluate.
            
        Returns:
            Dict[str, Any]: A dictionary containing the metric scores and explanations.
        """
        if not self._llm_client:
            raise ValueError("LLM client is required for this metric")
        
        # Get the answer text
        answer_text = answer.message if hasattr(answer, "message") else answer
        
        # Format the prompt
        prompt = self._prompt.format(
            question=input_sample,
            answer=answer_text
        )
        
        # Generate the evaluation
        evaluation = self._llm_client.generate_text(prompt)
        
        # Parse the evaluation
        try:
            import json
            result = json.loads(evaluation)
            
            # Extract the scores and explanations
            metrics = {}
            for criterion, data in result.items():
                score = data.get("score", 0)
                explanation = data.get("explanation", "")
                
                # Add the score and explanation to the metrics
                metrics[criterion] = score
                metrics[f"{criterion}_explanation"] = explanation
            
            return metrics
        except Exception as e:
            # If parsing fails, return a default result
            return {
                self.name: 0.0,
                f"{self.name}_explanation": f"Failed to parse evaluation: {str(e)}"
            }
```

## Advanced Prompt Techniques

### Using System and User Messages

For models that support system and user messages, you can create prompts with different roles:

```python
from merit.core.prompts import Prompt

# Create a prompt with system and user messages
system_user_prompt = Prompt(
    """
    <|system|>
    You are an expert evaluator of AI-generated text. Your task is to evaluate the quality of responses to user queries.
    </|system|>
    
    <|user|>
    Please evaluate the following response to the given query:
    
    Query: {query}
    
    Response: {response}
    
    Rate the response on a scale of 1-10 for accuracy, relevance, and helpfulness.
    </|user|>
    """
)
```

### Using Few-Shot Examples

You can include few-shot examples in your prompts:

```python
from merit.core.prompts import Prompt

# Create a prompt with few-shot examples
few_shot_prompt = Prompt(
    """
    Your task is to evaluate the factual accuracy of answers to questions.
    
    Here are some examples:
    
    Question: What is the capital of France?
    Answer: Paris is the capital of France.
    Evaluation: This answer is factually correct. The capital of France is indeed Paris.
    Score: 10/10
    
    Question: What is the tallest mountain in the world?
    Answer: The tallest mountain in the world is Mount Kilimanjaro.
    Evaluation: This answer is factually incorrect. The tallest mountain in the world is Mount Everest, not Mount Kilimanjaro.
    Score: 2/10
    
    Question: Who wrote the novel "Pride and Prejudice"?
    Answer: The novel "Pride and Prejudice" was written by Jane Austen in 1813.
    Evaluation: This answer is factually correct. Jane Austen did write "Pride and Prejudice" in 1813.
    Score: 10/10
    
    Now, evaluate the following answer:
    
    Question: {question}
    Answer: {answer}
    Evaluation:
    """
)
```

### Using Structured Output Formats

You can specify structured output formats in your prompts:

```python
from merit.core.prompts import Prompt

# Create a prompt with a structured output format
structured_prompt = Prompt(
    """
    Evaluate the following answer to the given question:
    
    Question: {question}
    
    Answer: {answer}
    
    Please provide your evaluation in the following JSON format:
    {
        "accuracy": {
            "score": 0.0,  // A score between 0.0 and 1.0
            "explanation": ""  // An explanation for the score
        },
        "completeness": {
            "score": 0.0,  // A score between 0.0 and 1.0
            "explanation": ""  // An explanation for the score
        },
        "clarity": {
            "score": 0.0,  // A score between 0.0 and 1.0
            "explanation": ""  // An explanation for the score
        }
    }
    """
)
```

## Best Practices for Custom Prompts

When creating custom prompts, follow these best practices:

### 1. Be Clear and Specific

Provide clear and specific instructions in your prompts:

```python
# Good prompt
good_prompt = Prompt(
    """
    Generate 5 diverse and challenging questions about artificial intelligence.
    The questions should:
    - Cover different aspects of AI (e.g., history, applications, ethics)
    - Vary in complexity (from beginner to advanced)
    - Be clear and unambiguous
    - Be answerable based on general knowledge about AI
    
    Format each question as a complete sentence ending with a question mark.
    """
)

# Bad prompt
bad_prompt = Prompt(
    """
    Generate some AI questions.
    """
)
```

### 2. Use Consistent Formatting

Use consistent formatting in your prompts:

```python
# Consistent formatting
consistent_prompt = Prompt(
    """
    Evaluate the following response to the given prompt:
    
    Prompt: {prompt}
    
    Response: {response}
    
    Please rate the response on a scale of 1-10 for the following criteria:
    - Criterion 1: Description of criterion 1
    - Criterion 2: Description of criterion 2
    - Criterion 3: Description of criterion 3
    
    For each criterion, provide a score and a brief explanation.
    
    Return your evaluation in the following JSON format:
    {
        "criterion_1": {"score": 0, "explanation": ""},
        "criterion_2": {"score": 0, "explanation": ""},
        "criterion_3": {"score": 0, "explanation": ""}
    }
    """
)
```

### 3. Include Examples

Include examples to guide the model:

```python
# Prompt with examples
example_prompt = Prompt(
    """
    Generate questions about the following document:
    
    {document_content}
    
    The questions should be diverse and cover different aspects of the document.
    
    Here are some examples of good questions:
    - What is the main topic of the document?
    - How does the author support their argument about X?
    - What evidence is presented for the claim that Y?
    - What are the implications of Z according to the document?
    - How does the document compare X and Y?
    
    Generate {num_questions} questions:
    """
)
```

### 4. Specify Output Format

Specify the desired output format:

```python
# Prompt with output format
format_prompt = Prompt(
    """
    Generate a summary of the following document:
    
    {document_content}
    
    Your summary should:
    - Be concise (no more than 3 paragraphs)
    - Capture the main points of the document
    - Be written in a neutral tone
    - Not include any information not present in the document
    
    Format your summary as plain text with no special formatting.
    """
)
```

### 5. Test and Refine

Test your prompts and refine them based on the results:

```python
from merit.core.prompts import Prompt
from merit.api.client import OpenAIClient

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Create a prompt
prompt = Prompt(
    """
    Generate {num_questions} diverse and challenging questions about artificial intelligence.
    """
)

# Test the prompt
formatted_prompt = prompt.format(num_questions=3)
result = client.generate_text(formatted_prompt)
print(result)

# Refine the prompt based on the results
refined_prompt = Prompt(
    """
    Generate {num_questions} diverse and challenging questions about artificial intelligence.
    The questions should:
    - Cover different aspects of AI (e.g., history, applications, ethics)
    - Vary in complexity (from beginner to advanced)
    - Be clear and unambiguous
    - Be answerable based on general knowledge about AI
    
    Format each question as a complete sentence ending with a question mark.
    """
)

# Test the refined prompt
formatted_refined_prompt = refined_prompt.format(num_questions=3)
refined_result = client.generate_text(formatted_refined_prompt)
print(refined_result)
```

## Next Steps

Now that you know how to create custom prompts, you can:

- Learn how to create [custom metrics](./custom_metrics.md) that use custom prompts
- Explore how to create [custom evaluators](./custom_evaluators.md) with custom prompts
- Discover how to customize [test set generation](../testset_generation/customizing_generation.md) with custom prompts
- Learn about [prompt engineering techniques](./prompt_engineering.md) for better results
