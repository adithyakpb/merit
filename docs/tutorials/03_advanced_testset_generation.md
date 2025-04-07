# Advanced Test Set Generation

This tutorial explores advanced techniques for generating high-quality test sets in MERIT. You'll learn how to customize the test set generation process, use example-guided generation, and create specialized test sets for specific evaluation scenarios.

## Prerequisites

Before you begin, make sure you have:

- Installed MERIT (see the [Installation Guide](../getting_started/installation.md))
- Completed the [Basic Workflow](./basic_workflow.md) tutorial
- An API key for OpenAI or another supported API provider

## Beyond Basic Test Set Generation

While the basic test set generation approach covered in the [Basic Workflow](./basic_workflow.md) tutorial is sufficient for many use cases, there are several advanced techniques that can help you create more effective test sets:

1. **Example-Guided Generation**: Using example inputs to guide the generation process
2. **Custom Generation Prompts**: Creating custom prompts for input and reference answer generation
3. **Filtering and Post-Processing**: Filtering and post-processing generated inputs
4. **Domain-Specific Generation**: Generating test sets for specific domains
5. **Adversarial Test Sets**: Creating challenging test cases to stress-test your system

Let's explore each of these techniques in detail.

## Example-Guided Generation

One of the most effective ways to control the test set generation process is to provide example inputs. These examples guide the generation process by showing the model what kind of inputs you're looking for.

```python
from merit.testset_generation import TestSetGenerator
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient

# Create a knowledge base
knowledge_base = KnowledgeBase.load("ai_knowledge_base.json")

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Create a test set generator
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    llm_client=client,
    language="en",
    agent_description="A chatbot that answers questions about artificial intelligence."
)

# Define example inputs
example_inputs = [
    "What is the difference between AI and machine learning?",
    "How does deep learning relate to machine learning?",
    "What are some real-world applications of artificial intelligence?",
    "What ethical concerns are associated with AI development?",
    "How has AI evolved over the past decade?"
]

# Generate a test set with example inputs
test_set = generator.generate(
    num_inputs=20,
    example_inputs=example_inputs
)

# Print the generated inputs
print(f"Generated {len(test_set.inputs)} test inputs:")
for i, input_sample in enumerate(test_set.inputs):
    print(f"{i+1}. {input_sample.input}")
```

### Different Types of Example Inputs

You can use different types of example inputs to guide the generation process:

#### Question Types

```python
# Different question types
example_inputs = [
    # What questions
    "What is artificial intelligence?",
    "What are the main branches of AI?",
    
    # How questions
    "How does machine learning work?",
    "How can businesses implement AI solutions?",
    
    # Why questions
    "Why is deep learning considered a breakthrough in AI?",
    "Why are ethical considerations important in AI development?",
    
    # Comparison questions
    "What's the difference between supervised and unsupervised learning?",
    "How does reinforcement learning compare to other machine learning approaches?"
]
```

#### Difficulty Levels

```python
# Different difficulty levels
example_inputs = [
    # Basic questions
    "What is artificial intelligence?",
    "What is machine learning?",
    
    # Intermediate questions
    "How do neural networks learn from data?",
    "What are the differences between CNN and RNN architectures?",
    
    # Advanced questions
    "How does attention mechanism improve transformer models?",
    "What are the limitations of current approaches to explainable AI?"
]
```

#### Domain-Specific Questions

```python
# Domain-specific questions
example_inputs = [
    # Healthcare
    "How is AI being used in medical diagnosis?",
    "What ethical concerns arise when using AI in healthcare?",
    
    # Finance
    "How are financial institutions using AI for fraud detection?",
    "What are the challenges of implementing AI in algorithmic trading?",
    
    # Education
    "How can AI personalize learning experiences?",
    "What are the potential drawbacks of AI-based assessment in education?"
]
```

## Custom Generation Prompts

You can customize the prompts used for generating test inputs and reference answers:

```python
from merit.core.prompts import Prompt
from merit.testset_generation import TestSetGenerator

# Define custom prompts
input_prompt = Prompt(
    """
    Generate {num_inputs} diverse and challenging questions about the following document:
    
    {document_content}
    
    The questions should:
    - Be varied in complexity (from basic to advanced)
    - Cover different aspects of the document
    - Include both factual and analytical questions
    - Be clear and unambiguous
    - Be answerable based on the document content
    
    Questions:
    """
)

reference_prompt = Prompt(
    """
    You are an expert in artificial intelligence. Answer the following question based on the provided document:
    
    Document:
    {document_content}
    
    Question:
    {input}
    
    Provide a comprehensive, accurate, and well-structured answer:
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

### Prompt Templates for Different Scenarios

Here are some prompt templates for different scenarios:

#### Technical Documentation

```python
# For technical documentation
input_prompt = Prompt(
    """
    Generate {num_inputs} questions that a developer might ask about the following technical documentation:
    
    {document_content}
    
    The questions should:
    - Focus on how to implement or use the described functionality
    - Include questions about parameters, return values, and error handling
    - Cover edge cases and potential pitfalls
    - Range from basic usage to advanced scenarios
    
    Questions:
    """
)
```

#### Educational Content

```python
# For educational content
input_prompt = Prompt(
    """
    Generate {num_inputs} questions that a student might ask about the following educational content:
    
    {document_content}
    
    The questions should:
    - Follow Bloom's taxonomy (knowledge, comprehension, application, analysis, synthesis, evaluation)
    - Help assess understanding of key concepts
    - Include both factual recall and conceptual understanding
    - Be suitable for a {difficulty_level} level student
    
    Questions:
    """
)
```

#### Customer Support

```python
# For customer support
input_prompt = Prompt(
    """
    Generate {num_inputs} questions that a customer might ask about the following product information:
    
    {document_content}
    
    The questions should:
    - Cover common customer concerns and issues
    - Include questions about features, pricing, troubleshooting, and support
    - Vary in complexity from basic to advanced
    - Be realistic and reflect actual customer inquiries
    
    Questions:
    """
)
```

## Filtering and Post-Processing

You can implement custom filtering and post-processing logic by extending the `TestSetGenerator` class:

```python
from merit.testset_generation import TestSetGenerator
from merit.core.models import TestItem, TestSet
from typing import List

class CustomTestSetGenerator(TestSetGenerator):
    """Custom test set generator with filtering and post-processing."""
    
    def filter_inputs(self, inputs: List[TestItem]) -> List[TestItem]:
        """
        Filter the generated inputs.
        
        Args:
            inputs: The generated inputs to filter.
            
        Returns:
            List[TestItem]: The filtered inputs.
        """
        filtered_inputs = []
        
        for input_sample in inputs:
            # Filter out inputs that are too short
            if len(input_sample.input.split()) < 5:
                continue
            
            # Filter out inputs that don't end with a question mark
            if not input_sample.input.strip().endswith("?"):
                continue
            
            # Filter out inputs that contain certain keywords
            if "keyword1" in input_sample.input.lower() or "keyword2" in input_sample.input.lower():
                continue
            
            # Add the input to the filtered list
            filtered_inputs.append(input_sample)
        
        return filtered_inputs
    
    def post_process_inputs(self, inputs: List[TestItem]) -> List[TestItem]:
        """
        Post-process the generated inputs.
        
        Args:
            inputs: The generated inputs to post-process.
            
        Returns:
            List[TestItem]: The post-processed inputs.
        """
        processed_inputs = []
        
        for input_sample in inputs:
            # Ensure the input ends with a question mark
            if not input_sample.input.strip().endswith("?"):
                input_sample.input = input_sample.input.strip() + "?"
            
            # Capitalize the first letter
            input_sample.input = input_sample.input[0].upper() + input_sample.input[1:]
            
            # Add the input to the processed list
            processed_inputs.append(input_sample)
        
        return processed_inputs
    
    def generate(self, num_inputs: int, **kwargs) -> TestSet:
        """
        Generate a test set with filtering and post-processing.
        
        Args:
            num_inputs: The number of test inputs to generate.
            **kwargs: Additional arguments for generation.
            
        Returns:
            TestSet: The generated test set.
        """
        # Generate inputs using the parent method
        test_set = super().generate(num_inputs, **kwargs)
        
        # Filter the inputs
        filtered_inputs = self.filter_inputs(test_set.inputs)
        
        # If we filtered out too many inputs, generate more
        while len(filtered_inputs) < num_inputs:
            # Calculate how many more inputs we need
            additional_inputs = num_inputs - len(filtered_inputs)
            
            # Generate additional inputs
            additional_test_set = super().generate(additional_inputs, **kwargs)
            
            # Filter the additional inputs
            additional_filtered = self.filter_inputs(additional_test_set.inputs)
            
            # Add the filtered additional inputs
            filtered_inputs.extend(additional_filtered)
            
            # Avoid infinite loops by limiting the number of iterations
            if not additional_filtered:
                break
        
        # Post-process the filtered inputs
        processed_inputs = self.post_process_inputs(filtered_inputs[:num_inputs])
        
        # Create a new test set with the processed inputs
        processed_test_set = TestSet(
            inputs=processed_inputs,
            metadata=test_set.metadata
        )
        
        return processed_test_set
```

## Domain-Specific Generation

You can create domain-specific test set generators by customizing the generation process for a particular domain:

```python
from merit.testset_generation import TestSetGenerator
from merit.core.models import TestSet
from typing import List, Optional

class MedicalTestSetGenerator(TestSetGenerator):
    """Test set generator for medical questions."""
    
    def __init__(self, medical_specialties=None, **kwargs):
        """
        Initialize the medical test set generator.
        
        Args:
            medical_specialties: The medical specialties to focus on.
            **kwargs: Additional arguments for the test set generator.
        """
        super().__init__(**kwargs)
        self.medical_specialties = medical_specialties or [
            "cardiology", "neurology", "oncology", "pediatrics", "psychiatry"
        ]
    
    def generate(self, num_inputs: int, **kwargs) -> TestSet:
        """
        Generate a medical test set.
        
        Args:
            num_inputs: The number of test inputs to generate.
            **kwargs: Additional arguments for generation.
            
        Returns:
            TestSet: The generated test set.
        """
        # Set a medical-specific agent description if not provided
        if not self.agent_description:
            self.agent_description = "A medical assistant that answers health-related questions based on medical literature."
        
        # Generate example inputs if not provided
        example_inputs = kwargs.get("example_inputs")
        if not example_inputs:
            example_inputs = self._generate_medical_examples()
            kwargs["example_inputs"] = example_inputs
        
        # Generate the test set using the parent method
        test_set = super().generate(num_inputs, **kwargs)
        
        # Add medical metadata to the test set
        test_set.metadata["domain"] = "medical"
        test_set.metadata["specialties"] = self.medical_specialties
        
        return test_set
    
    def _generate_medical_examples(self) -> List[str]:
        """
        Generate medical example inputs.
        
        Returns:
            List[str]: The generated example inputs.
        """
        examples = []
        
        for specialty in self.medical_specialties:
            if specialty == "cardiology":
                examples.extend([
                    "What are the symptoms of a heart attack?",
                    "How is hypertension diagnosed?",
                    "What are the risk factors for coronary artery disease?"
                ])
            elif specialty == "neurology":
                examples.extend([
                    "What are the early signs of Alzheimer's disease?",
                    "How is multiple sclerosis treated?",
                    "What causes migraines?"
                ])
            elif specialty == "oncology":
                examples.extend([
                    "What are the risk factors for breast cancer?",
                    "How is chemotherapy administered?",
                    "What is the survival rate for stage 4 lung cancer?"
                ])
            elif specialty == "pediatrics":
                examples.extend([
                    "What vaccines are recommended for infants?",
                    "How is ADHD diagnosed in children?",
                    "What are the signs of autism in toddlers?"
                ])
            elif specialty == "psychiatry":
                examples.extend([
                    "What are the symptoms of depression?",
                    "How is bipolar disorder treated?",
                    "What causes anxiety disorders?"
                ])
        
        return examples
```

## Adversarial Test Sets

You can create adversarial test sets to stress-test your system:

```python
from merit.testset_generation import TestSetGenerator
from merit.core.models import TestSet, TestItem
from merit.core.prompts import Prompt
from typing import List

class AdversarialTestSetGenerator(TestSetGenerator):
    """Test set generator for adversarial questions."""
    
    def __init__(self, **kwargs):
        """
        Initialize the adversarial test set generator.
        
        Args:
            **kwargs: Additional arguments for the test set generator.
        """
        # Create an adversarial input generation prompt
        adversarial_prompt = Prompt(
            """
            Generate {num_inputs} challenging and adversarial questions about the following document:
            
            {document_content}
            
            The questions should:
            - Be difficult to answer correctly
            - Require careful reading and understanding of the document
            - Include edge cases and corner cases
            - Potentially involve ambiguities or nuances
            - Test the limits of an AI system's understanding
            
            Examples of adversarial questions:
            - Questions that require reasoning about implicit information
            - Questions that involve counterfactuals
            - Questions that require understanding of subtle distinctions
            - Questions that might lead to hallucinations
            
            Questions:
            """
        )
        
        # Initialize the parent class with the adversarial prompt
        super().__init__(input_generation_prompt=adversarial_prompt, **kwargs)
    
    def generate(self, num_inputs: int, **kwargs) -> TestSet:
        """
        Generate an adversarial test set.
        
        Args:
            num_inputs: The number of test inputs to generate.
            **kwargs: Additional arguments for generation.
            
        Returns:
            TestSet: The generated test set.
        """
        # Generate the test set using the parent method
        test_set = super().generate(num_inputs, **kwargs)
        
        # Add adversarial metadata to the test set
        test_set.metadata["type"] = "adversarial"
        
        return test_set
```

## Combining Techniques

You can combine these techniques to create a comprehensive test set generation strategy:

```python
from merit.testset_generation import TestSetGenerator
from merit.core.models import TestSet, TestItem
from merit.core.prompts import Prompt
from typing import List, Dict, Any

class ComprehensiveTestSetGenerator(TestSetGenerator):
    """Comprehensive test set generator that combines multiple techniques."""
    
    def __init__(self, domain=None, difficulty_levels=None, question_types=None, **kwargs):
        """
        Initialize the comprehensive test set generator.
        
        Args:
            domain: The domain to generate questions for.
            difficulty_levels: The difficulty levels to include.
            question_types: The question types to include.
            **kwargs: Additional arguments for the test set generator.
        """
        super().__init__(**kwargs)
        self.domain = domain or "general"
        self.difficulty_levels = difficulty_levels or ["basic", "intermediate", "advanced"]
        self.question_types = question_types or ["factual", "conceptual", "analytical", "comparative"]
    
    def generate(self, num_inputs: int, **kwargs) -> TestSet:
        """
        Generate a comprehensive test set.
        
        Args:
            num_inputs: The number of test inputs to generate.
            **kwargs: Additional arguments for generation.
            
        Returns:
            TestSet: The generated test set.
        """
        # Calculate the number of inputs per category
        inputs_per_difficulty = num_inputs // len(self.difficulty_levels)
        inputs_per_type = num_inputs // len(self.question_types)
        
        # Generate inputs for each difficulty level
        difficulty_inputs = []
        for difficulty in self.difficulty_levels:
            # Create a prompt for this difficulty level
            prompt = Prompt(
                f"""
                Generate {inputs_per_difficulty} {difficulty} questions about the following document:
                
                {{document_content}}
                
                The questions should be at a {difficulty} level, suitable for {"beginners" if difficulty == "basic" else "intermediate learners" if difficulty == "intermediate" else "advanced users"}.
                
                Questions:
                """
            )
            
            # Create a generator for this difficulty level
            generator = TestSetGenerator(
                knowledge_base=self.knowledge_base,
                llm_client=self.llm_client,
                input_generation_prompt=prompt
            )
            
            # Generate inputs
            test_set = generator.generate(inputs_per_difficulty, **kwargs)
            
            # Add metadata
            for input_sample in test_set.inputs:
                input_sample.metadata["difficulty"] = difficulty
            
            # Add to the list
            difficulty_inputs.extend(test_set.inputs)
        
        # Generate inputs for each question type
        type_inputs = []
        for question_type in self.question_types:
            # Create a prompt for this question type
            prompt = Prompt(
                f"""
                Generate {inputs_per_type} {question_type} questions about the following document:
                
                {{document_content}}
                
                The questions should be {question_type} in nature, {"asking for specific facts" if question_type == "factual" else "exploring concepts and ideas" if question_type == "conceptual" else "requiring analysis and reasoning" if question_type == "analytical" else "comparing different aspects"}.
                
                Questions:
                """
            )
            
            # Create a generator for this question type
            generator = TestSetGenerator(
                knowledge_base=self.knowledge_base,
                llm_client=self.llm_client,
                input_generation_prompt=prompt
            )
            
            # Generate inputs
            test_set = generator.generate(inputs_per_type, **kwargs)
            
            # Add metadata
            for input_sample in test_set.inputs:
                input_sample.metadata["question_type"] = question_type
            
            # Add to the list
            type_inputs.extend(test_set.inputs)
        
        # Combine the inputs
        all_inputs = difficulty_inputs + type_inputs
        
        # Create a test set with all inputs
        test_set = TestSet(
            inputs=all_inputs[:num_inputs],  # Limit to the requested number of inputs
            metadata={
                "domain": self.domain,
                "difficulty_levels": self.difficulty_levels,
                "question_types": self.question_types
            }
        )
        
        return test_set
```

## Best Practices for Advanced Test Set Generation

Here are some best practices for advanced test set generation:

### 1. Balance Diversity and Relevance

Create diverse test sets that cover a wide range of scenarios, but ensure that all inputs are relevant to your use case:

```python
# Create a diverse but relevant test set
example_inputs = [
    # Different question types
    "What is artificial intelligence?",
    "How does machine learning work?",
    "Why is deep learning important?",
    
    # Different difficulty levels
    "What are the basic components of a neural network?",
    "How does backpropagation work in neural networks?",
    "What are the limitations of current deep learning approaches?",
    
    # Different domains
    "How is AI used in healthcare?",
    "What are the applications of machine learning in finance?",
    "How is computer vision used in autonomous vehicles?"
]

test_set = generator.generate(
    num_inputs=30,
    example_inputs=example_inputs
)
```

### 2. Include Edge Cases

Include edge cases and corner cases to test the limits of your system:

```python
# Include edge cases
edge_case_examples = [
    # Ambiguous questions
    "Can you explain the difference between AI and intelligence?",
    
    # Questions with implicit assumptions
    "Why hasn't artificial general intelligence been achieved yet?",
    
    # Questions requiring nuanced understanding
    "In what ways might current AI systems be considered intelligent, and in what ways are they not?",
    
    # Questions about limitations
    "What are the fundamental limitations of current machine learning approaches?"
]

test_set = generator.generate(
    num_inputs=20,
    example_inputs=edge_case_examples
)
```

### 3. Use Domain-Specific Knowledge

Leverage domain-specific knowledge to create more effective test sets:

```python
# Use domain-specific knowledge
medical_examples = [
    "What are the potential applications of AI in diagnosing rare diseases?",
    "How can machine learning improve the accuracy of medical imaging analysis?",
    "What ethical considerations should be taken into account when using AI for medical diagnosis?"
]

medical_prompt = Prompt(
    """
    Generate {num_inputs} questions about the application of artificial intelligence in medicine, based on the following document:
    
    {document_content}
    
    The questions should be relevant to healthcare professionals and cover various aspects of AI in medicine, including diagnosis, treatment, research, and ethics.
    
    Questions:
    """
)

generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    llm_client=client,
    input_generation_prompt=medical_prompt
)

test_set = generator.generate(
    num_inputs=20,
    example_inputs=medical_examples
)
```

### 4. Iterative Refinement

Refine your test sets iteratively based on evaluation results:

```python
# Iterative refinement
# Step 1: Generate an initial test set
initial_test_set = generator.generate(num_inputs=20)

# Step 2: Evaluate your system on the initial test set
# (Evaluation code here)

# Step 3: Identify areas for improvement
# (Analysis code here)

# Step 4: Generate additional test inputs focused on those areas
focused_examples = [
    # Examples focused on areas where the system performed poorly
]

additional_test_set = generator.generate(
    num_inputs=10,
    example_inputs=focused_examples
)

# Step 5: Combine the test sets
from merit.core.models import TestSet

combined_inputs = initial_test_set.inputs + additional_test_set.inputs
refined_test_set = TestSet(
    inputs=combined_inputs,
    metadata={"type": "refined"}
)
```

### 5. Document Your Test Sets

Document your test sets thoroughly to ensure reproducibility and facilitate analysis:

```python
# Document your test set
test_set.metadata = {
    "name": "AI Knowledge Test Set",
    "version": "1.0",
    "description": "A test set for evaluating AI systems' knowledge of artificial intelligence concepts",
    "creation_date": "2023-01-15",
    "creator": "Your Name",
    "generation_method": "Example-guided generation with custom prompts",
    "example_inputs": example_inputs,
    "domain": "artificial intelligence",
    "difficulty_levels": ["basic", "intermediate", "advanced"],
    "question_types": ["factual", "conceptual", "analytical", "comparative"]
}

# Save the test set with metadata
test_set.save("ai_knowledge_test_set.json")
```

## Complete Example

Here's a complete example that combines several advanced techniques:

```python
from merit.testset_generation import TestSetGenerator
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient
from merit.core.models import TestSet, TestItem
from merit.core.prompts import Prompt
from typing import List, Dict, Any

# Create a knowledge base
knowledge_base = KnowledgeBase.load("ai_knowledge_base.json")

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Define a custom test set generator
class AdvancedTestSetGenerator(TestSetGenerator):
    """Advanced test set generator with multiple customization options."""
    
    def __init__(self, domain=None, difficulty_levels=None, question_types=None, **kwargs):
        """Initialize the advanced test set generator."""
        super().__init__(**kwargs)
        self.domain = domain or "general"
        self.difficulty_levels = difficulty_levels or ["basic", "intermediate", "advanced"]
        self.question_types = question_types or ["factual", "conceptual", "analytical", "comparative"]
    
    def filter_inputs(self, inputs: List[TestItem]) -> List[TestItem]:
        """Filter the generated inputs."""
        filtered_inputs = []
        
        for input_sample in inputs:
            # Filter out inputs that are too short
            if len(input_sample.input.split()) < 5:
                continue
            
            # Filter out inputs that don't end with a question mark
            if not input_sample.input.strip().endswith("?"):
                continue
            
            # Add the input to the filtered list
            filtered_inputs.append(input_sample)
        
        return filtered_inputs
    
    def generate(self, num_inputs: int, **kwargs) -> TestSet:
        """Generate a test set with advanced customization."""
        # Generate example inputs if not provided
        example_inputs = kwargs.get("example_inputs")
        if not example_inputs:
            example_inputs = self._generate_examples()
            kwargs["example_inputs"] = example_inputs
        
        # Generate the test set using the parent method
        test_set = super().generate(num_inputs, **kwargs)
        
        # Filter the inputs
        filtered_inputs = self.filter_inputs(test_set.inputs)
        
        # If we filtered out too many inputs, generate more
        while len(filtered_inputs) < num_inputs:
            # Calculate how many more inputs we need
            additional_inputs = num_inputs - len(filtered_inputs)
            
            # Generate additional inputs
            additional_test_set = super().generate(additional_inputs, **kwargs)
            
            # Filter the additional inputs
            additional_filtered = self.filter_inputs(additional_test_set.inputs)
            
            # Add the filtered additional inputs
            filtered_inputs.extend(additional_filtered)
            
            # Avoid infinite loops by limiting the number of iterations
            if not additional_filtered:
                break
        
        # Create a new test set with the filtered inputs
        filtered_test_set = TestSet(
            inputs=filtered_inputs[:num_inputs],
            metadata={
                "domain": self.domain,
                "difficulty_levels": self.difficulty_levels,
                "question_types": self.question_types
            }
        )
        
        return filtered_test_set
    
    def _generate_examples(self) -> List[str]:
        """Generate example inputs based on domain, difficulty levels, and question types."""
        examples = []
        
        # Generate examples for each difficulty level and question type
        for difficulty in self.difficulty_levels:
            for question_type in self.question_types:
                if self.domain == "artificial_intelligence":
                    if difficulty == "basic" and question_type == "factual":
                        examples.append("What is artificial intelligence?")
                    elif difficulty == "intermediate" and question_type == "conceptual":
                        examples.append("How do neural networks learn from data?")
                    elif difficulty == "advanced" and question_type == "analytical":
                        examples.append("What are the limitations of current approaches to explainable AI?")
                    elif difficulty == "basic" and question_type == "comparative":
                        examples.append("What's the difference between AI and machine learning?")
                elif self.domain == "medical":
                    if difficulty == "basic" and question_type == "factual":
                        examples.append("What are the symptoms of diabetes?")
                    elif difficulty == "intermediate" and question_type == "conceptual":
                        examples.append("How does the immune system respond to viral infections?")
                    elif difficulty == "advanced" and question_type == "analytical":
                        examples.append("What are the ethical implications of using AI for medical diagnosis?")
                    elif difficulty == "basic" and question_type == "comparative":
                        examples.append("What's the difference between type 1 and type 2 diabetes?")
                else:  # general domain
                    if difficulty == "basic" and question_type == "factual":
                        examples.append("What is the capital of France?")
                    elif difficulty == "intermediate" and question_type == "conceptual":
                        examples.append("How does climate change affect biodiversity?")
                    elif difficulty == "advanced" and question_type == "analytical":
                        examples.append("What are the long-term economic implications of automation?")
                    elif difficulty == "basic" and question_type == "comparative":
                        examples.append("What's the difference between weather and climate?")
        
        return examples

# Create an advanced test set generator
advanced_generator = AdvancedTestSetGenerator(
    knowledge_base=knowledge_base,
    llm_client=client,
    domain="artificial_intelligence",
    difficulty_levels=["basic", "intermediate", "advanced"],
    question_types=["factual", "conceptual", "analytical", "comparative"],
    agent_description="A chatbot that answers questions about artificial intelligence."
)

# Generate a test set
test_set = advanced_generator.generate(num_inputs=20)

# Print the generated inputs
print(f"Generated {len(test_set.inputs)} test inputs:")
for i, input_sample in enumerate(test_set.inputs):
    print(f"{i+1}. {input_sample.input}")

# Save the test set
test_set.save("advanced_ai_test_set.json")
```

## Next Steps

Now that you've learned advanced test set generation techniques, you can:

- Explore [custom API evaluation](./custom_api_evaluation.md) to evaluate your own APIs
- Learn how to create [custom metrics](./custom_metrics_creation.md) for specialized evaluation
- Discover how to create [custom evaluators](../customization/custom_evaluators.md) for advanced evaluation scenarios
- Try the [End-to-End Workflow](./end_to_end_workflow.md) tutorial for a more comprehensive example
