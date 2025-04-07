# Customizing Test Set Generation

This guide explains how to customize the test set generation process in MERIT. While the basic generation methods are suitable for many use cases, you may want to customize the process to better suit your specific needs.

## Customization Options

MERIT provides several ways to customize the test set generation process:

- **Generator Parameters**: Configure the `TestSetGenerator` with various parameters
- **Example Inputs**: Guide the generation process with example inputs
- **Custom Prompts**: Define custom prompts for input and reference answer generation
- **Custom Filters**: Implement custom filters for generated inputs
- **Custom Generation Logic**: Extend the `TestSetGenerator` class with custom logic

## Generator Parameters

The `TestSetGenerator` class accepts several parameters that allow you to customize the generation process:

```python
from merit.testset_generation import TestSetGenerator
from merit.knowledge import KnowledgeBase
from merit.api.client import OpenAIClient

# Create a knowledge base
knowledge_base = KnowledgeBase(documents=[...])

# Create an API client
client = OpenAIClient(api_key="your-openai-api-key")

# Create a test set generator with custom parameters
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    llm_client=client,
    language="en",
    agent_description="A chatbot that answers questions about artificial intelligence.",
    batch_size=16,
    max_retries=3,
    timeout=30
)
```

### Available Parameters

- **knowledge_base**: The knowledge base to generate test inputs from
- **llm_client**: The API client to use for generation (optional)
- **language**: The language to generate test inputs in (default: "en")
- **agent_description**: A description of the agent being evaluated (optional)
- **batch_size**: The number of documents to process in each batch (default: 8)
- **max_retries**: The maximum number of retries for failed generations (default: 3)
- **timeout**: The timeout for API requests in seconds (default: 60)

## Generation Method Parameters

The `generate()` method also accepts several parameters for customizing the generation process:

```python
# Generate a test set with custom parameters
test_set = generator.generate(
    num_inputs=50,
    example_inputs=["What is artificial intelligence?", "How does machine learning work?"],
    remove_similar_examples=True,
    similarity_threshold=0.85,
    skip_relevance_check=False,
    metadata={"name": "AI Test Set", "version": "1.0"}
)
```

### Available Parameters

- **num_inputs**: The number of test inputs to generate
- **example_inputs**: Example inputs to guide the generation process (optional)
- **remove_similar_examples**: Whether to remove similar examples (default: False)
- **similarity_threshold**: The threshold for similarity detection (default: 0.85)
- **skip_relevance_check**: Whether to skip the relevance check (default: False)
- **metadata**: Metadata to include in the test set (optional)

## Using Custom Prompts

You can customize the prompts used for generating test inputs and reference answers:

```python
from merit.core.prompts import Prompt
from merit.testset_generation import TestSetGenerator

# Define custom prompts
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

## Implementing Custom Filters

You can implement custom filters for generated inputs by extending the `TestSetGenerator` class:

```python
from merit.testset_generation import TestSetGenerator
from merit.core.models import TestItem
from typing import List

class CustomTestSetGenerator(TestSetGenerator):
    """Custom test set generator with input filtering."""
    
    def filter_inputs(self, inputs: List[TestItem]) -> List[TestItem]:
        """
        Filter the generated inputs.
        
        Args:
            inputs: The generated inputs to filter.
            
        Returns:
            List[TestItem]: The filtered inputs.
        """
        # Filter out inputs that don't meet certain criteria
        filtered_inputs = []
        
        for input_sample in inputs:
            # Example: Filter out inputs that are too short
            if len(input_sample.input.split()) < 5:
                continue
            
            # Example: Filter out inputs that don't end with a question mark
            if not input_sample.input.strip().endswith("?"):
                continue
            
            # Example: Filter out inputs that contain certain keywords
            if "keyword1" in input_sample.input.lower() or "keyword2" in input_sample.input.lower():
                continue
            
            # Add the input to the filtered list
            filtered_inputs.append(input_sample)
        
        return filtered_inputs
    
    def generate(self, num_inputs: int, **kwargs) -> 'TestSet':
        """
        Generate a test set.
        
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
        
        # Create a new test set with the filtered inputs
        from merit.core.models import TestSet
        filtered_test_set = TestSet(
            inputs=filtered_inputs[:num_inputs],
            metadata=test_set.metadata
        )
        
        return filtered_test_set
```

## Implementing Custom Generation Logic

You can implement custom generation logic by extending the `TestSetGenerator` class:

```python
from merit.testset_generation import TestSetGenerator
from merit.core.models import TestSet, TestItem, Document
from typing import List, Optional

class TemplateBasedGenerator(TestSetGenerator):
    """Test set generator that uses templates for generation."""
    
    def __init__(self, templates: List[str], **kwargs):
        """
        Initialize the template-based generator.
        
        Args:
            templates: The templates to use for generation.
            **kwargs: Additional arguments to pass to the parent class.
        """
        super().__init__(**kwargs)
        self.templates = templates
    
    def generate_from_document(self, document: Document, num_inputs: int) -> List[TestItem]:
        """
        Generate test inputs from a document using templates.
        
        Args:
            document: The document to generate inputs from.
            num_inputs: The number of inputs to generate.
            
        Returns:
            List[TestItem]: The generated test inputs.
        """
        # Generate inputs using templates
        inputs = []
        
        # Extract key information from the document
        # (This is a simplified example; in practice, you might use NLP techniques)
        import re
        entities = re.findall(r'\b[A-Z][a-z]+\b', document.content)
        
        # Use templates to generate inputs
        for template in self.templates:
            # Skip if we have enough inputs
            if len(inputs) >= num_inputs:
                break
            
            # Replace placeholders in the template
            for entity in entities:
                # Skip if we have enough inputs
                if len(inputs) >= num_inputs:
                    break
                
                # Replace the placeholder with the entity
                input_text = template.replace("{entity}", entity)
                
                # Generate a reference answer
                reference_answer = self._generate_reference_answer(input_text, document)
                
                # Create a test input
                test_input = TestItem(
                    input=input_text,
                    reference_answer=reference_answer,
                    document=document,
                    metadata={"template": template, "entity": entity}
                )
                
                # Add the input to the list
                inputs.append(test_input)
        
        # If we don't have enough inputs, fall back to the parent method
        if len(inputs) < num_inputs:
            # Calculate how many more inputs we need
            additional_inputs = num_inputs - len(inputs)
            
            # Generate additional inputs using the parent method
            parent_inputs = super().generate_from_document(document, additional_inputs)
            
            # Add the additional inputs
            inputs.extend(parent_inputs)
        
        return inputs[:num_inputs]
    
    def _generate_reference_answer(self, input_text: str, document: Document) -> str:
        """
        Generate a reference answer for an input.
        
        Args:
            input_text: The input text.
            document: The document to generate the answer from.
            
        Returns:
            str: The generated reference answer.
        """
        # Use the parent method to generate a reference answer
        if self.llm_client:
            return super()._generate_reference_answer(input_text, document)
        
        # If no LLM client is available, use a simple approach
        # (This is a simplified example; in practice, you might use more sophisticated techniques)
        if "what is" in input_text.lower():
            # Extract the entity from the input
            entity = input_text.split("what is")[-1].strip().rstrip("?")
            
            # Find sentences in the document that mention the entity
            sentences = document.content.split(".")
            relevant_sentences = [s for s in sentences if entity.lower() in s.lower()]
            
            # Join the relevant sentences
            if relevant_sentences:
                return ". ".join(relevant_sentences) + "."
        
        # Default to returning the first few sentences of the document
        sentences = document.content.split(".")
        return ". ".join(sentences[:3]) + "."
```

## Example: Domain-Specific Generator

Here's an example of a domain-specific test set generator for medical questions:

```python
from merit.testset_generation import TestSetGenerator
from merit.core.models import TestSet, TestItem, Document
from typing import List, Optional

class MedicalTestSetGenerator(TestSetGenerator):
    """Test set generator for medical questions."""
    
    def __init__(self, medical_specialties: List[str] = None, **kwargs):
        """
        Initialize the medical test set generator.
        
        Args:
            medical_specialties: The medical specialties to focus on.
            **kwargs: Additional arguments to pass to the parent class.
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
        # Generate example inputs for each specialty
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
    
    def filter_document(self, document: Document) -> bool:
        """
        Filter documents based on medical relevance.
        
        Args:
            document: The document to filter.
            
        Returns:
            bool: True if the document should be included, False otherwise.
        """
        # Check if the document is related to medicine
        content = document.content.lower()
        medical_terms = ["patient", "disease", "treatment", "symptom", "diagnosis", "medical", "health", "doctor", "hospital"]
        
        # Count how many medical terms are in the document
        medical_term_count = sum(1 for term in medical_terms if term in content)
        
        # Include the document if it contains at least 2 medical terms
        return medical_term_count >= 2
    
    def generate_from_document(self, document: Document, num_inputs: int) -> List[TestItem]:
        """
        Generate test inputs from a document.
        
        Args:
            document: The document to generate inputs from.
            num_inputs: The number of inputs to generate.
            
        Returns:
            List[TestItem]: The generated test inputs.
        """
        # Filter out non-medical documents
        if not self.filter_document(document):
            return []
        
        # Generate inputs using the parent method
        return super().generate_from_document(document, num_inputs)
```

## Best Practices for Customizing Generation

When customizing the test set generation process, follow these best practices:

### 1. Start with the Basics

Start with the basic generation methods and parameters before implementing custom logic:

```python
# Start with basic parameters
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    llm_client=client,
    language="en",
    agent_description="A chatbot that answers questions about artificial intelligence."
)

# Generate a test set with basic parameters
test_set = generator.generate(num_inputs=10)

# If the results aren't satisfactory, try adding example inputs
test_set = generator.generate(
    num_inputs=10,
    example_inputs=["What is artificial intelligence?", "How does machine learning work?"]
)

# If you still need more customization, then consider extending the class
```

### 2. Use Example Inputs

Example inputs are a powerful way to guide the generation process without implementing custom logic:

```python
# Use diverse example inputs
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

# Generate a test set with example inputs
test_set = generator.generate(
    num_inputs=50,
    example_inputs=example_inputs
)
```

### 3. Customize Prompts

Customize the prompts to control the style and content of the generated inputs and answers:

```python
# Define a custom input generation prompt
input_prompt = Prompt(
    """
    Generate {num_inputs} questions about the following document that would be asked by a {user_type}:
    
    {document_content}
    
    The questions should be {difficulty_level} and focus on {focus_area}.
    
    Questions:
    """
)

# Define a custom reference generation prompt
reference_prompt = Prompt(
    """
    You are an expert in {document_topic}. Answer the following question based on the provided document:
    
    Document:
    {document_content}
    
    Question:
    {input}
    
    Provide a {answer_style} answer that is accurate and comprehensive:
    """
)

# Create a test set generator with custom prompts
generator = TestSetGenerator(
    knowledge_base=knowledge_base,
    llm_client=client,
    input_generation_prompt=input_prompt.format(
        user_type="beginner",
        difficulty_level="easy to moderate",
        focus_area="understanding basic concepts"
    ),
    reference_generation_prompt=reference_prompt.format(
        document_topic="artificial intelligence",
        answer_style="clear and concise"
    )
)
```

### 4. Implement Custom Filters

Implement custom filters to ensure the generated inputs meet your specific requirements:

```python
class FilteredTestSetGenerator(TestSetGenerator):
    def filter_inputs(self, inputs):
        # Filter out inputs that are too short
        inputs = [i for i in inputs if len(i.input.split()) >= 5]
        
        # Filter out inputs that don't end with a question mark
        inputs = [i for i in inputs if i.input.strip().endswith("?")]
        
        # Filter out inputs that contain certain keywords
        inputs = [i for i in inputs if "keyword1" not in i.input.lower() and "keyword2" not in i.input.lower()]
        
        return inputs
```

### 5. Test and Refine

Test your customizations and refine them based on the results:

```python
# Generate a small test set
test_set = generator.generate(num_inputs=5)

# Print the generated inputs
for i, input_sample in enumerate(test_set.inputs):
    print(f"\nInput {i+1}: {input_sample.input}")
    print(f"Reference answer: {input_sample.reference_answer}")

# Refine your customizations based on the results
# For example, adjust the prompts, add more example inputs, or modify the filters
```

## Next Steps

Now that you know how to customize the test set generation process, you can:

- Learn about [example-guided generation](./example_guided_generation.md) for more control over the generation process
- Explore how to [work with knowledge bases](../knowledge_bases/working_with_knowledge_bases.md) for test set generation
- Discover how to use test sets for [RAG evaluation](../evaluation/rag_evaluation.md)
