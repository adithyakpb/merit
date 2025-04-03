# Customization Overview

This guide provides an overview of customization in MERIT. MERIT is designed to be highly customizable, allowing you to adapt it to your specific needs and workflows.

## Why Customize MERIT?

Customization allows you to:

- **Adapt to Your Domain**: Tailor MERIT to your specific domain or industry
- **Implement Specialized Logic**: Add domain-specific logic to evaluations
- **Integrate with Your Systems**: Connect MERIT to your existing infrastructure
- **Extend Functionality**: Add new features and capabilities
- **Optimize for Your Use Case**: Fine-tune MERIT for your specific requirements

## Customization Points in MERIT

MERIT provides several key customization points:

### Knowledge Bases

You can create custom knowledge bases to store and retrieve documents for test set generation and evaluation. This allows you to integrate MERIT with your existing document storage systems.

```python
from merit.knowledge import KnowledgeBase

class CustomKnowledgeBase(KnowledgeBase):
    """Custom knowledge base implementation."""
    
    def __init__(self, connection_string, **kwargs):
        super().__init__(**kwargs)
        self.connection_string = connection_string
        self._connect()
    
    def _connect(self):
        # Connect to your custom document store
        pass
    
    def search(self, query, k=5):
        # Implement custom search logic
        pass
    
    def get_document(self, doc_id):
        # Retrieve a document from your store
        pass
```

### Prompts

You can define custom prompts for various MERIT components, including test set generation, reference answer generation, and evaluation. This allows you to control the instructions given to LLMs for different tasks.

```python
from merit.core.prompts import Prompt

# Define a custom prompt for generating test inputs
CUSTOM_INPUT_GENERATION_PROMPT = Prompt(
    """
    Generate {num_inputs} diverse and realistic questions about the following document:
    
    {document_content}
    
    The questions should be varied in complexity and cover different aspects of the document.
    
    Questions:
    """
)
```

### Evaluators

You can create custom evaluators to implement specialized evaluation logic for your AI systems. This allows you to define how your systems are evaluated and what metrics are applied.

```python
from merit.evaluation.evaluators.base import BaseEvaluator

class CustomEvaluator(BaseEvaluator):
    """Custom evaluator for specialized evaluation."""
    
    def __init__(self, custom_param, metrics=None):
        super().__init__(metrics)
        self.custom_param = custom_param
    
    def evaluate(self, model, dataset):
        # Implement custom evaluation logic
        pass
    
    def _evaluate_sample(self, sample, answer):
        # Implement sample-level evaluation
        pass
```

### Metrics

You can define custom metrics to measure specific aspects of your AI system's performance. This allows you to evaluate your systems based on criteria that are important for your use case.

```python
from merit.evaluation.metrics.base import BaseMetric

class CustomMetric(BaseMetric):
    """Custom metric for specialized evaluation."""
    
    def __init__(self, name="custom_metric", **kwargs):
        self.name = name
        self._kwargs = kwargs
    
    def __call__(self, input_sample, answer):
        # Implement metric calculation
        score = self._calculate_score(input_sample, answer)
        explanation = self._generate_explanation(input_sample, answer, score)
        
        return {
            self.name: score,
            f"{self.name}_explanation": explanation
        }
    
    def _calculate_score(self, input_sample, answer):
        # Calculate the metric score
        pass
    
    def _generate_explanation(self, input_sample, answer, score):
        # Generate an explanation for the score
        pass
```

### API Clients

You can create custom API clients to connect to specific AI APIs or services. This allows you to integrate MERIT with your preferred AI providers.

```python
from merit.api.client import AIAPIClient

class CustomAPIClient(AIAPIClient):
    """Custom client for a specific API."""
    
    def __init__(self, custom_param=None, **kwargs):
        super().__init__(**kwargs)
        self.custom_param = custom_param
        
    def _get_headers(self):
        headers = super()._get_headers()
        headers["Custom-Header"] = self.custom_param
        return headers
    
    def custom_method(self, data):
        # Implement custom API method
        pass
```

### Report Templates

You can create custom report templates to format evaluation results according to your preferences. This allows you to generate reports that meet your specific requirements.

```python
from merit.templates import ReportTemplate

class CustomReportTemplate(ReportTemplate):
    """Custom report template."""
    
    def generate(self, report):
        # Generate a custom report
        pass
    
    def to_html(self, report):
        # Generate an HTML report
        pass
    
    def to_markdown(self, report):
        # Generate a Markdown report
        pass
```

## Customization Workflow

Here's a typical workflow for customizing MERIT:

1. **Identify Customization Needs**: Determine what aspects of MERIT you need to customize for your use case
2. **Extend Base Classes**: Create custom classes that extend MERIT's base classes
3. **Implement Custom Logic**: Add your custom logic to the extended classes
4. **Test and Refine**: Test your customizations and refine them as needed
5. **Integrate with Your Workflow**: Integrate your customizations into your overall workflow

## Best Practices for Customization

When customizing MERIT, follow these best practices:

- **Extend, Don't Modify**: Extend MERIT's base classes rather than modifying them directly
- **Follow Interfaces**: Ensure your custom classes follow the same interfaces as the base classes
- **Document Your Customizations**: Document your custom classes and methods
- **Test Thoroughly**: Test your customizations thoroughly to ensure they work as expected
- **Share Your Customizations**: Consider sharing your customizations with the MERIT community

## Next Steps

Now that you understand the basics of customization in MERIT, you can:

- Learn how to create [custom knowledge bases](./custom_knowledgebases.md)
- Explore how to define [custom prompts](./custom_prompts.md)
- Discover how to create [custom evaluators](./custom_evaluators.md)
- Learn how to define [custom metrics](./custom_metrics.md)
- Understand how to create [custom report templates](./report_templates.md)
