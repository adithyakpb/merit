# Customization

This section covers how to customize various aspects of MERIT for your specific needs. MERIT is designed to be highly customizable, allowing you to adapt it to your unique requirements and workflows.

## In This Section

- [Overview](./01_overview.md): Introduction to customization in MERIT
- [Custom Knowledge Bases](../knowledge_bases/04_custom_knowledge_bases.md): How to create custom knowledge bases
- [Custom Prompts](./04_custom_prompts.md): How to define custom prompts for generation and evaluation
- [Custom Evaluators](./03_custom_evaluators.md): How to create custom evaluators
- [Custom Metrics](./02_custom_metrics.md): How to define custom evaluation metrics
- [Report Templates](./05_report_templates.md): How to create custom report templates

## Why Customize MERIT?

Customization allows you to:

- **Adapt to Your Domain**: Tailor MERIT to your specific domain or industry
- **Implement Specialized Logic**: Add domain-specific logic to evaluations
- **Integrate with Your Systems**: Connect MERIT to your existing infrastructure
- **Extend Functionality**: Add new features and capabilities
- **Optimize for Your Use Case**: Fine-tune MERIT for your specific requirements

## Key Customization Points

### Custom Knowledge Bases

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

### Custom Prompts

You can define custom prompts for various MERIT components, including test set generation, reference answer generation, and evaluation.

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

### Custom Evaluators

You can create custom evaluators to implement specialized evaluation logic for your AI systems.

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

### Custom Metrics

You can define custom metrics to measure specific aspects of your AI system's performance.

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

### Custom Report Templates

You can create custom report templates to format evaluation results according to your preferences.

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

## Next Steps

Start by reading the [Overview](./01_overview.md) to learn more about customization in MERIT. Then, depending on your needs, check out the specific guides for [Custom Knowledge Bases](../knowledge_bases/04_custom_knowledge_bases.md), [Custom Prompts](./04_custom_prompts.md), [Custom Evaluators](./03_custom_evaluators.md), [Custom Metrics](./02_custom_metrics.md), or [Report Templates](./05_report_templates.md).
