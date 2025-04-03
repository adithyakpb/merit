# MERIT: Monitoring, Evaluation, Reporting, Inspection, Testing

## Overview

MERIT is a comprehensive framework for evaluating and testing AI systems, particularly those powered by Large Language Models (LLMs). The name MERIT stands for:

- **M**onitoring: Track the performance of AI systems over time
- **E**valuation: Assess AI systems against defined metrics and benchmarks
- **R**eporting: Generate detailed reports on AI system performance
- **I**nspection: Analyze AI system behavior and outputs
- **T**esting: Create and run test sets to validate AI system capabilities

This package provides tools and utilities to help you implement rigorous evaluation and testing workflows for your AI applications.

## Current Implementation

The current version of MERIT implements several key components:

- **Test Set Generation**: Create comprehensive test sets for evaluating AI systems
- **API Client Integration**: Connect to various AI APIs with a unified interface
- **Evaluation Framework**: Evaluate AI systems using customizable metrics
- **Metrics Implementation**: Measure performance across various dimensions
- **Knowledge Base Integration**: Work with document collections for RAG evaluation

Future versions will expand on monitoring capabilities and reporting functionality.

## Key Features

### Test Set Generation

- Generate test sets from knowledge bases
- Create example-guided test sets based on user-provided examples
- Customize test set generation with various parameters
- Save and load test sets for reuse

### API Client Integration

- Connect to various AI APIs with a unified interface
- Built-in support for OpenAI and other providers
- Extensible client architecture for custom implementations
- Consistent response formats across different providers

### Evaluation Framework

- Evaluate RAG (Retrieval-Augmented Generation) systems
- Assess LLM outputs with customizable metrics
- Support for classification and other evaluation tasks
- Generate detailed evaluation reports

### Metrics Implementation

- RAG-specific metrics (correctness, faithfulness, relevance, etc.)
- Classification metrics for structured outputs
- LLM-based evaluation for qualitative assessment
- Extensible metric framework for custom implementations

## Documentation Structure

The documentation is organized into the following sections:

### [1. Getting Started](./getting_started/index.md)
Introduction to MERIT, installation instructions, key concepts, and quick start guides.

### [2. Test Set Generation](./testset_generation/index.md)
How to create and customize test sets for evaluating AI systems.

### [3. Knowledge Bases](./knowledge_bases/index.md)
How to create and use knowledge bases for test set generation and evaluation.

### [4. API Clients](./api_clients/index.md)
How to connect to AI APIs and create custom clients.

### [5. Evaluation](./evaluation/index.md)
How to evaluate AI systems using different metrics and frameworks.

### [6. Customization](./customization/index.md)
How to customize various aspects of MERIT for your specific needs.

### [7. Tutorials](./tutorials/index.md)
Step-by-step guides for common workflows and use cases.

## Getting Started

To get started with MERIT, check out the [Getting Started Guide](./getting_started/index.md) for installation instructions and a quick introduction to the package.

For a quick example of how to use MERIT, see the [Quick Start Guide](./getting_started/04_quick_start.md).

## Contributing

We welcome contributions to MERIT! If you'd like to contribute, please check out our [Contributing Guide](./contributing.md) for more information.

## License

MERIT is licensed under the [MIT License](../LICENSE).
