# MERIT Testing Framework

This directory contains the testing framework for the MERIT system. The tests are organized into a structured hierarchy to ensure comprehensive coverage and maintainability.

## Directory Structure

```
tests/
├── conftest.py              # Shared fixtures and configuration
├── fixtures/                # Test data files
├── pytest.ini               # pytest configuration
├── unit/                    # Unit tests for individual components
│   ├── core/                # Tests for core functionality
│   ├── api/                 # Tests for API functionality
│   ├── knowledge/           # Tests for knowledge base functionality
│   ├── testset_generation/  # Tests for test set generation
│   └── evaluation/          # Tests for evaluation functionality
└── integration/             # Integration tests across components
```

## Running Tests

### Basic Test Execution

To run all tests:

```bash
pytest
```

To run a specific test module:

```bash
pytest tests/unit/core/test_errors.py
```

To run a specific test class:

```bash
pytest tests/unit/core/test_errors.py::TestErrorHierarchy
```

To run a specific test:

```bash
pytest tests/unit/core/test_errors.py::TestErrorHierarchy::test_base_error_inheritance
```

### Running Tests by Category

We use pytest markers to categorize tests:

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run performance tests
pytest -m performance
```

## Testing Different Environments

The testing framework includes support for multiple simulated environments using the `test_environment` fixture:

```python
@pytest.mark.parametrize("test_environment", 
                        [TestEnvironment.STANDARD, TestEnvironment.ERROR_PRONE], 
                        indirect=True)
def test_something(test_environment):
    # This test will run twice:
    # 1. With standard settings
    # 2. With error-prone settings
    ...
```

Available environments:
- `STANDARD`: Normal operating conditions
- `MINIMAL`: Minimal configuration (e.g., no caching)
- `PERFORMANCE`: Optimized for performance testing
- `ERROR_PRONE`: Deliberately constrained to trigger error conditions
- `CUSTOM`: For user-defined test environments

## Testing Custom Implementations

MERIT is designed to be extensible, and the testing framework supports testing custom implementations:

### Testing Custom Metrics

```python
def test_custom_metric(custom_metric_loader):
    # Load your custom metric from your package
    CustomMetric = custom_metric_loader("your_package.metrics", "YourCustomMetric")
    
    # Initialize and test it
    metric = CustomMetric()
    result = metric.evaluate("test prediction", "test reference")
    assert 0 <= result.score <= 1
```

### Testing Custom Evaluators

```python
def test_custom_evaluator(custom_evaluator_loader, sample_knowledge_base, sample_documents):
    # Load your custom evaluator
    CustomEvaluator = custom_evaluator_loader("your_package.evaluators", "YourCustomEvaluator")
    
    # Initialize and test it
    evaluator = CustomEvaluator(knowledge_base=sample_knowledge_base)
    result = evaluator.evaluate(lambda q: "test response")
    assert hasattr(result, "metrics")
```

## Mocking Dependencies

The test framework provides several mock objects:

### Mock LLM Client

The `mock_llm_client` fixture provides a simulated LLM client:

```python
def test_with_mock_llm(mock_llm_client):
    # Configure pre-defined responses
    mock_llm_client.responses = {
        "What is the capital of France?": "Paris is the capital of France."
    }
    
    # Test your code that uses an LLM
    result = your_function_that_uses_llm(mock_llm_client)
    
    # Verify it was called correctly
    assert ("generate_text", "What is the capital of France?", {}) in mock_llm_client.call_history
```

## Writing New Tests

When writing new tests:

1. Place unit tests in the appropriate module directory under `tests/unit/`
2. Place integration tests in `tests/integration/`
3. Use descriptive test names (test_what_is_being_tested)
4. Use appropriate fixtures from `conftest.py`
5. Add docstrings to test classes and methods
6. Mark tests appropriately (e.g., `@pytest.mark.unit`)
7. For slow tests, consider adding `@pytest.mark.slow`

## Best Practices

- Ensure tests are independent and don't rely on state from other tests
- Mock external dependencies when appropriate
- Use parametrized tests for testing multiple input variations
- Aim for high code coverage, especially in critical components
- Include both positive and negative test cases (error conditions)
- For custom implementations, provide example test files
