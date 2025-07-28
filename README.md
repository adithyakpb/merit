# MERIT: Monitoring, Evaluation, Reporting, Inspection, Testing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.1.16-blue.svg)](https://pypi.org/project/merit-ai/)

A comprehensive framework for evaluating, monitoring, and testing AI systems, particularly those powered by Large Language Models (LLMs). MERIT provides tools for performance monitoring, evaluation metrics, RAG system testing, and comprehensive reporting.

## 🚀 Features

### 📊 **Monitoring & Observability**
- **Real-time LLM monitoring** with customizable metrics
- **Performance tracking** (latency, throughput, error rates)
- **Cost monitoring** and estimation
- **Usage analytics** and token volume tracking
- **Multi-backend storage** (SQLite, MongoDB, file-based)
- **Live dashboard** with interactive metrics

### 🧪 **Evaluation & Testing**
- **RAG system evaluation** with comprehensive metrics
- **LLM performance testing** with custom test sets
- **Automated evaluation** using LLM-based evaluators
- **Test set generation** for systematic testing
- **Multi-model evaluation** support

### 📈 **Metrics & Analytics**
- **Correctness, Faithfulness, Relevance** for RAG systems
- **Coherence and Fluency** metrics
- **Context Precision** evaluation
- **Custom metric development** framework
- **Performance benchmarking**

### 🔧 **Integration & APIs**
- **Simple 3-line integration** for existing applications
- **REST API** for remote monitoring
- **CLI tools** for configuration and execution
- **Multiple AI provider support** (OpenAI, Google, custom)

## 📦 Installation

### Basic Installation

```bash
pip install merit-ai
```

### Full Installation with All Dependencies

```bash
pip install merit-ai[all]
```

### Development Installation

```bash
git clone https://github.com/your-username/merit.git
cd merit
pip install -e .[dev]
```

## 🚀 Quick Start

### 1. Simple Integration (3 Lines!)

```python
from merit.monitoring.service import MonitoringService

# Initialize monitoring
monitor = MonitoringService()

# Log an interaction
monitor.log_simple_interaction({
    'user_message': 'Hello, how are you?',
    'llm_response': 'I am doing well, thank you!',
    'latency': 0.5,
    'model': 'gpt-3.5-turbo'
})
```

### 2. RAG System Evaluation

```python
from merit.evaluation.evaluators.rag import RAGEvaluator

# Initialize evaluator
evaluator = RAGEvaluator()

# Evaluate RAG response
results = evaluator.evaluate(
    query="What is machine learning?",
    response="Machine learning is a subset of AI...",
    context=["Document 1 content...", "Document 2 content..."]
)

print(f"Relevance: {results['relevance']}")
print(f"Faithfulness: {results['faithfulness']}")
```

### 3. CLI Usage

```bash
# Start evaluation with config file
merit start --config my_config.py

# Monitor your application
merit monitor --config monitoring_config.py
```

## 📚 Examples

### Basic Chat Application Integration

```python
from merit.monitoring.service import MonitoringService
from datetime import datetime

class ChatApp:
    def __init__(self):
        # Initialize MERIT monitoring
        self.monitor = MonitoringService()
    
    def process_message(self, user_message: str) -> str:
        start_time = datetime.now()
        
        # Your existing chat logic here
        response = self.llm_client.chat(user_message)
        
        end_time = datetime.now()
        
        # Log interaction with MERIT
        self.monitor.log_simple_interaction({
            'user_message': user_message,
            'llm_response': response,
            'latency': (end_time - start_time).total_seconds(),
            'model': 'gpt-3.5-turbo',
            'timestamp': end_time.isoformat()
        })
        
        return response
```

### Advanced RAG System with MERIT

```python
from merit.evaluation.evaluators.rag import RAGEvaluator
from merit.monitoring.service import MonitoringService

class RAGSystem:
    def __init__(self):
        self.evaluator = RAGEvaluator()
        self.monitor = MonitoringService()
    
    def query(self, user_question: str):
        # Retrieve relevant documents
        documents = self.retriever.search(user_question)
        
        # Generate response
        response = self.llm.generate(user_question, documents)
        
        # Evaluate with MERIT
        evaluation = self.evaluator.evaluate(
            query=user_question,
            response=response,
            context=[doc.content for doc in documents]
        )
        
        # Monitor performance
        self.monitor.log_simple_interaction({
            'query': user_question,
            'response': response,
            'evaluation_scores': evaluation,
            'num_documents': len(documents)
        })
        
        return response, evaluation
```

## 🏗️ Project Structure

```
merit/
├── api/                    # API clients (OpenAI, Google, etc.)
├── core/                   # Core models and utilities
├── evaluation/             # Evaluation framework
│   ├── evaluators/        # LLM and RAG evaluators
│   └── templates/         # Evaluation templates
├── knowledge/              # Knowledge base management
├── metrics/                # Metrics framework
│   ├── rag.py            # RAG-specific metrics
│   ├── llm_measured.py   # LLM-based metrics
│   └── monitoring.py     # Monitoring metrics
├── monitoring/             # Monitoring service
│   └── collectors/        # Data collectors
├── storage/               # Storage backends
├── templates/             # Dashboard and report templates
└── testset_generation/    # Test set generation tools
```

## 📊 Available Metrics

### RAG Metrics
- **Correctness**: Accuracy of generated responses
- **Faithfulness**: Adherence to source documents
- **Relevance**: Response relevance to query
- **Coherence**: Logical flow and consistency
- **Fluency**: Natural language quality
- **Context Precision**: Quality of retrieved context

### Monitoring Metrics
- **Latency**: Response time tracking
- **Throughput**: Requests per second
- **Error Rate**: Failure percentage
- **Cost**: Token usage and cost estimation
- **Usage**: Model and feature usage patterns

## 🔧 Configuration

### Basic Configuration File

```python
# merit_config.py
from merit.config.models import MeritMainConfig

config = MeritMainConfig(
    evaluation={
        "evaluator": "rag",
        "metrics": ["relevance", "faithfulness", "correctness"]
    },
    monitoring={
        "storage_type": "sqlite",
        "collection_interval": 60,
        "retention_days": 30
    }
)
```

### Advanced Configuration

```python
# advanced_config.py
config = MeritMainConfig(
    evaluation={
        "evaluator": "rag",
        "metrics": ["relevance", "faithfulness", "correctness"],
        "test_set": {
            "path": "test_questions.json",
            "size": 100
        }
    },
    monitoring={
        "storage_type": "mongodb",
        "storage_config": {
            "uri": "mongodb://localhost:27017",
            "database": "merit_metrics"
        },
        "metrics": ["latency", "cost", "error_rate"],
        "collection_interval": 30,
        "retention_days": 90
    },
    knowledge_base={
        "type": "vector_store",
        "path": "./knowledge_base"
    }
)
```

## 🎯 Use Cases

### 1. **Production LLM Monitoring**
Monitor your deployed LLM applications in real-time with performance metrics, cost tracking, and error monitoring.

### 2. **RAG System Development**
Evaluate and improve your RAG systems with comprehensive metrics and automated testing.

### 3. **Model Comparison**
Compare different models and configurations using standardized evaluation metrics.

### 4. **Quality Assurance**
Implement automated testing for LLM applications with custom test sets and evaluation criteria.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
git clone https://github.com/your-username/merit.git
cd merit
pip install -e .[dev]
pytest tests/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python practices and Pydantic for type safety
- Inspired by the need for comprehensive AI system evaluation
- Designed for simplicity and ease of integration

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/merit/issues)
- **Documentation**: [Full Documentation](https://merit.readthedocs.io)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/merit/discussions)

---

**MERIT**: Making AI systems more reliable, one evaluation at a time. 🚀