"""
Test Set Generation Examples

This file demonstrates various ways to generate test sets using MERIT.
"""

from merit.testset_generation import TestSetGenerator
from merit.knowledge import KnowledgeBase
from merit.core.models import Document
from merit.api.client import OpenAIClient
import os


def basic_testset_generation():
    """Basic test set generation from a knowledge base."""
    print("\n=== Basic Test Set Generation ===\n")
    
    # Create a knowledge base with sample documents
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
    print(f"Created knowledge base with {len(knowledge_base)} documents")
    
    # Create an API client for test set generation
    # Replace with your actual API key or use environment variables
    api_key = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
    client = OpenAIClient(api_key=api_key)
    
    # Create a test set generator
    generator = TestSetGenerator(
        knowledge_base=knowledge_base,
        llm_client=client,
        language="en",
        agent_description="A chatbot that answers questions about artificial intelligence and related topics."
    )
    
    # Generate a test set
    test_set = generator.generate(num_inputs=5)
    
    # Print the generated inputs
    print(f"Generated {len(test_set.inputs)} test inputs:")
    for i, input_sample in enumerate(test_set.inputs):
        print(f"\nInput {i+1}: {input_sample.input}")
        print(f"Reference answer: {input_sample.reference_answer[:100]}...")  # Print the first 100 characters
        print(f"Document ID: {input_sample.document.id}")
    
    # Save the test set for future use
    test_set.save("ai_test_set.json")
    print("\nTest set saved to ai_test_set.json")


def example_guided_generation():
    """Test set generation guided by example inputs."""
    print("\n=== Example-Guided Test Set Generation ===\n")
    
    # Create a knowledge base with sample documents
    documents = [
        Document(
            content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
            metadata={"source": "AI Overview", "topic": "AI Basics"},
            id="doc1"
        ),
        Document(
            content="Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
            metadata={"source": "Machine Learning", "topic": "ML Basics"},
            id="doc2"
        ),
        Document(
            content="Deep Learning is a subset of machine learning where artificial neural networks, algorithms inspired by the human brain, learn from large amounts of data.",
            metadata={"source": "Deep Learning", "topic": "DL Basics"},
            id="doc3"
        )
    ]
    
    knowledge_base = KnowledgeBase(documents=documents)
    
    # Create an API client for test set generation
    api_key = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
    client = OpenAIClient(api_key=api_key)
    
    # Define example inputs to guide the generation
    example_inputs = [
        "What is the difference between AI and machine learning?",
        "How does deep learning relate to machine learning?",
        "What are some real-world applications of artificial intelligence?",
        "What ethical concerns are associated with AI development?",
        "How has AI evolved over the past decade?"
    ]
    
    # Create a test set generator
    generator = TestSetGenerator(
        knowledge_base=knowledge_base,
        llm_client=client,
        language="en",
        agent_description="A chatbot that answers questions about artificial intelligence and related topics."
    )
    
    # Generate a test set with example inputs
    test_set = generator.generate(
        num_inputs=5,
        example_inputs=example_inputs
    )
    
    # Print the generated inputs
    print(f"Generated {len(test_set.inputs)} test inputs based on examples:")
    for i, input_sample in enumerate(test_set.inputs):
        print(f"\nInput {i+1}: {input_sample.input}")
        print(f"Reference answer: {input_sample.reference_answer[:100]}...")  # Print the first 100 characters
    
    # Save the test set for future use
    test_set.save("ai_example_guided_test_set.json")
    print("\nExample-guided test set saved to ai_example_guided_test_set.json")


def domain_specific_generation():
    """Generate test sets for specific domains."""
    print("\n=== Domain-Specific Test Set Generation ===\n")
    
    # Create a knowledge base with medical documents
    medical_documents = [
        Document(
            content="Diabetes is a chronic condition that affects how your body turns food into energy. Most of the food you eat is broken down into sugar (glucose) and released into your bloodstream. When your blood sugar goes up, it signals your pancreas to release insulin.",
            metadata={"source": "Diabetes Overview", "topic": "Medical Conditions"},
            id="med1"
        ),
        Document(
            content="Hypertension, also known as high blood pressure, is a long-term medical condition in which the blood pressure in the arteries is persistently elevated. High blood pressure typically does not cause symptoms but long-term high blood pressure is a major risk factor for stroke, coronary artery disease, heart failure, atrial fibrillation, and other conditions.",
            metadata={"source": "Hypertension Overview", "topic": "Medical Conditions"},
            id="med2"
        ),
        Document(
            content="Vaccines are biological preparations that provide active acquired immunity to a particular infectious disease. A vaccine typically contains an agent that resembles a disease-causing microorganism and is often made from weakened or killed forms of the microbe, its toxins, or one of its surface proteins.",
            metadata={"source": "Vaccines Overview", "topic": "Medical Treatments"},
            id="med3"
        )
    ]
    
    medical_kb = KnowledgeBase(documents=medical_documents)
    print(f"Created medical knowledge base with {len(medical_kb)} documents")
    
    # Create an API client for test set generation
    api_key = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
    client = OpenAIClient(api_key=api_key)
    
    # Create a test set generator for medical domain
    medical_generator = TestSetGenerator(
        knowledge_base=medical_kb,
        llm_client=client,
        language="en",
        agent_description="A medical assistant that answers health-related questions based on medical literature."
    )
    
    # Define medical example inputs
    medical_examples = [
        "What are the symptoms of diabetes?",
        "How is hypertension diagnosed?",
        "What are the side effects of vaccines?",
        "What lifestyle changes can help manage diabetes?",
        "How does high blood pressure affect heart health?"
    ]
    
    # Generate a medical test set
    medical_test_set = medical_generator.generate(
        num_inputs=5,
        example_inputs=medical_examples
    )
    
    # Print the generated medical inputs
    print(f"Generated {len(medical_test_set.inputs)} medical test inputs:")
    for i, input_sample in enumerate(medical_test_set.inputs):
        print(f"\nInput {i+1}: {input_sample.input}")
        print(f"Reference answer: {input_sample.reference_answer[:100]}...")  # Print the first 100 characters
    
    # Save the medical test set
    medical_test_set.save("medical_test_set.json")
    print("\nMedical test set saved to medical_test_set.json")


def custom_generation_parameters():
    """Test set generation with custom parameters."""
    print("\n=== Custom Generation Parameters ===\n")
    
    # Create a knowledge base with sample documents
    documents = [
        Document(
            content="Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions.",
            metadata={"source": "AI Overview", "topic": "AI Basics"},
            id="doc1"
        ),
        Document(
            content="Machine Learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
            metadata={"source": "Machine Learning", "topic": "ML Basics"},
            id="doc2"
        )
    ]
    
    knowledge_base = KnowledgeBase(documents=documents)
    
    # Create an API client for test set generation
    api_key = os.environ.get("OPENAI_API_KEY", "your-openai-api-key")
    client = OpenAIClient(api_key=api_key)
    
    # Create a test set generator with custom parameters
    generator = TestSetGenerator(
        knowledge_base=knowledge_base,
        llm_client=client,
        language="en",
        agent_description="A chatbot that answers questions about artificial intelligence and related topics.",
        temperature=0.8,  # Higher temperature for more diverse outputs
        max_tokens=150    # Limit token generation
    )
    
    # Generate a test set with custom parameters
    test_set = generator.generate(
        num_inputs=3,
        difficulty="advanced",  # Generate advanced questions
        question_types=["conceptual", "analytical"],  # Focus on conceptual and analytical questions
        max_retries=2  # Maximum number of retries for failed generations
    )
    
    # Print the generated inputs
    print(f"Generated {len(test_set.inputs)} test inputs with custom parameters:")
    for i, input_sample in enumerate(test_set.inputs):
        print(f"\nInput {i+1}: {input_sample.input}")
        print(f"Reference answer: {input_sample.reference_answer[:100]}...")  # Print the first 100 characters
    
    # Save the test set
    test_set.save("custom_params_test_set.json")
    print("\nCustom parameters test set saved to custom_params_test_set.json")


def load_and_analyze_test_set():
    """Load and analyze a previously generated test set."""
    print("\n=== Load and Analyze Test Set ===\n")
    
    try:
        # Load a previously generated test set
        from merit.core.models import TestSet
        test_set = TestSet.load("ai_test_set.json")
        
        print(f"Loaded test set with {len(test_set.inputs)} inputs")
        
        # Analyze the test set
        print("\nTest Set Analysis:")
        
        # Count question types
        question_types = {
            "what": 0,
            "how": 0,
            "why": 0,
            "when": 0,
            "where": 0,
            "which": 0,
            "who": 0,
            "other": 0
        }
        
        for input_sample in test_set.inputs:
            question = input_sample.input.lower()
            if question.startswith("what"):
                question_types["what"] += 1
            elif question.startswith("how"):
                question_types["how"] += 1
            elif question.startswith("why"):
                question_types["why"] += 1
            elif question.startswith("when"):
                question_types["when"] += 1
            elif question.startswith("where"):
                question_types["where"] += 1
            elif question.startswith("which"):
                question_types["which"] += 1
            elif question.startswith("who"):
                question_types["who"] += 1
            else:
                question_types["other"] += 1
        
        print("Question types distribution:")
        for qtype, count in question_types.items():
            if count > 0:
                print(f"  {qtype}: {count} ({count/len(test_set.inputs)*100:.1f}%)")
        
        # Analyze answer lengths
        answer_lengths = [len(input_sample.reference_answer.split()) for input_sample in test_set.inputs]
        avg_length = sum(answer_lengths) / len(answer_lengths)
        min_length = min(answer_lengths)
        max_length = max(answer_lengths)
        
        print(f"\nReference answer statistics:")
        print(f"  Average length: {avg_length:.1f} words")
        print(f"  Minimum length: {min_length} words")
        print(f"  Maximum length: {max_length} words")
        
        # Document distribution
        doc_counts = {}
        for input_sample in test_set.inputs:
            doc_id = input_sample.document.id
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        print("\nDocument distribution:")
        for doc_id, count in doc_counts.items():
            print(f"  {doc_id}: {count} ({count/len(test_set.inputs)*100:.1f}%)")
    
    except FileNotFoundError:
        print("Test set file not found. Run basic_testset_generation() first to create the file.")
    except Exception as e:
        print(f"Error analyzing test set: {str(e)}")


if __name__ == "__main__":
    print("Test Set Generation Examples")
    print("===========================")
    print("Note: Replace 'your-openai-api-key' with your actual OpenAI API key to run these examples.")
    print("You can also set the OPENAI_API_KEY environment variable instead.")
    
    # Run the examples
    print("\nRunning basic test set generation example...")
    basic_testset_generation()
    
    print("\nRunning example-guided generation example...")
    example_guided_generation()
    
    print("\nRunning domain-specific generation example...")
    domain_specific_generation()
    
    print("\nRunning custom generation parameters example...")
    custom_generation_parameters()
    
    print("\nRunning load and analyze test set example...")
    load_and_analyze_test_set()
