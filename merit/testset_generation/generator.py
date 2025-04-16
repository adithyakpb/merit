"""
MERIT RAG Test Set Generator

This module provides a class-based approach for generating test sets for RAG evaluation.
It encapsulates the functionality for test set generation in an object-oriented design.

The TestSetGenerator class provides a flexible and maintainable API for generating
test sets for RAG evaluation.
"""
import numpy as np
import os
import concurrent.futures
import threading
from functools import partial
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

from ..core.models import TestSet, TestItem, ExampleItem, ExampleSet, Document
from ..knowledge import knowledgebase
from ..core.utils import batch_iterator, parse_json
from ..core.logging import get_logger
from ..api.client import AIAPIClient
from .prompts import (
    TEST_INPUT_GENERATION_PROMPT, 
    REFERENCE_ANSWER_GENERATION_PROMPT,
    ADAPTIVE_TEST_INPUT_GENERATION_PROMPT
)
from .analysis import analyze_examples
logger = get_logger(__name__)

DEFAULT_NUM_ITEMS = 50
DEFAULT_ITEMS_PER_DOCUMENT = 3
DEFAULT_BATCH_SIZE = 32

DISTRIBUTION_RANDOM = "random"
DISTRIBUTION_REPRESENTATIVE = "representative" 
DISTRIBUTION_PER_DOCUMENT = "per_document"

class TestSetGenerator:
    """
    A class for generating test sets for RAG evaluation.
    
    This class encapsulates the functionality for generating test sets,
    including both standard generation and example-guided generation.
    """
    
    def __init__(
        self,
        knowledge_base: knowledgebase.KnowledgeBase,
        language: str = "en",
        agent_description: str = "A chatbot that answers inputs based on a knowledge base.",
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """
        Initialize the TestSetGenerator.
        
        Args:
            knowledge_base: The knowledge base to generate inputs from.
            language: The language to generate inputs in.
            agent_description: A description of the agent being evaluated.
            batch_size: The batch size to use for input generation.
        """
        self.knowledge_base = knowledge_base
        self.language = language
        self.agent_description = agent_description
        self.batch_size = batch_size
    
    def generate(
        self,
        num_items: int = DEFAULT_NUM_ITEMS,
        example_items: Optional[Union[str, List[Dict[str, Any]], List[str], Dict[str, Any], ExampleItem, ExampleSet]] = None,
        similarity_threshold: float = 0.85,
        skip_relevance_check: bool = False,
        distribution_strategy: str = DISTRIBUTION_RANDOM, 
        items_per_document: int = DEFAULT_ITEMS_PER_DOCUMENT,
    ) -> TestSet:
        """
        Generate a test set for RAG evaluation.
        
        Args:
            num_items: The number of TestItems to generate.
            example_items: Optional example items to guide generation.
                Can be:
                - An ExampleSet object
                - An ExampleItem object
                - A file path (string) to a JSON file containing example inputs
                - A list of strings (inputs)
                - A list of dictionaries (structured inputs)
                - A dictionary with a "inputs" key
            similarity_threshold: Threshold for similarity detection (0.0-1.0).
            skip_relevance_check: Whether to skip document relevance check during generation.
                If True, all documents will be considered relevant for all inputs.
            distribution_strategy: Strategy for distributing inputs across the knowledge base:
                - "random": Randomly samples documents from the knowledge base (default).
                - "representative": Distributes inputs proportionally to represent the knowledge base structure.
                - "per_document": Generates a fixed number of inputs per document.
            items_per_document: Number of items to generate per document. 
                Only used when distribution_strategy is "per_document".
                
        Returns:
            TestSet: The generated test set.
        """
        logger.info(f"Generating test set with {num_items} items in {self.language}")
        example_analysis_result = None
        example_set = None
        
        # Process example items if provided
        if example_items:
            #TODO validate that the exampleset has been reduced to it's representative state 
            if not isinstance(example_items, ExampleSet):
                example_set = ExampleSet(inputs=example_items)
            else:
                example_set = example_items

            if example_set.length() > 0:
                logger.info(f"Using {example_set.length()} example items")
            
                options = {
                    "language": self.language,
                    "agent_description": self.agent_description,
                    "batch_size": self.batch_size,
                    "similarity_threshold": similarity_threshold,
                    "skip_relevance_check": skip_relevance_check,
                    "distribution_strategy": distribution_strategy,
                    "items_per_document": items_per_document,
                }
            
                # Directly analyze example items
                logger.info("Analyzing example items")
                # Call analyze_examples with the ExampleSet object directly
                example_analysis_result = analyze_examples(
                    example_set,  # Pass the ExampleSet object directly
                    ai_client=self.knowledge_base._client,
                    use_llm=True,
                    analysis_type="all"
                )
        
        # Sample documents with input allocation based on the distribution strategy
        logger.info(f"Using {distribution_strategy} distribution strategy to generate {num_items} items")
        doc_input_pairs = self.knowledge_base.sample_documents_with_input_allocation(
            n=num_items,
            strategy=distribution_strategy,
            items_per_document=items_per_document
        )
        
        # Generate test items
        test_items = []
        
        for document, items_to_generate in doc_input_pairs:
            logger.info(f"Generating {items_to_generate} items for document {document.id}")
            
            # Generate the specified number of items for this document
            items_for_doc = self.generate_items(
                document=document,
                ai_client=self.knowledge_base._client,
                example_set=example_set if example_items else None,
                num_items=items_to_generate
            )
            
            # Add the generated test items to our collection
            test_items.extend(items_for_doc)
                
            # Stop if we've reached the requested number of items
            if len(test_items) >= num_items:
                break
        
        # Create and return test set
        test_set = TestSet(
            inputs=test_items,
            metadata={
                "language": self.language,
                "agent_description": self.agent_description,
                "num_items": len(test_items),
                "num_documents": len(set(item.document.id for item in test_items if item.document)),
                "num_topics": len(set(item.metadata.get("topic_id") for item in test_items if item.metadata.get("topic_id") is not None)),
                "source": "example_guided" if example_analysis_result and example_items else "standard",
                "distribution_strategy": distribution_strategy
            }
        )
        
        logger.info(f"Generated test set with {len(test_set.inputs)} items using {distribution_strategy} distribution")
        return test_set
   
    
    def generate_items(
        self,
        document: Document,
        ai_client: AIAPIClient,
        example_set: Optional[ExampleSet] = None,
        num_items: int = DEFAULT_NUM_ITEMS
    ) -> List[TestItem]:
        """
        Generate test items for a given document, with or without a seed ExampleSet
        
        This method generates both the test inputs and reference answers for each test item.
        
        Args:
            document: The document to generate items for
            ai_client: The AI client to use for generation
            example_set: Optional example set to guide generation
            num_items: The number of items to generate
            
        Returns:
            List[TestItem]: List of TestItem objects with inputs and reference answers
        """
        logger.info(f"Generating {num_items} items for document {document.id}")
        
        # Import TestItem and Document here to ensure they're in scope
        from ..core.models import TestItem, TestSet
        
        # Calculate number of documents to sample
        num_docs = min(len(self.knowledge_base), (num_items + DEFAULT_ITEMS_PER_DOCUMENT - 1) // DEFAULT_ITEMS_PER_DOCUMENT)
        
        # Sample documents
        documents = self.knowledge_base.get_random_documents(num_docs)
        
        # Generate items for each document in parallel
        all_test_inputs = []
        if example_set:
            logger.info(f"Generating items for document {document.id} using example set")
            
            # Get analysis results for formatting prompt
            # Call analyze_examples with the ExampleSet object directly
            analysis = analyze_examples(
                example_set,  # Pass the ExampleSet object directly
                ai_client=ai_client,
                use_llm=True,
                analysis_type="all"
            )
            
            # Create the prompt
            prompt = ADAPTIVE_TEST_INPUT_GENERATION_PROMPT.safe_format(
                document_content=document.content,
                example_section=example_set,
                style_guidance=analysis,
                num_items=num_items
            )
            
            # Generate items using the AI client
            logger.info("Generating items using adaptive prompt")
            response = ai_client.generate_text(prompt)
            
            # Parse the response to extract inputs
            try:
                response_json = parse_json(response)
                test_inputs = response_json.get("test_inputs", [])
                
                # Convert items to the right format
                if isinstance(test_inputs, list):
                    for test_input_instance in test_inputs:
                        if isinstance(test_input_instance, dict) and "test_input" in test_input_instance:
                            all_test_inputs.append((document, test_input_instance["test_input"]))
                        elif isinstance(test_input_instance, dict) and "input" in test_input_instance:
                            all_test_inputs.append((document, test_input_instance["input"]))
                        elif isinstance(test_input_instance, str):
                            all_test_inputs.append((document, test_input_instance))
                elif isinstance(test_inputs, dict):
                    for test_input_instance in test_inputs.values():
                        if test_input_instance and isinstance(test_input_instance, str):
                            all_test_inputs.append((document, test_input_instance))
                        if test_input_instance in test_inputs.values() and isinstance(test_input_instance, list) and isinstance(test_input_instance[0], str):
                            all_test_inputs.append((document, test_input_instance[0]))
            except Exception as e:
                logger.error(f"Failed to parse generated test input for document - {document.id} due to error: {str(e)}")
                # Fallback: try to extract any string that looks like an item
        else:
            # Use standard input generation without examples
            try:
                # Format the prompt for standard generation
                prompt = TEST_INPUT_GENERATION_PROMPT.safe_format(
                    document_content=document.content,
                    system_description=self.agent_description,
                    num_items=num_items,
                    language=self.language
                )
                
                # Generate items using the AI client
                logger.info(f"Generating {num_items} standard items for document {document.id}")
                response = ai_client.generate_text(prompt)
                
                # Parse the response to extract inputs
                try:
                    response_json = parse_json(response)
                    test_inputs = response_json.get("test_inputs", [])
                    
                    # Convert items to the right format
                    if isinstance(test_inputs, list):
                        for test_input_instance in test_inputs:
                            if isinstance(test_input_instance, dict) and "test_input" in test_input_instance:
                                all_test_inputs.append((document, test_input_instance["test_input"]))
                            elif isinstance(test_input_instance, dict) and "input" in test_input_instance:
                                all_test_inputs.append((document, test_input_instance["input"]))
                            elif isinstance(test_input_instance, str):
                                all_test_inputs.append((document, test_input_instance))
                    elif isinstance(test_inputs, dict):
                        for test_input_instance in test_inputs.values():
                            if test_input_instance and isinstance(test_input_instance, str):
                                all_test_inputs.append((document, test_input_instance))
                            if test_input_instance in test_inputs.values() and isinstance(test_input_instance, list) and isinstance(test_input_instance[0], str):
                                all_test_inputs.append((document, test_input_instance[0]))
                except Exception as e:
                    logger.error(f"Failed to parse generated items: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to generate items for document {document.id}: {str(e)}")
        
        # Limit to the requested number of items
        all_test_inputs = all_test_inputs[:num_items]
        
        # Generate test items with reference answers
        test_items = []
        for document, test_input in all_test_inputs:
            try:
                # Format the prompt for reference answer generation
                if example_set:
                    # If we have an example set, include style guidance
                    prompt = REFERENCE_ANSWER_GENERATION_PROMPT.safe_format(
                        document_content=document.content,
                        test_input=test_input,
                        language=self.language,
                        style_guidance=analysis
                    )
                else:
                    # If no example set, don't include style guidance
                    prompt = REFERENCE_ANSWER_GENERATION_PROMPT.safe_format(
                        document_content=document.content,
                        test_input=test_input,
                        language=self.language
                    )
                
                # Generate reference answer
                logger.info(f"Generating reference answer for input: {test_input[:50]}...")
                reference_answer = ai_client.generate_text(prompt)
                
                # Create test item
                test_item = TestItem(
                    input=test_input,
                    reference_answer=reference_answer,
                    document=document,
                    metadata={
                        "language": self.language,
                        "topic_id": document.topic_id if hasattr(document, "topic_id") else None,
                        "topic_name": self.knowledge_base.topics.get(document.topic_id, "Unknown") if hasattr(document, "topic_id") else "Unknown",
                        "example_guided": example_set is not None
                    }
                )
                
                test_items.append(test_item)
            except Exception as e:
                logger.error(f"Failed to generate reference answer for input '{test_input}': {str(e)}")
        
        return test_items



# TODO multi document testitem generation
# TODO support for different input-response types: negative response, confusing input
