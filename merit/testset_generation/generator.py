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
from ..testset_generation.prompts import (
    INPUT_GENERATION_PROMPT, 
    REFERENCE_ANSWER_PROMPT,
    STYLE_ANALYSIS_PROMPT,
    INPUT_STYLE_ANALYSIS_PROMPT,
    ADAPTIVE_INPUT_GENERATION_PROMPT
)
from .analysis import analyze_examples, _create_sub_clusters

logger = get_logger(__name__)

DEFAULT_NUM_INPUTS = 50
DEFAULT_INPUTS_PER_DOCUMENT = 3
DEFAULT_BATCH_SIZE = 32

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
        num_inputs: int = DEFAULT_NUM_INPUTS,
        example_inputs: Optional[Union[str, List[Dict[str, Any]], List[str], Dict[str, Any], ExampleItem, ExampleSet]] = None,
        remove_similar_examples: bool = False,
        similarity_threshold: float = 0.85,
        skip_relevance_check: bool = False,
    ) -> TestSet:
        """
        Generate a test set for RAG evaluation.
        
        Args:
            num_inputs: The number of inputs to generate.
            example_inputs: Optional example inputs to guide generation.
                Can be:
                - An ExampleSet object
                - An ExampleItem object
                - A file path (string) to a JSON file containing example inputs
                - A list of strings (inputs)
                - A list of dictionaries (structured inputs)
                - A dictionary with a "inputs" key
            remove_similar_examples: Whether to remove similar example inputs.
            similarity_threshold: Threshold for similarity detection (0.0-1.0).
            skip_relevance_check: Whether to skip document relevance check during generation.
                If True, all documents will be considered relevant for all inputs.
            
        Returns:
            TestSet: The generated test set.
        """
        logger.info(f"Generating test set with {num_inputs} inputs in {self.language}")
        
        # Generate test set based on whether we have example inputs
        if example_inputs:
            if not isinstance(example_inputs, ExampleSet):
                example_set = ExampleSet(inputs=example_inputs)
            else:
                example_set = example_inputs

            if len(example_set.inputs) > 0:
                logger.info(f"Using {len(example_set.inputs)} example inputs")
            
            # Set up options
            options = {
                "language": self.language,
                "agent_description": self.agent_description,
                "batch_size": self.batch_size,
                "remove_similar": remove_similar_examples,
                "similarity_threshold": similarity_threshold,
                "skip_relevance_check": skip_relevance_check,
            }
            
            # Process example inputs
            logger.info("Processing example inputs")
            result = self._process_example_inputs(
                example_inputs=example_set,
                options=options,
            )
            
            if "test_set" in result:
                example_test_set = result["test_set"]
                
                # If we don't have enough inputs, generate more
                remaining_inputs = num_inputs - len(example_test_set.inputs)
                if remaining_inputs > 0:
                    logger.info(f"Generated {len(example_test_set.inputs)} inputs from examples, generating {remaining_inputs} more")
                    
                    # Generate additional inputs using standard approach
                    # Check if we have a document in the metadata or in the first input
                    document = example_test_set.metadata.get("document")
                    if document is None and example_test_set.inputs:
                        document = example_test_set.inputs[0].document
                    
                    if document is not None:
                        additional_inputs = generate_document_specific_test_inputs(
                            document=document,
                            client=self.knowledge_base._client,
                            example_inputs=[inp.input for inp in example_test_set.inputs],
                            input_patterns={},
                            num_inputs=remaining_inputs,
                            language=self.language,
                        )
                    else:
                        # If no document is available, generate standard inputs
                        logger.warning("No document found in test set, generating standard inputs")
                        additional_inputs = []
                    
                    # Create test inputs from additional inputs
                    additional_test_inputs = []
                    for input_text in additional_inputs:
                        doc = example_test_set.inputs[0].document if example_test_set.inputs else None
                        if doc:
                            reference_answer = generate_reference_answer(
                                document=doc,
                                input=input_text,
                                client=self.knowledge_base._client,
                                language=self.language,
                            )
                            
                            test_input = TestItem(
                                input=input_text,
                                reference_answer=reference_answer,
                                document=doc,
                                metadata={
                                    "source": "additional_generation",
                                    "language": self.language,
                                }
                            )
                            additional_test_inputs.append(test_input)
                    
                    # Combine inputs
                    combined_inputs = example_test_set.inputs + additional_test_inputs
                    
                    # Create combined metadata
                    combined_metadata = example_test_set.metadata.copy()
                    combined_metadata.update({
                        "num_inputs": len(combined_inputs),
                        "num_example_inputs": len(example_test_set.inputs),
                        "num_generated_inputs": len(additional_test_inputs),
                        "source": "combined",
                    })
                    
                    # Create combined test set
                    combined_test_set = TestSet(
                        inputs=combined_inputs,
                        metadata=combined_metadata,
                    )
                    
                    return combined_test_set
                
                return example_test_set
        
        # Standard input generation (no examples or processing failed)
        logger.info(f"Generating standard test set with {num_inputs} inputs")
        return self._generate_standard_inputs(num_inputs=num_inputs)
    
    def _generate_standard_inputs(
        self,
        num_inputs: int = DEFAULT_NUM_INPUTS,
    ) -> TestSet:
        """
        Generate a test set using the standard approach (without example inputs).
        
        Args:
            num_inputs: The number of inputs to generate.
            
        Returns:
            TestSet: The generated test set.
        """
        logger.info(f"Generating {num_inputs} standard inputs")
        
        # Import TestItem and Document here to ensure they're in scope
        from ..core.models import TestItem, Document, TestSet
        
        # Calculate number of documents to sample
        num_docs = min(len(self.knowledge_base), (num_inputs + DEFAULT_INPUTS_PER_DOCUMENT - 1) // DEFAULT_INPUTS_PER_DOCUMENT)
        
        # Sample documents
        documents = self.knowledge_base.get_random_documents(num_docs)
        
        # Generate inputs for each document in parallel
        all_inputs = []
        
        def process_document(doc):
            """Process a single document to generate inputs."""
            try:
                inputs = generate_inputs_for_document(
                    document=doc,
                    client=self.knowledge_base._client,
                    num_inputs=DEFAULT_INPUTS_PER_DOCUMENT,
                    language=self.language,
                )
                return [(doc, input_text) for input_text in inputs]
            except Exception as e:
                logger.error(f"Failed to generate inputs for document {doc.id}: {str(e)}")
                return []
        
        # Process documents in parallel batches
        total_batches = (len(documents) + self.batch_size - 1) // self.batch_size
        
        for batch_idx, doc_batch in enumerate(batch_iterator(documents, self.batch_size)):
            logger.info(f"Processing batch {batch_idx+1}/{total_batches}")
            
            # Process batch in parallel
            max_workers = min(os.cpu_count() * 2 or 4, len(doc_batch))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                batch_results = list(executor.map(process_document, doc_batch))
                for doc_inputs in batch_results:
                    all_inputs.extend(doc_inputs)
        
        # Shuffle inputs
        np.random.shuffle(all_inputs)
        
        # Limit to requested number of inputs
        all_inputs = all_inputs[:num_inputs]
        
        # Generate reference answers in parallel
        def generate_reference_for_pair(idx_doc_input):
            """Generate reference answer for a document-input pair."""
            idx, (doc, input_text) = idx_doc_input
            try:
                logger.info(f"Generating answer {idx+1}/{len(all_inputs)}")
                reference_answer = generate_reference_answer(
                    document=doc,
                    input=input_text,
                    client=self.knowledge_base._client,
                    language=self.language,
                )
                
                return TestItem(
                    input=input_text,
                    reference_answer=reference_answer,
                    document=doc,
                    metadata={
                        "language": self.language,
                        "topic_id": doc.topic_id,
                        "topic_name": self.knowledge_base.topics.get(doc.topic_id, "Unknown"),
                    },
                )
            except Exception as e:
                logger.error(f"Failed to generate reference answer for input '{input_text}': {str(e)}")
                return None
        
        # Process in parallel with a reasonable number of workers
        input_samples = []
        
        # Only process if we have inputs
        if all_inputs:
            max_workers = min(os.cpu_count() * 2 or 4, len(all_inputs))

            # Add indices to inputs for logging
            indexed_inputs = [(i, pair) for i, pair in enumerate(all_inputs)]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for sample in executor.map(generate_reference_for_pair, indexed_inputs):
                    if sample is not None:
                        input_samples.append(sample)
        
        # Create test set
        test_set = TestSet(
            inputs=input_samples,
            metadata={
                "language": self.language,
                "agent_description": self.agent_description,
                "num_inputs": len(input_samples),
                "num_documents": len(documents),
                "num_topics": len(set(doc.topic_id for doc in documents if doc.topic_id is not None)),
                "source": "standard",
            },
        )
        
        logger.info(f"Generated test set with {len(test_set.inputs)} inputs")
        return test_set
    
    def _process_example_inputs(
        self,
        example_inputs: ExampleSet,
        options: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Process example inputs based on their type.
        
        Args:
            example_inputs: The example inputs to process.
            options: Processing options.
            
        Returns:
            Dict[str, Any]: Processing results.
        """
        logger.info(f"Processing example inputs: {len(example_inputs.inputs)} total examples")
        
        # Default options
        if options is None:
            options = {
                "remove_similar": False,
                "similarity_threshold": 0.85,
                "language": self.language,
                "agent_description": self.agent_description,
            }
        
        # Categorize inputs
        inputs_with_refs = [q for q in example_inputs.inputs if q.reference_answer]
        inputs_with_responses = [q for q in example_inputs.inputs if q.response]
        inputs_only = [q for q in example_inputs.inputs 
                         if not q.reference_answer and not q.response]
        
        logger.info(f"Example input types: {len(inputs_with_refs)} with reference answers, " +
                   f"{len(inputs_with_responses)} with responses, {len(inputs_only)} with inputs only")
        
        # Log the first few examples for debugging
        if len(example_inputs.inputs) > 0:
            logger.info(f"First example input: {example_inputs.inputs[0].input}")
        
        # Process based on predominant type
        if len(inputs_with_refs) > 0:
            logger.info("Processing examples with reference answers")
            return self._process_example_inputs_with_reference_answers(
                inputs_with_refs, options
            )
        elif len(inputs_with_responses) > 0:
            logger.info("Processing examples with responses")
            return self._process_inputs_with_responses(
                inputs_with_responses, options
            )
        else:
            logger.info("Processing examples with inputs only")
            return self._process_inputs_only(
                inputs_only, options
            )
    
    def _process_inputs_only(
        self,
        example_inputs: List[ExampleItem],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process example inputs when only inputs are provided.
        
        Args:
            example_inputs: List of example input objects
            options: Processing options
            
        Returns:
            Dict[str, Any]: Processing results
        """
        logger.info(f"Processing {len(example_inputs)} example inputs with inputs only")
        
        # Extract just the input strings
        input_texts = [q.input for q in example_inputs]
        logger.info(f"Extracted {len(input_texts)} input texts")
        
        # Log the first few examples for debugging
        if len(input_texts) > 0:
            logger.info(f"First example input text: {input_texts[0]}")
            if len(input_texts) > 1:
                logger.info(f"Second example input text: {input_texts[1]}")
        
        # Remove similar inputs if requested
        if options.get("remove_similar", False):
            logger.info(f"Removing similar inputs with threshold {options.get('similarity_threshold', 0.85)}")
            input_texts = remove_similar_inputs(
                input_texts, 
                self.knowledge_base._client, 
                options.get("similarity_threshold", 0.85)
            )
            logger.info(f"After removing similar inputs: {len(input_texts)} inputs remaining")
        
        # Analyze inputs with different levels of detail
        logger.info("Analyzing example inputs with LLM")
        style_analysis = analyze_examples(
            input_texts,
            self.knowledge_base._client,
            use_llm=True,
            analysis_type="all"# these should be arguments passed to the constructor  
        )
        logger.info("Completed style analysis of example inputs")
        
        # Generate document-specific inputs based on example styles
        all_inputs = []
        num_docs_to_sample = min(10, len(self.knowledge_base))
        logger.info(f"Sampling {num_docs_to_sample} documents from knowledge base")
        documents = self.knowledge_base.get_random_documents(num_docs_to_sample)
        
        # Check if we should skip the relevance check
        skip_relevance_check = options.get("skip_relevance_check", False)
        
        # Process documents in parallel
        def process_document(doc_info):
            doc_idx, doc = doc_info
            logger.info(f"Processing document {doc_idx+1}/{len(documents)} (ID: {doc.id})")
            
            if skip_relevance_check:
                # Skip relevance check and consider all documents relevant
                is_relevant = True
                logger.info(f"Document {doc.id} considered relevant (relevance check skipped)")
                relevant_for_inputs = []
            else:
                # Check if document is relevant to any example input
                relevant_for_inputs = []
                
                # Process in smaller batches to avoid overwhelming the API
                check_inputs = input_texts[:5]  # Use only first 5 for efficiency
                max_workers = min(len(check_inputs), 5)  # Limit concurrent API calls
                
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit relevance check tasks
                    future_to_idx = {}
                    for q_idx, q in enumerate(check_inputs):
                        future = executor.submit(check_document_relevance, doc, q, self.knowledge_base._client)
                        future_to_idx[future] = q_idx
                    
                    # Collect results
                    for future in concurrent.futures.as_completed(future_to_idx):
                        q_idx = future_to_idx[future]
                        try:
                            is_relevant_for_input = future.result()
                            if is_relevant_for_input:
                                relevant_for_inputs.append(q_idx)
                        except Exception as e:
                            logger.error(f"Error checking relevance for document {doc.id}, input {q_idx}: {str(e)}")
                
                is_relevant = len(relevant_for_inputs) > 0
            
            if is_relevant:
                logger.info(f"Document {doc.id} is relevant for {len(relevant_for_inputs)} inputs: {relevant_for_inputs}")
                
                # Generate example-matched inputs for this document
                logger.info(f"Generating example-matched inputs for document {doc.id}")
                inputs = generate_example_matched_inputs(
                    document=doc,
                    client=self.knowledge_base._client,
                    example_inputs=input_texts,
                    style_analysis=style_analysis,
                    num_inputs=3,
                    language=options.get("language", "en"),
                )

                # Handle the case where inputs is None
                if inputs is None:
                    logger.warning(f"Failed to generate inputs for document {doc.id}")
                    inputs = []

                logger.info(f"Generated {len(inputs)} example-matched inputs for document {doc.id}")

                return [(doc, input_text) for input_text in inputs]
            else:
                logger.info(f"Document {doc.id} is not relevant for any example inputs")
                return []
        
        # Process documents in parallel
        max_workers = min(os.cpu_count() * 2 or 4, len(documents))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit document processing tasks
            doc_infos = [(i, doc) for i, doc in enumerate(documents)]
            results = list(executor.map(process_document, doc_infos))
            
            # Flatten results
            for doc_inputs in results:
                all_inputs.extend(doc_inputs)
        
        relevant_docs_count = sum(1 for result in results if result)
        logger.info(f"Found {relevant_docs_count} relevant documents out of {len(documents)}")
        logger.info(f"Generated a total of {len(all_inputs)} inputs from relevant documents")
        
        # Generate reference answers in parallel
        def generate_reference_for_pair(doc_input_pair):
            doc, input_text = doc_input_pair
            try:
                reference_answer = generate_reference_answer(
                    document=doc,
                    input=input_text,
                    client=self.knowledge_base._client,
                    language=options.get("language", "en"),
                )
                
                return TestItem(
                    input=input_text,
                    reference_answer=reference_answer,
                    document=doc,
                    metadata={
                        "language": options.get("language", "en"),
                        "topic_id": doc.topic_id,
                        "topic_name": self.knowledge_base.topics.get(doc.topic_id, "Unknown"),
                        "source": "example_guided",
                    },
                )
            except Exception as e:
                logger.error(f"Failed to generate reference answer for input '{input_text}': {str(e)}")
                return None
        
        # Process in parallel with a reasonable number of workers
        input_samples = []
        
        # Only process if we have inputs
        if all_inputs:
            max_workers = min(os.cpu_count() * 2 or 4, len(all_inputs))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                for sample in executor.map(generate_reference_for_pair, all_inputs):
                    if sample is not None:
                        input_samples.append(sample)
        
        # Create test set
        test_set = TestSet(
            inputs=input_samples,
            metadata={
                "language": options.get("language", "en"),
                "agent_description": options.get("agent_description", "A customer support chatbot that answers inputs based on a knowledge base."),
                "num_inputs": len(input_samples),
                "source": "example_guided",
                "example_inputs": {
                    "original_count": len(example_inputs),
                    "processed_count": len(input_texts),
                    "patterns": analyze_examples(input_texts, self.knowledge_base._client, analysis_type="basic"),
                    "style_analysis": style_analysis,
                    "full_analysis": analyze_examples(input_texts, self.knowledge_base._client, use_llm=True, analysis_type="all"),
                    "generation_method": "adaptive_prompt"
                },
            },
        )
        
        return {
            "test_set": test_set,
            "metadata": {
                "original_example_count": len(example_inputs),
                "processed_example_count": len(input_texts),
                "input_patterns": analyze_examples(input_texts, self.knowledge_base._client, analysis_type="basic"),
                "style_analysis": style_analysis,
            },
        }
    
    def _process_inputs_with_responses(
        self,
        example_inputs: List[ExampleItem],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process example inputs when inputs and responses are provided.
        
        Args:
            example_inputs: List of example input objects
            options: Processing options
            
        Returns:
            Dict[str, Any]: Processing results
        """
        # Filter to inputs that have responses
        inputs_with_responses = [q for q in example_inputs if q.response]
        
        # Extract just the input strings for analysis
        input_texts = [q.input for q in inputs_with_responses]
        
        # Comprehensive analysis of example inputs
        input_analysis = analyze_examples(
            input_texts,
            self.knowledge_base._client,
            use_llm=True,
            analysis_type="all"
        )
        
        # Check if we should skip the relevance check
        skip_relevance_check = options.get("skip_relevance_check", False)
        
        # Find relevant documents for each input
        for q in inputs_with_responses:
            if skip_relevance_check:
                # If skipping relevance check, just get the first document from the knowledge base
                doc = self.knowledge_base.get_random_documents(1)[0]
                q.metadata["document_id"] = doc.id
                q.metadata["document_content"] = doc.content
                q.metadata["relevance_score"] = 1.0  # Assign a perfect score since we're skipping the check
                logger.info(f"Assigned random document {doc.id} to input (relevance check skipped)")
            else:
                # Normal relevance check using search
                relevant_docs = self.knowledge_base.search(q.input, k=1)
                if relevant_docs:
                    doc, score = relevant_docs[0]
                    q.metadata["document_id"] = doc.id
                    q.metadata["document_content"] = doc.content
                    q.metadata["relevance_score"] = score
        
        # Generate reference answers for inputs without them
        for q in inputs_with_responses:
            if not q.reference_answer and "document_content" in q.metadata:
                # Create document object
                doc = Document(
                    content=q.metadata["document_content"],
                    metadata={},
                    id=q.metadata.get("document_id"),
                )
                
                # Generate reference answer
                q.reference_answer = generate_reference_answer(
                    document=doc,
                    input=q.input,
                    client=self.knowledge_base._client,
                    language=options.get("language", "en"),
                )
        
        # Create input samples
        input_samples = []
        for q in inputs_with_responses:
            if q.reference_answer and "document_content" in q.metadata:
                # Create document object
                doc = Document(
                    content=q.metadata["document_content"],
                    metadata={},
                    id=q.metadata.get("document_id"),
                )
                
                # Create input sample
                sample = TestItem(
                    input=q.input,
                    reference_answer=q.reference_answer,
                    document=doc,
                    metadata={
                        "source": "example_with_response",
                        "relevance_score": q.metadata.get("relevance_score", 0),
                        "has_feedback": q.feedback is not None,
                        "feedback": q.feedback,
                        "response": q.response,
                    },
                )
                
                input_samples.append(sample)
        
        # Create test set
        test_set = TestSet(
            inputs=input_samples,
            metadata={
                "language": options.get("language", "en"),
                "agent_description": options.get("agent_description", "A customer support chatbot that answers inputs based on a knowledge base."),
                "num_inputs": len(input_samples),
                "source": "example_with_responses",
                "example_inputs": {
                    "original_count": len(example_inputs),
                    "with_responses_count": len(inputs_with_responses),
                    "with_feedback_count": sum(1 for q in inputs_with_responses if q.feedback),
                    "patterns": input_analysis,
                },
            },
        )
        
        return {
            "test_set": test_set,
            "metadata": {
                "original_example_count": len(example_inputs),
                "with_responses_count": len(inputs_with_responses),
                "with_feedback_count": sum(1 for q in inputs_with_responses if q.feedback),
                "input_patterns": input_analysis,
            },
        }
    
    def _process_inputs_with_reference_answers(
        self,
        example_inputs: List[ExampleItem],
        options: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process example inputs when inputs and reference answers are provided.
        
        Args:
            example_inputs: List of example input objects
            options: Processing options
            
        Returns:
            Dict[str, Any]: Processing results
        """
        # Filter to inputs that have reference answers
        inputs_with_refs = [q for q in example_inputs if q.reference_answer]
        
        # Extract just the input strings for analysis
        input_texts = [q.input for q in inputs_with_refs]
        
        # Comprehensive analysis of example inputs
        input_analysis = analyze_examples(
            input_texts,
            self.knowledge_base._client,
            use_llm=True,
            analysis_type="all"
        )
        
        # Check if we should skip the relevance check
        skip_relevance_check = options.get("skip_relevance_check", False)
        
        # Find relevant documents for each input
        for q in inputs_with_refs:
            if skip_relevance_check:
                # If skipping relevance check, just get the first document from the knowledge base
                doc = self.knowledge_base.get_random_documents(1)[0]
                q.metadata["document_id"] = doc.id
                q.metadata["document_content"] = doc.content
                q.metadata["relevance_score"] = 1.0  # Assign a perfect score since we're skipping the check
                logger.info(f"Assigned random document {doc.id} to input (relevance check skipped)")
            else:
                # Normal relevance check using search
                relevant_docs = self.knowledge_base.search(q.input, k=1)
                if relevant_docs:
                    doc, score = relevant_docs[0]
                    q.metadata["document_id"] = doc.id
                    q.metadata["document_content"] = doc.content
                    q.metadata["relevance_score"] = score
        
        # Create input samples
        input_samples = []
        for q in inputs_with_refs:
            if "document_content" in q.metadata:
                # Create document object
                doc = Document(
                    content=q.metadata["document_content"],
                    metadata={},
                    id=q.metadata.get("document_id"),
                )
                
                # Create input sample
                sample = TestItem(
                    input=q.input,
                    reference_answer=q.reference_answer,
                    document=doc,
                    metadata={
                        "source": "example_with_reference",
                        "relevance_score": q.metadata.get("relevance_score", 0),
                    },
                )
                
                input_samples.append(sample)
        
        # Create test set
        test_set = TestSet(
            inputs=input_samples,
            metadata={
                "language": options.get("language", "en"),
                "agent_description": options.get("agent_description", "A customer support chatbot that answers inputs based on a knowledge base."),
                "num_inputs": len(input_samples),
                "source": "example_with_references",
                "example_inputs": {
                    "original_count": len(example_inputs),
                    "with_references_count": len(inputs_with_refs),
                    "matched_with_documents": len(input_samples),
                    "analysis": input_analysis,
                },
            },
        )
        
        return {
            "test_set": test_set,
            "metadata": {
                "original_example_count": len(example_inputs),
                "with_references_count": len(inputs_with_refs),
                "matched_with_documents": len(input_samples),
                "input_analysis": input_analysis,
            },
        }
