"""
MERIT RAG Test Set Generator

This module provides a class-based approach for generating synthetic test sets for RAG evaluation.
It encapsulates the functionality for test set generation in an object-oriented design.

The TestSetGenerator class provides a flexible and maintainable API for generating
test sets for RAG evaluation.
"""
from typing import Dict, Any, List, Optional, Union
import datetime 

from ..core.models import TestSet, TestItem, ExampleItem, ExampleSet, Document, Input, Response
from ..knowledge import knowledgebase
from ..core.utils import parse_json
from ..core.logging import get_logger
from ..api.base import BaseAPIClient
from .prompts import (
    TEST_INPUT_GENERATION_PROMPT, 
    REFERENCE_ANSWER_GENERATION_PROMPT,
    ADAPTIVE_TEST_INPUT_GENERATION_PROMPT
)
from .analysis import analyze_examples
logger = get_logger(__name__)

DEFAULT_NUM_ITEMS = 10 # Lowered for easier debugging initially
DEFAULT_ITEMS_PER_DOCUMENT = 2 # Lowered for easier debugging
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
        example_set_obj: Optional[ExampleSet] = None
        example_analysis_result: Optional[Dict[str, Any]] = None
        
        if example_items:
            if not isinstance(example_items, ExampleSet):
                example_set_obj = ExampleSet(inputs=example_items)
            else:
                example_set_obj = example_items
            
            if example_set_obj and example_set_obj.length() > 0:
                logger.info(f"Using {example_set_obj.length()} example items for guidance.")
                try:
                    logger.info("Analyzing example items for generate() method...")
                    # Ensure _client attribute exists and is the correct AI client
                    if not hasattr(self.knowledge_base, '_client') or not isinstance(self.knowledge_base._client, BaseAPIClient):
                        logger.error("KnowledgeBase does not have a valid AI client for example analysis.")
                        # Decide how to handle: raise error, or proceed without example analysis
                        # For now, proceed without, but this is a potential issue.
                    else:
                        example_analysis_result = analyze_examples(
                            example_set_obj, 
                            ai_client=self.knowledge_base._client, 
                            use_llm=True, 
                            analysis_type="all"
                        )
                        logger.info("Example items analyzed successfully for generate() method.")
                except Exception as e:
                    logger.error(f"Failed to analyze example items in generate(): {e}", exc_info=True)
                    example_analysis_result = None # Proceed without analysis if it fails
        
        logger.info(f"Using '{distribution_strategy}' distribution strategy to generate up to {num_items} items.")
        doc_input_pairs = self.knowledge_base.sample_documents_with_input_allocation(
            n=num_items,
            strategy=distribution_strategy,
            items_per_document=items_per_document
        )
        print(f"[DEBUG TestSetGenerator.generate] doc_input_pairs from KB sampling: {doc_input_pairs}")
        
        final_test_items_list: List[TestItem] = []
        processed_items_count = 0
        processed_doc_ids_in_current_run = set()

        for document, num_items_allocated_for_doc in doc_input_pairs:
            if processed_items_count >= num_items:
                logger.info("Target number of items reached. Stopping document processing loop.")
                break
            
            # How many items we still need vs. how many this doc is allocated
            items_to_attempt_for_this_doc = min(num_items_allocated_for_doc, num_items - processed_items_count)

            if items_to_attempt_for_this_doc <= 0:
                continue # No items to generate for this doc in this pass

            logger.info(f"Calling generate_items for document ID: {document.id} to generate {items_to_attempt_for_this_doc} items.")
            items_generated_for_doc = self.generate_items(
                document=document,
                ai_client=self.knowledge_base._client, 
                example_set=example_set_obj,
                example_analysis=example_analysis_result,
                num_items_to_generate=items_to_attempt_for_this_doc
            )
            
            final_test_items_list.extend(items_generated_for_doc)
            processed_items_count += len(items_generated_for_doc)
            processed_doc_ids_in_current_run.add(document.id)
            logger.info(f"Generated {len(items_generated_for_doc)} items for doc {document.id}. Total items so far: {processed_items_count}")
            print(f"[DEBUG TestSetGenerator.generate] items_generated_for_doc for doc {document.id} (length {len(items_generated_for_doc)}): {items_generated_for_doc}") # DEBUG PRINT ADDED
        
        if len(final_test_items_list) > num_items:
            logger.info(f"Truncating generated items from {len(final_test_items_list)} to requested {num_items}.")
            final_test_items_list = final_test_items_list[:num_items]

        kb_doc_count = len(self.knowledge_base.documents) if self.knowledge_base.documents else 0
        num_topics_val = 0
        if final_test_items_list:
             num_topics_val = len(set(item.metadata.get("topic_id") for item in final_test_items_list if item.document and item.metadata and item.metadata.get("topic_id") is not None))

        timestamp_str = datetime.datetime.now().isoformat()
        test_set_metadata = {
            "language": self.language,
            "agent_description": self.agent_description,
            "num_items_requested": num_items,
            "num_items_generated": len(final_test_items_list),
            "num_documents_sampled_for_generation": len(doc_input_pairs),
            "num_distinct_documents_in_testset": len(set(item.document.id for item in final_test_items_list if item.document)),
            "num_documents_in_kb": kb_doc_count,
            "source": "example_guided" if example_set_obj and example_set_obj.length() > 0 else "standard",
            "distribution_strategy": distribution_strategy,
            "items_per_document_setting" : items_per_document if distribution_strategy == DISTRIBUTION_PER_DOCUMENT else None,
            "skip_relevance_check_setting": skip_relevance_check,
            "similarity_threshold_setting": similarity_threshold,
            "timestamp": timestamp_str
        }
        if num_topics_val > 0 : test_set_metadata["num_topics_covered"] = num_topics_val

        test_set = TestSet(
            inputs=final_test_items_list, 
            metadata=test_set_metadata
        )
        logger.info(f"Finished generating test set. Total items generated: {len(final_test_items_list)}")
        return test_set

    def generate_items(
        self,
        document: Document, 
        ai_client: BaseAPIClient,
        example_set: Optional[ExampleSet] = None,
        example_analysis: Optional[Dict[str, Any]] = None,
        num_items_to_generate: int = DEFAULT_ITEMS_PER_DOCUMENT 
    ) -> List[TestItem]:
        """
        Generates a list of TestItem objects for a single document.

        This method uses an AI client to generate potential test inputs based on the provided document.
        It can operate in two modes:
        1. Example-guided: If 'example_set' and 'example_analysis' are provided, it uses them
           to guide the generation of inputs that are stylistically similar or cover similar aspects.
        2. Standard: If no examples are provided, it generates inputs based on a general prompt
           instructing the AI to create diverse inputs answerable by the document.

        For each generated input, it then calls the AI client again to generate a reference answer
        based on that input and the original document content.

        Args:
            document: The Document object to generate test items from.
            ai_client: The BaseAPIClient instance to use for generating text (inputs and answers).
            example_set: Optional. An ExampleSet object containing examples to guide input generation.
            example_analysis: Optional. Analysis results from the example_set, used in example-guided mode.
            num_items_to_generate: The target number of test inputs to generate for this document.

        Returns:
            A list of TestItem objects, each containing a generated input, a reference answer,
            and associated metadata.
        """
        logger.info(f"generate_items called for document ID: {document.id}, aiming for {num_items_to_generate} items.")
        
        # Ensure necessary models are imported locally if not at module level for clarity
        from ..core.models import Input, Response, TestItem 
        # Prompts are imported at module level
        # parse_json is imported at module level

        generated_input_strings: List[str] = [] 

        if example_set and example_set.length() > 0:
            logger.info(f"Using example-guided input generation for document {document.id}")
            try:
                current_example_analysis = example_analysis
                if not current_example_analysis:
                    logger.warning("Example set provided to generate_items but no analysis result; re-analyzing.")
                    current_example_analysis = analyze_examples(
                        example_set, ai_client=ai_client, use_llm=True, analysis_type="all"
                    )
                
                prompt_str = ADAPTIVE_TEST_INPUT_GENERATION_PROMPT.safe_format(
                    document_content=document.content,
                    example_section=str(example_set), 
                    style_guidance=str(current_example_analysis), 
                    num_inputs=num_items_to_generate 
                )
                print(f"\n[DEBUG generate_items EG] Document ID: {document.id}")
                print(f"[DEBUG generate_items EG] Input Gen Prompt:\n{prompt_str}\n")
                print(f"[DEBUG generate_items EG PRE-CALL] ai_client type: {type(ai_client)}, model: {getattr(ai_client, 'model', 'N/A')}, api_key: {getattr(ai_client, 'api_key', 'N/A')[:10]}..., prompt_str[:100]: {prompt_str[:100]}...") # DEBUG PRINT ADDED
                response_str = ai_client.generate_text(prompt_str)
                print(f"[DEBUG generate_items EG] Raw Input Gen LLM Response:\n{response_str}\n")
                response_json = parse_json(response_str)
                print(f"[DEBUG generate_items EG] Parsed Input Gen JSON:\n{response_json}\n")
                extracted_inputs = response_json.get("test_inputs", [])
                print(f"[DEBUG generate_items EG] Extracted 'test_inputs' list:\n{extracted_inputs}\n")
                for item_data in extracted_inputs:
                    input_text = None
                    if isinstance(item_data, dict) and "test_input" in item_data:
                        input_text = item_data["test_input"]
                    elif isinstance(item_data, str):
                        input_text = item_data
                    if input_text and isinstance(input_text, str):
                        generated_input_strings.append(input_text)
                print(f"[DEBUG generate_items EG] Collected input strings for doc {document.id}: {generated_input_strings}")
            except Exception as e:
                logger.error(f"Failed during example-guided input generation for doc {document.id}: {e}", exc_info=True)
        else:
            logger.info(f"Using standard input generation for document {document.id}, for {num_items_to_generate} inputs.")
            try:
                prompt_str = TEST_INPUT_GENERATION_PROMPT.safe_format(
                    document_content=document.content,
                    system_description=self.agent_description,
                    num_inputs=num_items_to_generate, 
                    language=self.language
                )
                print(f"\n[DEBUG generate_items STD] Document ID: {document.id}")
                print(f"[DEBUG generate_items STD] Input Gen Prompt:\n{prompt_str}\n")
                print(f"[DEBUG generate_items STD PRE-CALL] ai_client type: {type(ai_client)}, model: {getattr(ai_client, 'model', 'N/A')}, api_key: {getattr(ai_client, 'api_key', 'N/A')[:10]}..., prompt_str[:100]: {prompt_str[:100]}...") # DEBUG PRINT ADDED
                response_str = ai_client.generate_text(prompt_str)
                print(f"[DEBUG generate_items STD] Raw Input Gen LLM Response:\n{response_str}\n")
                response_json = parse_json(response_str)
                print(f"[DEBUG generate_items STD] Parsed Input Gen JSON:\n{response_json}\n")
                extracted_inputs = response_json.get("test_inputs", [])
                print(f"[DEBUG generate_items STD] Extracted 'test_inputs' list:\n{extracted_inputs}\n")
                for item_data in extracted_inputs:
                    input_text = None
                    if isinstance(item_data, dict) and "test_input" in item_data:
                        input_text = item_data["test_input"]
                    elif isinstance(item_data, str):
                        input_text = item_data
                    if input_text and isinstance(input_text, str):
                        generated_input_strings.append(input_text)
                print(f"[DEBUG generate_items STD] Collected input strings for doc {document.id}: {generated_input_strings}")
            except Exception as e:
                logger.error(f"Failed during standard input generation for doc {document.id}: {e}", exc_info=True)

        final_test_items: List[TestItem] = []
        if not generated_input_strings:
            logger.warning(f"No input strings were generated for document {document.id}. Returning empty list for this doc.")
            return []

        logger.info(f"Attempting to generate reference answers for {len(generated_input_strings)} inputs from doc {document.id}.")
        for input_text_str in generated_input_strings:
            try:
                input_obj = Input(user_input=input_text_str)
                ref_prompt_str = REFERENCE_ANSWER_GENERATION_PROMPT.safe_format(
                    document_content=document.content,
                    test_input=input_text_str,
                    language=self.language
                )
                # print(f"[DEBUG generate_items REF] Ref Answer Prompt for '{input_text_str[:30]}...':\n{ref_prompt_str}\n") # Verbose
                ref_answer_str = ai_client.generate_text(ref_prompt_str)
                # print(f"[DEBUG generate_items REF] Raw Ref Answer for '{input_text_str[:30]}...':\n{ref_answer_str}\n") # Verbose
                ref_answer_obj = Response(content=ref_answer_str)
                
                test_item = TestItem(
                    input=input_obj,
                    reference_answer=ref_answer_obj,
                    document=document,
                    metadata={
                        "language": self.language,
                        "topic_id": getattr(document, 'topic_id', None),
                        "topic_name": self.knowledge_base.topics.get(getattr(document, 'topic_id', None), "Unknown") if self.knowledge_base.topics else "Unknown",
                        "example_guided": example_set is not None
                    }
                )
                final_test_items.append(test_item)
                logger.debug(f"Successfully created TestItem for input: {input_text_str[:50]}...")
            except Exception as e:
                logger.error(f"Failed creating TestItem for input '{input_text_str}' from doc {document.id}: {e}", exc_info=True)
        
        logger.info(f"generate_items finished for doc {document.id}. Generated {len(final_test_items)} TestItems.")
        print(f"[DEBUG TestSetGenerator.generate_items] Returning for doc {document.id} (length {len(final_test_items)}): {final_test_items}") # DEBUG PRINT ADDED
        return final_test_items
