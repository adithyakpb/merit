"""
MERIT Core Models

This module contains the core data models used in MERIT.
These models represent documents, inputs, responses, test sets, and example inputs for evaluation.
"""

import json
import uuid
import os
import csv
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from .utils import parse_json

from .logging import get_logger

logger = get_logger(__name__)

@dataclass
class Input:
    """
    A class representing an input to the evaluated system.
    
    Attributes:
        user_input: The raw user question/input
        prompt_prefix: Text before user input in the full prompt
        prompt_suffix: Text after user input in the full prompt
        metadata: Additional metadata
        id: The ID of the input
    """
    
    user_input: str
    prompt_prefix: str = ""
    prompt_suffix: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict) 
    id: str = None
    
    
    @property
    def content(self) -> str:
        """Reconstruct the complete content of the Input"""
        return f"{self.prompt_prefix}{self.user_input}{self.prompt_suffix}"
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Input':
        """Create from dictionary."""
        if isinstance(data, str):
            return cls(user_input=data) # Changed content to user_input
        
        return cls(
            id=data.get("id"),
            user_input=data.get("content", ""), # Changed content to user_input
            metadata=data.get("metadata", {}),
        )

@dataclass
class Response:
    """
    A class representing a response from the evaluated system.
    
    Attributes:
        content: The raw response content (e.g., LLM completion)
        documents: The documents used to generate the answer
        metadata: Additional metadata
        id: The ID of the response
    """
    
    content: str
    documents: Optional[List[Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None # Added error field
    id: str = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    @property
    def message(self) -> str:
        """
        Get the message content (alias for content).
        
        Returns:
            str: The content of the response
        """
        return self.content
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        serializable_documents = None
        if self.documents:
            serializable_documents = []
            for doc in self.documents:
                if hasattr(doc, 'to_dict'):
                    serializable_documents.append(doc.to_dict())
                elif isinstance(doc, Document): # If it's a Document object without to_dict
                    serializable_documents.append({
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "topic_id": doc.topic_id
                    })
                else:
                    serializable_documents.append(doc) # Fallback for other types

        return {
            "id": self.id,
            "content": self.content,
            "documents": serializable_documents,
            "metadata": self.metadata,
            "error": self.error, # Added error to to_dict
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Response':
        """Create from dictionary."""
        if isinstance(data, str):
            return cls(content=data)
        
        return cls(
            id=data.get("id"),
            content=data.get("content", ""),
            documents=data.get("documents"),
            metadata=data.get("metadata", {}),
            error=data.get("error"), # Added error to from_dict
        )

@dataclass
class Document:
    """
    A document in the knowledge base. 
    
    A 'document' in your knowledge base is a single 'chunk' of text content. Chunk the content knowledge base into documents that are small enough to be processed by the model in a single step. For example, a document could be a single paragraph, a section of a webpage, a single sentence, or a single list item. 
    
    MERIT will process each document independently, so it's important to chunk the content in a way that makes sense for processing.
    
    Attributes:
        content: The content of the document.
        metadata: Metadata about the document.
        id: The ID of the document.
        embeddings: (Optional) The embeddings of the document.
        reduced_embeddings: (Optional) The reduced embeddings of the document for visualization.
        topic_id: (Optional) The ID of the topic the document belongs to.
    """
    
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict) # Added default_factory
    id: str = None
    embeddings: Optional[List[float]] = None
    reduced_embeddings: Optional[List[float]] = None
    topic_id: Optional[int] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())

@dataclass
class TestItem:
    """
    A input sample in a test set.
    
    Attributes:
        input: The input text.
        reference_answer: The reference answer as a Response object.
        document: The document the input is based on.
        id: The ID of the input sample.
        metadata: Additional metadata about the input sample.
    """
    input: str | Input
    reference_answer: str | Response
    document: Document
    id: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
        
        # Convert string input to Input object
        if isinstance(self.input, str):
            self.input = Input(user_input=self.input) # Changed content to user_input
    
        # Convert string reference_answer to Response object
        if isinstance(self.reference_answer, str):
            self.reference_answer = Response(content=self.reference_answer)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Test Item to a dictionary.
        
        Returns:
            Dict[str, Any]: The input sample as a dictionary.
        """
        # Convert Input and Response objects to their dictionary representations
        input_data = self.input.to_dict() if hasattr(self.input, 'to_dict') else self.input
        ref_answer_data = self.reference_answer.to_dict() if hasattr(self.reference_answer, 'to_dict') else self.reference_answer
        
        return {
            "id": self.id,
            "input": input_data,
            "reference_answer": ref_answer_data,
            "document_id": self.document.id,
            "document_content": self.document.content,
            "metadata": self.metadata,
        }
    @classmethod
    def from_dict(cls, data: Dict[str, Any], document: Optional[Document] = None) -> 'TestItem':
        """
        Create a input sample from a dictionary.
        
        Args:
            data: The dictionary to create the input sample from.
            document: The document the input is based on.
            
        Returns:
            TestItem: The created input sample.
        """
        # If document is not provided, create a dummy document
        if document is None:
            document = Document(
                content=data.get("document_content", ""),
                metadata={},
                id=data.get("document_id"),
            )
        
        # Get input data
        input_data = data.get("input", "")
        if isinstance(input_data, dict):
            input_obj = Input.from_dict(input_data)
        else:
            input_obj = input_data
        
        # Get reference answer data
        ref_answer_data = data.get("reference_answer", "")
        if isinstance(ref_answer_data, dict):
            ref_answer_obj = Response.from_dict(ref_answer_data)
        else:
            ref_answer_obj = ref_answer_data  # Will be converted to Response in __post_init__
        
        return cls(
            id=data.get("id"),
            input=input_obj,
            reference_answer=ref_answer_obj,
            document=document,
            metadata=data.get("metadata", {}),
        )

@dataclass
class TestSet:
    """
    A test set for RAG evaluation.
    
    Attributes:
        inputs: The inputs in the test set, as a list of TestItem objects.
        metadata: Additional metadata about the test set.
    """
    
    inputs: List[TestItem]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the test set to a dictionary.
        
        Returns:
            Dict[str, Any]: The test set as a dictionary.
        """
        return {
            "inputs": [q.to_dict() for q in self.inputs],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], documents: Optional[Dict[str, Document]] = None) -> 'TestSet':
        """
        Create a test set from a dictionary.
        
        Args:
            data: The dictionary to create the test set from.
            documents: A dictionary mapping document IDs to documents.
            
        Returns:
            TestSet: The created test set.
        """
        inputs = []
        for q_data in data.get("inputs", []):
            # Get document if available
            document = None
            if documents is not None and "document_id" in q_data:
                document = documents.get(q_data["document_id"])
            
            # Create input sample
            input = TestItem.from_dict(q_data, document)
            inputs.append(input)
        
        return cls(
            inputs=inputs,
            metadata=data.get("metadata", {}),
        )
    
    def save(self, file_path: str) -> bool:
        """
        Save the test set to a file.
        
        Args:
            file_path: The path to save the test set to.
            
        Returns:
            bool: True if the test set was saved successfully, False otherwise.
        """
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved test set with {len(self.inputs)} inputs to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save test set to {file_path}: {str(e)}")
            return False
    
    @classmethod
    def load(cls, file_path: str, documents: Optional[Dict[str, Document]] = None) -> 'TestSet':
        """
        Load a test set from a file (JSON or CSV).
        
        Args:
            file_path: The path to load the test set from.
            documents: A dictionary mapping document IDs to documents.
            
        Returns:
            TestSet: The loaded test set.
        """
        from .utils import parse_json
        
        # Determine file type based on extension
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.json':
                # Load from JSON using parse_json from utils
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                
                data = parse_json(file_content)
                test_set = cls.from_dict(data, documents)
                logger.info(f"Loaded test set with {len(test_set.inputs)} inputs from JSON file {file_path}")
                return test_set
            
            elif file_ext == '.csv':
                # Load from CSV
                inputs = []
                with open(file_path, "r", encoding="utf-8") as f:
                    csv_reader = csv.DictReader(f)
                    
                    for row in csv_reader:
                        # Extract required fields
                        input_text = row.get('input', '')
                        reference_answer = row.get('reference_answer', '')
                        document_id = row.get('document_id')
                        document_content = row.get('document_content', '')
                        
                        # Create document
                        if documents and document_id and document_id in documents:
                            # Use existing document if available
                            document = documents[document_id]
                        else:
                            # Create new document
                            document = Document(
                                content=document_content,
                                metadata=row.get("document_metadata", {}), # Allow loading metadata if present
                                id=document_id
                            )
        
        # Create TestItem
                        metadata = {k: v for k, v in row.items() 
                                   if k not in ['input', 'reference_answer', 'document_id', 'document_content']}
                        
                        test_input = TestItem(
                            input=input_text,
                            reference_answer=reference_answer,
                            document=document,
                            id=row.get('id'),
                            metadata=metadata
                        )
                        
                        inputs.append(test_input)
                
                test_set = cls(inputs=inputs)
                logger.info(f"Loaded test set with {len(test_set.inputs)} inputs from CSV file {file_path}")
                return test_set
            
            else:
                # Unsupported file type
                logger.error(f"Unsupported file type: {file_ext}. Supported types are .json and .csv")
                return cls(inputs=[])
                
        except Exception as e:
            logger.error(f"Failed to load test set from {file_path}: {str(e)}")
            return cls(inputs=[])

@dataclass
class ExampleItem:
    """
    An example input provided by the user.
    
    Attributes:
        input: The input text.
        reference_answer: Optional reference answer.
        feedback: Optional feedback on the response.
        metadata: Additional metadata.
    """
    
    input: str
    reference_answer: Optional[Response | str] = None
    feedback: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        # Convert Response objects to their dictionary representations
        ref_answer_data = self.reference_answer.to_dict() if hasattr(self.reference_answer, 'to_dict') else self.reference_answer
        
        return {
            "input": self.input,
            "reference_answer": ref_answer_data,
            "feedback": self.feedback,
            "metadata": self.metadata,
        }
    
    def __post_init__(self):
        """Process the inputs after initialization if needed."""
        # Convert string reference_answer to Response object
        if isinstance(self.reference_answer, str):
            self.reference_answer = Response(content=self.reference_answer)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExampleItem':
        """Create from dictionary."""
        if isinstance(data, str):
            return cls(input=data)
        
        # Get reference answer data
        ref_answer_data = data.get("reference_answer")
        if isinstance(ref_answer_data, dict):
            ref_answer_obj = Response.from_dict(ref_answer_data)
        else:
            ref_answer_obj = ref_answer_data  # Will be converted to Response in __post_init__ if it's a string
            
        return cls(
            input=data.get("input", ""),
            reference_answer=ref_answer_obj,
            feedback=data.get("feedback"),
            metadata=data.get("metadata", {}),
        )

@dataclass
class ExampleSet:
    """
    A collection of example item.
    
    Attributes:
        examples: The example items. Can be:
            - A list of ExampleItem objects
            - A list of strings (inputs)
            - A list of dictionaries (inputs, reference_answers)
            - A dictionary with a "inputs" key
            - A file path (string) to a JSON file
            - A single ExampleItem object
        metadata: Additional metadata.
        remove_similar: Whether to remove similar inputs during initialization.
        similarity_threshold: Threshold for similarity detection.
        client: The API client to use for embeddings (required if remove_similar is True).
    """
    
    examples: Union[List[ExampleItem], List[str], List[Dict[str, Any]], Dict[str, Any], str, ExampleItem]
    metadata: Dict[str, Any] = field(default_factory=dict)
    remove_similar: bool = False
    similarity_threshold: float = 0.85
    client: Optional[Any] = None
    
    def __post_init__(self):
        """Process the examples after initialization if needed."""
        # Process the input if it's not already a list of ExampleItem objects
        if not (isinstance(self.examples, list) and 
                all(isinstance(q, ExampleItem) for q in self.examples)):
            self.examples = self._process_input(self.examples)
        
        # Remove similar examples if requested and client is provided
        if self.remove_similar and self.client and len(self.examples) > 1:
            self._remove_similar_example_inputs()
    
    def _process_input(self, input_data) -> List[ExampleItem]:
        """Convert various example item formats to a list of ExampleItem objects."""
        # If it's a single ExampleItem, wrap it in a list
        if isinstance(input_data, ExampleItem):
            return [input_data]
        
        # If it's a string (file path), load from file
        if isinstance(input_data, str):
            try:
                logger.info(f"Loading example inputs from file: {input_data}")
                with open(input_data, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return self._process_input(data)
            except Exception as e:
                logger.error(f"Failed to load example inputs from file: {str(e)}")
                return []
        
        # If it's a list of strings, dictionaries, or ExampleItem objects
        if isinstance(input_data, list):
            return [q if isinstance(q, ExampleItem) else ExampleItem.from_dict(q) for q in input_data]
        
        # If it's a dictionary with a "inputs" key
        if isinstance(input_data, dict) and "inputs" in input_data:
            inputs = input_data.get("inputs", [])
            # Update metadata if available
            if "metadata" in input_data:
                self.metadata.update(input_data.get("metadata", {}))
            return self._process_input(inputs)
        
        # Default case
        logger.warning(f"Unrecognized example inputs format: {type(input_data)}")
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "inputs": [q.to_dict() for q in self.inputs],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExampleSet':
        """Create from dictionary."""
        # Simply pass the data to the constructor, which will handle processing
        return cls(inputs=data, metadata={} if isinstance(data, list) else data.get("metadata", {}))
    
    @classmethod
    def load(cls, file_path: str) -> 'ExampleSet':
        """Load from a JSON file."""
        # Simply pass the file path to the constructor, which will handle loading
        return cls(inputs=file_path)
    
    def _remove_similar_example_inputs(self):
        """Remove similar inputs from the set."""
        from ..testset_generation.generator import remove_similar_inputs
        
        # Extract input texts
        input_texts = [q.input for q in self.inputs]
        
        # Use the existing remove_similar_inputs function
        filtered_texts = remove_similar_inputs(
            input_texts,
            self.client,
            self.similarity_threshold
        )
        
        # Map back to original inputs
        filtered_indices = []
        for text in filtered_texts:
            try:
                idx = input_texts.index(text)
                filtered_indices.append(idx)
            except ValueError:
                continue
        
        # Update inputs
        self.inputs = [self.inputs[i] for i in filtered_indices]
        
        # Update metadata
        self.metadata["original_count"] = len(input_texts)
        self.metadata["filtered_count"] = len(self.inputs)
        self.metadata["removed_count"] = len(input_texts) - len(self.inputs)
        
        logger.info(f"Removed {len(input_texts) - len(self.inputs)} similar inputs from example set")
    
    def save(self, file_path: str) -> bool:
        """Save to a JSON file."""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved {len(self.inputs)} example inputs to {file_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save example inputs: {e}")
            return False

    def length(self, selective: str = "") -> int:
        if selective == "reference_answers":
            length = 0
            for reference_answer in self.examples:
                length += len(reference_answer.split())
        elif selective == "inputs":
            return len(self.examples)
        return len(self.examples)

@dataclass
class EvaluationResult:
    """
    A result of evaluating a single input.
    
    Attributes:
        input: The input that was evaluated
        reference: The reference answer if available
        response: The response from the system
        metrics: Dictionary of metric names to scores
        pass_fail: Whether the result passes or fails
        metadata: Additional metadata
    """
    
    input: Any
    response: Any
    metrics: List[Dict[str, Any]] = field(default_factory=list)
    reference: Optional[Any] = None
    pass_fail: Optional[bool] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "metrics": self.metrics,
            "metadata": self.metadata
        }
        
        # Add optional fields if present
        if self.input is not None:
            if hasattr(self.input, "to_dict"):
                result["input"] = self.input.to_dict()
            else:
                result["input"] = self.input
                
        if self.response is not None:
            if hasattr(self.response, "to_dict"):
                result["response"] = self.response.to_dict()
            else:
                result["response"] = self.response
                
        if self.reference is not None:
            if hasattr(self.reference, "to_dict"):
                result["reference"] = self.reference.to_dict()
            else:
                result["reference"] = self.reference
                
        if self.pass_fail is not None:
            result["pass_fail"] = self.pass_fail
            
        return result

@dataclass
class EvaluationReport:
    """
    A report containing multiple evaluation results.
    
    Attributes:
        results: List of evaluation results
        summary: Summary of all results
        metadata: Additional metadata
    """
    
    results: List[EvaluationResult]
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "results": [r.to_dict() for r in self.results],
            "summary": self.summary,
            "metadata": self.metadata
        }
    
    def save(self, file_path: str, generate_html: bool = True) -> bool:
        """
        Save the evaluation report to a file.
        
        Args:
            file_path: Path to save the report to
            generate_html: Whether to also generate an HTML report
            
        Returns:
            bool: Whether the save was successful
        """
        try:
            # Save JSON report
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved evaluation report to {file_path}")
            
            # Generate HTML if requested
            if generate_html:
                html_path = file_path.replace('.json', '.html')
                html_success = self.save_as_html(html_path)
                if html_success:
                    logger.info(f"Generated HTML report at {html_path}")
                else:
                    logger.warning(f"Failed to generate HTML report at {html_path}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to save evaluation report: {str(e)}")
            return False
    
    def save_as_html(self, file_path: str) -> bool:
        """
        Save the evaluation report as an HTML file.
        
        Args:
            file_path: Path to save the HTML report to
            
        Returns:
            bool: Whether the save was successful
        """
        try:
            import os
            
            # Get the HTML template
            template_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates')
            template_path = os.path.join(template_dir, 'report_template.html')
            
            with open(template_path, 'r', encoding='utf-8') as f:
                template = f.read()
            
            # Convert report to JSON string
            report_json = json.dumps(self.to_dict(), indent=2)
            
            # Insert the report data into the template
            html_content = template.replace('const REPORT_DATA = {};', 
                                          f'const REPORT_DATA = {report_json};')
            
            # Write the HTML file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save HTML report: {str(e)}")
            return False
