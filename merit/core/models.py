"""
Merit Core Models

This module contains the core data models used in the Merit system.
These models represent documents, inputs, test sets, and example inputs for evaluation.
"""

import json
import uuid
import os
import csv
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """
    A document in the knowledge base. 
    
    A 'document' in your knowledge base is a single 'chunk' of text content. Chunk the content knowledge base into documents that are small enough to be processed by the model in a single step. For example, a document could be a single paragraph, a section of a webpage, a single sentence, or a single list item. 
    
    The model will process each document independently, so it's important to chunk the content in a way that makes sense for the model to process.
    
    Attributes:
        content: The content of the document.
        metadata: Metadata about the document.
        id: The ID of the document.
        embeddings: The embeddings of the document.
        reduced_embeddings: The reduced embeddings of the document for visualization.
        topic_id: The ID of the topic the document belongs to.
    """
    
    content: str
    metadata: Dict[str, Any]
    id: str = None
    embeddings: Optional[List[float]] = None
    reduced_embeddings: Optional[List[float]] = None
    topic_id: Optional[int] = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())


@dataclass
class TestInput:
    """
    A input sample in a test set.
    
    Attributes:
        input: The input text.
        reference_answer: The reference answer.
        document: The document the input is based on.
        id: The ID of the input sample.
        metadata: Additional metadata about the input sample.
    """
    
    input: str
    reference_answer: str
    document: Document
    id: str = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.id is None:
            self.id = str(uuid.uuid4())
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the input sample to a dictionary.
        
        Returns:
            Dict[str, Any]: The input sample as a dictionary.
        """
        return {
            "id": self.id,
            "input": self.input,
            "reference_answer": self.reference_answer,
            "document_id": self.document.id,
            "document_content": self.document.content,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], document: Optional[Document] = None) -> 'TestInput':
        """
        Create a input sample from a dictionary.
        
        Args:
            data: The dictionary to create the input sample from.
            document: The document the input is based on.
            
        Returns:
            TestInput: The created input sample.
        """
        # If document is not provided, create a dummy document
        if document is None:
            document = Document(
                content=data.get("document_content", ""),
                metadata={},
                id=data.get("document_id"),
            )
        
        return cls(
            id=data.get("id"),
            input=data.get("input", ""),
            reference_answer=data.get("reference_answer", ""),
            document=document,
            metadata=data.get("metadata", {}),
        )


@dataclass
class TestSet:
    """
    A test set for RAG evaluation.
    
    Attributes:
        inputs: The inputs in the test set, as a list of TestInput objects.
        metadata: Additional metadata about the test set.
    """
    
    inputs: List[TestInput]
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
            input = TestInput.from_dict(q_data, document)
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
                                metadata={},
                                id=document_id
                            )
                        
                        # Create TestInput
                        metadata = {k: v for k, v in row.items() 
                                   if k not in ['input', 'reference_answer', 'document_id', 'document_content']}
                        
                        test_input = TestInput(
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
class ExampleInput:
    """
    An example input provided by the user.
    
    Attributes:
        input: The input text.
        reference_answer: Optional reference answer.
        response: Optional model response.
        feedback: Optional feedback on the response.
        metadata: Additional metadata.
    """
    
    input: str
    reference_answer: Optional[str] = None
    response: Optional[str] = None
    feedback: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "input": self.input,
            "reference_answer": self.reference_answer,
            "response": self.response,
            "feedback": self.feedback,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExampleInput':
        """Create from dictionary."""
        if isinstance(data, str):
            # Handle case where data is just a input string
            return cls(input=data)
        
        return cls(
            input=data.get("input", ""),
            reference_answer=data.get("reference_answer"),
            response=data.get("response"),
            feedback=data.get("feedback"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ExampleSet:
    """
    A collection of example inputs.
    
    Attributes:
        inputs: The example inputs. Can be:
            - A list of ExampleInput objects
            - A list of strings (inputs)
            - A list of dictionaries (structured inputs)
            - A dictionary with a "inputs" key
            - A file path (string) to a JSON file
            - A single ExampleInput object
        metadata: Additional metadata.
        remove_similar: Whether to remove similar inputs during initialization.
        similarity_threshold: Threshold for similarity detection.
        client: The API client to use for embeddings (required if remove_similar is True).
    """
    
    inputs: Union[List[ExampleInput], List[str], List[Dict[str, Any]], Dict[str, Any], str, ExampleInput]
    metadata: Dict[str, Any] = field(default_factory=dict)
    remove_similar: bool = False
    similarity_threshold: float = 0.85
    client: Optional[Any] = None
    
    def __post_init__(self):
        """Process the inputs after initialization if needed."""
        # Process the input if it's not already a list of ExampleInput objects
        if not (isinstance(self.inputs, list) and 
                all(isinstance(q, ExampleInput) for q in self.inputs)):
            self.inputs = self._process_input(self.inputs)
        
        # Remove similar inputs if requested and client is provided
        if self.remove_similar and self.client and len(self.inputs) > 1:
            self._remove_similar_inputs()
    
    def _process_input(self, input_data) -> List[ExampleInput]:
        """Convert various input formats to a list of ExampleInput objects."""
        # If it's a single ExampleInput, wrap it in a list
        if isinstance(input_data, ExampleInput):
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
        
        # If it's a list of strings, dictionaries, or ExampleInput objects
        if isinstance(input_data, list):
            return [q if isinstance(q, ExampleInput) else ExampleInput.from_dict(q) for q in input_data]
        
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
    
    def _remove_similar_inputs(self):
        """Remove similar inputs from the set."""
        from ..evaluation.generation.generator import remove_similar_inputs
        
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
                # This shouldn't happen, but just in case
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
