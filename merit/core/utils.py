import json
import re
import numpy as np
from typing import Any, List, Iterable, Iterator, Tuple, TypeVar, Optional, Callable
from merit.core.logging import get_logger

logger = get_logger(__name__)

T = TypeVar('T')
U = TypeVar('U')

def detect_language(text: str) -> str:
    """
    Detect the language of a text.
    
    This function uses a simple heuristic approach to detect common languages.
    For more accurate language detection, consider using a dedicated library.
    
    Args:
        text: The text to detect the language of
        
    Returns:
        str: The language code (e.g., "en", "fr", "es")
    """
    # For now, we'll use a very simple approach
    # In a real implementation, you might want to use a library like langdetect
    
    # Check if it's empty
    if not text or not text.strip():
        return "en"  # Default to English
        
    # Simplified language detection based on common words/characters
    text = text.lower()
    
    # Check for Chinese characters
    if re.search(r'[\u4e00-\u9fff]', text):
        return "zh"
        
    # Check for Japanese characters
    if re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
        return "ja"
        
    # Check for Korean characters
    if re.search(r'[\uac00-\ud7af\u1100-\u11ff]', text):
        return "ko"
        
    # Check for Arabic characters
    if re.search(r'[\u0600-\u06ff]', text):
        return "ar"
        
    # Check for Russian/Cyrillic characters
    if re.search(r'[\u0400-\u04ff]', text):
        return "ru"
        
    # Check for common words in European languages
    spanish_words = ['el', 'la', 'los', 'las', 'y', 'es', 'por', 'que', 'con', 'para']
    french_words = ['le', 'la', 'les', 'et', 'est', 'pas', 'pour', 'dans', 'avec', 'ce']
    german_words = ['der', 'die', 'das', 'und', 'ist', 'nicht', 'für', 'mit', 'auf', 'dem']
    italian_words = ['il', 'la', 'i', 'le', 'e', 'è', 'non', 'per', 'con', 'che']
    
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count occurrences
    es_count = sum(word in spanish_words for word in words)
    fr_count = sum(word in french_words for word in words)
    de_count = sum(word in german_words for word in words)
    it_count = sum(word in italian_words for word in words)
    
    # Determine language based on highest count
    counts = [
        (es_count, "es"),
        (fr_count, "fr"),
        (de_count, "de"),
        (it_count, "it"),
    ]
    
    max_count = max(counts, key=lambda x: x[0])
    
    # If we found a reasonable number of words, return that language
    if max_count[0] >= 3:
        return max_count[1]
    
    # Default to English if no other language is detected
    return "en"


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        float: Cosine similarity between the vectors
    """
    # Ensure vectors are numpy arrays
    a = np.array(a)
    b = np.array(b)
    
    # Calculate dot product
    dot_product = np.dot(a, b)
    
    # Calculate magnitudes
    magnitude_a = np.linalg.norm(a)
    magnitude_b = np.linalg.norm(b)
    
    # Handle zero magnitudes
    if magnitude_a == 0 or magnitude_b == 0:
        return 0
    
    # Calculate cosine similarity
    return dot_product / (magnitude_a * magnitude_b)

def batch_iterator(items: Iterable[T], batch_size: int = 16, 
                  process_fn: Optional[Callable[[T], U]] = None) -> Iterator[List[T] | List[U]]:
    """
    Split an iterable into batches for efficient processing.
    
    This utility helps with batched processing which is useful for:
    - Processing large datasets in manageable chunks
    - Working with APIs that have rate limits or batch size constraints
    - Improving memory efficiency by processing data incrementally
    
    Args:
        items: The iterable to batch (list, array, DataFrame, etc.)
        batch_size: The size of each batch
        process_fn: Optional function to apply to each item in the batch
        
    Returns:
        Iterator: An iterator yielding batches of items
        
    Examples:
        # Simple batching
        for batch in batch_iterator(['a', 'b', 'c', 'd', 'e'], batch_size=2):
            print(batch)  # ['a', 'b'], then ['c', 'd'], then ['e']
            
        # With processing function
        for batch in batch_iterator(range(5), batch_size=2, process_fn=lambda x: x*2):
            print(batch)  # [0, 2], then [4, 6], then [8]
    """
    batch = []
    for i, item in enumerate(items):
        if process_fn is not None:
            item = process_fn(item)
            
        batch.append(item)
        
        if len(batch) == batch_size or i == len(items) - 1:
            yield batch
            batch = []

def parse_json(text: str = None, file_path: str = None, return_type: str = "any") -> Any:
    """
    Parse a JSON string with fallbacks for common errors, optionally extracting specific structures.
    
    Args:
        text: The JSON string to parse.
        file_path: Path to a JSON file to parse.
        return_type: The desired return type - "any" (default), "array", or "object"
        
    Returns:
        Any: The parsed JSON data, with type based on return_type parameter.
        
    Raises:
        ValueError: If both text and file_path are provided.
        FileNotFoundError: If the specified file path does not exist.
    """
    if file_path and text:
        error_msg = "Cannot provide both text and file_path. Use only one parameter."
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    if file_path and not text:
        logger.debug(f"Parsing JSON file: {file_path} with return type '{return_type}'")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        except FileNotFoundError:
            logger.error(f"JSON file not found: {file_path}")
            raise
        except PermissionError:
            logger.error(f"Permission denied when accessing JSON file: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading JSON file {file_path}: {str(e)}")
            raise
            
    if not text:
        logger.warning("No JSON text or file provided.")
        return [] if return_type == "array" else {}
        
    logger.debug(f"Parsing JSON text with return type '{return_type}'")
    # Helper function to handle return type conversion
    def handle_return_type(parsed_data):
        if return_type == "array" and not isinstance(parsed_data, list):
            # If result is a dict, try to extract an array from it
            if isinstance(parsed_data, dict):
                for value in parsed_data.values():
                    if isinstance(value, list):
                        return value
            # If we couldn't extract an array, return empty list
            return []
        elif return_type == "object" and not isinstance(parsed_data, dict):
            # If we wanted an object but got something else, return empty dict
            return {}
        else:
            # Return whatever we parsed
            return parsed_data
    
    # Try direct parsing first
    try:
        parsed = json.loads(text)
        return handle_return_type(parsed)
    except json.JSONDecodeError:
        pass
    
    # Try extracting from markdown code block
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group(1))
            return handle_return_type(parsed)
        except json.JSONDecodeError:
            pass
    
    # Try fixing missing quotes around keys
    fixed_text = re.sub(r'(\s*?)(\w+)(\s*?):', r'\1"\2"\3:', text)
    try:
        parsed = json.loads(fixed_text)
        return handle_return_type(parsed)
    except json.JSONDecodeError:
        pass
    
    # Try fixing single quotes
    fixed_text = text.replace("'", '"')
    try:
        parsed = json.loads(fixed_text)
        return handle_return_type(parsed)
    except json.JSONDecodeError:
        pass
    
    # If we're looking for an array specifically, try regex as last resort
    if return_type == "array":
        array_match = re.search(r"\[\s*.*?\s*\]", text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group(0))
            except json.JSONDecodeError:
                pass
        logger.error(f"Failed to extract JSON array from: {text[:100]}...")
        return []
    
    # Return appropriate empty value based on return_type
    logger.warning("Failed to parse JSON even with fallbacks.")
    return [] if return_type == "array" else {}
