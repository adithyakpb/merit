import json
# import numpy as np
# import pandas as pd
from typing import Any
from merit.core.logging import get_logger

logger = get_logger(__name__)
#TODO add a log for when the function is called
def parse_json(text: str, return_type: str = "any") -> Any:
    """
    Parse a JSON string with fallbacks for common errors, optionally extracting specific structures.
    
    Args:
        text: The JSON string to parse.
        return_type: The desired return type - "any" (default), "array", or "object"
        
    Returns:
        Any: The parsed JSON data, with type based on return_type parameter.
    """
    import re
    
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