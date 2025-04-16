"""
MERIT RAG Prompts

This module contains all the prompts used in the MERIT RAG system. All prompts
are exposed as variables that can be customized by the user.
"""

import inspect
import sys

def get_all_prompts():
    """
    Get all Prompt objects defined at runtime in this module.
    
    Returns:
        dict: Dictionary of Prompt objects with their variable names as keys.
    """
    module = sys.modules[__name__]
    return {name: value for name, value in inspect.getmembers(module)
            if isinstance(value, Prompt) and name.isupper()}

class Prompt(str):
    """
    A string subclass specifically for prompts that adds a safe_format method.
    
    This class inherits from str, so it has all the properties and methods of a string,
    plus the additional safe_format method and default values support.
    
    Args:
        content: The prompt string content.
        defaults: Optional dictionary of default values for variables in the prompt.
    """
    def __new__(cls, content, defaults=None):
        instance = super().__new__(cls, content)
        instance.defaults = defaults or {}
        return instance
    
    def safe_format(self, **kwargs):
        """
        A safe version of str.format() that doesn't raise KeyError for missing keys,
        handles invalid format specifiers, and uses default values if provided.
        
        Args:
            **kwargs: The keyword arguments to format the string with.
            
        Returns:
            str: The formatted string.
        """
        import re
        result = self
        
        # First, identify all format placeholders in the string
        format_placeholders = re.findall(r'\{([^{}:]+)(?::[^{}]*)?\}', result)
        
        # Process each placeholder according to these rules:
        # 1. If the key is in kwargs, use that value (highest priority)
        # 2. If the key is in defaults but not in kwargs, use the default value
        # 3. If the key is neither in kwargs nor defaults, leave it unchanged
        for key in format_placeholders:
            pattern = r'\{' + re.escape(key) + r'(?::[^{}]*)?\}'
            
            if key in kwargs:
                replacement = str(kwargs[key])
                result = re.sub(pattern, lambda m: replacement, result)
            elif key in self.defaults:
                replacement = str(self.defaults[key])
                result = re.sub(pattern, lambda m: replacement, result)
        
        return result
    
    def get_variables(self):
        """
        Extract all format placeholders (variables) from the prompt string.
        
        Returns:
            list: List of variable names found in the prompt string.
        """
        import re
        # Find all format placeholders in the string
        # Pattern matches {variable_name} or {variable_name:format_spec}
        format_placeholders = re.findall(r'\{([^{}:]+)(?::[^{}]*)?\}', self)
        
        # Return unique variable names
        return list(set(format_placeholders))