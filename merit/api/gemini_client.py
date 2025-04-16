"""
Gemini API Client for MERIT.

This module provides a client for the Google Gemini API that implements
the AIAPIClient interface.
"""
from typing import List, Dict, Any, Optional, Union

from google import genai
from google.genai import types

from .client import AIAPIClient

class GeminiClient(AIAPIClient):
    """
    A client for the Google Gemini API.
    
    This client implements the AIAPIClient interface and uses the
    Google Generative AI (genai) library to interact with the Gemini models.
    """
    
    def __init__(
        self,
        api_key: str,
        generation_model: str = "gemini-2.0-flash",
        embedding_model: str = "gemini-embedding-exp-03-07",
        max_output_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 40,
    ):
        """
        Initialize the Gemini client.
        
        Args:
            api_key: The API key for the Gemini API.
            generation_model: The model to use for text generation.
            embedding_model: The model to use for embeddings.
            max_output_tokens: The maximum number of tokens to generate.
            temperature: The temperature for text generation.
            top_p: The top-p value for text generation.
            top_k: The top-k value for text generation.
        """
        self.client = genai.Client(api_key=api_key)
        self.generation_model = generation_model
        self.embedding_model = embedding_model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
    
    def generate_text(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate text based on a prompt.
        
        Args:
            prompt: The prompt to generate text from.
            max_tokens: The maximum number of tokens to generate.
            temperature: The temperature for text generation.
            system_prompt: A system prompt to use.
            
        Returns:
            str: The generated text.
        """
        # Set up the generation config
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens or self.max_output_tokens,
            temperature=temperature or self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        
        # Add system prompt if provided
        if system_prompt:
            config.system_instruction = system_prompt
        
        # Generate the content
        response = self.client.models.generate_content(
            model=self.generation_model,
            contents=prompt,
            config=config
        )
        
        # Return the generated text
        return response.text
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """
        Get embeddings for the given texts.
        
        Args:
            texts: A string or list of strings to get embeddings for.
            
        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
            For a single input text, still returns a list containing one embedding.
        """
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for text in texts:
            result = self.client.models.embed_content(
                model=self.embedding_model,
                contents=text
            )
            
            # Convert ContentEmbedding to a list of floats
            if hasattr(result.embeddings, 'values'):
                # If it's a ContentEmbedding object with a 'values' attribute
                if hasattr(result.embeddings.values, 'tolist'):
                    embedding_values = result.embeddings.values.tolist()
                else:
                    embedding_values = result.embeddings.values
            elif isinstance(result.embeddings, list) and len(result.embeddings) == 1:
                # If it's a list containing a single embedding
                if hasattr(result.embeddings[0], 'values'):
                    # If the embedding has a 'values' attribute
                    if hasattr(result.embeddings[0].values, 'tolist'):
                        embedding_values = result.embeddings[0].values.tolist()
                    else:
                        embedding_values = result.embeddings[0].values
                else:
                    # If the embedding is already a list
                    embedding_values = result.embeddings[0]
            else:
                # Fallback to using the embeddings as is
                embedding_values = result.embeddings
            
            embeddings.append(embedding_values)
        
        return embeddings
