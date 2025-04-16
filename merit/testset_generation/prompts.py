from ..core.prompts import Prompt

INPUT_STYLE_ANALYSIS_PROMPT = Prompt("""You are an AI assistant helping to analyze the style of example inputs for a LLM-powered system.

I will provide you with a set of example inputs, and I want you to analyze their style, grammar, and linguistic patterns.

The inputs are:
{example_inputs}

Provide a detailed analysis of:
1. Structure patterns
2. Grammatical patterns
3. Vocabulary and formality level
4. Distinctive style elements
5. Word usage patterns across all inputs

Format your response as a JSON object with these keys:
- structure_patterns: Array of common structural patterns
- grammar_patterns: Array of common grammatical structures
- vocabulary_level: String describing vocabulary (e.g., "technical", "conversational")
- formality_level: Number from 0-1 where 0 is very informal and 1 is very formal
- distinctive_elements: Array of distinctive style elements to replicate
- example_templates: Array of templates that capture the input patterns
- linguistic_features: Object with additional linguistic observations
- common_phrases: Array of frequently used phrases or word combinations
- notes: Additional notes or observations.
""")

ADAPTIVE_TEST_INPUT_GENERATION_PROMPT = Prompt("""You are an AI assistant helping to generate inputs for evaluating a RAG system.

I will provide you with:
1. A document
2. {example_type} for style reference
3. Style characteristics to match

Your task is to generate {num_inputs} inputs that:
1. Are directly related to the document content
2. Match the style, grammar, and patterns of the example inputs
3. Address specific information in the document

Here is the document:
<document>
{document_content}
</document>

{example_section}

Style characteristics to match:
{style_guidance}

Generate {num_inputs} inputs based on this document. Format your response as a JSON array of Input strings, like this:
{
  "test_inputs":
  [
    {
      "test_input": "<Input 1>?",
      "test_input_type": "<input_type>"
    },
    {
      "test_input": "<Input 2>?",
      "test_input_type": "<input_type>"
    },
  ]
}
""", 
  defaults={
      "num_inputs": 3,
      "example_type": "Example inputs",
  }
)

TEST_INPUT_GENERATION_PROMPT = Prompt("""You are an AI assistant helping to generate inputs for evaluating a RAG (Retrieval-Augmented Generation) system.

The description of the RAG system and its domain is given as: {system_description}
I will provide you with a document, and I want you to generate {num_inputs} diverse inputs that could be answered using the information in this document.

The generated inputs should be:
1. Directly answerable from the document
2. Diverse in their type and complexity
3. Natural and conversational in tone
4. In the {language} language

Here is the document:

<document>
{document_content}
</document>

Generate {num_inputs} inputs based on this document. Format your response as a JSON array of strings, like this:
{
  "test_inputs":
  [
    {
      "test_input": "<Input 1>?",
      "test_input_type": "<input_type>"
    },
    {
      "test_input": "<Input 2>?",
      "test_input_type": "<input_type>"
    },
  ]
}
""", 
  defaults={
      "system_description": "A customer support chatbot that answers inputs based on a knowledge base.",
      "num_inputs": 3,
      "language": "en"
  }
)

REFERENCE_ANSWER_GENERATION_PROMPT = Prompt("""You are an AI assistant helping to generate reference answers for evaluating a LLM powered system.

I will provide you with a document and a input, and I want you to generate a comprehensive and accurate answer to the input based solely on the information in the document.

The answer should be:
1. Directly based on the information in the document
2. Comprehensive and complete, without ambiguity
3. Accurate and factual
4. In the {language} language
{style_guidance}

Here is the document:

<document>
{document_content}
</document>

Here is the input:

<input>
{test_input}
</input>

Provide a comprehensive answer to the input based solely on the information in the document. If the input cannot be answered based on the document, state that clearly.
""", 
  defaults={
      "language": "en",
      "style_guidance": ""
  }
)
