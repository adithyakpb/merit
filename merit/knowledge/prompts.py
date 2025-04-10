"""
MERIT Knowledge Base Prompts

This module contains all the prompts used in the MERIT Knowledge Base system.
"""

from ..core.prompts import Prompt

TOPIC_GENERATION_PROMPT = Prompt("""Your task is to define the topic which best represents a set of documents.

Your are given below a list of documents and you must extract the topic best representing ALL contents.
- The topic name should be between 1 to 5 words
- Provide the topic in this language: {language}

Make sure to only return the topic name between quotes, and nothing else.

EXAMPLE:
<documents>
Camembert is a moist, soft, creamy, surface-ripened cow's milk cheese.
----------
Bleu d'Auvergne is a French blue cheese, named for its place of origin in the Auvergne region.
----------
Roquefort is a sheep milk cheese from the south of France.
</documents>

The topic is:
OUTPUT:
"French Cheese"

Now it's your turn. Here is the list of documents:

<documents>
{topics_elements}
</documents>

The topic is:
""", 
  defaults={
      "language": "en"
  }
)
