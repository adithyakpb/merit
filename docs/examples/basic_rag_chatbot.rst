Basic RAG Chatbot
================

This example demonstrates how to create a simple RAG (Retrieval-Augmented Generation) chatbot using MERIT.

Overview
--------

The basic RAG chatbot example shows how to:

- Set up a knowledge base with documents
- Create a simple retrieval system
- Build a chatbot that uses retrieved context
- Monitor the chatbot's performance

Code Example
-----------

.. literalinclude:: ../../examples/simple_rag_chatbot.py
   :language: python
   :caption: Basic RAG Chatbot Implementation

Running the Example
------------------

To run this example:

.. code-block:: bash

   cd examples
   python simple_rag_chatbot.py

Key Features
-----------

- **Document Loading**: Loads documents into a vector store
- **Retrieval**: Finds relevant documents for user queries
- **Generation**: Uses retrieved context to generate responses
- **Monitoring**: Tracks response quality and performance

Configuration
------------

The example uses a simple configuration:

.. code-block:: python

   config = {
       "knowledge_base": {
           "documents": ["path/to/documents"],
           "vector_store": "file"
       },
       "chatbot": {
           "model": "gpt-3.5-turbo",
           "temperature": 0.7
       }
   } 