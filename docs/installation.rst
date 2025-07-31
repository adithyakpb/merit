Installation
============

Prerequisites
------------

MERIT requires Python 3.8 or higher. We recommend using a virtual environment for installation.

Installation Options
-------------------

Using pip
~~~~~~~~~

The simplest way to install MERIT is using pip:

.. code-block:: bash

   pip install merit-ai

For development installation with all dependencies:

.. code-block:: bash

   pip install merit-ai[all]

Installation with Specific Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can install MERIT with specific feature sets:

.. code-block:: bash

   # Core installation
   pip install merit-ai

   # With OpenAI support
   pip install merit-ai[openai]

   # With Google AI support
   pip install merit-ai[google]

   # With analysis tools
   pip install merit-ai[analysis]

   # With development tools
   pip install merit-ai[dev]

   # With documentation tools
   pip install merit-ai[docs]

From Source
~~~~~~~~~~

To install from source:

.. code-block:: bash

   git clone https://github.com/your-username/merit.git
   cd merit
   pip install -e .

Verification
-----------

After installation, you can verify that MERIT is working correctly:

.. code-block:: python

   import merit
   print(merit.__version__)

Or using the command line:

.. code-block:: bash

   merit --help 