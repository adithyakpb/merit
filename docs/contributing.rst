Contributing to MERIT
====================

Thank you for your interest in contributing to MERIT! This document provides guidelines for contributing to the project.

Getting Started
--------------

1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a virtual environment**:

   .. code-block:: bash

      python -m venv merit-venv
      source merit-venv/bin/activate  # On Windows: merit-venv\Scripts\activate

4. **Install development dependencies**:

   .. code-block:: bash

      pip install -e ".[dev]"

Development Setup
----------------

Install pre-commit hooks for code quality:

.. code-block:: bash

   pre-commit install

Running Tests
------------

Run the test suite:

.. code-block:: bash

   pytest

Run with coverage:

.. code-block:: bash

   pytest --cov=merit

Code Style
----------

MERIT uses several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

Run all style checks:

.. code-block:: bash

   black merit/
   isort merit/
   flake8 merit/
   mypy merit/

Documentation
------------

To build the documentation:

.. code-block:: bash

   cd docs
   make html

The documentation will be built in `docs/_build/html/`.

Pull Request Process
-------------------

1. **Create a feature branch** from `main`
2. **Make your changes** following the coding standards
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Run the test suite** to ensure everything works
6. **Submit a pull request** with a clear description

Commit Message Format
--------------------

Use conventional commit messages:

.. code-block::

   feat: add new metric for response time
   fix: resolve issue with MongoDB connection
   docs: update installation instructions
   test: add unit tests for BaseMetric class

Issue Reporting
--------------

When reporting issues, please include:

- **Description** of the problem
- **Steps to reproduce**
- **Expected behavior**
- **Actual behavior**
- **Environment details** (OS, Python version, etc.)
- **Error messages** if applicable

Code of Conduct
---------------

We are committed to providing a welcoming and inspiring community for all. Please be respectful and inclusive in all interactions.

Contact
-------

If you have questions about contributing, please:

- Open an issue on GitHub
- Join our community discussions
- Reach out to the maintainers 