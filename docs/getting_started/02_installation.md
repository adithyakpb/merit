# Installing MERIT

This guide provides instructions for installing MERIT and its dependencies. MERIT is a Python package that can be installed using pip, the Python package manager.

## Prerequisites

Before installing MERIT, ensure you have the following prerequisites:

- **Python**: MERIT requires Python 3.8 or later. You can check your Python version by running:
  ```bash
  python --version
  ```

- **pip**: The Python package manager. It's usually installed with Python. You can check your pip version by running:
  ```bash
  pip --version
  ```

- **Virtual Environment** (recommended): It's a good practice to install Python packages in a virtual environment to avoid conflicts with other packages. You can use `venv` (built into Python) or `conda` (if you're using Anaconda).

## Installation Options

### Option 1: Install from PyPI (Recommended)

The simplest way to install MERIT is from PyPI using pip:

```bash
pip install merit
```

This will install MERIT and its required dependencies.

### Option 2: Install from Source

If you want to install the latest development version or contribute to MERIT, you can install it from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/merit.git
   cd merit
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

   This installs MERIT in "editable" mode, meaning changes to the source code will be immediately available without reinstalling.

## Installing Optional Dependencies

MERIT has several optional dependencies for specific features:

### OpenAI Integration

To use MERIT with OpenAI's APIs:

```bash
pip install merit[openai]
```

### Visualization Support

To enable visualization features:

```bash
pip install merit[viz]
```

### Development Tools

If you're developing MERIT or running tests:

```bash
pip install merit[dev]
```

### All Optional Dependencies

To install all optional dependencies:

```bash
pip install merit[all]
```

## Verifying the Installation

To verify that MERIT is installed correctly, you can run:

```python
import merit
print(merit.__version__)
```

This should print the version of MERIT that you installed.

## Troubleshooting

### Common Issues

#### Package Not Found

If you get a "Package not found" error, ensure that:
- You're using the correct package name (`merit`)
- Your internet connection is working
- PyPI is accessible from your network

#### Dependency Conflicts

If you encounter dependency conflicts, try installing MERIT in a fresh virtual environment:

```bash
# Create a new virtual environment
python -m venv merit-env

# Activate the environment
# On Windows:
merit-env\Scripts\activate
# On macOS/Linux:
source merit-env/bin/activate

# Install MERIT
pip install merit
```

#### Permission Errors

If you get permission errors during installation, you might need to:
- Use `sudo` on Linux/macOS: `sudo pip install merit`
- Run the command prompt as administrator on Windows
- Use the `--user` flag: `pip install --user merit`

### Getting Help

If you encounter issues that aren't covered here, you can:
- Check the [GitHub Issues](https://github.com/yourusername/merit/issues) for similar problems
- Open a new issue on GitHub
- Reach out to the community for help

## Next Steps

Now that you have MERIT installed, you can:

- Read the [Key Concepts](./key_concepts.md) guide to understand MERIT's core concepts
- Follow the [Quick Start](./quick_start.md) guide to get started with MERIT
- Explore the [Tutorials](../tutorials/index.md) for step-by-step examples
