#!/usr/bin/env python3
"""
Script to build MERIT documentation.
"""

import os
import subprocess
import sys
from pathlib import Path

def main():
    """Build the documentation."""
    # Change to the docs directory
    docs_dir = Path(__file__).parent / "docs"
    os.chdir(docs_dir)
    
    # Install requirements if needed
    print("Installing documentation requirements...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], check=True)
    
    # Build the documentation
    print("Building documentation...")
    subprocess.run(["make", "html"], check=True)
    
    print("Documentation built successfully!")
    print(f"Open docs/_build/html/index.html in your browser to view the documentation.")

if __name__ == "__main__":
    main() 