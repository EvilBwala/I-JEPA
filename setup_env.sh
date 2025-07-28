#!/bin/bash
# Setup script for brain_jepa environment

echo "Setting up brain_jepa conda environment..."

# Create conda environment from yml file
conda env create -f environment.yml

echo "Environment created successfully!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate brain_jepa"
echo ""
echo "If you prefer using pip instead of conda, you can:"
echo "  1. Create a virtual environment: python -m venv venv"
echo "  2. Activate it: source venv/bin/activate  # On Windows: venv\\Scripts\\activate"
echo "  3. Install dependencies: pip install -r requirements.txt"
echo ""
echo "Note: Adjust CUDA version in environment.yml if needed (currently set to 11.8)"