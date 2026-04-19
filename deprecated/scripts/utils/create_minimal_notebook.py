#!/usr/bin/env python3
"""
Create a minimal valid notebook from the existing notebook structure.
"""

import json
import sys

input_file = "feature_analysis.ipynb"
output_file = "feature_analysis_minimal.ipynb"

print(f"Reading {input_file}...")
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Original notebook: {len(data['cells'])} cells")
print(f"Format: {data.get('nbformat')}.{data.get('nbformat_minor')}")

# Create minimal notebook with just structure
minimal_notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

# Copy cells but ensure clean structure
for i, cell in enumerate(data['cells']):
    new_cell = {
        "cell_type": cell.get("cell_type", "code"),
        "metadata": {},
        "source": cell.get("source", [])
    }
    
    # Ensure source is a list
    if isinstance(new_cell["source"], str):
        new_cell["source"] = new_cell["source"].splitlines(keepends=True)
    
    # For code cells, add empty outputs
    if new_cell["cell_type"] == "code":
        new_cell["execution_count"] = None
        new_cell["outputs"] = []
    else:
        # For markdown cells, ensure metadata is minimal
        pass
    
    minimal_notebook["cells"].append(new_cell)

print(f"Created minimal notebook with {len(minimal_notebook['cells'])} cells")

# Write minimal notebook
print(f"Writing to {output_file}...")
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(minimal_notebook, f, indent=1, ensure_ascii=False)

print(f"Successfully created {output_file}")
print(f"File size: {Path(output_file).stat().st_size / 1024:.2f} KB")

