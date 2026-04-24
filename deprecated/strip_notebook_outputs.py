#!/usr/bin/env python3
"""
Script to remove outputs from a Jupyter notebook to reduce file size.
"""

import json
import sys
from pathlib import Path

def strip_outputs(input_file, output_file=None):
    """Remove outputs from notebook cells to reduce file size."""
    if output_file is None:
        output_file = input_file.replace('.ipynb', '_no_outputs.ipynb')
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: File {input_file} not found")
        return False
    
    print(f"Reading {input_file}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    original_size = input_path.stat().st_size
    print(f"Original file size: {original_size / 1024 / 1024:.2f} MB")
    print(f"Number of cells: {len(data['cells'])}")
    
    # Remove outputs from all cells
    cells_modified = 0
    for cell in data['cells']:
        if 'outputs' in cell:
            if len(cell['outputs']) > 0:
                cell['outputs'] = []
                cells_modified += 1
        if 'execution_count' in cell:
            cell['execution_count'] = None
        if 'metadata' in cell:
            # Keep metadata but remove execution-related metadata if needed
            pass
    
    print(f"Removed outputs from {cells_modified} cells")
    
    # Write cleaned notebook
    print(f"Writing cleaned notebook to {output_file}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=1, ensure_ascii=False)
    
    new_size = output_path.stat().st_size
    print(f"New file size: {new_size / 1024 / 1024:.2f} MB")
    print(f"Size reduction: {(original_size - new_size) / 1024 / 1024:.2f} MB ({100 * (original_size - new_size) / original_size:.1f}%)")
    print(f"Successfully created {output_file}")
    return True

if __name__ == "__main__":
    input_file = "feature_analysis.ipynb"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    output_file = None
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    success = strip_outputs(input_file, output_file)
    sys.exit(0 if success else 1)

