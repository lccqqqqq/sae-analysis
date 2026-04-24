#!/usr/bin/env python3
"""
Script to fix corrupted Jupyter notebook JSON files.
"""

import json
import sys
from pathlib import Path

def fix_notebook(input_file, output_file=None):
    """Fix a corrupted notebook file."""
    if output_file is None:
        output_file = input_file
    
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    if not input_path.exists():
        print(f"Error: File {input_file} not found")
        return False
    
    print(f"Reading {input_file}...")
    
    # Try to read the file with different encodings
    content = None
    for encoding in ['utf-8', 'utf-8-sig', 'latin-1']:
        try:
            with open(input_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            print(f"Successfully read file with {encoding} encoding")
            break
        except Exception as e:
            print(f"Failed to read with {encoding}: {e}")
            continue
    
    if content is None:
        print("Error: Could not read file with any encoding")
        return False
    
    # Try to parse JSON
    try:
        data = json.loads(content)
        print("JSON is valid!")
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Error at line {e.lineno}, column {e.colno}")
        print(f"Error message: {e.msg}")
        
        # Try to fix common issues
        print("\nAttempting to fix JSON...")
        
        # Remove trailing commas before closing braces/brackets
        import re
        # Fix trailing commas
        content = re.sub(r',\s*}', '}', content)
        content = re.sub(r',\s*]', ']', content)
        
        try:
            data = json.loads(content)
            print("Fixed trailing commas - JSON is now valid!")
        except json.JSONDecodeError as e2:
            print(f"Still invalid after fixing: {e2}")
            return False
    
    # Validate notebook structure
    if 'cells' not in data:
        print("Warning: 'cells' key not found - this may not be a valid notebook")
        return False
    
    print(f"Notebook has {len(data['cells'])} cells")
    print(f"Notebook format: {data.get('nbformat', 'unknown')}.{data.get('nbformat_minor', 'unknown')}")
    
    # Write fixed notebook
    print(f"\nWriting fixed notebook to {output_file}...")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=1, ensure_ascii=False)
        print(f"Successfully wrote fixed notebook to {output_file}")
        return True
    except Exception as e:
        print(f"Error writing file: {e}")
        return False

if __name__ == "__main__":
    input_file = "feature_analysis.ipynb"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    output_file = input_file  # Overwrite by default
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    success = fix_notebook(input_file, output_file)
    sys.exit(0 if success else 1)

