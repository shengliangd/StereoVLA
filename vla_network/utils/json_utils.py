import json
import os
from pathlib import Path

def load_json(path: str):
    """
    Load a JSON file.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Parsed JSON content
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"JSON file not found: {path}")
    
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path: str, indent=2):
    """
    Save data to a JSON file.
    
    Args:
        data: Data to save
        path: Path to save the JSON file
        indent: Indentation for pretty printing
    """
    # Ensure the directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent)
