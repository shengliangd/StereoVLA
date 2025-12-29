import os
from pathlib import Path

# Base paths for model storage
def get_model_dir() -> str:
    """
    Get the base directory for model storage.
    
    By default, uses './models' relative to the current directory,
    but can be overridden with STORAGE_PATH environment variable.
    """
    storage_path = os.environ["STORAGE_PATH"]
    return storage_path

def get_dir_ckpt() -> str:
    """
    Get the directory for model checkpoints.
    """
    return os.path.join(get_model_dir(), "ckpt")

def get_path_ckpt(exp_path: str, iter: int = None) -> str:
    """
    Get the path to a specific model checkpoint.
    
    Args:
        exp_path: Path to the experiment
        iter: Iteration number (optional)
        
    Returns:
        Path to the checkpoint file
    """
    if iter is not None:
        return os.path.join(exp_path, f"checkpoint-{iter}", "model.safetensors")
    # Get latest checkpoint if iteration not specified
    checkpoint_dirs = [d for d in os.listdir(exp_path) if d.startswith("checkpoint-")]
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {exp_path}")
    
    # Sort by iteration number
    checkpoint_dirs.sort(key=lambda x: int(x.split("-")[1]))
    latest_checkpoint = checkpoint_dirs[-1]
    
    return os.path.join(exp_path, latest_checkpoint, "model.safetensors")

def get_path_exp(exp_name: str) -> str:
    """
    Get the path to an experiment directory.
    
    Args:
        exp_name: Name of the experiment
        
    Returns:
        Path to the experiment directory
    """
    return os.path.join(get_dir_ckpt(), "exp", exp_name)

def get_path_exp_from_ckpt(ckpt_path: str) -> str:
    """
    Get the experiment path from a checkpoint path.
    
    Args:
        ckpt_path: Path to the checkpoint
        
    Returns:
        Path to the experiment directory
    """
    # Expected format: /path/to/exp/checkpoint-X/model.safetensors
    path = Path(ckpt_path)
    if path.name == "model.safetensors":
        return str(path.parent.parent)
    return str(path)

def get_path_exp_config(exp_path: str) -> str:
    """
    Get the path to the experiment configuration file.
    
    Args:
        exp_path: Path to the experiment
        
    Returns:
        Path to the configuration file
    """
    return os.path.join(exp_path, "config.json")
