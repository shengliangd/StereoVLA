from pathlib import Path

from vla_network.utils.file_manager import get_dir_ckpt, get_path_exp

PRETRAINED = "pretrained"
PREPROCESSOR_NPZ = "preprocessor.npz"

def get_dir_pretrained() -> str:
    return str(Path(get_dir_ckpt()) / PRETRAINED)

def get_path_pretrained(path: str) -> str:
    return str(Path(get_dir_pretrained()) / path)

def get_path_preprocessor(exp: str = None, exp_path: str = None) -> str:
    if exp is not None:
        exp_path = get_path_exp(exp)
    return str(Path(exp_path) / PREPROCESSOR_NPZ)
