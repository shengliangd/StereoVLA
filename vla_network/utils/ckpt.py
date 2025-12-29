from safetensors.torch import load_file
from torch import nn


def load_safetensors(path: str) -> dict:
    return load_file(path)


def load_model(model: nn.Module, ckpt_path: str) -> nn.Module:
    ckpt = load_safetensors(ckpt_path)
    model.load_state_dict(ckpt)
    return model
