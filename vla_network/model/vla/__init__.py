from vla_network.config.define import VLAModelConfig
from .base import BaseVLA
from .assembled_vla import AssembledVLA


REGISTRY = {
    'AssembledVLA': AssembledVLA,
}


def VLA(config: VLAModelConfig) -> BaseVLA:
    return REGISTRY[config.model_type](config)
