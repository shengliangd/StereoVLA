from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from vla_network.config.define import VLAModelConfig
from vla_network.datasample import BaseSample


def get_frame_indices(
    data_cfg: VLAModelConfig,
    obs_traj: np.ndarray,
    dt_steps: int,
    frame: int,
) -> Tuple[int, List[int], List[int], List[int]]:
    length = len(obs_traj)

    def clip(index: int) -> int:
        return min(length - 1, max(0, index))

    state_indices = [
        clip(frame + i * dt_steps)
        for i in range(-data_cfg.proprio_steps + 1, 1)
    ]
    ref_state_indices = [
        clip(frame + i * dt_steps)
        for i in range(data_cfg.action_steps)
    ]
    action_indices = [
        clip(frame + i * dt_steps + (dt_steps - 1))
        for i in range(data_cfg.action_steps)
    ]
    return frame, state_indices, ref_state_indices, action_indices


class BaseDataset(ABC):
    config: VLAModelConfig

    def __init__(self, config: VLAModelConfig, path: str, train: bool):
        self.config = config
        self.path = path
        self.train = train
        self.is_action_dataset = True

    def __len__(self) -> int:
        return int(1e7)

    @abstractmethod
    def load_raw(self) -> BaseSample:
        raise NotImplementedError()


def cycle_iter(iterator):
    while True:
        for item in iterator:
            yield item
