from collections import OrderedDict

import numpy as np
from PIL import Image

from vla_network.config.define import VLAModelConfig
from vla_network.datasample import VLASample

from .common import BaseDataset


class ExampleDataset(BaseDataset):
    def __init__(self, config: VLAModelConfig, path: str, train: bool):
        super().__init__(config=config, path=path, train=train)
        self._zero_image = Image.new("RGB", (config.image_size, config.image_size), color=(0, 0, 0))

    def load_raw(self) -> VLASample:
        image_steps = self.config.image_steps
        image_keys = self.config.image_keys or ["left", "right"]
        goal_dim = self.config.max_goal_dim or self.config.max_action_dim

        return VLASample(
            dataset_name="example_dataset_dummy",
            embodiment="dummy_robot",
            instruction="pick up red cube",
            images=OrderedDict(
                (key, [self._zero_image.copy() for _ in range(image_steps)])
                for key in image_keys
            ),
            bboxs=OrderedDict(
                (key, [np.zeros((4,), dtype=np.float32) for _ in range(image_steps)])
                for key in image_keys
            ),
            proprio=np.zeros((self.config.proprio_steps, self.config.max_proprio_dim), dtype=np.float32),
            goal=np.zeros((goal_dim,), dtype=np.float32),
            action=np.zeros((self.config.action_steps, self.config.max_action_dim), dtype=np.float32),
        )
