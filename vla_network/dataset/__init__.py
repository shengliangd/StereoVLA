import os
import pickle
import random
import tempfile
import time
from typing import List, Optional

from torch.utils.data import Dataset

from vla_network.config.define import MixDatasetConfig, VLAConfig, VLAModelConfig
from vla_network.datasample import BaseSample
from vla_network.preprocessor.preprocess import DataPreprocessor
from vla_network.utils.logger import log
from vla_network.utils.path import get_path_preprocessor
from vla_network.utils.utils import is_main_rank

from .common import BaseDataset


def init_dataset(config: VLAModelConfig, dataset_type: str, path: str, train: bool) -> BaseDataset:
    if dataset_type == "example_dataset":
        from .example_dataset import ExampleDataset

        return ExampleDataset(config=config, path=path, train=train)
    raise ValueError(
        f"Unknown dataset type: {dataset_type}. Add it to "
        "vla_network/dataset/__init__.py; see docs/training.md."
    )


class DatasetMixer(Dataset):
    datasets: List[BaseDataset]
    weights: List[float]

    def __init__(
        self,
        mix_config: MixDatasetConfig,
        config: VLAConfig,
        exp_name: str,
        train: bool,
        setup: bool,
        shuffle_buffer_size: int,
        reuse_preprocessor_from: Optional[str] = None,
    ):
        super().__init__()
        self.mix_config = mix_config
        self.config = config
        self.exp_name = exp_name
        self.train = train
        self.setup_flag = setup
        self.shuffle_buffer_size_orig = shuffle_buffer_size
        self.reuse_preprocessor_from = reuse_preprocessor_from

        self.specify_len: Optional[int] = None
        self.init_pid = os.getpid()
        self._init_datasets()

    def _init_datasets(self):
        log.info(f"Initializing DatasetMixer (PID: {os.getpid()})")
        self.datasets = []
        if not self.mix_config.datasets:
            raise ValueError("No datasets configured. Set train.train_datasets in the config or CLI.")
        for name, dataset_type in zip(self.mix_config.datasets, self.mix_config.datasets_type):
            dataset = init_dataset(self.config.model, dataset_type, name, self.train)
            self.datasets.append(dataset)

        weight_sum = sum(self.mix_config.datasets_weight)
        self.weights = [weight / weight_sum for weight in self.mix_config.datasets_weight]
        self.preprocessor = DataPreprocessor(self.config)

        num_workers = self.config.train.num_workers
        self.shuffle_buffer_size = max(self.shuffle_buffer_size_orig // max(num_workers, 1), 1)
        self._buffer: List[BaseSample] = []
        self.setup_preprocessor(self.train, self.setup_flag, self.exp_name, self.reuse_preprocessor_from)

    def _check_reinit(self):
        if os.getpid() != self.init_pid:
            self._init_datasets()
            self.init_pid = os.getpid()

    def specify_length(self, length: int):
        self.specify_len = length

    def __len__(self) -> int:
        return self.specify_len or sum(len(dataset) for dataset in self.datasets)

    def _get_processed_sample(self) -> BaseSample:
        dataset = random.choices(self.datasets, weights=self.weights)[0]
        sample = dataset.load_raw()
        return self.preprocessor.process_input(sample, normalize=True)

    def __getitem__(self, idx: Optional[int] = None) -> BaseSample:
        self._check_reinit()
        if self.train:
            while len(self._buffer) < self.shuffle_buffer_size:
                self._buffer.append(self._get_processed_sample())
            sample_idx = random.randrange(len(self._buffer))
            sample = self._buffer[sample_idx]
            self._buffer[sample_idx] = self._get_processed_sample()
            return sample
        return self._get_processed_sample()

    def setup_preprocessor(
        self,
        train: bool,
        setup: bool,
        exp_name: str,
        reuse_preprocessor_from: Optional[str],
    ):
        def get_data():
            action_datasets, action_weights = [], []
            for dataset, weight in zip(self.datasets, self.weights):
                if dataset.is_action_dataset:
                    action_datasets.append(dataset)
                    action_weights.append(weight)
            dataset = random.choices(action_datasets, weights=action_weights)[0]
            return dataset.load_raw()

        if reuse_preprocessor_from is not None:
            path = get_path_preprocessor(exp_path=reuse_preprocessor_from)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Preprocessor not found: {path}")
        else:
            path = get_path_preprocessor(exp=exp_name)
            os.makedirs(os.path.dirname(path), exist_ok=True)

        is_original_process = os.getpid() == self.init_pid
        if train and setup and is_main_rank() and is_original_process:
            if os.path.exists(path):
                with open(path, "rb") as f:
                    self.preprocessor = pickle.load(f)
                log.info("Preprocessor loaded from existing file")
            else:
                self.preprocessor.setup(get_data)

            current_path = get_path_preprocessor(exp=exp_name)
            fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(current_path), suffix=".tmp")
            with os.fdopen(fd, "wb") as f:
                pickle.dump(self.preprocessor, f)
            os.replace(tmp_path, current_path)
        else:
            while not os.path.exists(path):
                time.sleep(10)
            with open(path, "rb") as f:
                self.preprocessor = pickle.load(f)
            log.info("Preprocessor loaded successfully")
