from urchin import URDF, Collision, Sphere, Geometry

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
from transformers import Trainer
from transformers.trainer_utils import get_last_checkpoint

from vla_network.config.define import VLAConfig
from vla_network.dataset import DatasetMixer
from vla_network.model.vla import BaseVLA, VLA
from vla_network.utils.file_manager import get_path_exp, get_path_exp_config
from vla_network.utils.json_utils import load_json, save_json
from vla_network.utils.logger import log
from vla_network.utils.utils import is_main_rank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a StereoVLA training JSON config.")
    parser.add_argument("--exp_name", default=None)
    parser.add_argument("--ckpt", default=None, help="Optional checkpoint used to initialize/fine-tune the model.")
    parser.add_argument("--resume_from", default=None, help="Optional Trainer checkpoint directory to resume from.")
    parser.add_argument("--train_datasets", default=None, help="Format: /path/to/data,dataset_type,weight[;/path2,type2,weight2].")
    parser.add_argument("--val_datasets", default=None)
    parser.add_argument("--global_bs", type=int, default=None)
    parser.add_argument("--device_bs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_step", type=int, default=None)
    parser.add_argument("--eval_step", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--shuffle_buffer_size", type=int, default=None)
    parser.add_argument("--normalizer_num_samples", type=int, default=None)
    parser.add_argument("--eval_each", type=int, default=None)
    parser.add_argument("--resume_preprocessor", type=int, choices=[0, 1], default=None)
    return parser


def apply_overrides(config: VLAConfig, args: argparse.Namespace) -> VLAConfig:
    if args.exp_name is not None:
        config.train.exp_name = args.exp_name
    if args.ckpt is not None:
        config.model.ckpt = args.ckpt
    if args.resume_from is not None:
        config.train.resume_from = args.resume_from
    if args.train_datasets is not None:
        config.train.train_datasets = args.train_datasets
    if args.val_datasets is not None:
        config.train.val_datasets = args.val_datasets
    if args.global_bs is not None:
        config.train.global_batch_size = args.global_bs
    if args.device_bs is not None:
        config.train.device_batch_size = args.device_bs
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.save_step is not None:
        config.train.save_step = args.save_step
    if args.eval_step is not None:
        config.train.eval_step = args.eval_step
    if args.num_workers is not None:
        config.train.num_workers = args.num_workers
    if args.shuffle_buffer_size is not None:
        config.train.shuffle_buffer_size = args.shuffle_buffer_size
    if args.normalizer_num_samples is not None:
        config.train.normalizer_num_samples = args.normalizer_num_samples
    if args.eval_each is not None:
        config.train.eval_each = args.eval_each
    if args.resume_preprocessor is not None:
        config.train.resume_preprocessor = bool(args.resume_preprocessor)
    config.train.setup()
    return config


def load_config(config_path: str, args: argparse.Namespace) -> VLAConfig:
    config = VLAConfig.model_validate(load_json(config_path))
    return apply_overrides(config, args)


def save_config(config: VLAConfig):
    if not is_main_rank():
        return
    exp_root = get_path_exp(config.train.exp_name)
    Path(exp_root).mkdir(parents=True, exist_ok=True)
    config_dict = config.model_dump(mode="json", exclude={"train": {"args"}})
    config_dict["train"]["args"] = config.train.args.to_dict()
    save_json(config_dict, get_path_exp_config(exp_root))


def resolve_resume_checkpoint(config: VLAConfig) -> Optional[str]:
    if config.train.resume_from:
        return config.train.resume_from
    output_dir = config.train.args.output_dir
    if os.path.isdir(output_dir):
        return get_last_checkpoint(output_dir)
    return None


def main():
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config, args)
    save_config(config)

    model: BaseVLA = VLA(config.model)
    model.requires_grad_(True)

    checkpoint_dir = resolve_resume_checkpoint(config)
    if checkpoint_dir is not None:
        log.info(f"Resuming training from {checkpoint_dir}")

    if checkpoint_dir is not None and config.train.resume_preprocessor:
        reuse_preprocessor_from = os.path.dirname(checkpoint_dir)
    elif config.model.ckpt is not None and config.train.resume_preprocessor:
        reuse_preprocessor_from = os.path.dirname(os.path.dirname(config.model.ckpt))
    else:
        reuse_preprocessor_from = None

    train_dataset = DatasetMixer(
        config.train.train_datasets,
        config,
        config.train.exp_name,
        train=True,
        setup=True,
        reuse_preprocessor_from=reuse_preprocessor_from,
        shuffle_buffer_size=config.train.shuffle_buffer_size,
    )
    if config.train.val_datasets is None:
        config.train.val_datasets = config.train.train_datasets
    val_dataset = DatasetMixer(
        config.train.val_datasets,
        config,
        config.train.exp_name,
        train=False,
        setup=False,
        reuse_preprocessor_from=reuse_preprocessor_from,
        shuffle_buffer_size=0,
    )
    if config.train.full_bf16:
        model = model.to(torch.bfloat16)
    val_dataset.specify_length(config.train.eval_each * config.train.global_batch_size)

    trainer = Trainer(
        model=model,
        args=config.train.args,
        train_dataset=train_dataset,
        data_collator=model.collate,
        eval_dataset=val_dataset,
    )

    log.info("Start training")
    trainer.train(resume_from_checkpoint=checkpoint_dir)


if __name__ == "__main__":
    main()
