# Training StereoVLA

This document introduces the public StereoVLA training interface. To train or fine-tune StereoVLA, implement a data loader as described below and launch the provided training script.

## 1. Prepare the environment

Follow the README setup first and make sure inference runs with the released StereoVLA checkpoint. Training uses the same environment, pretrained assets, and `STORAGE_PATH`; outputs are written under `ckpt/exp/<exp_name>` inside that storage directory.

## 2. Adapt your dataset

Use `vla_network/dataset/example_dataset.py` as a reference, implement your own data loader, and register it in `vla_network/dataset/__init__.py`. The loader should inherit `BaseDataset` and return a `VLASample` from `load_raw()`.

Required fields:

- `instruction`: natural-language task instruction.
- `images`: ordered stereo images. For the released model config, use `left` then `right`, each containing `image_steps` PIL RGB images.
- `bboxs`: ordered bounding boxes aligned with `images`. For the released training template, provide one box per configured image view/time step. If your data does not contain bounding box labels, fill zeros, but this may degrade the grounding performance.
- `proprio`: shape `(proprio_steps, D)`, usually robot state history.
- `goal`: shape `(D_goal,)`. The released template predicts a 6D goal (x, y, z, roll, pitch, yaw). If your dataset or task is not suitable to provide a goal pose, zeros can be used for compatibility, but this may reduce performance.
- `action`: shape `(action_steps, D_action)`, the future action chunk supervised in training.
- `embodiment`: string key used for action/proprio normalization statistics.

## 3. Normalization

The preprocessor computes min/max normalization statistics by sampling `normalizer_num_samples` raw `VLASample`s per embodiment. Keep the action/proprio conventions consistent within an embodiment; use a different `embodiment` string if dimensions or coordinate frames change.

## 4. Start training

Start from the template config and override paths from the command line:

```bash
torchrun --nproc_per_node=<num_gpus> -m vla_network.scripts.train_vla \
  --config configs/train_stereovla_template.json \
  --exp_name <experiment_name> \
  --train_datasets /path/to/your_dataset,<dataset_type>,1 \
  --val_datasets /path/to/val_dataset,<dataset_type>,1
```
Batch size, number of training steps, learning rate schedule, evaluation cadence, and checkpoint cadence depend on the embodiment, dataset size, and task difficulty. Treat the template config as a starting point rather than a universal recipe.

For fine-tuning from the released StereoVLA checkpoint, add `--ckpt /path/to/stereovla/checkpoint/model.safetensors` to initialize model weights before training. Batch size and training length should still be tuned for your robot embodiment and task set.

By default, fine-tuning reuses the preprocessor of the released checkpoint. If you want to recompute normalization statistics from your own dataset instead, pass `--resume_preprocessor 0`.

## 5. Optional auxiliary tasks

To co-train depth estimation or camera-parameter estimation, adapt your dataset class, or add a new one under `vla_network/dataset`, so that it can return `ValueQASample` samples.

1. Implement a dataset whose `load_raw()` returns `ValueQASample`, and set `self.is_action_dataset = False` in its constructor.
2. For depth estimation, sample a pixel and return a caption such as `the depth at (0.432, 0.615)` with value `[depth]`. We use normalized `(x, y)` image coordinates in `[0, 1]`.
3. For camera-parameter estimation, return a caption such as `the camera extrinsics and intrinsics` with values for camera pose and intrinsics.
4. Register the dataset in `vla_network/dataset/__init__.py` and mix it with the action dataset through `--train_datasets`.

For example, in the dataset loader you can construct samples like:

```python
# Depth query
sample = ValueQASample(
    dataset_name="my_depth_dataset",
    images=OrderedDict((k, v[-1]) for k, v in images.items()),
    image_key="left",
    caption=f"the depth at ({query_x / width:.3f}, {query_y / height:.3f})",
    values=np.array([depth], dtype=np.float32),
    precision=np.array([0.001], dtype=np.float32),
)

# Camera-parameter query
sample = ValueQASample(
    dataset_name="my_camera_dataset",
    images=OrderedDict((k, v[-1]) for k, v in images.items()),
    image_key="left",
    caption="the camera extrinsics and intrinsics",
    values=np.array([
        *camera_position,
        *camera_euler_xyz,
        fx, fy, cx, cy,
    ], dtype=np.float32),
    precision=np.array([
        0.001, 0.001, 0.001,
        0.01, 0.01, 0.01,
        0.1, 0.1, 0.1, 0.1,
    ], dtype=np.float32),
)
```
