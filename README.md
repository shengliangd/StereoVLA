# StereoVLA: Enhancing Vision-Language-Action Models with Stereo Vision

[![arXiv](https://img.shields.io/badge/arXiv-2505.03233-df2a2a.svg)](https://arxiv.org/pdf/2512.21970)
[![Static Badge](https://img.shields.io/badge/Project-Page-a)](https://shengliangd.github.io/StereoVLA-Webpage/)

Stereo cameras closely mimic human binocular vision, providing rich spatial cues critical for precise robotic manipulation. Despite their advantage, the adoption of stereo vision in vision-language-action models (VLAs) remains underexplored. In this work, we present StereoVLA, a VLA model that leverages rich geometric cues from stereo vision. We propose a novel Geometric-Semantic Feature Extraction module that utilizes vision foundation models to extract and fuse two key features: 1\) geometric features from subtle stereo-view differences for spatial perception; 2\) semantic-rich features from the monocular view for instruction following. Additionally, we propose an auxiliary Interaction-Region Depth Estimation task to further enhance spatial perception and accelerate model convergence. Extensive experiments show that our approach outperforms baselines by a large margin in diverse tasks under the stereo setting, and demonstrates strong robustness to camera pose variations.

## Updates

- [2025/12/29] Release the paper and model.

## Model Server

### Environment Setup

1. Create a directory to place the model code and weight in the subsequent steps, and cd to it.

2. Create a new conda environment:
   ```bash
   conda create -p ./env python=3.12
   conda activate ./env
   ```

3. Download the model weights with `hf download --local-dir ./storage shengliangd/StereoVLA`.

4. Clone this repo, cd to the cloned directory, and install dependencies with `pip install -r requirements.txt`.

5. Clone our modified FoundationStereo repo and install with `pip install -e .`.

### Running the Server

To run the model server:

```bash
STORAGE_PATH=`readlink -f ../storage` python -m vla_network.scripts.serve --path ../storage/stereovla/checkpoint/model.safetensors --port 6666
```

For faster inference, add `--compile` to the command. It will speed up the inference around 50\% with a cost of slower model loading.

## Real-World Control Interface

Setup real-world controller following [this repo](https://github.com/shengliangd/StereoVLA-real-world-controller).

## Citation

```bibtex
@misc{deng2025stereovlaenhancingvisionlanguageactionmodels,
      title={StereoVLA: Enhancing Vision-Language-Action Models with Stereo Vision}, 
      author={Shengliang Deng and Mi Yan and Yixin Zheng and Jiayi Su and Wenhao Zhang and Xiaoguang Zhao and Heming Cui and Zhizheng Zhang and He Wang},
      year={2025},
      eprint={2512.21970},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2512.21970}, 
}
```

[![License](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](LICENSE)

