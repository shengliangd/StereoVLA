import torch
import torch.nn.functional as F
import copy
import os
from typing import List, Union, Tuple

from typing import Optional

from vla_network.config.define import ImageTransform

import torchvision.transforms as tvtf

from . import Backbone2D, Backbone2DConfig
from .dinosiglip_vit import DINOSigLIP_NAMES, ViT, CombineImageTransform
from .common import create_timm_vit_and_transform
from foundation_stereo.core.foundation_stereo import FoundationStereo
from omegaconf import OmegaConf

from vla_network.utils.path import get_path_pretrained


class FoundationStereoViT(torch.nn.Module):
    def __init__(self, model: FoundationStereo, feature: str) -> None:
        super().__init__()
        self.model = model
        self.feature = feature

    @property
    def embed_dim(self) -> int:
        return {"vcorr": 56, "vcp": 2912, "vc": 3328, "lookup": 522}[self.feature]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (Bx2, C, H, W)
        returns:
            (B, N, embed_dim)
        """
        # reshape to (B, 2, C, H, W)
        x_b_2_c_h_w = x.view(-1, 2, *x.shape[1:])
        left_img = x_b_2_c_h_w[:, 0]  # (B, C, H, W)
        right_img = x_b_2_c_h_w[:, 1]  # (B, C, H, W)
        
        # Pass stereo pair through foundation stereo model
        return self.model(left_img, right_img, feature=self.feature, test_mode=True)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def create_foundation_stereo_and_transform(ckpt_path: str, image_size: int, feature: Optional[str]=None) -> tuple[ViT, ImageTransform]:
    cfg = OmegaConf.load(f'{os.path.dirname(ckpt_path)}/cfg.yaml')
    args = OmegaConf.create(cfg)

    model = FoundationStereo(args)
    ckpt = torch.load(ckpt_path, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    model = FoundationStereoViT(model, feature)

    transform = tvtf.Compose([
        tvtf.Resize(image_size),
        tvtf.ToTensor()
    ])

    return model, transform


class DINOSigLIPStereoBackbone(Backbone2D):
    """
    Vision encoder utilizing SigLIP and FoundationStereo.
    Input: left_rgb, right_rgb.
    Output: SigLIP(left_rgb) & FoundationStereo(left_rgb, right_rgb), concatenated along the feature dimension.
    """
    # from parent class
    config: Backbone2DConfig
    image_transform: CombineImageTransform

    models: List[str]
    siglip: ViT
    stereo: ViT

    def __init__(self, config: Backbone2DConfig, stereo_feature, pooling=True) -> None:
        super().__init__(config)
        # Initialize SigLIP model
        name_siglip = DINOSigLIP_NAMES[config.image_size]["siglip"]
        self.siglip, transform_siglip = create_timm_vit_and_transform(name_siglip, config.image_size)

        name_dino = DINOSigLIP_NAMES[config.image_size]["dino"]
        self.dino, transform_dino = create_timm_vit_and_transform(name_dino, config.image_size)
        
        # Initialize Stereo model
        self.feature = stereo_feature
        self.stereo, transform_stereo = create_foundation_stereo_and_transform(
            f'{get_path_pretrained("foundation_stereo")}/model_best_bp2.pth',
            config.image_size,
            stereo_feature
        )
        self.pooling = pooling
        
        # Combine transforms
        self.image_transform = CombineImageTransform([transform_siglip, transform_dino, transform_stereo])

        # Optional similarity loss functionality for backward compatibility
        self.enable_similarity_loss = getattr(config, 'enable_similarity_loss', False)
        if self.enable_similarity_loss:
            # Create a frozen copy of DINO for similarity loss
            self.dino_fixed = copy.deepcopy(self.dino)
            self.similarity_loss_weight = config.similarity_loss_weight
            self.similarity_loss_type = config.similarity_loss_type

    @property
    def feature_dim(self) -> int:
        return self.siglip.embed_dim + self.dino.embed_dim + self.stereo.embed_dim

    def compute_similarity_loss(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        """Compute similarity loss between two feature tensors"""
        if self.similarity_loss_type == "cosine":
            # Cosine similarity loss
            feat1_norm = F.normalize(feat1, dim=-1)
            feat2_norm = F.normalize(feat2, dim=-1)
            similarity = torch.sum(feat1_norm * feat2_norm, dim=-1)
            loss = 1.0 - similarity.mean()
        else:  # l2
            # L2 distance loss
            loss = F.mse_loss(feat1, feat2)
        
        return loss * self.similarity_loss_weight

    def forward(self, images: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, n, _, *chw = images.shape
        assert n == 2, "expected stereo pair of images"
        
        # Process with SigLIP (only uses left image)
        siglip_feat = self.siglip(images[:, :1, 0].reshape(b * 1, *chw))
        siglip_feat = siglip_feat.reshape(b, 16, 16, siglip_feat.shape[-1])

        dino_feat = self.dino(images[:, :1, 1].reshape(b * 1, *chw))
        dino_feat = dino_feat.reshape(b, 16, 16, dino_feat.shape[-1])
        
        # Process with Stereo (uses both images)
        with torch.set_grad_enabled(False):
            stereo_feat = self.stereo(images[:, :, 2].reshape(b * n, *chw)*255)
        if self.pooling:
            # Use adaptive avg pooling to match stereo_feat spatial size to siglip_feat
            stereo_feat = torch.nn.functional.adaptive_avg_pool2d(
                stereo_feat.permute(0, 3, 1, 2),  # (b, c, h, w)
                output_size=(16, 16),
            ).permute(0, 2, 3, 1)  # (b, h, w, c)

        similarity_loss = None
        if self.enable_similarity_loss:
            with torch.no_grad():
                dino_fixed_feat = self.dino_fixed(images[:, :1, 1].reshape(b * 1, *chw))
            dino_fixed_feat = dino_fixed_feat.reshape(b, 16, 16, dino_fixed_feat.shape[-1])
            similarity_loss = self.compute_similarity_loss(dino_feat, dino_fixed_feat)

        feat = torch.cat([siglip_feat, dino_feat, stereo_feat], dim=-1)
        feat = feat.reshape(b, -1, feat.shape[-1])
        
        if similarity_loss is not None:
            return feat, similarity_loss
        else:
            return feat
