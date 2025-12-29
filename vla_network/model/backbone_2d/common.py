import timm
import torch
from torch import nn
import torchvision.transforms as tvtf
import math

from vla_network.config.define import ImageTransform
from timm.models.vision_transformer import VisionTransformer


class ViT(nn.Module):
    model: VisionTransformer

    def __init__(self, model: VisionTransformer) -> None:
        super().__init__()
        self.model = model
        self.n = len(self.model.blocks) - 2

    @property
    def embed_dim(self) -> int:
        return self.model.embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.get_intermediate_layers(x, n={self.n})[0]


def create_timm_vit_and_transform(name: str, image_size: int) -> tuple[ViT, ImageTransform]:
    model: ViT = ViT(
        timm.create_model(
            name, pretrained=True, num_classes=0, img_size=image_size
        )
    )
    model.eval()

    model_cfg = timm.data.resolve_model_data_config(model.model)
    model_cfg["input_size"] = (3, image_size, image_size)
    transform = timm.data.create_transform(**model_cfg, is_training=False)

    # Replace the resize transform with the target size
    target_size = (image_size, image_size)
    resize_transform = tvtf.Compose(
        [
            tvtf.Resize(
                target_size, interpolation=transform.transforms[0].interpolation
            ),
            *transform.transforms[1:],
        ]
    )

    return model, resize_transform


def concat_multi_scale_features(P, Q):
    """
    Concatenate features from different patch sizes.
    
    Args:
        P (torch.Tensor): Tensor of shape (batch_size, A, A, C_p) - larger patches
        Q (torch.Tensor): Tensor of shape (batch_size, B, B, C_q) - smaller patches
        
    Returns:
        torch.Tensor: Combined features of shape (batch_size, A, A, C_p + variable)
    """
    batch_size, A, _, C_p = P.shape
    _, B, _, C_q = Q.shape
    assert B >= A
    scale = B / A
    combined_features = []
    for i in range(A):
        for j in range(A):
            q_start_i = math.floor(i * scale)
            q_end_i = math.ceil((i + 1) * scale)
            q_start_j = math.floor(j * scale)
            q_end_j = math.ceil((j + 1) * scale)
            q_features = Q[:, q_start_i:q_end_i, q_start_j:q_end_j, :].reshape(Q.shape[0], -1)
            p_feature = P[:, i, j, :]
            
            combined = torch.cat([p_feature, q_features], dim=-1)  # Shape: (batch_size, C_p + n*C_q)
            combined_features.append(combined)
    result = torch.stack(combined_features, dim=1).reshape(batch_size, A, A, -1)
    return result
