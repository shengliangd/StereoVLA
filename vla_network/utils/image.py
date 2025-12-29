import numpy as np
from typing import Tuple, Optional
from PIL import Image
import random


RGB_BRIGHTNESS = np.array([0.299, 0.587, 0.114])


def get_brightness(img: np.ndarray):
    return np.sum(img * RGB_BRIGHTNESS, axis=-1).mean()


def crop_img(img: np.ndarray, mode: str):
    h, w = img.shape[-3:-1]
    if mode == "center":
        return img[..., (w - h) // 2 : (w + h) // 2, :]
    elif mode == "left":
        return img[..., :h, :]
    elif mode == None:
        return img
    else:
        raise ValueError(f"Unknown mode: {mode}")


def crop_img_bbox(orig_shape: Tuple[int, int], bbox: np.ndarray, mode: str):
    h, w = orig_shape
    if mode == 'center':
        delta_w = (w - h) // 2
        xmin, ymin, xmax, ymax = bbox
        new_xmin = max(xmin - delta_w, 0)
        new_xmax = max(xmax - delta_w, 0)
        return np.array([new_xmin, ymin, new_xmax, ymax])
    elif mode == 'left':
        xmin, ymin, xmax, ymax = bbox
        new_xmin = min(xmin, h)
        new_xmax = min(xmax, h)
        return np.array([new_xmin, ymin, new_xmax, ymax])
    elif mode == None:
        return bbox
    else:
        raise ValueError(f"Unknown mode: {mode}")


def resize_image_with_bbox(
    image: Image.Image,
    bbox: Optional[np.ndarray],
    target_size: Tuple[int, int],
    random_padding: bool = True,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """
    Resize the image to target size. Pad if necessary.
    Also computes the bbox on the resized & padded image.
    """
    original_size = image.size
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    image = image.resize(new_size, Image.LANCZOS)

    new_image = Image.new("RGB", target_size)
    if random_padding:
        paste_x = random.randint(0, target_size[0] - new_size[0])
        paste_y = random.randint(0, target_size[1] - new_size[1])
    else:
        paste_x = (target_size[0] - new_size[0]) // 2
        paste_y = (target_size[1] - new_size[1]) // 2
    new_image.paste(image, (paste_x, paste_y))

    if bbox is not None:
        new_bbox = bbox * ratio
        new_bbox[0] += paste_x
        new_bbox[1] += paste_y
        new_bbox[2] += paste_x
        new_bbox[3] += paste_y
        new_bbox = np.array([int(t) for t in new_bbox])
    else:
        new_bbox = None

    return new_image, new_bbox


def resize_image_with_position(
    image: Image.Image,
    position: Optional[Tuple[int, int]],
    target_size: Tuple[int, int],
    random_padding: bool = True,
) -> Tuple[Image.Image, Optional[np.ndarray]]:
    """
    Resize the image to target size. Pad if necessary.
    Also computes the bbox on the resized & padded image.
    """
    original_size = image.size
    ratio = min(target_size[0] / original_size[0], target_size[1] / original_size[1])
    new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
    image = image.resize(new_size, Image.LANCZOS)

    new_image = Image.new("RGB", target_size)
    if random_padding:
        paste_x = random.randint(0, target_size[0] - new_size[0])
        paste_y = random.randint(0, target_size[1] - new_size[1])
    else:
        paste_x = (target_size[0] - new_size[0]) // 2
        paste_y = (target_size[1] - new_size[1]) // 2
    new_image.paste(image, (paste_x, paste_y))

    if position is not None:
        new_position = np.array(position) * ratio
        new_position[0] += paste_x
        new_position[1] += paste_y
        new_position = tuple(int(v) for v in new_position)
    else:
        new_position = None

    return new_image, new_position
