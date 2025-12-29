from .base import BaseSample

from typing import Dict, List, OrderedDict
from PIL import Image
from vla_network.utils.dtype import NdArray


class BBoxSample(BaseSample):
    caption: str
    images: OrderedDict[str, List[Image.Image]]
    bboxs: OrderedDict[str, List[NdArray]]
