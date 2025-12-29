from .base import BaseSample

from pydantic import ConfigDict
from typing import Optional, List, OrderedDict, Tuple
from PIL import Image

from vla_network.utils.dtype import NdArray


class ValueQASample(BaseSample):
    """
    A data sample for value QA.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    images: OrderedDict[str, Image.Image]
    """Note that there is no T dim."""
    image_key: str
    """The image key."""
    caption: str
    """The caption of the query, e.g., 'the depth at (16, 21)'.
    Do not use a whole sentence/question, so that the model can customize the prompt.
    """
    values: NdArray
    """A 1D float array to answer the query."""
    precision: NdArray
    """A 1D float array for the precision of each value, mainly for the language models to quantize the answer."""
