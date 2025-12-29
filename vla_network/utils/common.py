"""
Common utility functions used throughout the codebase
"""
from typing import Any, Iterable, List


def to_flatten_list(x: Any) -> List:
    """Convert nested iterable to flatten list."""
    if x is None:
        return []
    if isinstance(x, (str, bytes)) or not isinstance(x, Iterable):
        return [x]
    ret = []
    for i in x:
        ret.extend(to_flatten_list(i))
    return ret
