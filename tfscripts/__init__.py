import tensorflow as tf

from .__about__ import (
    __version_major__,
    __version_minor__,
    __version_patch__,
    __version_info__,
    __version__,
    __description__,
    __url__,
)

__all__ = [
    "__version_major__",
    "__version_minor__",
    "__version_patch__",
    "__version_info__",
    "__version__",
    "__description__",
    "__url__",
    "FLOAT_PRECISION",
]

# constants
FLOAT_PRECISION = tf.float32
