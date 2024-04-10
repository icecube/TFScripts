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

try:
    import tensorflow as tf

    # constants
    FLOAT_PRECISION = tf.float32
except ImportError:
    FLOAT_PRECISION = "float32"
    print("Tensorflow not found. Some functions will not work.")
