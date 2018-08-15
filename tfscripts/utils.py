'''
Some utility functions for tfscripts
'''

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# constants
FLOAT_PRECISION = tf.float32


def countParameters(var_list=None):
    """Count number of trainable parameters

    Parameters
    ----------
    var_list : None, optional
        If a var_list (list of tensors) is given, the number of parameters
        is calculated.
        If var_list is None, all trainable paramters in the current graph
        are counted

    Returns
    -------
    int
        Number of parameters.
    """
    if var_list is None:
        var_list = tf.trainable_variables()
    return np.sum([np.prod(x.get_shape().as_list()) for x in var_list])


def get_angle(vec1, vec2, dtype=FLOAT_PRECISION):
    """ Get the opening angle between two direction vectors.

    vec1/2 : shape: [?,3] or [3]
    https://www.cs.berkeley.edu/~wkahan/Mindless.pdf

    Parameters
    ----------
    vec1 : tf.Tensor
        A tensor of shape [?,3] or [3].
        Direction vectors.
    vec2 : tf.Tensor
        A tensor of shape [?,3] or [3].
        Direction vectors.
    dtype : tf.Dtype, optional
        The dtype for the tensorflow tensors.

    Returns
    -------
    tf.tensor
        Description
    """

    assert vec1.get_shape().as_list()[-1] == 3, \
        "Expect shape [?,3] or [3], but got {!r}".format(vec1.get_shape())
    assert vec2.get_shape().as_list()[-1] == 3, \
        "Expect shape [?,3] or [3], but got {!r}".format(vec2.get_shape())

    norm1 = tf.norm(vec1, axis=-1, keepdims=True)
    norm2 = tf.norm(vec2, axis=-1, keepdims=True)
    tmp1 = vec1 * norm2
    tmp2 = vec2 * norm1

    tmp3 = tf.norm(tmp1 - tmp2, axis=-1)
    tmp4 = tf.norm(tmp1 + tmp2, axis=-1)

    theta = 2*tf.atan2(tmp3, tmp4)
    return theta
