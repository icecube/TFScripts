'''
Some utility functions for tfscripts
'''

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# constants
from tfscripts import FLOAT_PRECISION


def count_parameters(var_list=None):
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
        var_list = tf.compat.v1.trainable_variables()
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

    msg = 'Expect shape [?,3] or [3], but got {!r}'
    assert vec1.get_shape().as_list()[-1] == 3, msg.format(vec1.get_shape())
    assert vec2.get_shape().as_list()[-1] == 3, msg.format(vec2.get_shape())

    norm1 = tf.norm(tensor=vec1, axis=-1, keepdims=True)
    norm2 = tf.norm(tensor=vec2, axis=-1, keepdims=True)
    tmp1 = vec1 * norm2
    tmp2 = vec2 * norm1

    tmp3 = tf.norm(tensor=tmp1 - tmp2, axis=-1)
    tmp4 = tf.norm(tensor=tmp1 + tmp2, axis=-1)

    theta = 2*tf.atan2(tmp3, tmp4)
    return theta


def polynomial_interpolation(x, x_ref_min, x_ref_max, y_ref,
                             polynomial_order=2,
                             fill_value=None,
                             axis=-1,
                             equidistant_query_points=False,
                             dtype=FLOAT_PRECISION):
    """Performs a 1-dimensional polynomial interpolation of degree 1 or 2
    along the specified axis.

    x and y_ref must have the same shape in all, but the interpolation axis
    (Batch axis is also allowed to differ if it exists).

    This should be equivalent to
        tensorflow_probability.math.interp_regular_1d_grid
    in the case of rank 1 tensors x and y_ref and polynomial_order == 1.

    Parameters
    ----------
    x : tf.Tensor
        The points at which to interpolate.
    x_ref_min : scalar tf.Tensor or float
        Scalar Tensor of same dtype as x. The minimum value of the
        (implicitly defined) reference x_ref.
    x_ref_max : scalar tf.Tensor or float
        Scalar Tensor of same dtype as x. The maximum value of the
        (implicitly defined) reference x_ref.
    y_ref : tf.Tensor
        The reference values at the (implicitly defined) reference points
        x_ref.
    polynomial_order : int, optional
        The order of the polynomial interpolation. Supported values are
        1 and 2.
    fill_value : None, optional
        If None, interpolation points outside of [x_ref_min, x_ref_max]
        are set to the value at the boundary. Otherwise they are set
        to the specified fill_value.
    axis : int, optional
        The axis of y_ref along which to perform the interpolation
    equidistant_query_points : bool, optional
        If True, it is assumed that the points x are sorted and equidistant
        with the same spacing as y_ref. This potentially saves some
        computation costs. If you use this, make sure, that x qualify
        the above requirements!
    dtype : TYPE, optional
        The float dtype to be used.

    Returns
    -------
    tf.Tensor
        Same shape and type as x.
        The interpolated values at the points x.

    Raises
    ------
    ValueError
        If the arguments are not in range.
    """
    if axis != -1:
        # Transpose if necessary: last axis corresponds to interp points
        perm = [i for i in range(len(y_ref.get_shape()))
                if i != axis] + [axis]
        y_ref = tf.transpose(a=y_ref, perm=perm)
        x = tf.transpose(a=x, perm=perm)

    # get shapes and make sure they amtch
    y_shape = y_ref.get_shape().as_list()
    x_shape = x.get_shape().as_list()
    nbins = y_shape[-1]

    # sanity checks to make sure shapes match
    if x_shape[0] != y_shape[0]:
        assert x_shape[0] is None or y_shape[0] is None, '{!r} != {!r}'.format(
                                                x_shape[0], y_shape[0])
    assert x_shape[1:-1] == y_shape[1:-1], '{!r} != {!r}'.format(
                                                x_shape[:-1], y_shape[:-1])

    # We can take advantage of equidistant binning to compute indices
    bin_width = (x_ref_max - x_ref_min) / nbins
    indices = tf.histogram_fixed_width_bins(x, [x_ref_min, x_ref_max],
                                            nbins=nbins)

    lower_boundary = indices < 1
    upper_boundary = indices >= nbins - 1
    indices_sel = tf.clip_by_value(indices, 1, nbins - 2)

    indices_sel_float = tf.cast(indices_sel, dtype)
    x_0 = (indices_sel_float - 1) * bin_width + x_ref_min
    x_1 = (indices_sel_float) * bin_width + x_ref_min
    x_2 = (indices_sel_float + 1) * bin_width + x_ref_min

    # --------------------------------
    #  Pick correct reference points y
    # --------------------------------
    if y_shape[0] is None or x_shape[0] is None:
        # Need to obtain y_ref shape dynamically in this case
        index_offset = tf.tile(
            tf.range(tf.math.reduce_prod(
                input_tensor=tf.shape(input=y_ref)[:-1])) * nbins,
            tf.expand_dims(x_shape[-1], axis=-1))
        index_offset = tf.reshape(index_offset, [-1] + x_shape[1:])
    else:
        index_offset = np.tile(np.arange(np.prod(y_shape[:-1])) * nbins,
                               x_shape[-1])
        index_offset = np.reshape(index_offset, x_shape)

    y_ref_flat = tf.reshape(y_ref, [-1])
    indices_sel_offset = index_offset + indices_sel

    y_0 = tf.gather(y_ref_flat, indices_sel_offset - 1)
    y_1 = tf.gather(y_ref_flat, indices_sel_offset)
    y_2 = tf.gather(y_ref_flat, indices_sel_offset + 1)
    # ------------------------------

    if polynomial_order == 2:
        if equidistant_query_points:
            bin_width_squared = bin_width * bin_width
            L_0 = (x - x_1) * (x - x_2) / (2 * bin_width_squared)
            L_1 = (x - x_0) * (x - x_2) / (-bin_width_squared)
            L_2 = (x - x_0) * (x - x_1) / (2 * bin_width_squared)

        else:
            L_0 = (x - x_1) * (x - x_2) / ((x_0 - x_1)*(x_0 - x_2))
            L_1 = (x - x_0) * (x - x_2) / ((x_1 - x_0)*(x_1 - x_2))
            L_2 = (x - x_0) * (x - x_1) / ((x_2 - x_0)*(x_2 - x_1))

        result = y_0 * L_0 + y_1 * L_1 + y_2 * L_2

    elif polynomial_order == 1:
        if equidistant_query_points:
            result = y_1 + (x - x_1) / (bin_width) * (y_2 - y_1)
        else:
            result = y_1 + (x - x_1) / (x_2 - x_1) * (y_2 - y_1)

    else:
        raise ValueError('Interpolation order {!r} not supported'.format(
                                                        polynomial_order))

    if fill_value is None:
        result = tf.where(lower_boundary,
                          tf.zeros_like(result)
                          + tf.expand_dims(y_ref[..., 0], axis=-1),
                          result)
        result = tf.where(upper_boundary,
                          tf.zeros_like(result)
                          + tf.expand_dims(y_ref[..., -1], axis=-1),
                          result)
    else:
        result = tf.where(tf.logical_or(lower_boundary, upper_boundary),
                          tf.zeros_like(result) + fill_value,
                          result)

    if axis != -1:
        # Transpose back if necessary
        result = tf.transpose(a=result, perm=perm)

    return result
