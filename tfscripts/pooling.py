'''
Pooling functions for tfscripts
'''

from __future__ import division, print_function

import tensorflow as tf

# tfscripts specific imports
from tfscripts.conv import get_conv_slice, get_start_index


def pool3d(layer, ksize, strides, padding, pooling_type):
    """Convenience function to perform pooling in 3D

    Parameters
    ----------
    layer : tf.Tensor
        Input tensor.
    ksize : list of int
        Size of pooling kernel in each dimension:
            [size_batch, size_x, size_y, size_z, size_channel]
    strides : list of int
        Stride along each dimension:
            [stride_batch, stride_x, stride_y, stride_z, stride_channel]
    padding : str
      The type of padding to be used. 'SAME' or 'VALID' are supported.
    pooling_type : str
        The type of pooling to be used.

    Returns
    -------
    tf.Tensor
        The pooled output tensor.
    """

    # tensorflow's pooling operations do not support float64, so
    # use workaround with casting to float32 and then back again
    if layer.dtype == tf.float64:
        layer = tf.cast(layer, tf.float32)
        was_float64 = True
    else:
        was_float64 = False

    # pool over depth, if necessary:
    if ksize[-1] != 1 or strides[-1] != 1:
        layer = pool_over_depth(layer,
                                ksize=ksize[-1],
                                stride=strides[-1],
                                padding=padding,
                                pooling_type=pooling_type)
    ksize = list(ksize)
    strides = list(strides)
    ksize[-1] = 1
    strides[-1] = 1

    # Use pooling to down-sample the image resolution?
    if ksize[1:-1] != [1, 1, 1] or strides[1:-1] != [1, 1, 1]:
        if pooling_type == 'max':
            layer = tf.nn.max_pool3d(input=layer,
                                     ksize=ksize,
                                     strides=strides,
                                     padding=padding)
        elif pooling_type == 'avg':
            layer = tf.nn.avg_pool3d(input=layer,
                                     ksize=ksize,
                                     strides=strides,
                                     padding=padding)

        elif pooling_type == 'max_avg':
            layer_max = tf.nn.max_pool3d(input=layer,
                                         ksize=ksize,
                                         strides=strides,
                                         padding=padding)
            layer_avg = tf.nn.avg_pool3d(input=layer,
                                         ksize=ksize,
                                         strides=strides,
                                         padding=padding)
            layer = (layer_avg + layer_max) / 2.

    if was_float64:
        layer = tf.cast(layer, tf.float64)

    return layer


def pool2d(layer, ksize, strides, padding, pooling_type):
    """Convenience function to perform pooling in 2D

    Parameters
    ----------
    layer : tf.Tensor
        Input tensor.
    ksize : list of int
        Size of pooling kernel in each dimension:
            [size_batch, size_x, size_y, size_channel]
    strides : list of int
        Stride along each dimension:
            [stride_batch, stride_x, stride_y, stride_channel]
    padding : str
      The type of padding to be used. 'SAME' or 'VALID' are supported.
    pooling_type : str
        The type of pooling to be used.

    Returns
    -------
    tf.Tensor
        The pooled output tensor.
    """

    # tensorflow's pooling operations do not support float64, so
    # use workaround with casting to float32 and then back again
    if layer.dtype == tf.float64:
        layer = tf.cast(layer, tf.float32)
        was_float64 = True
    else:
        was_float64 = False

    # pool over depth, if necessary:
    if ksize[-1] != 1 or strides[-1] != 1:
        layer = pool_over_depth(layer,
                                ksize=ksize[-1],
                                stride=strides[-1],
                                padding=padding,
                                pooling_type=pooling_type)
        ksize = list(ksize)
        strides = list(strides)
        ksize[-1] = 1
        strides[-1] = 1

    # Use pooling to down-sample the image resolution?
    if pooling_type == 'max':
        layer = tf.nn.max_pool2d(input=layer,
                                 ksize=ksize,
                                 strides=strides,
                                 padding=padding)
    elif pooling_type == 'avg':
        layer = tf.nn.avg_pool2d(input=layer,
                                 ksize=ksize,
                                 strides=strides,
                                 padding=padding)
    elif pooling_type == 'max_avg':
        layer_max = tf.nn.max_pool2d(input=layer,
                                     ksize=ksize,
                                     strides=strides,
                                     padding=padding)
        layer_avg = tf.nn.avg_pool2d(input=layer,
                                     ksize=ksize,
                                     strides=strides,
                                     padding=padding)
        layer = (layer_avg + layer_max) / 2.

    if was_float64:
        layer = tf.cast(layer, tf.float64)

    return layer


def pool_over_depth(layer, ksize, stride, padding, pooling_type):
    '''
    Performs pooling over last dimension of layer.
    Assumes that the last dimension of layer
    is the depth channel.

    Parameters
    ----------
    layer : tf.Tensor.
        Input layer.
    ksize : int
        Size of pooling kernel across channel dimension
    stride : int
        Stride along channel dimension
    padding : str
        The type of padding to be used. 'SAME' or 'VALID' are supported.
    pooling_type : str
        The type of pooling to be used.

    Returns
    -------
    tf.Tensor
        The pooled output tensor.
    '''

    num_channels = layer.get_shape().as_list()[-1]

    # get start index
    start_index = get_start_index(input_length=num_channels,
                                  filter_size=ksize,
                                  padding=padding,
                                  stride=stride,
                                  dilation=1)

    input_patches = []
    # ---------------------------
    # loop over all channel positions
    # ---------------------------
    for index in range(start_index, num_channels, stride):

        # get slice for patch along channel-axis
        slice_c, padding_c = get_conv_slice(index,
                                            input_length=num_channels,
                                            filter_size=ksize,
                                            stride=stride,
                                            dilation=1)

        if padding == 'VALID' and padding_c != (0, 0):
            # skip this c position, since it does not provide
            # a valid patch for padding 'VALID'
            continue

        # ------------------------------------------
        # Get input patch at filter position c
        # ------------------------------------------
        if pooling_type == 'max':
            input_patches.append(tf.reduce_max(
                                    input_tensor=layer[..., slice_c], axis=-1))

        elif pooling_type == 'avg':
            input_patches.append(tf.reduce_mean(
                                    input_tensor=layer[..., slice_c], axis=-1))

        else:
            raise ValueError('Pooling_type {!r} is unknown.'.format(
                                                                pooling_type))

    if pooling_type in ['max', 'avg']:
        layer = tf.stack(input_patches, axis=-1)

    return layer


def max_pool4d_stacked(input, ksize, strides, padding):
    '''max_pool4d equivalent

        Equivalent to tensorflows max_pool3d, but in 4D.

        avg_pool4d_stacked uses tensorflows avg_pool3d and stacks results for
        t-Dimension. Does not (yet) support pooling in time dimension:
            ksize != 1 in time dimension!

    Parameters
    ----------
    input : A Tensor. Must be one of the following types:
            float32, float64, int64, int32, uint8, uint16, int16, int8,
            complex64, complex128, qint8, quint8, qint32, half.
            Shape [batch, in_depth, in_height, in_width, in_channels].

    ksize : A list of ints that has length >= 6. 1-D tensor of length 6.
            The size of the window for each dimension of the input tensor.
            Must have ksize[0] = strides[4] = ksize[5] = 1.

    strides : A list of ints that has length >= 6. 1-D tensor of length 6.
                The stride of the sliding window for each dimension of input.
                Must have strides[0] = strides[5] = 1.
    padding : str
        The type of padding to be used. 'SAME' or 'VALID' are supported.

    Returns
    -------
    A Tensor. Has the same type as input. The max pooled output tensor.

    Raises
    ------
    ValueError
        Description
    '''
    if ksize[4] != 1:
        raise ValueError("max_pool4d_stacked does not yet support "
                         "pooling in t dimension.")

    # unpack along t dimension
    tensors_t = tf.unstack(input, axis=4)

    len_ts = ksize[4]
    size_of_t_dim = len(tensors_t)

    # loop over all z_j in t
    result_t = []
    for i in range(0, size_of_t_dim, strides[4]):
        result_t.append(tf.nn.max_pool3d(input=tensors_t[i],
                                         ksize=ksize[:4]+ksize[5:],
                                         strides=strides[:4]+strides[5:],
                                         padding=padding))
    # stack together
    return tf.stack(result_t, axis=4)


def avg_pool4d_stacked(input, ksize, strides, padding):
    '''avg_pool4d equivalent
    Equivalent to tensorflows avg_pool3d, but in 4D.
    This method is slightly slower, but appears to use less vram.

    avg_pool4d_stacked uses tensorflows avg_pool3d and stacks results
    for t-Dimension.

    Parameters
    ----------
    input : A Tensor. Must be one of the following types:
        float32, float64, int64, int32, uint8, uint16, int16, int8, complex64,
        complex128, qint8, quint8, qint32, half.
        Shape [batch, in_depth, in_height, in_width, in_channels].

    ksize : A list of ints that has length >= 6. 1-D tensor of length 6.
        The size of the window for each dimension of the input tensor.
        Must have ksize[0] = ksize[5] = 1.

    strides : A list of ints that has length >= 6. 1-D tensor of length 6.
            The stride of the sliding window for each dimension of input.
            Must have strides[0] = strides[5] = 1.
    padding : str
        The type of padding to be used. 'SAME' or 'VALID' are supported.

    Returns
    -------
    A Tensor. Has the same type as input.The average pooled output tensor.
    '''

    # unpack along t dimension
    tensors_t = tf.unstack(input, axis=4)

    len_ts = ksize[4]
    size_of_t_dim = len(tensors_t)

    if len_ts % 2 == 1:
        # uneven filter size: same size to left and right
        filter_l = int(len_ts/2)
        filter_r = int(len_ts/2)
    else:
        # even filter size: one more to right
        filter_l = int(len_ts/2) - 1
        filter_r = int(len_ts/2)

    # The start index is important for strides
    # The strides start with the first element
    # that works and is VALID:
    start_index = 0
    if padding == 'VALID':
        for i in range(size_of_t_dim):
            if len(range(max(i - filter_l, 0), min(i + filter_r+1,
                   size_of_t_dim))) == len_ts:
                # we found the first index that doesn't need padding
                break
        start_index = i

    # loop over all z_j in t
    result_t = []
    for i in range(start_index, size_of_t_dim, strides[4]):

            if padding == 'VALID':
                # Get indices z_s
                indices_t_s = range(max(i - filter_l, 0),
                                    min(i + filter_r+1, size_of_t_dim))

                # check if Padding = 'VALID'
                if len(indices_t_s) == len_ts:

                    tensors_t_averaged = []
                    # sum over all remaining index_z_i in indices_t_s
                    for j, index_z_i in enumerate(indices_t_s):
                            tensors_t_averaged.append(
                                        tf.nn.avg_pool3d(
                                            input=tensors_t[index_z_i],
                                            ksize=ksize[:4]+ksize[5:],
                                            strides=strides[:4]+strides[5:],
                                            padding=padding)
                                            )
                    avg_tensors_t_s = tf.divide(tf.add_n(tensors_t_averaged),
                                                len_ts)

                    # put together
                    result_t.append(avg_tensors_t_s)

            elif padding == 'SAME':
                tensors_t_averaged = []
                for kernel_j, j in enumerate(range(i - filter_l,
                                                   (i + 1) + filter_r)):
                    # we can just leave out the invalid t coordinates
                    # since they will be padded with 0's and therfore
                    # don't contribute to the sum
                    if 0 <= j < size_of_t_dim:
                        tensors_t_averaged.append(
                            tf.nn.avg_pool3d(input=tensors_t[j],
                                             ksize=ksize[:4]+ksize[5:],
                                             strides=strides[:4]+strides[5:],
                                             padding=padding)
                                             )
                avg_tensors_t_s = tf.divide(tf.add_n(tensors_t_averaged),
                                            len_ts)

                # put together
                result_t.append(avg_tensors_t_s)

    # stack together
    return tf.stack(result_t, axis=4)
