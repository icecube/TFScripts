'''
Conv functions for tfscripts.compat.v1:
    convolution helper functions,
    locally connected 2d and 3d convolutions,
    dynamic 2d and 3d convolution,
    local trafo 2d and 3d,
    wrapper: trafo on patch 2d and 3d
    stacked convolution 3d and 4d


'''

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# tfscripts.compat.v1 specific imports
from tfscripts.compat.v1.weights import new_weights


def conv_output_length(input_length, filter_size, padding, stride, dilation=1):
    """Determines output length of a convolution given input length.

    Helper function to compute output length of a dimension.

    ADOPTED FROM
    ------------
    https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
    contrib/keras/python/keras/utils/conv_utils.py

    Parameters
    ----------
    input_length : int
        The length of the input.
    filter_size : int
        The size of the filter along the dimension.
    padding : str
        The type of padding to be used. 'SAME' or 'VALID' are supported.
    stride : int
        The stride along the dimension.
    dilation : int, optional
        Dilation rate along the dimension.

    Returns
    -------
    int
        The output length.
    """
    if input_length is None:
        return None
    assert padding in {'SAME', 'VALID'}

    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)

    if padding == 'SAME':
        output_length = input_length
    elif padding == 'VALID':
        output_length = input_length - dilated_filter_size + 1

    return (output_length + stride - 1) // stride


def get_filter_lr(filter_size):
    '''
    Get the number of elements left and right
    of the filter position. If filtersize is
    even, there will be one more element to the
    right, than to the left.

    Parameters
    ----------
    filter_size : int
        Size of filter along the given dimension.

    Returns
    -------
    (int, int)
        Number of elemnts left and right
        of the filter position.


    '''
    if filter_size % 2 == 1:
        # uneven filter size: same size to left and right
        filter_l = filter_size // 2
        filter_r = filter_size // 2
    else:
        # even filter size: one more to right
        filter_l = max((filter_size // 2) - 1, 0)
        filter_r = filter_size // 2

    return filter_l, filter_r


def get_start_index(input_length, filter_size, padding, stride, dilation=1):
    '''
    Get start index for a convolution along an axis.
    This will be the index of the input axis
    for the first valid convolution position.

    Parameters
    ----------
    input_length : int
        The length of the input.
    filter_size : int
        The size of the filter along the dimension.
    padding : str
        The type of padding to be used. 'SAME' or 'VALID' are supported.
    stride : int
        The stride along the dimension.
    dilation : int, optional
        Dilation rate along the dimension.

    Returns
    -------
    int
        The start index for a convolution.

    Raises
    ------
    ValueError
        Description
    '''
    filter_l, filter_r = get_filter_lr(filter_size)

    # The start index is important for strides and dilation
    # The strides start with the first element
    # that works and is VALID:
    start_index = 0
    found_valid_position = False
    if padding == 'VALID':
        for i in range(input_length):
            if len(range(max(i - dilation*filter_l, 0),
                         min(i + dilation*filter_r + 1, input_length),
                         dilation)) == filter_size:
                # we found the first index that doesn't need padding
                found_valid_position = True
                break

        if not found_valid_position:
            raise ValueError('Input dimension is too small for "VALID" patch')
        start_index = i

    return start_index


def get_conv_indices(position, input_length, filter_size, stride, dilation=1):
    '''
    Get indices for a convolution patch.
    The indices will correspond to the elements
    being convolved with the filter of size
    filter_size.
    Note: indices get cropped at 0 and maximum value

    Parameters
    ----------
    position : int
        Current position of filter along a given axis.
    input_length : int
        The length of the input.
    filter_size : int
        The size of the filter along the dimension.
    stride : int
        The stride along the dimension.
    dilation : int, optional
        Dilation rate along the dimension.

    Returns
    -------
    list of int
        Indices of a convolution patch
    '''
    filter_l, filter_r = get_filter_lr(filter_size)

    indices = range(max(position - dilation*filter_l, 0),
                    min(position + dilation*filter_r + 1, input_length),
                    dilation)
    return indices


def get_conv_slice(position, input_length, filter_size, stride, dilation=1):
    '''
    Get slice for a convolution patch.
    The slice will correspond to the elements
    being convolved with the filter of size
    filter_size.
    Note: slice gets cropped at 0 and maximum value

    Parameters
    ----------
    position : int
        Current position of filter along a given axis.
    input_length : int
        The length of the input.
    filter_size : int
        The size of the filter along the dimension.
    stride : int
        The stride along the dimension.
    dilation : int, optional
        Dilation rate along the dimension.

    Returns
    -------
    (slice, (int, int))
        slice of input being convolved with the filter
        at 'position'. And number of elements cropped at
        either end, which are needed for padding
    '''
    filter_l, filter_r = get_filter_lr(filter_size)

    min_index = position - dilation*filter_l
    max_index = position + dilation*filter_r + 1

    conv_slice = slice(max(min_index, 0),
                       min(max_index, input_length),
                       dilation)

    padding_left = int(np.ceil(max(- min_index, 0) / dilation))
    padding_right = int(np.ceil(max(max_index - input_length, 0) / dilation))

    return conv_slice, (padding_left, padding_right)


def locally_connected_2d(input,
                         num_outputs,
                         filter_size,
                         kernel=None,
                         strides=[1, 1],
                         padding='SAME',
                         dilation_rate=None):
    '''
    Like conv2d, but doesn't share weights.
    (not tested/validated yet!!)

    Parameters
    ----------
    input : A Tensor. Must be one of the following types:
        float32, float64, int64, int32, uint8, uint16, int16, int8, complex64,
        complex128, qint8, quint8, qint32, half.
        Shape [batch, in_depth, in_height, in_width, in_channels].
    num_outputs : int
          Number of output channels
    filter_size : list of int of size 2
        [filter x size, filter y size]
    kernel : tf.Tensor, optional
        The kernel weights. If none are provided, new kernel weights will be
        created.
    strides : A list of ints that has length = 2. 1-D tensor of length 2.
              The stride of the sliding window for each dimension of input.
    padding : A string from: "SAME", "VALID".
        The type of padding algorithm to use.

    dilation_rate : None or list of int of length 2
        [dilattion in x, dilation in y]
        defines dilattion rate to be used

    Returns
    -------
    2 Tensors: result and kernels.
    Have the same type as input.
    '''

    if dilation_rate is None:
        dilation_rate = [1, 1]

    # ------------------
    # get shapes
    # ------------------
    input_shape = input.get_shape().as_list()

    # sanity checks
    assert len(filter_size) == 2, \
        'Filter size must be of shape [x,y], but is {!r}'.format(filter_size)
    assert np.prod(filter_size) > 0, \
        'Filter sizes must be greater than 0'
    assert len(input_shape) == 4, \
        'Shape is expected to be of length 4, but is {!r}'.format(input_shape)

    # calculate output shape
    output_shape = np.empty(4, dtype=int)
    for i in range(2):
        output_shape[i+1] = conv_output_length(input_length=input_shape[i + 1],
                                               filter_size=filter_size[i],
                                               padding=padding,
                                               stride=strides[i],
                                               dilation=dilation_rate[i])
    output_shape[0] = -1
    output_shape[3] = num_outputs

    num_inputs = input_shape[3]

    kernel_shape = (np.prod(output_shape[1:-1]),
                    np.prod(filter_size) * num_inputs,
                    num_outputs)

    # ------------------
    # 1x1 convolution
    # ------------------
    # fast shortcut
    if list(filter_size) == [1, 1]:
        if kernel is None:
            kernel = new_weights(shape=input_shape[1:] + [num_outputs])
        output = tf.reduce_sum(
            input_tensor=tf.expand_dims(input, axis=4) * kernel, axis=3)
        return output, kernel

    # ------------------
    # get slices
    # ------------------
    start_indices = [get_start_index(input_length=input_shape[i + 1],
                                     filter_size=filter_size[i],
                                     padding=padding,
                                     stride=strides[i],
                                     dilation=dilation_rate[i])
                     for i in range(2)]

    input_patches = []
    # ---------------------------
    # loop over all x positions
    # ---------------------------
    for x in range(start_indices[0], input_shape[1], strides[0]):

        # get slice for patch along x-axis
        slice_x, padding_x = get_conv_slice(x,
                                            input_length=input_shape[1],
                                            filter_size=filter_size[0],
                                            stride=strides[0],
                                            dilation=dilation_rate[0])

        if padding == 'VALID' and padding_x != (0, 0):
            # skip this x position, since it does not provide
            # a valid patch for padding 'VALID'
            continue

        # ---------------------------
        # loop over all y positions
        # ---------------------------
        for y in range(start_indices[1], input_shape[2], strides[1]):

            # get indices for patch along y-axis
            slice_y, padding_y = get_conv_slice(y,
                                                input_length=input_shape[2],
                                                filter_size=filter_size[1],
                                                stride=strides[1],
                                                dilation=dilation_rate[1])

            if padding == 'VALID' and padding_y != (0, 0):
                # skip this y position, since it does not provide
                # a valid patch for padding 'VALID'
                continue

            # At this point, slice_x/y either correspond
            # to a vaild patch, or padding is 'SAME'
            # Now we need to pick slice and add it to
            # input patches. These will later be convolved
            # with the kernel.

            # ------------------------------------------
            # Get input patch at filter position x,y
            # ------------------------------------------
            input_patch = input[:, slice_x, slice_y, :]

            if padding == 'SAME':
                # pad with zeros
                paddings = [(0, 0), padding_x, padding_y, (0, 0)]
                if paddings != [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]:
                    input_patch = tf.pad(tensor=input_patch,
                                         paddings=paddings,
                                         mode='CONSTANT',
                                         )

            # reshape
            input_patch = tf.reshape(
                    input_patch, [-1, 1, np.prod(filter_size) * num_inputs, 1])

            # append to list
            input_patches.append(input_patch)
            # ------------------------------------------

    # concat input patches
    input_patches = tf.concat(input_patches, axis=1)

    # ------------------
    # get kernel
    # ------------------
    if kernel is None:
        kernel = new_weights(shape=kernel_shape)

    # ------------------
    # perform convolution
    # ------------------
    output = input_patches * kernel
    output = tf.reduce_sum(input_tensor=output, axis=2)
    output = tf.reshape(output, output_shape)
    return output, kernel


def locally_connected_3d(input,
                         num_outputs,
                         filter_size,
                         kernel=None,
                         strides=[1, 1, 1],
                         padding='SAME',
                         dilation_rate=None):
    '''
    Like conv3d, but doesn't share weights.

    Parameters
    ----------
    input : A Tensor. Must be one of the following types:
        float32, float64, int64, int32, uint8, uint16, int16, int8, complex64,
        complex128, qint8, quint8, qint32, half.
        Shape [batch, in_depth, in_height, in_width, in_channels].
    num_outputs : int
        Number of output channels
    filter_size : list of int of size 3
            [filter x size, filter y size, filter z size]
    kernel : tf.Tensor, optional
        The kernel weights. If none are provided, new kernel weights will be
        created.
    strides : A list of ints that has length >= 5. 1-D tensor of length 5.
            The stride of the sliding window for each dimension of input.
            Must have strides[0] = strides[4] = 1.
    padding : A string from: "SAME", "VALID".
        The type of padding algorithm to use.
    dilation_rate : None or list of int of length 3
        [dilattion in x, dilation in y, dilation in z]
        defines dilattion rate to be used

    Returns
    -------
    2 Tensors: result and kernels.
    Have the same type as input.
    '''

    if dilation_rate is None:
        dilation_rate = [1, 1, 1]

    # ------------------
    # get shapes
    # ------------------
    input_shape = input.get_shape().as_list()

    # sanity checks
    assert len(filter_size) == 3, \
        'Filter size must be of shape [x,y,z], but is {!r}'.format(filter_size)
    assert np.prod(filter_size) > 0, 'Filter sizes must be greater than 0'
    assert len(input_shape) == 5, \
        'Shape is expected to be of length 5, but is {!r}'.format(input_shape)

    # calculate output shape
    output_shape = np.empty(5, dtype=int)
    for i in range(3):
        output_shape[i+1] = conv_output_length(input_length=input_shape[i + 1],
                                               filter_size=filter_size[i],
                                               padding=padding,
                                               stride=strides[i],
                                               dilation=dilation_rate[i])
    output_shape[0] = -1
    output_shape[4] = num_outputs

    num_inputs = input_shape[4]

    kernel_shape = (np.prod(output_shape[1:-1]),
                    np.prod(filter_size) * num_inputs,
                    num_outputs)

    # ------------------
    # 1x1x1 convolution
    # ------------------
    # fast shortcut
    if list(filter_size) == [1, 1, 1]:
        if kernel is None:
            kernel = new_weights(shape=input_shape[1:] + [num_outputs])
        output = tf.reduce_sum(
            input_tensor=tf.expand_dims(input, axis=5) * kernel, axis=4)
        return output, kernel

    # ------------------
    # get slices
    # ------------------
    start_indices = [get_start_index(input_length=input_shape[i + 1],
                                     filter_size=filter_size[i],
                                     padding=padding,
                                     stride=strides[i],
                                     dilation=dilation_rate[i])
                     for i in range(3)]

    input_patches = []
    # ---------------------------
    # loop over all x positions
    # ---------------------------
    for x in range(start_indices[0], input_shape[1], strides[0]):

        # get slice for patch along x-axis
        slice_x, padding_x = get_conv_slice(x,
                                            input_length=input_shape[1],
                                            filter_size=filter_size[0],
                                            stride=strides[0],
                                            dilation=dilation_rate[0])

        if padding == 'VALID' and padding_x != (0, 0):
            # skip this x position, since it does not provide
            # a valid patch for padding 'VALID'
            continue

        # ---------------------------
        # loop over all y positions
        # ---------------------------
        for y in range(start_indices[1], input_shape[2], strides[1]):

            # get indices for patch along y-axis
            slice_y, padding_y = get_conv_slice(y,
                                                input_length=input_shape[2],
                                                filter_size=filter_size[1],
                                                stride=strides[1],
                                                dilation=dilation_rate[1])

            if padding == 'VALID' and padding_y != (0, 0):
                # skip this y position, since it does not provide
                # a valid patch for padding 'VALID'
                continue

            # ---------------------------
            # loop over all z positions
            # ---------------------------
            for z in range(start_indices[2], input_shape[3], strides[2]):

                # get indices for patch along y-axis
                slice_z, padding_z = get_conv_slice(
                                        z,
                                        input_length=input_shape[3],
                                        filter_size=filter_size[2],
                                        stride=strides[2],
                                        dilation=dilation_rate[2])

                if padding == 'VALID' and padding_z != (0, 0):
                    # skip this z position, since it does not provide
                    # a valid patch for padding 'VALID'
                    continue

                # At this point, slice_x/y/z either correspond
                # to a vaild patch, or padding is 'SAME'
                # Now we need to pick slice and add it to
                # input patches. These will later be convolved
                # with the kernel.

                # ------------------------------------------
                # Get input patch at filter position x,y,z
                # ------------------------------------------
                input_patch = input[:, slice_x, slice_y, slice_z, :]

                if padding == 'SAME':
                    # pad with zeros
                    paddings = [(0, 0), padding_x, padding_y,
                                padding_z, (0, 0)]
                    if paddings != [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]:
                        input_patch = tf.pad(tensor=input_patch,
                                             paddings=paddings,
                                             mode='CONSTANT',
                                             )

                # reshape
                input_patch = tf.reshape(
                    input_patch, [-1, 1, np.prod(filter_size) * num_inputs, 1])

                # append to list
                input_patches.append(input_patch)
                # ------------------------------------------

    # concat input patches
    input_patches = tf.concat(input_patches, axis=1)

    # ------------------
    # get kernel
    # ------------------
    if kernel is None:
        kernel = new_weights(shape=kernel_shape)

    # ------------------
    # perform convolution
    # ------------------
    output = input_patches * kernel
    output = tf.reduce_sum(input_tensor=output, axis=2)
    output = tf.reshape(output, output_shape)
    return output, kernel


def local_translational3d_trafo(input,
                                num_outputs,
                                filter_size,
                                fcn=None,
                                weights=None,
                                strides=[1, 1, 1],
                                padding='SAME',
                                dilation_rate=None,
                                is_training=True,
                                ):
    '''
    Applies a transformation defined by the callable fcn(input_patch)
    to the input_patch. Returns the output of fcn.
    Similiar to conv3d, but instead of a convolution, the transformation
    defined by fcn is performed. The transformation is learnable and shared
    accross input_patches similar to how the convolutional kernel is sahred.

    Parameters
    ----------
    input : A Tensor. Must be one of the following types:
        float32, float64, int64, int32, uint8, uint16, int16, int8, complex64,
        complex128, qint8, quint8, qint32, half.
        Shape [batch, in_depth, in_height, in_width, in_channels].

    num_outputs : int
        Number of output channels

    filter_size : list of int of size 3
            [filter x size, filter y size, filter z size]

    fcn : callable: fcn(input_patch)
            Defines the transformation:
              input_patch -> output
              with output.shape = [-1, num_outputs]

    weights : None, optional
        Description
    strides : A list of ints that has length >= 5. 1-D tensor of length 5.
            The stride of the sliding window for each dimension of input.
            Must have strides[0] = strides[4] = 1.
    padding : A string from: "SAME", "VALID".
        The type of padding algorithm to use.

    dilation_rate :None or list of int of length 3
        [dilattion in x, dilation in y, dilation in z]
        defines dilattion rate to be used

    is_training : bool, optional
        Indicates whether currently in training or inference mode.
        True: in training mode
        False: inference mode.

    Returns
    -------
    2 Tensors: result and kernels.
    Have the same type as input.
    '''

    if dilation_rate is None:
        dilation_rate = [1, 1, 1]

    # ------------------
    # get shapes
    # ------------------
    input_shape = input.get_shape().as_list()

    # sanity checks
    assert len(filter_size) == 3, \
        'Filter size must be of shape [x,y,z], but is {!r}'.format(filter_size)
    assert np.prod(filter_size) > 0, 'Filter sizes must be greater than 0'
    assert len(input_shape) == 5, \
        'Shape is expected to be of length 5, but is {!r}'.format(input_shape)

    # calculate output shape
    output_shape = np.empty(5, dtype=int)
    for i in range(3):
        output_shape[i+1] = conv_output_length(input_length=input_shape[i + 1],
                                               filter_size=filter_size[i],
                                               padding=padding,
                                               stride=strides[i],
                                               dilation=dilation_rate[i])
    output_shape[0] = -1
    output_shape[4] = num_outputs

    num_inputs = input_shape[4]

    # ------------------
    # get slices
    # ------------------
    start_indices = [get_start_index(input_length=input_shape[i + 1],
                                     filter_size=filter_size[i],
                                     padding=padding,
                                     stride=strides[i],
                                     dilation=dilation_rate[i])
                     for i in range(3)]

    output = []
    # ---------------------------
    # loop over all x positions
    # ---------------------------
    for x in range(start_indices[0], input_shape[1], strides[0]):

        # get slice for patch along x-axis
        slice_x, padding_x = get_conv_slice(x,
                                            input_length=input_shape[1],
                                            filter_size=filter_size[0],
                                            stride=strides[0],
                                            dilation=dilation_rate[0])

        if padding == 'VALID' and padding_x != (0, 0):
            # skip this x position, since it does not provide
            # a valid patch for padding 'VALID'
            continue

        # ---------------------------
        # loop over all y positions
        # ---------------------------
        for y in range(start_indices[1], input_shape[2], strides[1]):

            # get indices for patch along y-axis
            slice_y, padding_y = get_conv_slice(y,
                                                input_length=input_shape[2],
                                                filter_size=filter_size[1],
                                                stride=strides[1],
                                                dilation=dilation_rate[1])

            if padding == 'VALID' and padding_y != (0, 0):
                # skip this y position, since it does not provide
                # a valid patch for padding 'VALID'
                continue

            # ---------------------------
            # loop over all z positions
            # ---------------------------
            for z in range(start_indices[2], input_shape[3], strides[2]):

                # get indices for patch along y-axis
                slice_z, padding_z = get_conv_slice(
                                        z,
                                        input_length=input_shape[3],
                                        filter_size=filter_size[2],
                                        stride=strides[2],
                                        dilation=dilation_rate[2])

                if padding == 'VALID' and padding_z != (0, 0):
                    # skip this z position, since it does not provide
                    # a valid patch for padding 'VALID'
                    continue

                # At this point, slice_x/y/z either correspond
                # to a vaild patch, or padding is 'SAME'
                # Now we need to pick slice and add it to
                # input patches. These will later be convolved
                # with the kernel.

                # ------------------------------------------
                # Get input patch at filter position x,y,z
                # ------------------------------------------
                # input_patch = tf.expand_dims(
                #                    input[:, slice_x, slice_y, slice_z, :], 5)
                input_patch = input[:, slice_x, slice_y, slice_z, :]

                if padding == 'SAME':
                    # pad with zeros
                    paddings = [(0, 0), padding_x, padding_y,
                                padding_z, (0, 0)]
                    if paddings != [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]:
                        input_patch = tf.pad(tensor=input_patch,
                                             paddings=paddings,
                                             mode='CONSTANT',
                                             )

                # ------------------------------
                # Perform trafo on input_patch
                # ------------------------------

                if weights is not None:
                    expanded_input = tf.expand_dims(input_patch, -1)
                    output_patch = tf.reduce_sum(
                        input_tensor=expanded_input * weights,
                        axis=[1, 2, 3, 4],
                        keepdims=False,
                    )
                elif fcn is not None:
                    output_patch = fcn(input_patch)

                # append to list
                output.append(output_patch)
                # ------------------------------------------

    # concat input patches
    output = tf.stack(output, axis=1)
    output = tf.reshape(output, output_shape)

    return output


def dynamic_conv(
                input,
                filter,
                batch_size=None,
                strides=[1, 1, 1],
                padding='SAME',
                dilation_rate=None,
                ):
    '''
    Equivalent to tf.nn.convolution, but filter has additional
    batch dimension. This allows the filter to be a function
    of some input, hence, enabling dynamic convolutions.

    Parameters
    ----------
    input : A Tensor. Must be one of the following types:
            float32, float64, int64, int32, uint8, uint16, int16,
            int8, complex64, complex128, qint8, quint8, qint32, half.

            2d case:
            Shape [batch, in_depth, in_height, in_channels].
            3d case:
            Shape [batch, in_depth, in_height, in_width, in_channels].

    filter : A Tensor. Must have the same type as input.
            in_channels must match between input and filter.
            2d case:
            Shape [batch, filter_x, filter_y, in_ch, out_ch].
            3d case:
            Shape [batch, filter_x, filter_y, filter_z, in_ch, out_ch] .

    batch_size : int, optional
        The batch size.

    strides : A list of ints that has length >= 2.
        1-D tensor of length 2 (2D) or 3(3D).
        The stride of the sliding window for each spatial
        dimension of input.

    padding : A string from: "SAME", "VALID".
        The type of padding algorithm to use.

    dilation_rate : Optional.
        Sequence of N ints >= 1.
        Specifies the filter upsampling/input downsampling rate.
        In the literature, the same parameter is sometimes called input stride
        or dilation. The effective filter size used for the convolution will be
        spatial_filter_shape + (spatial_filter_shape - 1) * (rate - 1),
        obtained by inserting (dilation_rate[i]-1) zeros between consecutive
        elements of the original filter in each spatial dimension i.
        If any value of dilation_rate is > 1, then all values of strides
        must be 1.

    Returns
    -------
    A Tensor. Has the same type as input.
    '''

    input_shape = input.get_shape().as_list()
    filter_shape = filter.get_shape().as_list()

    assert len(filter_shape) == len(input_shape) + 1
    assert filter_shape[0] == input_shape[0]

    if batch_size is None:
        batch_size = tf.shape(input=input)[0]

        try:
            batch_size = dynamic_conv.BATCH_SIZE
        except Exception as e:
            batch_size = input.get_shape().as_list()[0]

    split_inputs = tf.split(input,
                            batch_size,
                            axis=0)
    split_filters = tf.unstack(filter,
                               batch_size,
                               axis=0)

    output_list = []
    for split_input, split_filter in zip(split_inputs, split_filters):
        output_list.append(
              tf.nn.convolution(split_input,
                                split_filter,
                                strides=strides,
                                padding=padding,
                                dilations=dilation_rate,
                                )
          )
    output = tf.concat(output_list, axis=0)
    return output


def trans3d_op(input,
               num_out_channel,
               filter_size,
               method,
               trafo=None,
               filter=None,
               strides=[1, 1, 1],
               padding='SAME',
               dilation_rate=None,
               stack_axis=None,
               ):
    '''
    Applies a transformation to the input_patch. Input patches are obtained
    as it is done in conv3d. The transformation that is performed on this input
    patch is defined by the method argument:

    method:

      1) method == 'dynamic_convolution'
          Performs a convolution over the input patch:
          output_patch = reduce_sum(input_patch * weights)
          The convolution filter must be given and has
          the shape:
          [batch, filter_x, filter_y, filter_z, in_channel, num_out_channel]

      2) method == 'locally_connected'
          Performs a convolution as it is done in conv3d,
          but does not share weights.
          The convolution filter must be given and has the shape:
          [batch, x,y,z, filter_x, filter_y, filter_z, in_channel, out_channel]
          (Broadcasting is supported??)

      3) method == 'local_trafo'
          Performs a local trafo to the input patch.
          The trafo is defined by the callable trafo.
          output_patch = trafo(input_patch)
          The function trafo must return an output of the shape
          [batch, num_out_channel]

    Parameters
    ----------
    input : A Tensor. Must be one of the following types:
        float32, float64, int64, int32, uint8, uint16, int16, int8, complex64,
        complex128, qint8, quint8, qint32, half.
        Shape [batch, in_depth, in_height, in_width, in_channels].

    num_out_channel : int
        Number of output channels

    filter_size : list of int of size 3
        [filter x size, filter y size, filter z size]

    method : str
        Defines the transformation that is applied to the input patch.
        See above for possible options.

    trafo : callable: trafo(input_patch)
        Defines the transformation:
            input_patch -> output
            with output.shape = [-1, num_out_channel]

    filter : Defines the filter for the convolution, if method is
        either 'dynamic_convolution' or 'locally_connected'

    strides : A list of ints that has length >= 5. 1-D tensor of length 5.
                The stride of the sliding window for each dimension of input.
                Must have strides[0] = strides[4] = 1.

    padding : A string from: "SAME", "VALID".
        The type of padding algorithm to use.

    dilation_rate : None or list of int of length 3
                [dilattion in x, dilation in y, dilation in z]
                defines dilattion rate to be used

    stack_axis : Int
              Spatial axis along which the input_patches will be stacked.
                0: x_dimension
                1: y_dimension
                2: z_dimension
                where input is of the shape:
                [batch, x_dimension, y_dimension, z_dimension, in_channels]

              The input_patches along the remaining two spatial dimensions
              are obtained though tf.extract_image_patches .
              By default the axis with the lowest output dimensionality will be
              chosen.

    Returns
    -------
    Tensors: result
    Has the same type as input.

    Raises
    ------
    NotImplementedError
        Description
    ValueError
        Description
    '''

    if dilation_rate is None:
        dilation_rate = [1, 1, 1]

    # ------------------
    # get shapes
    # ------------------
    input_shape = input.get_shape().as_list()

    # sanity checks
    if method not in ['dynamic_convolution', 'locally_connected',
                      'local_trafo']:
        raise ValueError('Method unknown: {!r}'.format(method))
    assert len(filter_size) == 3, \
        'Filter size must be of shape [x,y,z], but is {!r}'.format(filter_size)
    assert np.prod(filter_size) > 0, 'Filter sizes must be greater than 0'
    assert len(input_shape) == 5, \
        'Shape is expected to be of length 5, but is {}'.format(input_shape)

    # calculate output shape
    output_shape = np.empty(5, dtype=int)
    for i in range(3):
        output_shape[i+1] = conv_output_length(input_length=input_shape[i + 1],
                                               filter_size=filter_size[i],
                                               padding=padding,
                                               stride=strides[i],
                                               dilation=dilation_rate[i])
    output_shape[0] = -1
    output_shape[4] = num_out_channel

    num_in_channel = input_shape[4]

    # pick stack_axis
    if stack_axis is None:
        stack_axis = np.argmin(output_shape[1:4])

    # shape of patches
    # [batch, x, y, z, filter_x, filter_y, filter_z, num_in_channel]
    patches_shape = [-1] + input_shape[1:4] + filter_size + [num_in_channel]
    # shape of patches for a given position s along stack_axis
    # if stack_axis is 2 (z-dimension):
    # [batch, x, y, filter_x, filter_y, num_in_channel]
    s_patches_shape = list(patches_shape)
    del s_patches_shape[1+stack_axis]
    del s_patches_shape[3+stack_axis]

    # define parameters for extract_image_patches
    ksizes = [1] + list(filter_size) + [1]
    del ksizes[1 + stack_axis]
    strides2d = [1] + list(strides) + [1]
    del strides2d[1 + stack_axis]
    rates = [1] + list(dilation_rate) + [1]
    del rates[1 + stack_axis]

    # unstack along stack_axis
    unstacked_input = tf.unstack(input, axis=1 + stack_axis)

    # ------------------
    # get slices
    # ------------------
    start_indices = [get_start_index(input_length=input_shape[i + 1],
                                     filter_size=filter_size[i],
                                     padding=padding,
                                     stride=strides[i],
                                     dilation=dilation_rate[i])
                     for i in range(3)]

    output = []
    # ---------------------------------------
    # loop over all positions in dim #stack_axis
    # ----------------------------------------
    for s in range(start_indices[stack_axis], input_shape[stack_axis + 1],
                   strides[stack_axis]):

        # get slice for patch along stack_axis
        slice_s, padding_s = get_conv_slice(
                                s,
                                input_length=input_shape[stack_axis + 1],
                                filter_size=filter_size[stack_axis],
                                stride=strides[stack_axis],
                                dilation=dilation_rate[stack_axis])

        if padding == 'VALID' and padding_s != (0, 0):
            # skip this axis position, since it does not provide
            # a valid patch for padding 'VALID'
            continue

        # At this point, slice_s either corresponds
        # to a vaild patch, or padding is 'SAME'
        # Now we need to combine the input_patches

        # ----------------------------------------------------
        # Get input patch at filter position s on stack_axis
        # ----------------------------------------------------

        # Now go through all possible positions along stack_axis
        # within the input_s slice
        input_patches = []
        for input_s in unstacked_input[slice_s]:
            # if stack_axis is 2 (z-dimension),
            # input_s has shape:
            # [batch, filter_x, filter_y, in_channel]
            # input_patches has shape:
            # [batch, x,y, filter_x*filter_y*in_channel]
            input_s_patches = tf.image.extract_patches(input_s,
                                                       sizes=ksizes,
                                                       strides=strides2d,
                                                       rates=rates,
                                                       padding=padding)

            # reshape input_patches to
            # [batch, x,y, filter_x,filter_y,in_channel]
            # (assuming stack_axis=2)
            reshaped_input_s_patches = tf.reshape(input_s_patches,
                                                  shape=s_patches_shape)
            input_patches.append(reshaped_input_s_patches)

        # input_patches almost has correct shape:
        # filter_z(possibly less) * [batch, x,y, filter_x,filter_y, in_channel]
        # (assuming stack_axis=2)
        # However, padding must still be applied

        # Pad input_patches along stack_axis dimension
        if padding == 'SAME' and method in ['locally_connected',
                                            'local_trafo']:
            if padding_s != (0, 0):
                zeros = tf.zeros_like(input_patches[0])

                # prepend zeros
                for i in range(padding_s[0]):
                    input_patches.insert(0, zeros)

                # append zeros
                for i in range(padding_s[1]):
                    input_patches.append(zeros)

        # stack input_patches along correct filter stack_axis
        input_patches = tf.stack(input_patches, axis=3 + stack_axis)

        # input_patches now has the correct shape
        # [batch, x, y, filter_x, filter_y, filter_z, in_channels]
        # So now we can perform the convolution or
        # transformation as defined by the method keyword.

        # ------------------------------
        # Perform dynamic convolution
        # ------------------------------
        if method == 'dynamic_convolution':
            # filter has shape
            # [filter_x, filter_y, filter_z, in_channels, out_channels]
            # Dimensions need to match:
            expanded_input_patches = tf.expand_dims(input_patches, -1)
            begin = [0]*5
            size = [-1]*5
            begin[stack_axis] = padding_s[0]
            size[stack_axis] = (filter_size[stack_axis] - padding_s[1]
                                - padding_s[0])
            sliced_filter = tf.slice(filter, begin, size)
            output_patch = tf.reduce_sum(
                input_tensor=expanded_input_patches * sliced_filter,
                axis=[3, 4, 5, 6],
                keepdims=False,
            )
            output.append(output_patch)
        # ------------------------------
        # Locally connected
        # ------------------------------
        elif method == 'locally_connected':
            raise NotImplementedError()

        # ------------------------------
        # Perform trafo on input_patch
        # ------------------------------
        elif method == 'local_trafo':

            output_patch = trafo(input_patches)
            output.append(output_patch)

    # concat input patches
    output = tf.stack(output, axis=1 + stack_axis)
    # output = tf.reshape(output, output_shape)

    return output


def conv3d_stacked(input, filter, strides=[1, 1, 1, 1, 1], padding='SAME'):
    '''
    Equivalent to tensorflows conv3d.
    This method is slightly slower, but appears to use less vram.

    conv3d_stacked uses tensorflows conv2d and stacks results for z-Dimension.
    Results are the same except for some settings of strides when
    padding == 'SAME'. conv3d sometimes starts the strides on index 0
    and sometimes on index 1. conv3d_stacked will always start on index 0
    and pick the strides from there.
    Example: strides = [1,1,1,3,1], padding = 'SAME'
            This means that the stride in the z-dimension is 3.
            The elements chosen are with the z index 0,3,6,9,...
            [ As opposed to 1,4,7,10... or 2,5,8,11,...]

    Parameters
    ----------
    input : A Tensor. Must be one of the following types:
        float32, float64, int64, int32, uint8, uint16, int16, int8, complex64,
        complex128, qint8, quint8, qint32, half.
        Shape [batch, in_depth, in_height, in_width, in_channels].

    filter : A Tensor. Must have the same type as input.
        Shape [filter_x, filter_y, filter_z, in_channels, out_channels].
        in_channels must match between input and filter.

    strides : A list of ints that has length >= 5. 1-D tensor of length 5.
            The stride of the sliding window for each dimension of input.
            Must have strides[0] = strides[4] = 1.
    padding : A string from: "SAME", "VALID".
        The type of padding algorithm to use.

    Returns
    -------
    A Tensor. Has the same type as input.
    '''

    # unpack along z dimension
    tensors_z = tf.unstack(input, axis=3)
    kernel_z = tf.unstack(filter, axis=2)

    len_zs = filter.get_shape().as_list()[2]
    size_of_z_dim = len(tensors_z)

    if len_zs % 2 == 1:
        # uneven filter size: same size to left and right
        filter_l = int(len_zs/2)
        filter_r = int(len_zs/2)
    else:
        # even filter size: one more to right
        filter_l = int(len_zs/2) - 1
        filter_r = int(len_zs/2)

    # The start index is important for strides
    # The strides start with the first element
    # that works and is VALID:
    start_index = 0
    if padding == 'VALID':
        for i in range(size_of_z_dim):
            if len(range(max(i - filter_l, 0),
                         min(i + filter_r+1, size_of_z_dim))) == len_zs:
                # we found the first index that doesn't need padding
                break
        start_index = i

    # loop over all z_j in z
    result_z = []
    for i in range(start_index, size_of_z_dim, strides[3]):

        if padding == 'VALID':
            # Get indices z_s
            indices_z_s = range(max(i - filter_l, 0),
                                min(i + filter_r+1, size_of_z_dim))

            # check if Padding = 'VALID'
            if len(indices_z_s) == len_zs:

                tensors_z_convoluted = []
                # sum over all remaining index_z_i in indices_z_s
                for j, index_z_i in enumerate(indices_z_s):
                    tensors_z_convoluted.append(
                                tf.nn.conv2d(input=tensors_z[index_z_i],
                                             filters=kernel_z[j],
                                             strides=strides[:3]+strides[4:],
                                             padding=padding)
                                )
                sum_tensors_z_s = tf.add_n(tensors_z_convoluted)

                # put together
                result_z.append(sum_tensors_z_s)

        elif padding == 'SAME':
            tensors_z_convoluted = []
            for kernel_j, j in enumerate(range(i - filter_l,
                                               (i + 1) + filter_r)):
                # we can just leave out the invalid z coordinates
                # since they will be padded with 0's and therfore
                # don't contribute to the sum
                if 0 <= j < size_of_z_dim:
                    tensors_z_convoluted.append(
                                tf.nn.conv2d(input=tensors_z[j],
                                             filters=kernel_z[kernel_j],
                                             strides=strides[:3]+strides[4:],
                                             padding=padding)
                                )
            sum_tensors_z_s = tf.add_n(tensors_z_convoluted)
            # put together
            result_z.append(sum_tensors_z_s)

    # stack together
    return tf.stack(result_z, axis=3)


def conv4d_stacked(input, filter,
                   strides=[1, 1, 1, 1, 1, 1],
                   padding='SAME',
                   dilation_rate=None,
                   stack_axis=None,
                   stack_nested=False,
                   ):
    '''
    Computes a convolution over 4 dimensions.
    Python generalization of tensorflow's conv3d with dilation.
    conv4d_stacked uses tensorflows conv3d and stacks results along
    stack_axis.

    Parameters
    ----------
    input : A Tensor.
        Shape [batch, x_dim, y_dim, z_dim, t_dim, in_channels]

    filter : A Tensor. Must have the same type as input.
        Shape [x_dim, y_dim, z_dim, t_dim, in_channels, out_channels].
        in_channels must match between input and filter

    strides : A list of ints that has length 6. 1-D tensor of length 6.
         The stride of the sliding window for each dimension of input.
         Must have strides[0] = strides[5] = 1.

    padding : A string from: "SAME", "VALID".
        The type of padding algorithm to use.

    dilation_rate : None or list of int of length 3
                [dilattion in x, dilation in y, dilation in z]
                defines dilattion rate to be used

    stack_axis : Int
          Axis along which the convolutions will be stacked.
          By default the axis with the lowest output dimensionality will be
          chosen. This is only an educated guess of the best choice!

    stack_nested : Bool
        If set to True, this will stack in a for loop seperately and afterwards
        combine the results.
        In most cases slower, but maybe less memory needed.

    Returns
    -------
    A Tensor. Has the same type as input.
    '''

    # heuristically choose stack_axis
    if stack_axis is None:
        if dilation_rate is None:
            dil_array = np.ones(4)
        else:
            dil_array = np.asarray(dilation_rate)
        outputsizes = (np.asarray(input.get_shape().as_list()[1:5]) /
                       np.asarray(strides[1:5]))
        outputsizes -= dil_array*(
                            np.asarray(filter.get_shape().as_list()[:4])-1)
        stack_axis = np.argmin(outputsizes)+1

    if dilation_rate is not None:
        dilation_along_stack_axis = dilation_rate[stack_axis-1]
    else:
        dilation_along_stack_axis = 1

    tensors_t = tf.unstack(input, axis=stack_axis)
    kernel_t = tf.unstack(filter, axis=stack_axis-1)

    noOfInChannels = input.get_shape().as_list()[-1]
    len_ts = filter.get_shape().as_list()[stack_axis-1]
    size_of_t_dim = input.get_shape().as_list()[stack_axis]

    if len_ts % 2 == 1:
        # uneven filter size: same size to left and right
        filter_l = int(len_ts/2)
        filter_r = int(len_ts/2)
    else:
        # even filter size: one more to right
        filter_l = int(len_ts/2) - 1
        filter_r = int(len_ts/2)

    # The start index is important for strides and dilation
    # The strides start with the first element
    # that works and is VALID:
    start_index = 0
    if padding == 'VALID':
        for i in range(size_of_t_dim):
            if len(range(max(i - dilation_along_stack_axis*filter_l, 0),
                         min(i + dilation_along_stack_axis*filter_r+1,
                             size_of_t_dim),
                         dilation_along_stack_axis)) == len_ts:
                # we found the first index that doesn't need padding
                break
        start_index = i

    # loop over all t_j in t
    result_t = []
    for i in range(start_index, size_of_t_dim, strides[stack_axis]):

        kernel_patch = []
        input_patch = []
        tensors_t_convoluted = []

        if padding == 'VALID':

            # Get indices t_s
            indices_t_s = range(max(i - dilation_along_stack_axis*filter_l, 0),
                                min(i + dilation_along_stack_axis*filter_r+1,
                                    size_of_t_dim),
                                dilation_along_stack_axis)

            # check if Padding = 'VALID'
            if len(indices_t_s) == len_ts:

                # sum over all remaining index_t_i in indices_t_s
                for j, index_t_i in enumerate(indices_t_s):
                    if not stack_nested:
                        kernel_patch.append(kernel_t[j])
                        input_patch.append(tensors_t[index_t_i])
                    else:
                        if dilation_rate is not None:
                            tensors_t_convoluted.append(
                                tf.nn.convolution(
                                    tensors_t[index_t_i],
                                    kernel_t[j],
                                    strides=(strides[1:stack_axis+1]
                                             + strides[stack_axis:5]),
                                    padding=padding,
                                    dilations=(
                                            dilation_rate[:stack_axis-1]
                                            + dilation_rate[stack_axis:]))
                                )
                        else:
                            tensors_t_convoluted.append(
                                tf.nn.conv3d(input=tensors_t[index_t_i],
                                             filters=kernel_t[j],
                                             strides=(strides[:stack_axis] +
                                                      strides[stack_axis+1:]),
                                             padding=padding)
                                )
                if stack_nested:
                    sum_tensors_t_s = tf.add_n(tensors_t_convoluted)
                    # put together
                    result_t.append(sum_tensors_t_s)

        elif padding == 'SAME':

            # Get indices t_s
            indices_t_s = range(i - dilation_along_stack_axis*filter_l,
                                (i + 1) + dilation_along_stack_axis*filter_r,
                                dilation_along_stack_axis)

            for kernel_j, j in enumerate(indices_t_s):
                # we can just leave out the invalid t coordinates
                # since they will be padded with 0's and therfore
                # don't contribute to the sum

                if 0 <= j < size_of_t_dim:
                    if not stack_nested:
                        kernel_patch.append(kernel_t[kernel_j])
                        input_patch.append(tensors_t[j])
                    else:
                        if dilation_rate is not None:
                            tensors_t_convoluted.append(
                                tf.nn.convolution(
                                    tensors_t[j],
                                    kernel_t[kernel_j],
                                    strides=(strides[1:stack_axis+1] +
                                             strides[stack_axis:5]),
                                    padding=padding,
                                    dilations=(
                                        dilation_rate[:stack_axis-1] +
                                        dilation_rate[stack_axis:]))
                                )
                        else:
                            tensors_t_convoluted.append(
                                tf.nn.conv3d(input=tensors_t[j],
                                             filters=kernel_t[kernel_j],
                                             strides=(strides[:stack_axis] +
                                                      strides[stack_axis+1:]),
                                             padding=padding)
                                        )
            if stack_nested:
                sum_tensors_t_s = tf.add_n(tensors_t_convoluted)
                # put together
                result_t.append(sum_tensors_t_s)

        if not stack_nested:
            if kernel_patch:
                kernel_patch = tf.concat(kernel_patch, axis=3)
                input_patch = tf.concat(input_patch, axis=4)
                if dilation_rate is not None:
                    result_patch = tf.nn.convolution(
                        input_patch,
                        kernel_patch,
                        strides=(strides[1:stack_axis] +
                                 strides[stack_axis+1:5]),
                        padding=padding,
                        dilations=(dilation_rate[:stack_axis-1] +
                                   dilation_rate[stack_axis:]))
                else:
                    result_patch = tf.nn.conv3d(
                                            input=input_patch,
                                            filters=kernel_patch,
                                            strides=(strides[:stack_axis] +
                                                     strides[stack_axis+1:]),
                                            padding=padding)
                result_t.append(result_patch)

    # stack together
    return tf.stack(result_t, axis=stack_axis)
