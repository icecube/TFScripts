'''
tfscripts.compat.v1 hexagonal convolution utility functions
    Hex utility functions
    hex conv 3d and 4d

ToDo:
    - Remove duplicate code
'''

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# tfscripts.compat.v1 specific imports
from tfscripts.compat.v1.weights import (
    new_weights, new_biases, new_kernel_weights)
from tfscripts.compat.v1.hex.visual import print_hex_data
from tfscripts.compat.v1.hex import rotation
from tfscripts.compat.v1.hex.icecube import get_icecube_kernel
from tfscripts.compat.v1.conv import dynamic_conv, conv4d_stacked

# constants
from tfscripts.compat.v1 import FLOAT_PRECISION


def get_num_hex_points(edge_length):
    """Get number of hexagonal points for a hexagon with a given edge length.

    Parameters
    ----------
    edge_length : int
        Edge length of a hexagon: number of points along one edge of the
        symmetric hexagon.

    Returns
    -------
    int
        Number of points in a hexagon with a given edge length.

    Raises
    ------
    ValueError
        Description
    """
    if edge_length < 0:
        raise ValueError('get_num_hex_points: expected edge_length >= 0')
    if edge_length == 0:
        return 1
    return (edge_length-1)*6 + get_num_hex_points(edge_length-1)


def hex_distance(h1, h2):
    """Get hexagonal distance (manhattan distance) of two hexagon points
    given by the hexagonal coordinates h1 and h2

    Parameters
    ----------
    h1 : int, int
        Hexagonal coordinates of point 1.
    h2 : int, int
        Hexagonal coordinates of point 2.

    Returns
    -------
    float
        distance
    """
    a1, b1 = h1
    a2, b2 = h2
    c1 = -a1-b1
    c2 = -a2-b2
    return (abs(a1 - a2) + abs(b1 - b2) + abs(c1 - c2)) / 2


def get_hex_kernel(filter_size, print_kernel=False, get_ones=False):
    '''Get hexagonal convolution kernel

    Create Weights for a hexagonal kernel.
    The Kernel will be of a hexagonal shape in the first two dimensions,
    while the other dimensions are normal.
    The hexagonal kernel is off the shape:
    [kernel_edge_points, kernel_edge_points, *filter_size[2:]]
    But elments with coordinates in the first two dimensions, that don't belong
    to the hexagon are set to a tf.Constant 0.

    The hexagon is defined by filter_size[0:2].
    filter_size[0] defines the size of the hexagon and
    filter_size[1] the orientation.

    Parameters
    ----------
    filter_size : A list of int
        filter_size = [s, o, 3. dim(e.g. z), 4. dim(e.g. t),...]
        s: size of hexagon
        o: orientation of hexagon

        Examples:

                  s = 2, o = 0:
                                    1   1   0             1  1

                                 1   1   1             1   1   1

                               0   1   1                 1   1

                  s = 3, o = 2:
                          0   1   0   0   0   0   0               1

                       0   1   1   1   1   0   0               1   1   1   1

                     1   1   1   1   1   0   0           1   1   1   1   1

                   0   1   1   1   1   1   0               1   1   1   1   1

                 0   0   1   1   1   1   1                   1   1   1   1   1

               0   0   1   1   1   1   0                    1   1   1   1

             0   0   0   0   0   1   0                               1

    print_kernel : bool.
      True: print first two dimensions of kernel.
            0 represents a const 0 Tensor of shape filter_size[2:]
            1 represents a trainable Tensor of shape filter_size[2:]
            This can be used to verify the shape of the hex kernel
      False: do not print

    get_ones : bool, optional
        If True, returns constant ones for elements in hexagon.
        If False, return trainable tf.tensor for elements in hexagon.
        In both cases, constant zeros are returned for elements outside of
        hexagon.

    Returns
    -------
    tf.Tensor
        A Tensor with shape: [ s, s, *filter_size[2:] ]
        where s = 2*filter_size[0] -1 if x == o
                            [hexagon is parallel to axis of first dimension]
                = 2*filter_size[0] +1 if x != o
                         [hexagon is tilted to axis of first dimension]

    Raises
    ------
    ValueError
        Description
    '''
    k = filter_size[0]
    x = filter_size[1]

    if x >= k:
        raise ValueError("get_hex_kernel: filter_size (k,x,z) must fulfill "
                         "x < k: ({}, {}, {})".format(k, x, filter_size[2]))
    if x == 0:
        kernel_edge_points = 2*k - 1
    else:
        kernel_edge_points = 2*k + 1

    zeros = tf.zeros(filter_size[2:], dtype=FLOAT_PRECISION)
    ones = tf.ones(filter_size[2:], dtype=FLOAT_PRECISION)

    a_list = []
    test_hex_dict = {}
    for a in range(kernel_edge_points):
        b_list = []
        for b in range(kernel_edge_points):

            # -------------------------
            # regular aligned hexagons
            # -------------------------
            if x == 0:
                if a+b < k - 1 or a + b > 3*k - 3:
                    weights = zeros
                    test_hex_dict[(a, b)] = 0
                else:
                    if get_ones:
                        weights = ones
                    else:
                        weights = new_weights(filter_size[2:])
                    test_hex_dict[(a, b)] = 1

            # -------------------------
            # tilted hexagons
            # -------------------------
            else:
                inHexagon = False
                # check if inside normal k.0 aligned hexagon
                #   |----inside normal k.0 rhombus -----------|
                if ((a > 0 and a < 2*k) and (b > 0 and b < 2*k) and
                    #   |--in k.0 aligned hexagon-|
                        (a+b > k and a + b < 3*k)):

                    if a+b > k and a + b < 3*k:
                        inHexagon = True
                else:
                    # add 6 additional edges outside of k.0 aligned hexagon
                    if a == 2*k-x and b == 0:  # Edge 1
                        inHexagon = True
                    elif a == k - x and b == x:  # Edge 2
                        inHexagon = True
                    elif a == 0 and b == k+x:  # Edge 3
                        inHexagon = True
                    elif a == x and b == 2*k:  # Edge 4
                        inHexagon = True
                    elif a == k+x and b == 2*k-x:  # Edge 5
                        inHexagon = True
                    elif a == 2*k and b == k-x:  # Edge 6
                        inHexagon = True
                # get weights or constant 0 depending on if point is in hexagon
                if inHexagon:
                    if get_ones:
                        weights = ones
                    else:
                        weights = new_weights(filter_size[2:])
                    test_hex_dict[(a, b)] = 1
                else:
                    weights = zeros
                    test_hex_dict[(a, b)] = 0

            b_list.append(weights)
        a_list.append(tf.stack(b_list))
    hexKernel = tf.stack(a_list)
    if print_kernel:
        print_hex_data(test_hex_dict)
    return hexKernel


def conv_hex(input_data,
             filter_size,
             num_filters,
             padding='SAME',
             strides=[1, 1, 1, 1, 1],
             num_rotations=1,
             dilation_rate=None,
             zero_out=False,
             kernel=None,
             azimuth=None,
             ):
    """Convolve a hex2d or hex3d layer (2d hex + 1d cartesian)

    Parameters
    ----------
    input_data : tf.Tensor
        Input data.
    filter_size : A list of int
        filter_size = [s, o, z]
        s: size of hexagon
        o: orientation of hexagon
        z: size along z axis

        The hexagonal filter along axis x and y is put together from s and o.

        Examples:

                  s = 2, o = 0:
                                    1   1   0             1  1

                                 1   1   1             1   1   1

                               0   1   1                 1   1

                  s = 3, o = 2:
                          0   1   0   0   0   0   0               1

                       0   1   1   1   1   0   0               1   1   1   1

                     1   1   1   1   1   0   0           1   1   1   1   1

                   0   1   1   1   1   1   0               1   1   1   1   1

                 0   0   1   1   1   1   1                   1   1   1   1   1

               0   0   1   1   1   1   0                    1   1   1   1

             0   0   0   0   0   1   0                               1
    num_filters : int
        The number of filters to use in the convolution operation.
    padding : str, optional
        The padding method to be used for the convolution operation.
        Options are 'VALID', 'SAME'.
    strides : list, optional
        The strides to be used for the convolution operation.
        Shape: [1, stride_x, stride_y, stride_z, stride_channel]
        Examples:
            [1, 1, 1, 1, 1]: a stride of 1 is used along all axes.
            [1, 1, 2, 1, 1]: a stride of 2 is used along the y axis.
    num_rotations : int, optional
        If num_rotations >= 1: weights of a kernel will be shared over
            'num_rotations' many rotated versions of that kernel.
    dilation_rate : None or list of int, optional
        The dilation rate to be used for the layer.
        Dilation rate is given by: [dilation_x, dilation_y, dilation_z]
        If the dilation rate is None, no dilation is applied in convolution.
    zero_out : bool, optional
        If True, elements in result tensor which are not part of hexagon or
        IceCube strings (if shape in x and y dimensions is 10x10), will be
        set to zero.
    kernel :  None or tf.Tensor, optional
        Optionally, the weights to be used as the kernel can be provided.
        If None, new kernel weights are created.
    azimuth : float or scalar float tf.Tensor
        Hexagonal kernel is turned by the angle 'azimuth' [given in degrees]
        in counterclockwise direction

    Returns
    -------
    tf.Tensor, tf.Tensor
        Result and kernel weights

    Raises
    ------
    ValueError
        Description
    """

    # make sure it is a 2d or 3d convolution
    assert len(input_data.get_shape()) == 4 or len(input_data.get_shape()) == 5

    # allocate random weights for kernel
    num_channels = input_data.get_shape().as_list()[4]

    if kernel is None:
        if azimuth is not None and filter_size[:2] != [1, 0]:
            kernel = rotation.get_dynamic_rotation_hex_kernel(
                            filter_size+[num_channels, num_filters], azimuth)
        else:
            if num_rotations > 1:
                kernel = rotation.get_rotated_hex_kernel(
                        filter_size+[num_channels, num_filters], num_rotations)
            else:
                kernel = get_hex_kernel(
                                filter_size+[num_channels, num_filters])

    if azimuth is not None and filter_size[:2] != [1, 0]:
        result = dynamic_conv(
                            input=input_data,
                            filter=kernel,
                            strides=strides[1:-1],
                            padding=padding,
                            dilation_rate=dilation_rate,
                            )
    else:
        result = tf.nn.convolution(input_data,
                                   kernel,
                                   strides=strides[1:-1],
                                   padding=padding,
                                   dilations=dilation_rate)

    # zero out elements that don't belong on hexagon or IceCube Strings
    if zero_out:

        if result.get_shape().as_list()[1:3] == [10, 10]:
            # IceCube shape
            print('Assuming IceCube shape for layer', result)

            zero_out_matrix = get_icecube_kernel(
                                            result.get_shape().as_list()[3:],
                                            get_ones=True)
            result = result*zero_out_matrix

        else:
            # Generic hexagonal shape
            zero_out_matrix = get_hex_kernel(
                                [(result.get_shape().as_list()[1]+1) // 2, 0,
                                 result.get_shape().as_list()[3],
                                 num_filters*num_rotations],
                                get_ones=True)
            if result.get_shape()[1:] == zero_out_matrix.get_shape():
                result = result*zero_out_matrix
            else:
                print(result, zero_out_matrix)
                raise ValueError("conv_hex3d: Shapes do not match for "
                                 "zero_out_matrix and result. "
                                 " {!r} != {!r}".format(
                                                result.get_shape()[1:],
                                                zero_out_matrix.get_shape()))

    return result, kernel


# create explicit 2d and 3d versions
conv_hex2d = conv_hex
conv_hex3d = conv_hex


def conv_hex4d(input_data,
               filter_size,
               num_filters,
               padding='VALID',
               strides=[1, 1, 1, 1, 1, 1],
               num_rotations=1,
               dilation_rate=None,
               kernel=None,
               azimuth=None,
               stack_axis=None,
               zero_out=False):
    """Convolve a hex4d layer (2d hex + 1d cartesian + 1d time)

    Parameters
    ----------
    input_data : tf.Tensor
        Input data.
    filter_size : A list of int
        filter_size = [s, o, z, t]
        s: size of hexagon
        o: orientation of hexagon
        z: size along z axis
        t: size along t axis

        The hexagonal filter along axis x and y is put together from s and o.

        Examples:

                  s = 2, o = 0:
                                    1   1   0             1  1

                                 1   1   1             1   1   1

                               0   1   1                 1   1

                  s = 3, o = 2:
                          0   1   0   0   0   0   0               1

                       0   1   1   1   1   0   0               1   1   1   1

                     1   1   1   1   1   0   0           1   1   1   1   1

                   0   1   1   1   1   1   0               1   1   1   1   1

                 0   0   1   1   1   1   1                   1   1   1   1   1

               0   0   1   1   1   1   0                    1   1   1   1

             0   0   0   0   0   1   0                               1
    num_filters : int
        The number of filters to use in the convolution operation.
    padding : str, optional
        The padding method to be used for the convolution operation.
        Options are 'VALID', 'SAME'.
    strides : list, optional
        The strides to be used for the convolution operation.
        Shape: [1, stride_x, stride_y, stride_z, stride_t, stride_channel]
        Examples:
            [1, 1, 1, 1, 1, 1]: a stride of 1 is used along all axes.
            [1, 1, 2, 1, 2, 1]: a stride of 2 is used along the y and t axis.
    num_rotations : int, optional
        If num_rotations >= 1: weights of a kernel will be shared over
            'num_rotations' many rotated versions of that kernel.
    dilation_rate : None or list of int, optional
        The dilation rate to be used for the layer.
        Dilation rate is given by: [dilation_x, dilation_y, dilation_z]
        If the dilation rate is None, no dilation is applied in convolution.
    kernel :  None or tf.Tensor, optional
        Optionally, the weights to be used as the kernel can be provided.
        If None, new kernel weights are created.
    azimuth : float or scalar float tf.Tensor
        Hexagonal kernel is turned by the angle 'azimuth' [given in degrees]
        in counterclockwise direction
    stack_axis : Int
          Axis along which the convolutions will be stacked.
          By default the axis with the lowest output dimensionality will be
          chosen. This is only an educated guess of the best choice!
    zero_out : bool, optional
        If True, elements in result tensor which are not part of hexagon or
        IceCube strings (if shape in x and y dimensions is 10x10), will be
        set to zero.

    Returns
    -------
    tf.Tensor, tf.Tensor
        Result and kernel weights

    Raises
    ------
    ValueError
        Description
    """
    # allocate random weights for kernel
    if kernel is None:
        num_in_channels = input_data.get_shape().as_list()[5]
        if azimuth is not None:
            kernel = rotation.get_dynamic_rotation_hex_kernel(
                            filter_size+[num_channels, num_filters], azimuth)
        else:
            if num_rotations > 1:
                kernel = rotation.get_rotated_hex_kernel(
                    filter_size+[num_in_channels, num_filters], num_rotations)
            else:
                kernel = get_hex_kernel(
                    filter_size+[num_in_channels, num_filters])

    # convolve with tf conv4d_stacked
    result = conv4d_stacked(input=input_data,
                            filter=kernel,
                            strides=strides,
                            padding=padding,
                            dilation_rate=dilation_rate,
                            stack_axis=stack_axis)

    # zero out elements that don't belong on hexagon
    if zero_out:
        zero_out_matrix = get_hex_kernel(
                                [int((result.get_shape().as_list()[1]+1)/2), 0,
                                 result.get_shape().as_list()[3],
                                 result.get_shape().as_list()[4],
                                 num_filters*num_rotations],
                                get_ones=True)

        if result.get_shape()[1:] == zero_out_matrix.get_shape():
            result = result*zero_out_matrix
        else:
            print(result, zero_out_matrix)
            raise ValueError("conv_hex4d: Shapes do not match for "
                             "zero_out_matrix and result. {!r} != {!r}".format(
                                                result.get_shape()[1:],
                                                zero_out_matrix.get_shape()))

    return result, kernel


def create_conv_hex_layers_weights(num_input_channels,
                                   filter_size_list,
                                   num_filters_list,
                                   num_rotations_list=1,
                                   azimuth_list=None,
                                   ):
    '''Create weights and biases for conv hex n-dimensional layers with n >= 2

    Parameters
    ----------
    num_input_channels : int
        Number of channels of input layer.
    filter_size_list : list of int or list of list of int
        A list of filter sizes.
        If only one filter_size is given, this will be used for all layers.
        filter_size : A list of int
        filter_size = [s, o, 3. dim(e.g. z), 4. dim(e.g. t),...]
        s: size of hexagon
        o: orientation of hexagon

        Examples:

                  s = 2, o = 0:
                                    1   1   0             1  1

                                 1   1   1             1   1   1

                               0   1   1                 1   1

                  s = 3, o = 2:
                          0   1   0   0   0   0   0               1

                       0   1   1   1   1   0   0               1   1   1   1

                     1   1   1   1   1   0   0           1   1   1   1   1

                   0   1   1   1   1   1   0               1   1   1   1   1

                 0   0   1   1   1   1   1                   1   1   1   1   1

               0   0   1   1   1   1   0                    1   1   1   1

             0   0   0   0   0   1   0                               1
    num_filters_list : list of int
        A list of int where each int denotes the number of filters in
        that layer.
    num_rotations_list : int or list of int, optional
        The number of rotations to use for each layer.
        If num_rotations >= 1: weights of a kernel will be shared over
            'num_rotations' many rotated versions of that kernel.
        If only a single number is given, the same number of rotations will be
        used for all layers.
    azimuth_list : None, optional
        A list of floats or scalar tf.tensors denoting the azimuth angle by
        which the kernel of each layer is rotated.
        Hexagonal kernel is turned by the angle 'azimuth' [given in degrees]
        in counterclockwise direction.
        If only a single azimuth angle is given, the same rotation is used for
        all layers.
        If azimuth is None, the hexagonal kernel is not rotated.

    Returns
    -------
    list of tf.Tensor, list of tf.Tensor
        Returns the list of weight and bias tensors for each layer
    '''

    # create num_rotations_list
    if isinstance(num_rotations_list, int):
        num_rotations_list = [num_rotations_list for i
                              in range(len(num_filters_list))]
    # create azimuth_list
    if azimuth_list is None or tf.is_tensor(azimuth_list):
        azimuth_list = [azimuth_list for i in range(noOfLayers)]

    weights_list = []
    biases_list = []
    for filter_size, num_filters, num_rotations, azimuth in zip(
                                        filter_size_list,
                                        num_filters_list,
                                        num_rotations_list,
                                        azimuth_list,
                                        ):
        if azimuth is not None:
            kernel = rotation.get_dynamic_rotation_hex_kernel(filter_size,
                                                              azimuth)
        else:
            if num_rotations > 1:
                kernel = rotation.get_rotated_hex_kernel(
                                        filter_size +
                                        [num_input_channels, num_filters],
                                        num_rotations)
            else:
                kernel = get_hex_kernel(filter_size+[num_input_channels,
                                                     num_filters])

        weights_list.append(kernel)
        biases_list.append(new_biases(length=num_filters*num_rotations))
        num_input_channels = num_filters

    return weights_list, biases_list
