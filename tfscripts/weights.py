'''
Core functions of tfscripts:

    Create weights and biases
'''

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# constants
from tfscripts import FLOAT_PRECISION


def new_weights(shape, stddev=1.0, name="weights",
                float_precision=FLOAT_PRECISION):
    """Helper-function to create new weights

    Parameters
    ----------
    shape : list of int
        The desired shape.
    stddev : float, optional
        The initial values are sampled from a truncated gaussian with this
        std. deviation.
    name : str, optional
        The name of the tensor.
    float_precision : tf.dtype, optional
        The tensorflow dtype describing the float precision to use.

    Returns
    -------
    tf.Tensor
        A tensor with the weights.
    """
    return tf.Variable(tf.random.truncated_normal(
                            shape, stddev=stddev, dtype=float_precision),
                       name=name,
                       dtype=float_precision)


def new_locally_connected_weights(shape, stddev=1.0, name="weights",
                                  shared_axes=None,
                                  float_precision=FLOAT_PRECISION):
    """Helper-function to create new weights

    Parameters
    ----------
    shape : list of int
        The desired shape.
    stddev : float, optional
        The initial values are sampled from a truncated gaussian with this
        std. deviation.
    name : str, optional
        The name of the tensor.
    shared_axes : list of int, optional
        A list of axes over which the same initial values will be chosen.
    float_precision : tf.dtype, optional
        The tensorflow dtype describing the float precision to use.

    Returns
    -------
    tf.Tensor
        A tensor with the weights.
    """
    if shared_axes is None:
        shared_axes = []

    shape_init = []
    multiples = []
    for index, dim in enumerate(shape):
        if index in shared_axes:
            shape_init.append(1)
            multiples.append(dim)
        else:
            shape_init.append(dim)
            multiples.append(1)

    # sample initial values
    initial_value = tf.random.truncated_normal(
        shape_init, stddev=stddev, dtype=float_precision)

    # tile over shared axes
    initial_value = tf.tile(initial_value, multiples=multiples)

    return tf.Variable(initial_value, name=name, dtype=float_precision)


def new_kernel_weights(shape, stddev=0.01, name="weights",
                       float_precision=FLOAT_PRECISION):
    '''
    Get weights for a convolutional kernel. The weights will be initialised,
    so that convolution performs matrix multiplication over a single pixel.

    Assumes that the last two dimensions of shape are num_inputs, num_outputs.
    Normal distributed noise with stddev is added to break symmetry.

    Parameters
    ----------
    shape : list of int
        Shape of weights.
    stddev : float, optional
        Noise is sampled from a gaussian with this std. deviation.
    name : str, optional
        The name of the tensor.
    float_precision : tf.dtype, optional
        The tensorflow dtype describing the float precision to use.

    Returns
    -------
    tf.Tensor
        A tensor with the weights.

    '''
    weight_initialisation = np.zeros(shape)
    spatial_shape = shape[:-2]
    middle_index = [slice((dim - 1) // 2,
                    (dim - 1) // 2 + 1) for dim in spatial_shape]

    # Set value in the middle to 1 and divide by the sqrt of
    # num_inputs to maintain normal distributed output.
    # (assumes input filters are normally distributed)
    weight_initialisation[middle_index] = 1.0 / np.sqrt(shape[-2])

    # add random noise to break symmetry
    weight_initialisation += np.random.normal(size=shape, loc=0.0,
                                              scale=stddev)

    return tf.Variable(weight_initialisation,
                       name=name, dtype=float_precision)


def new_biases(length, stddev=1.0, name='biases',
               float_precision=FLOAT_PRECISION):
    """Get new biases.

    Parameters
    ----------
    length : int
        Number of biases to get.
    stddev : float, optional
        The initial values are sampled from a truncated gaussian with this
        std. deviation.
    name : str, optional
        The name of the tensor.
    float_precision : tf.dtype, optional
        The tensorflow dtype describing the float precision to use.

    Returns
    -------
    tf.Tensor
        A tensor with the biases.
    """
    return tf.Variable(tf.random.truncated_normal(shape=[length],
                                                  stddev=stddev,
                                                  dtype=float_precision),
                       name=name, dtype=float_precision)
    # return tf.Variable(tf.random_normal(shape=[length],
    #                                     stddev=2.0/length,
    #                                     dtype=float_precision),
    #                    name=name, dtype=float_precision)


def create_conv_nd_layers_weights(num_input_channels,
                                  filter_size_list,
                                  num_filters_list,
                                  name='conv_{}d',
                                  float_precision=FLOAT_PRECISION):
    '''Create weights and biases for conv 3d layers

    Parameters
    ----------
    num_input_channels : int
        Number of channels of input layer.
    filter_size_list : list of list of int
        A list of filtersizes.
        If a single filtersize is given, this will be used for every layer.
        Otherwise, a filtersize must be given for each layer.
        3D case:
            Single: filtersize = [filter_x, filter_y, filter_z]
            Multiple: [[x1, y1, z1], ... , [xn, yn, zn] ]
                      where [xi, yi, zi] is the filter size of the ith layer.
        2D case:
            Single: filtersize = [filter_x, filter_y]
            Multiple: [[x1, y1], ... , [xn, yn] ]
                      where [xi, yi] is the filter size of the ith layer.
    num_filters_list : list of int
        A list of int where each int denotes the number of filters in
        that layer.
    name : str, optional
        Name of weights and biases.
    float_precision : tf.dtype, optional
        The tensorflow dtype describing the float precision to use.

    Returns
    -------
    list of tf.Tensor, list of tf.Tensor
        Returns the list of weight and bias tensors for each layer
    '''

    num_dims = len(filter_size_list[0])
    name = name.format(num_dims)

    weights_list = []
    biases_list = []
    for i, (filter_size, num_filters) in enumerate(zip(filter_size_list,
                                                       num_filters_list)):

        # Shape of the filter-weights for the convolution.
        shape = list(filter_size) + [num_input_channels, num_filters]
        if num_dims == 1:
            shape = shape.insert(1, 1)

        weight_name = 'weights_{}_{:03d}'.format(name, i)
        bias_name = 'biases_{}_{:03d}'.format(name, i)

        # weights_list.append(new_kernel_weights(shape=shape, name=weight_name))
        weights_list.append(new_weights(shape=shape, name=weight_name,
                                        float_precision=float_precision))
        biases_list.append(new_biases(length=num_filters, name=bias_name,
                                      float_precision=float_precision))

        # update number of input channels for next layer
        num_input_channels = num_filters

    return weights_list, biases_list


def create_fc_layers_weights(num_inputs,
                             fc_sizes,
                             max_out_size_list=None,
                             name='fc',
                             float_precision=FLOAT_PRECISION):
    '''
    Create weights and biases for
    fully connected layers

    Parameters
    ----------
    num_inputs : int
        Number of inputs of the input layer.
    fc_sizes : list of int
        Number of nodes per layer.
    max_out_size_list : None, optional
        If a list of int is given, it is interpreted as the maxout size for
        each layer.
    name : str, optional
        Name of weights and biases.
    float_precision : tf.dtype, optional
        The tensorflow dtype describing the float precision to use.

    Returns
    -------
    list of tf.Tensor, list of tf.Tensor
        Returns the list of weight and bias tensors for each layer
    '''
    # create max out array
    if max_out_size_list is None:
        max_out_size_list = [None for i in range(len(fc_sizes))]

    weights_list = []
    biases_list = []
    for i, (num_outputs, max_out_size) in enumerate(zip(fc_sizes,
                                                        max_out_size_list)):

        weight_name = 'weights_{}_{:03d}'.format(name, i)
        bias_name = 'biases_{}_{:03d}'.format(name, i)

        weights_list.append(new_weights(shape=[num_inputs, num_outputs],
                                        name=weight_name,
                                        float_precision=float_precision))
        biases_list.append(new_biases(length=num_outputs, name=bias_name,
                                      float_precision=float_precision))

        if max_out_size is None:
            num_inputs = num_outputs
        else:
            num_inputs = num_outputs // max_out_size

    return weights_list, biases_list
