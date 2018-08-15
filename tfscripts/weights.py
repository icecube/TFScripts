'''
Core functions of tfscripts:

    Create weights and biases
'''

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# constants
FLOAT_PRECISION = tf.float32


def new_weights(shape, stddev=1.0, name="weights"):
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

    Returns
    -------
    tf.Tensor
        A tensor with the weights.
    """
    return tf.Variable(tf.truncated_normal(shape,
                                           stddev=stddev,
                                           dtype=FLOAT_PRECISION),
                                           name=name,
                                           dtype=FLOAT_PRECISION)


def new_kernel_weights(shape, stddev=0.01, name="weights"):
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
                       name=name, dtype=FLOAT_PRECISION)


def new_biases(length, name='biases'):
    """Get new biases.

    Parameters
    ----------
    length : int
        Number of biases to get.
    name : str, optional
        The name of the tensor.

    Returns
    -------
    tf.Tensor
        A tensor with the biases.
    """
    return tf.Variable(tf.random_normal(shape=[length],
                                        stddev=2.0/length,
                                        dtype=float_precision),
                       name=name, dtype=float_precision)


def create_conv_layers_weights(num_input_channels,
                               filter_size_list,
                               num_filters_list,
                               name="conv1d"):
    ''' Create weights and biases for conv 1d layers

    Parameters
    ----------
    num_input_channels : int
        Number of channels of input layer.
    filter_size_list : list of int
        A list of int where each int is the filter size for that layer.
    num_filters_list : list of int
        A list of int where each int denotes the number of filters in
        that layer.
    name : str, optional
        Name of weights and biases.

    Returns
    -------
    list of tf.Tensor, list of tf.Tensor
        Returns the list of weight and bias tensors for each layer
    '''
    weights_list = []
    biases_list = []
    for i, (filter_size, num_filters) in enumerate(zip(filter_size_list,
                                                       num_filters_list)):
        shape = [filter_size, 1, num_input_channels, num_filters]

        weight_name = 'weights_{}_{:03d}'.format(name, i)
        bias_name = 'biases_{}_{:03d}'.format(name, i)

        weights_list.append(new_kernel_weights(shape=shape, name=weight_name))
        biases_list.append(new_biases(length=num_filters, name=bias_name))
        num_input_channels = num_filters

    return weights_list, biases_list


def create_conv2d_layers_weights(num_input_channels,
                                 filter_size_list,
                                 num_filters_list,
                                 name='conv2d'):
    '''Create weights and biases for conv 2d layers

    Parameters
    ----------
    num_input_channels : int
        Number of channels of input layer.
    filter_size_list : list of (int, int)
        A list of tuples (int, int) which denote the filter size along the
        x and y axis for each layer.

    num_filters_list : list of int
        A list of int where each int denotes the number of filters in
        that layer.
    name : str, optional
        Name of weights and biases.

    Returns
    -------
    list of tf.Tensor, list of tf.Tensor
        Returns the list of weight and bias tensors for each layer
    '''
    weights_list = []
    biases_list = []
    for i, (filter_size, num_filters) in enumerate(zip(filter_size_list,
                                                       num_filters_list)):
        shape = [filter_size[0],
                 filter_size[1],
                 num_input_channels,
                 num_filters]

        weight_name = 'weights_{}_{:03d}'.format(name, i)
        bias_name = 'biases_{}_{:03d}'.format(name, i)

        weights_list.append(new_kernel_weights(shape=shape, name=weight_name))
        biases_list.append(new_biases(length=num_filters, name=bias_name))
        num_input_channels = num_filters

    return weights_list, biases_list


def create_conv3d_layers_weights(num_input_channels,
                                 filterXYZ_list,
                                 num_filters_list,
                                 name='conv3d'):
    '''Create weights and biases for conv 3d layers

    Parameters
    ----------
    num_input_channels : int
        Number of channels of input layer.
    filterXYZ_list : list of (int, int, int)
        A list of tuples (int, int, int) which denote the filter size along the
        x, y, and z axis for each layer.
    num_filters_list : list of int
        A list of int where each int denotes the number of filters in
        that layer.
    name : str, optional
        Name of weights and biases.

    Returns
    -------
    list of tf.Tensor, list of tf.Tensor
        Returns the list of weight and bias tensors for each layer
    '''
    weights_list = []
    biases_list = []
    for i, (filterXYZ, num_filters) in enumerate(zip(filterXYZ_list,
                                                     num_filters_list)):
        shape = [filterXYZ[0],
                 filterXYZ[1],
                 filterXYZ[2],
                 num_input_channels,
                 num_filters]

        weight_name = 'weights_{}_{:03d}'.format(name, i)
        bias_name = 'biases_{}_{:03d}'.format(name, i)

        weights_list.append(new_kernel_weights(shape=shape, name=weight_name))
        biases_list.append(new_biases(length=num_filters, name=bias_name))
        num_input_channels = num_filters

    return weights_list, biases_list


def create_fc_layers_weights(num_inputs,
                             fc_sizes,
                             max_out_size_list=None,
                             name='fc'):
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
                                        name=weight_name))
        biases_list.append(new_biases(length=num_outputs, name=bias_name))

        if max_out_size is None:
            num_inputs = num_outputs
        else:
            num_inputs = num_outputs // max_out_size

    return weights_list, biases_list
