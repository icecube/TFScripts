'''
Core functions of tfscripts.compat.v1:

    Add residuals, batch normalisation, activation,
'''

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# tfscripts.compat.v1 specific imports
from tfscripts.compat.v1.weights import new_weights

# constants
from tfscripts.compat.v1 import FLOAT_PRECISION


def add_residual(input, residual, strides=None, use_scale_factor=True,
                 scale_factor=0.001):
    '''Convenience function to add a residual

    Will add input + scale*residual where these overlap in the last dimension
    currently only supports input and residual tensors of same shape in
    other dimensions

    Parameters
    ----------
    input : tf.Tensor
        Input tensor.
    residual : tf.Tensor
        Residual to be added to the input tensor
    strides : list of int, optional
        strides must define a stride (int) for each dimension of input.
    use_scale_factor : bool, optional
        If true, the residuals will be scaled by the scale_factor prior
        to addition.
    scale_factor : float, optional
        Defines how much the residuals will be scaled prior to addition if
        use_scale_factor is True.

    Returns
    -------
    tf.Tensor
        The output Tensor: input + scale * residual(if use_scale_factor)
    '''

    # ----------------------
    # strides for mismatching
    # dimensions other than channel
    # dimension
    # (Post Masterthesis)
    # ----------------------
    if strides is not None:

        assert len(strides) == len(input.get_shape().as_list()), \
            'Number of dimensions of strides and input must match'
        assert strides[0] == 1, 'stride in batch dimension must be 1'

        if not strides == [1 for s in strides]:
            begin = [0 for s in strides]
            end = [0] + input.get_shape().as_list()[1:]
            input = tf.strided_slice(input,
                                     begin=begin,
                                     end=end,
                                     strides=strides,
                                     begin_mask=1,
                                     end_mask=1,
                                     )
    # ----------------------

    num_outputs = residual.get_shape().as_list()[-1]
    num_inputs = input.get_shape().as_list()[-1]

    # Residuals added over multiple layers accumulate.
    # A scale factor < 1 reduces instabilities in beginnning
    if use_scale_factor:
        scale = new_weights([num_outputs], stddev=scale_factor)
        residual = residual*scale
        if num_inputs == num_outputs:
            output = residual + input
        elif num_inputs > num_outputs:
            output = residual + input[..., :num_outputs]
        elif num_inputs < num_outputs:
            output = tf.concat([residual[..., :num_inputs] + input,
                                residual[..., num_inputs:]], axis=-1)
    else:
        if num_inputs == num_outputs:
            output = (residual + input)/np.sqrt(2.)
        elif num_inputs > num_outputs:
            output = (residual + input[..., :num_outputs])/np.sqrt(2.)
        elif num_inputs < num_outputs:
            output = tf.concat(
                        [(residual[..., :num_inputs] + input)/np.sqrt(2.),
                         residual[..., num_inputs:]],
                        axis=-1)

    return output


def activation(layer, activation_type,
               use_batch_normalisation=False,
               is_training=None,
               verbose=True):
    '''
    Helper-functions to perform activation on a layer

    for parametric activation functions this assumes that the first
    dimension is batch size and that for each of the other dimensions
    seperate parametrizations should be learned

    Parameters
    ----------
    layer : tf.Tensor
        Input tensor.
    activation_type : str or callable
        The activation type to be used.
    use_batch_normalisation : bool, optional
        True: use batch normalisation
    is_training : None, optional
        Indicates whether currently in training or inference mode.
        True: in training mode
        False: inference mode.
    verbose : bool, optional
        If true, more verbose output is printed.

    Returns
    -------
    tf.Tensor
        The output tensor.

    Raises
    ------
    ValueError
        If wrong settings passed.
    '''

    # Use batch normalisation?
    if use_batch_normalisation:
        if verbose:
            print('Using Batch Normalisation')
        if is_training is None:
            raise ValueError('To use batch normalisation a boolean is_training'
                             ' needs to be passed')
        layer = batch_norm_wrapper(layer, is_training)

    if activation_type == '':
        return layer

    if hasattr(tf.nn, activation_type):
        layer = getattr(tf.nn, activation_type)(layer)

    elif hasattr(tf, activation_type):
        layer = getattr(tf, activation_type)(layer)

    elif activation_type == 'leaky':
        layer = tf.multiply(tf.maximum(-0.01*layer, layer), tf.sign(layer))
    # todo: NecroRelu
    # https://stats.stackexchange.com/questions/176794/
    #    how-does-rectilinear-activation-function-solve-the-
    #    vanishing-gradient-problem-in
    # https://github.com/ibmua/learning-to-make-nn-in-python/
    #    blob/master/nn_classifier.py
    elif activation_type == 'requ':
        layer = tf.where(tf.less(layer, tf.constant(0, dtype=FLOAT_PRECISION)),
                         tf.zeros_like(layer, dtype=FLOAT_PRECISION),
                         tf.square(layer))

    elif activation_type == 'selu':
        lam = 1.0507
        alpha = 1.6733
        # from https://arxiv.org/abs/1706.02515
        #   self normalizing networks
        layer = tf.where(tf.less(layer, tf.constant(0, dtype=FLOAT_PRECISION)),
                         tf.exp(layer) * tf.constant(alpha,
                                                     dtype=FLOAT_PRECISION)
                         - tf.constant(alpha,dtype=FLOAT_PRECISION),
                         layer)
        layer = layer * tf.constant(lam, dtype=FLOAT_PRECISION)

    elif activation_type == 'centeredRelu':
        layer = tf.nn.relu6(layer) - tf.constant(3, dtype=FLOAT_PRECISION)

    elif activation_type == 'negrelu':
        layer = -tf.nn.relu(layer)

    elif activation_type == 'invrelu':
        layer = tf.where(tf.less(layer, tf.constant(0,
                         dtype=FLOAT_PRECISION)), layer, (layer+1e-8)**-1)

    elif activation_type == 'sign':
        layer = tf.where(tf.less(layer, tf.constant(0, dtype=FLOAT_PRECISION)),
                         layer, tf.sign(layer))

    elif activation_type == 'prelu':
        slope = new_weights(layer.get_shape().as_list()[1:]) + 1.0
        layer = tf.where(tf.less(layer, tf.constant(0, dtype=FLOAT_PRECISION)),
                         layer*slope, layer)

    elif activation_type == 'pelu':
        a = new_weights(layer.get_shape().as_list()[1:]) + 1.0
        b = new_weights(layer.get_shape().as_list()[1:]) + 1.0
        layer = tf.where(tf.less(layer,
                         tf.constant(0, dtype=FLOAT_PRECISION)),
                         (tf.exp(layer/b) - 1)*a, layer*(a/b))

    elif activation_type == 'gaussian':
        layer = tf.exp(-tf.square(layer))

    elif activation_type == 'pgaussian':
        sigma = new_weights(layer.get_shape().as_list()[1:]) + \
                tf.constant(1.0, dtype=FLOAT_PRECISION)
        mu = new_weights(layer.get_shape().as_list()[1:])
        layer = tf.exp(tf.square((layer - mu) / sigma) *
                       tf.constant(-0.5, dtype=FLOAT_PRECISION)) / (sigma)

    elif callable(activation_type):
        layer = activation_type(layer)

    else:
        raise ValueError('activation: Unknown activation type: {!r}'.format(
                                                            activation_type))

    return layer


def batch_norm_wrapper(inputs, is_training, decay=0.99, epsilon=1e-6):
    ''' Batch normalisation

        Adopted from:
        http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

        Performs batch normalisation on the inputs according to
        BN2015 paper by Sergey Ioffe and Christian Szegedy

    Parameters
    ----------
    inputs:  A Tensor on which to perform batch normalisation
             Tensor will be normalised in all but

    is_training : tf.placeholder of type bool.
                  Indicates wheter the network is being trained
                  or whether it is being used in inference mode.
                  If set to true, the population mean and variance
                  will be updated and learned.

    decay :   Decay of moving exponential average

    epsilon : Small constant used in normalisation to prevent
              division by zero.

    Returns
    -------
            A Tensor. Has the same type as inputs.
            The batch normalized input
    '''
    norm_shape = inputs.get_shape().as_list()[1:]
    scale = tf.Variable(tf.ones(norm_shape, dtype=FLOAT_PRECISION),
                        name='BN_scale', dtype=FLOAT_PRECISION)
    beta = tf.Variable(tf.zeros(norm_shape,dtype=FLOAT_PRECISION),
                       name='BN_beta', dtype=FLOAT_PRECISION)
    pop_mean = tf.Variable(tf.zeros(norm_shape, dtype=FLOAT_PRECISION),
                           trainable=False,
                           name='BN_pop_mean',
                           dtype=FLOAT_PRECISION)
    pop_var = tf.Variable(tf.ones(norm_shape, dtype=FLOAT_PRECISION),
                          trainable=False,
                          name='BN_pop_var',
                          dtype=FLOAT_PRECISION)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(x=inputs, axes=[0], keepdims=False)
        train_mean = tf.compat.v1.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.compat.v1.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(
                inputs,
                batch_mean, batch_var, beta, scale, epsilon,  # R2RT's blog
                # pop_mean, pop_var, beta, scale, epsilon,
                )
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta,
                                         scale, epsilon)
