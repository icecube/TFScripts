'''
Core functions of tfscripts:

    Add residuals, batch normalisation, activation,
'''

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# tfscripts specific imports
from tfscripts.weights import new_weights

# constants
from tfscripts import FLOAT_PRECISION


class AddResidual(tf.Module):
    """Convenience Module to add a residual

    Will add input + scale*residual where these overlap in the last dimension
    currently only supports input and residual tensors of same shape in
    other dimensions
    """

    def __init__(self,
                 residual_shape,
                 strides=None,
                 use_scale_factor=True,
                 scale_factor=0.001,
                 float_precision=FLOAT_PRECISION,
                 name=None):
        """Initialize object

        Parameters
        ----------
        residual_shape : TensorShape, or list of int
            The shape of the residual.
        strides : list of int, optional
            strides must define a stride (int) for each dimension of input.
        use_scale_factor : bool, optional
            If true, the residuals will be scaled by the scale_factor prior
            to addition.
        scale_factor : float, optional
            Defines how much the residuals will be scaled prior to addition if
            use_scale_factor is True.
        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        name : None, optional
            The name of the tensorflow module.
        """
        super(AddResidual, self).__init__(name=name)

        self.num_outputs = residual_shape[-1]
        self.use_scale_factor = use_scale_factor
        self.scale_factor = scale_factor
        self.strides = strides

        # Residuals added over multiple layers accumulate.
        # A scale factor < 1 reduces instabilities in beginnning
        if self.use_scale_factor:
            self.scale = new_weights([self.num_outputs],
                                     stddev=self.scale_factor,
                                     float_precision=float_precision)

    def __call__(self, input, residual):
        '''Apply residual additions

        Parameters
        ----------
        input : tf.Tensor
            Input tensor.
        residual : tf.Tensor
            Residual to be added to the input tensor

        Returns
        -------
        tf.Tensor
            The output Tensor: input + scale * residual(if use_scale_factor)
        '''

        # make sure the shape of the residual is correct in last dimension
        assert residual.get_shape()[-1] == self.num_outputs

        # ----------------------
        # strides for mismatching dimensions other than channel dimension
        # ----------------------
        if self.strides is not None:

            msg = 'Number of dimensions of strides and input must match'
            assert len(self.strides) == len(input.get_shape().as_list()), msg
            assert self.strides[0] == 1, 'stride in batch dimension must be 1'

            if not self.strides == [1 for s in self.strides]:
                begin = [0 for s in self.strides]
                end = [0] + input.get_shape().as_list()[1:]
                input = tf.strided_slice(input,
                                         begin=begin,
                                         end=end,
                                         strides=self.strides,
                                         begin_mask=1,
                                         end_mask=1,
                                         )
        # ----------------------

        num_outputs = residual.get_shape().as_list()[-1]
        num_inputs = input.get_shape().as_list()[-1]

        # Residuals added over multiple layers accumulate.
        # A scale factor < 1 reduces instabilities in beginnning
        if self.use_scale_factor:
            residual = residual * self.scale
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


class Activation(tf.Module):
    """Helper-Module to perform activation on a layer

    For parametric activation functions this assumes that the first
    dimension is batch size and that for each of the other dimensions
    seperate parametrizations should be learned.

    """

    def __init__(self,
                 activation_type,
                 input_shape=None,
                 use_batch_normalisation=False,
                 float_precision=FLOAT_PRECISION,
                 name=None):
        """Initialize object

        Parameters
        ----------
        activation_type : str or callable
            The activation type to be used.
        input_shape : TensorShape, or list of int
            The shape of the inputs.
        use_batch_normalisation : bool, optional
        True: use batch normalisation
        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        name : None, optional
            The name of the tensorflow module.
        """
        super(Activation, self).__init__(name=name)

        self.activation_type = activation_type
        self.use_batch_normalisation = use_batch_normalisation
        self.float_precision = float_precision

        if self.use_batch_normalisation:
            self.batch_norm_layer = BatchNormWrapper(
                                        input_shape=input_shape,
                                        float_precision=float_precision,
                                        name=name)

        if activation_type == 'prelu':
            self.slope_weight = new_weights(input_shape[1:],
                                            float_precision=float_precision)

        elif activation_type == 'pelu':
            self.a_weight = new_weights(input_shape[1:],
                                        float_precision=float_precision)
            self.b_weight = new_weights(input_shape[1:],
                                        float_precision=float_precision)

        elif activation_type == 'pgaussian':
            self.sigma_weight = new_weights(input_shape[1:],
                                            float_precision=float_precision)
            self.mu = new_weights(input_shape[1:],
                                  float_precision=float_precision)

    def __call__(self, layer, is_training=None):
        """Apply Activation Module.

        Parameters
        ----------
        layer : tf.Tensor
            Input tensor.
        is_training : None, optional
            Indicates whether currently in training or inference mode.
            True: in training mode
            False: inference mode.

        Returns
        -------
        tf.Tensor
            The output tensor.

        Raises
        ------
        ValueError
            If wrong settings passed.
        """
        activation_type = self.activation_type

        # Use batch normalisation?
        if self.use_batch_normalisation:
            if is_training is None:
                raise ValueError('To use batch normalisation a boolean '
                                 'is_training needs to be passed')
            layer = self.batch_norm_layer(layer, is_training)

        if activation_type == '' or activation_type is None:
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
            layer = tf.where(tf.less(layer, 0.),
                             tf.zeros_like(layer, dtype=self.float_precision),
                             tf.square(layer))

        elif activation_type == 'selu':
            lam = 1.0507
            alpha = 1.6733
            # from https://arxiv.org/abs/1706.02515
            #   self normalizing networks
            layer = tf.where(tf.less(layer, 0.),
                             tf.exp(layer) * alpha - alpha,
                             layer)
            layer = layer * lam

        elif activation_type == 'centeredRelu':
            layer = tf.nn.relu6(layer) - 3.

        elif activation_type == 'negrelu':
            layer = -tf.nn.relu(layer)

        elif activation_type == 'invrelu':
            layer = tf.where(tf.less(layer, 0.), layer, (layer+1e-8)**-1)

        elif activation_type == 'sign':
            layer = tf.where(tf.less(layer, 0.), layer, tf.sign(layer))

        elif activation_type == 'prelu':
            slope = self.slope_weight + 1.0
            layer = tf.where(tf.less(layer, 0.), layer*slope, layer)

        elif activation_type == 'pelu':
            a = self.a_weight + 1.0
            b = self.b_weight + 1.0
            layer = tf.where(tf.less(layer, 0.),
                             (tf.exp(layer/b) - 1)*a, layer*(a/b))

        elif activation_type == 'gaussian':
            layer = tf.exp(-tf.square(layer))

        elif activation_type == 'pgaussian':
            sigma = self.sigma_weight + 1.0
            layer = tf.exp(tf.square((layer - self.mu) / sigma) *
                           (-0.5)) / (sigma)

        elif callable(activation_type):
            layer = activation_type(layer)

        else:
            msg = 'Unknown activation type: {!r}'
            raise ValueError(msg.format(activation_type))

        return layer


class BatchNormWrapper(tf.Module):
    """Batch normalisation

    Adopted from:
    http://r2rt.com/implementing-batch-normalization-in-tensorflow.html

    Performs batch normalisation on the inputs according to
    BN2015 paper by Sergey Ioffe and Christian Szegedy
    """

    def __init__(self,
                 input_shape,
                 decay=0.99,
                 epsilon=1e-6,
                 float_precision=FLOAT_PRECISION,
                 name=None):
        """Initialize object

        Parameters
        ----------
        input_shape : TensorShape, or list of int
            The shape of the inputs.
        decay : float, optional
            Decay of moving exponential average
        epsilon : float, optional
            Small constant used in normalisation to prevent division by zero.
        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        name : None, optional
            The name of the tensorflow module.
        """
        super(BatchNormWrapper, self).__init__(name=name)

        norm_shape = input_shape[1:]
        self.epsilon = epsilon
        self.decay = decay
        self.scale = tf.Variable(tf.ones(norm_shape, dtype=float_precision),
                                 name='BN_scale', dtype=float_precision)
        self.beta = tf.Variable(tf.zeros(norm_shape, dtype=float_precision),
                                name='BN_beta', dtype=float_precision)
        self.pop_mean = tf.Variable(tf.zeros(norm_shape,
                                             dtype=float_precision),
                                    trainable=False,
                                    name='BN_pop_mean',
                                    dtype=float_precision)
        self.pop_var = tf.Variable(tf.ones(norm_shape, dtype=float_precision),
                                   trainable=False,
                                   name='BN_pop_var',
                                   dtype=float_precision)

    def __call__(self, inputs, is_training):
        """Apply Batch Normalization Wrapper

        Parameters
        ----------
        inputs : tf.Tensor
            A Tensor on which to perform batch normalisation.
            Tensor will be normalised in all but batch dimension.

        is_training : bool, or tf.Tensor of type bool.
          Indicates wheter the network is being trained
          or whether it is being used in inference mode.
          If set to true, the population mean and variance
          will be updated and learned.

        Returns
        -------
        tf.Tensor
            The batch normalized output
        """
        if is_training:
            batch_mean, batch_var = tf.nn.moments(x=inputs,
                                                  axes=[0],
                                                  keepdims=False)
            train_mean = tf.compat.v1.assign(
                self.pop_mean,
                self.pop_mean * self.decay + batch_mean * (1 - self.decay))
            train_var = tf.compat.v1.assign(
                self.pop_var,
                self.pop_var * self.decay + batch_var * (1 - self.decay))

            with tf.control_dependencies([train_mean, train_var]):
                return tf.nn.batch_normalization(
                    inputs,
                    batch_mean, batch_var,  # R2RT's blog
                    # self.pop_mean, self.pop_var,
                    self.beta, self.scale, self.epsilon,
                    )
        else:
            return tf.nn.batch_normalization(inputs, self.pop_mean,
                                             self.pop_var, self.beta,
                                             self.scale, self.epsilon)


def maxout(inputs, num_units, axis=-1):
    """Applies Maxout to the input.

    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
    Courville, Yoshua Bengio. https://arxiv.org/abs/1302.4389
    Usually the operation is performed in the filter/channel dimension. This
    can also be used after Dense layers to reduce number of features.

    Adopted from:
        https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/
        layers/maxout.py

    Parameters
    ----------
    inputs : nD tensor
        The input data tensor on which to apply the the maxout operation.
        shape: (batch_size, ..., axis_dim, ...)
    num_units
        Specifies how many features will remain after maxout
        in the axis dimension (usually channel).
        This must be a factor of number of features.
    axis
        The dimension where max pooling will be performed. Default is the
        last dimension.

    Returns
    -------
    nD tensor
        The output tensor after applying maxout operation.
        shape: (batch_size, ..., num_units, ...)
    """
    inputs = tf.convert_to_tensor(inputs)
    shape = inputs.get_shape().as_list()

    # Dealing with batches with arbitrary sizes
    for i in range(len(shape)):
        if shape[i] is None:
            shape[i] = tf.shape(inputs)[i]

    num_channels = shape[axis]
    if (not isinstance(num_channels, tf.Tensor)
            and num_channels % num_units):
        raise ValueError('number of features({}) is not '
                         'a multiple of num_units({})'.format(
                             num_channels, num_units))

    if axis < 0:
        axis = axis + len(shape)
    else:
        axis = axis
    assert axis >= 0, 'Find invalid axis: {}'.format(axis)

    expand_shape = shape[:]
    expand_shape[axis] = num_units
    k = num_channels // num_units
    expand_shape.insert(axis, k)

    outputs = tf.math.reduce_max(
        tf.reshape(inputs, expand_shape), axis, keepdims=False)
    return outputs
