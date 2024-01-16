'''
tfscripts layers are defined here
'''

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# tfscripts specific imports
from tfscripts.weights import new_weights, new_biases, new_kernel_weights
from tfscripts.weights import new_locally_connected_weights
from tfscripts import conv
from tfscripts import core
from tfscripts import pooling
from tfscripts.hex import conv as hx

# constants
from tfscripts import FLOAT_PRECISION


def flatten_layer(layer):
    '''
    Helper-function for flattening a layer

    Parameters
    ----------
    layer : tf.Tensor
        Input layer that is to be flattened

    Returns
    -------
    tf.Tensor
        Flattened layer.
    '''
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_of_events, *dimensions_to_be_flattened]

    num_features = layer_shape[1:].num_elements()

    # Reshape the layer to [num_of_events, num_features].
    layer_flat = tf.reshape(layer, [-1, num_features])

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features


def flatten_hex_layer(hex_layer):
    '''Helper-function for flattening an hexagonally shaped layer

    Assumes hexagonal shape in the first two spatial dimensions
    of the form:
                                    1   1   0             1  1

                                 1   1   1             1   1   1

                               0   1   1                 1   1
    Elements with indices a,b satisfying:

      a + b < (s-1)//2 or a + b > 2*(s-1) - (s-1)//2

      where s is the size of the x- and y-dimension

    will be discarded.
    These correspond to the elments outside of the hexagon

    Parameters
    ----------
    hex_layer : tf.Tensor
        Tensor of shape [batch, x, y, ...]
        Hexagonal layer to be flattended.

    Returns
    -------
    tf.Tensor
        A Tensor with shape: [batch,x*y*...]
        Flattened hexagonally shaped layer.

    Raises
    ------
    ValueError
        Description
    '''

    # Get the shape of the input layer.
    layer_shape = hex_layer.get_shape().as_list()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_of_events, *dimensions_to_be_flattened]

    size = layer_shape[1]
    if layer_shape[2] != size:
        raise ValueError("flatten_hex_layer: size of x- and y- dimension must "
                         "match, but are {!r}".format(layer_shape[1:3]))

    num_features = np.prod(layer_shape[3:])

    total_num_features = 0
    flat_elements = []
    for a in range(size):
        for b in range(size):
            if not (a + b < (size-1)//2 or a + b > 2*(size-1) - (size-1)//2):
                flattened_element = tf.reshape(hex_layer[:, a, b, ],
                                               [-1, num_features])
                flat_elements.append(flattened_element)
                total_num_features += num_features

    hex_layer_flat = tf.concat(flat_elements, axis=1)
    return hex_layer_flat, total_num_features


class ConvNdLayer(tf.Module):
    """TF Module for creating a new nD Convolutional Layer

    2 <= n <=4 are supported.
    For n == 3 (3 spatial dimensions x, y, and z):
        input: (n+2)-dim tensor of shape [batch, x, y, z, num_input_channels]
        output: (n+2)-dim tensor of shape [batch, x_p, y_p, z_p, num_filters]
            Where x_p, y_p, z_p may differ from the input spatial dimensions
            due to downsizing in pooling operation.
    """

    def __init__(self,
                 input_shape,
                 filter_size,
                 num_filters,
                 pooling_type=None,
                 pooling_strides=None,
                 pooling_ksize=None,
                 pooling_padding='SAME',
                 use_dropout=False,
                 activation='elu',
                 strides=None,
                 padding='SAME',
                 use_batch_normalisation=False,
                 dilation_rate=None,
                 use_residual=False,
                 method='convolution',
                 repair_std_deviation=True,
                 var_list=None,
                 weights=None,
                 biases=None,
                 trafo=None,
                 hex_num_rotations=1,
                 hex_azimuth=None,
                 hex_zero_out=False,
                 float_precision=FLOAT_PRECISION,
                 name=None):
        """Initialize object

        Parameters
        ----------
        input_shape : TensorShape, or list of int
            The shape of the inputs.
        filter_size : list of int
            The size of the convolution kernel: [filter_1, ..., filter_n]
            Example n == 3:
                [3, 3, 5] will perform a convolution with a 3x3x5 kernel.

            if method == 'hex_convolution':
                filter_size = [s, o, z, t]
                s: size of hexagon
                o: orientation of hexagon
                z: size along z axis [if n>= 3]
                t: size along t axis [if n>= 4]

                The hexagonal filter along axis x and y is put together
                from s and o.

                Examples:

                      s = 2, o = 0:
                                        1   1   0             1  1

                                     1   1   1             1   1   1

                                   0   1   1                 1   1

                      s = 3, o = 2:
                              0   1   0   0   0   0   0            1

                           0   1   1   1   1   0   0            1   1   1   1

                         1   1   1   1   1   0   0        1   1   1   1   1

                       0   1   1   1   1   1   0            1   1   1   1   1

                     0   0   1   1   1   1   1                1   1   1   1   1

                   0   0   1   1   1   1   0                 1   1   1   1

                 0   0   0   0   0   1   0                            1
        num_filters : int
            The number of filters to use in the convolution operation.
        pooling_type : str, optional
            The pooling method to use, e.g. 'max', 'avg', 'max_avg'
            If None, no pooling is applied.
        pooling_strides : list, optional
            The strides to be used for the pooling operation.
            Shape: [1, stride_1, ..., stride_n, stride_channel]
            Example n == 3:
                [1, 2, 2, 2, 1]: a stride of 2 is used along the x, y, and
                z axes.
        pooling_ksize : list, optional
            The pooling window size to be used.
            Shape: [1, pool_1, ..., pool_n, pool_channel]
            Example n == 3:
                [1, 2, 2, 2, 1]
                Will apply a pooling window of 2x2x2 in the spatial
                coordinates.
        pooling_padding : str, optional
            The padding method to be used for the pooling operation.
            Options are 'VALID', 'SAME'.
        use_dropout : bool, optional
            If True, dropout will be used.
        activation : str or callable, optional
            The type of activation function to be used.
        strides : list, optional
            The strides to be used for the convolution operation.
            Shape: [1, stride_1, ..., stride_n, stride_channel]
            Examples n == 3:
                [1, 1, 1, 1, 1]: a stride of 1 is used along all axes.
                [1, 1, 2, 1, 1]: a stride of 2 is used along the y axis.
        padding : str, optional
            The padding method to be used for the convolution operation.
            Options are 'VALID', 'SAME'.
        use_batch_normalisation : bool, optional
            If True, batch normalisation will be used.
        dilation_rate : None or list of int, optional
            The dilation rate to be used for the layer.
            Dilation rate is given by: [dilation_1, ..., dilation_n]
            where dilation_i specifies the dilation rate for axis_i.
            If the dilation rate is None, no dilation is applied in
            convolution.
        use_residual : bool, optional
            If True, layer result will be added as a residual to the
            input layer.
        method : str, optional
            Which convolution method to use for the layer, e.g.:
                'convolution', 'dynamic_convolution', 'local_trafo'.
                For details, see: tfscripts.conv.trans3d_op
        repair_std_deviation : bool, optional
            If true, the std deviation of the output will be repaired to be
            closer to 1. This helps to maintain a normalized throughput
            throughout the network.
            Note: this is only approximative and assume that the input has
            a std deviation of 1.
        var_list : list of tf.Variable, optional
            If 'weights' is provided, var_list must be passed as well.
            It must include a list of Variables of which the weights
            consist of.
        weights : None or tf.Tensor, optional
            Optionally, the weights to be used in the layer can be provided.
            If None, new weights are created.
        biases : None or tf.Tensor, optional
            Optionally, the biases to be used in the layer can be provided.
            If None, new biases are created.
        trafo : None or callable, optional
            If the convolution method is 'local_trafo', the callable provided
            by the 'trafo_list' will be used as the transformation on the input
            patch.
        hex_num_rotations : int, optional
            Only used if method == 'hex_convolution'.
            If num_rotations >= 1: weights of a kernel will be shared over
            'num_rotations' many rotated versions of that kernel.
        hex_azimuth : None or float or scalar float tf.Tensor
            Only used if method == 'hex_convolution'.
            Hexagonal kernel is turned by the angle 'azimuth'
            [given in degrees] in counterclockwise direction.
            If azimuth is None, the kernel will not be rotated dynamically.
        hex_zero_out : bool, optional
            Only used if method == 'hex_convolution'.
            If True, elements in result tensor which are not part of hexagon or
            IceCube strings (if shape in x and y dimensions is 10x10), will be
            set to zero.
        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        name : None, optional
            The name of the tensorflow module.

        Raises
        ------
        NotImplementedError
            Description
        ValueError
            Description
        """
        super(ConvNdLayer, self).__init__(name=name)

        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        # check dimension of input
        num_dims = len(input_shape) - 2
        if num_dims == 4:
            # 4D convolution
            if pooling_strides is None:
                pooling_strides = [1, 2, 2, 2, 2, 1]
            if pooling_ksize is None:
                pooling_ksize = [1, 2, 2, 2, 2, 1]
            if strides is None:
                strides = [1, 1, 1, 1, 1, 1]

        elif num_dims == 3:
            # 3D convolution
            if pooling_strides is None:
                pooling_strides = [1, 2, 2, 2, 1]
            if pooling_ksize is None:
                pooling_ksize = [1, 2, 2, 2, 1]
            if strides is None:
                strides = [1, 1, 1, 1, 1]

        elif num_dims == 2:
            # 2D convolution
            if pooling_strides is None:
                pooling_strides = [1, 2, 2, 1]
            if pooling_ksize is None:
                pooling_ksize = [1, 2, 2, 1]
            if strides is None:
                strides = [1, 1, 1, 1]

        else:
            msg = 'Currently only 2D, 3D or 4D supported {!r}'
            raise ValueError(msg.format(input_shape))

        # make sure inferred dimension matches filter_size
        if not len(filter_size) == num_dims:
            err_msg = 'Filter size {!r} does not fit to input shape {!r}'
            raise ValueError(err_msg.format(input_shape))

        num_input_channels = input_shape[-1]

        # Shape of the filter-weights for the convolution.
        shape = list(filter_size) + [num_input_channels, num_filters]

        # Create new weights aka. filters with the given shape.
        if method.lower() == 'convolution':
            if weights is None:
                # weights = new_kernel_weights(shape=shape)
                weights = new_weights(shape=shape,
                                      float_precision=float_precision)

            # Create new biases, one for each filter.
            if biases is None:
                biases = new_biases(length=num_filters,
                                    float_precision=float_precision)

            if num_dims == 2 or num_dims == 3:
                # create a temp function with all parameters set
                def temp_func(inputs):
                    return tf.nn.convolution(input=inputs,
                                             filters=weights,
                                             strides=strides[1:-1],
                                             padding=padding,
                                             dilations=dilation_rate)
                self.conv_layer = temp_func

            elif num_dims == 4:

                # create a temp function with all parameters set
                def temp_func(inputs):
                    return conv.conv4d_stacked(input=inputs,
                                               filter=weights,
                                               strides=strides,
                                               padding=padding,
                                               dilation_rate=dilation_rate)
                self.conv_layer = temp_func

        # ---------------------
        # Hexagonal convolution
        # ---------------------
        elif method.lower() == 'hex_convolution':
            if num_dims == 2 or num_dims == 3:
                self.conv_layer = hx.ConvHex(
                                            input_shape=input_shape,
                                            filter_size=filter_size,
                                            num_filters=num_filters,
                                            padding=padding,
                                            strides=strides,
                                            num_rotations=hex_num_rotations,
                                            azimuth=hex_azimuth,
                                            dilation_rate=dilation_rate,
                                            zero_out=hex_zero_out,
                                            kernel=weights,
                                            var_list=var_list,
                                            float_precision=float_precision,
                                            )
            elif num_dims == 4:
                self.conv_layer = hx.ConvHex4d(
                                            input_shape=input_shape,
                                            filter_size=filter_size,
                                            num_filters=num_filters,
                                            padding=padding,
                                            strides=strides,
                                            num_rotations=hex_num_rotations,
                                            azimuth=hex_azimuth,
                                            dilation_rate=dilation_rate,
                                            zero_out=hex_zero_out,
                                            kernel=weights,
                                            var_list=var_list,
                                            float_precision=float_precision,
                                            )

            # Create new biases, one for each filter.
            if biases is None:
                biases = new_biases(length=num_filters*hex_num_rotations,
                                    float_precision=float_precision)

        # -------------------
        # locally connected
        # -------------------
        elif method.lower() == 'locally_connected':

            if num_dims == 2:
                self.conv_layer = conv.LocallyConnected2d(
                                            input_shape=input_shape,
                                            num_outputs=num_filters,
                                            filter_size=filter_size,
                                            kernel=weights,
                                            strides=strides[1:-1],
                                            padding=padding,
                                            dilation_rate=dilation_rate,
                                            float_precision=float_precision)
            elif num_dims == 3:
                self.conv_layer = conv.LocallyConnected3d(
                                            input_shape=input_shape,
                                            num_outputs=num_filters,
                                            filter_size=filter_size,
                                            kernel=weights,
                                            strides=strides[1:-1],
                                            padding=padding,
                                            dilation_rate=dilation_rate,
                                            float_precision=float_precision)
            elif num_dims == 4:
                raise NotImplementedError(
                    '4D locally connected not implemented!')

            # Create new biases, one for each filter and position
            if biases is None:
                biases = new_locally_connected_weights(
                    shape=self.conv_layer.output_shape[1:],
                    shared_axes=[i for i in range(num_dims)],
                    float_precision=float_precision)

        # -------------------
        # local trafo
        # -------------------
        elif method.lower() == 'local_trafo':

            assert (weights is None and biases is None)

            if num_dims == 3:

                # create a temp function with all parameters set
                def temp_func(inputs):
                    return conv.trans3d_op(
                                    input=inputs,
                                    num_out_channel=num_filters,
                                    filter_size=filter_size,
                                    method=method,
                                    trafo=trafo,
                                    filter=weights,
                                    strides=strides[1:-1],
                                    padding=padding,
                                    dilation_rate=dilation_rate,
                                    stack_axis=None,
                                    )
                self.conv_layer = temp_func
            else:
                raise NotImplementedError('local_trafo currently only for 3D')

        # --------------------
        # dynamic convolution
        # --------------------
        elif method.lower() == 'dynamic_convolution':

            assert weights is not None

            if num_dims == 2 or num_dims == 3:
                # create a temp function with all parameters set
                def temp_func(inputs):
                    return conv.dynamic_conv(inputs=inputs,
                                             filters=weights,
                                             strides=strides[1:-1],
                                             padding=padding,
                                             dilation_rate=dilation_rate)
                self.conv_layer = temp_func

            elif num_dims == 4:
                raise NotImplementedError(
                    '4D dynamic_convolution not implemented')

            if biases is None:
                biases = new_biases(length=num_filters,
                                    float_precision=float_precision)

        else:
            raise ValueError('Unknown method: {!r}'.format(method))

        self.biases = biases
        self.weights = weights

        # get shape of activation input
        # todo: figure out better way to obtain this
        dummy_data = tf.ones([1] + input_shape[1:], dtype=float_precision)
        conv_layer_output = self.conv_layer(dummy_data)

        self.activation = core.Activation(
                            activation_type=activation,
                            input_shape=conv_layer_output.shape,
                            use_batch_normalisation=use_batch_normalisation,
                            float_precision=float_precision)

        # assing and keep track of settings
        self.input_shape = input_shape
        self.num_dims = num_dims
        self.filter_size = filter_size
        self.use_residual = use_residual
        self.pooling_type = pooling_type
        self.pooling_ksize = pooling_ksize
        self.pooling_strides = pooling_strides
        self.pooling_padding = pooling_padding
        self.use_dropout = use_dropout
        self.method = method
        self.repair_std_deviation = repair_std_deviation
        self.float_precision = float_precision

        # get shape of output
        # todo: figure out better way to obtain this
        output = self.activation(conv_layer_output, is_training=False)
        output = self._apply_pooling(output)
        self.output_shape = output.shape.as_list()

        # Use as Residual
        if use_residual:
            self.residual_add = core.AddResidual(
                                    residual_shape=self.output_shape,
                                    strides=strides,
                                    float_precision=float_precision)

    def __call__(self, inputs, is_training, keep_prob=None):
        """Apply Module.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        is_training : None, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        keep_prob : None, optional
            The keep probability to be used for dropout.
            Can either be a float or a scalar float tf.Tensor.

        Returns
        -------
        tf.Tensor
            The output tensor.

        Raises
        ------
        NotImplementedError
            Description
        ValueError
            Description
        """

        inputs = tf.convert_to_tensor(inputs)

        # check shape of input tensor
        if inputs.shape[1:].as_list() != self.input_shape[1:]:
            raise ValueError('Shape mismatch: {!r} != {!r}'.format(
                inputs.shape[1:].as_list(), self.input_shape[1:]))

        # Perform convolution/transformation
        layer = self.conv_layer(inputs)

        num_input_channels = inputs.get_shape().as_list()[-1]

        # repair to get std dev of 1
        # In convolution operation, a matrix multiplication is performed
        # over the image patch and the kernel. Afterwards, a reduce_sum
        # is called. Assuming that all inputs and weights are normalized,
        # the result of the matrix multiplication will approximately still
        # have std dev 1.
        # However, the reduce_sum operation over the filter size and number
        # of input channels will add up
        #   n = np.prod(filter_size) * num_input_channels
        # variables which each have a std of 1. In the case of normal
        # distributed values, this results in a resulting std deviation
        # of np.sqrt(np.prod(filter_size) * num_input_channels). To ensure
        # that the result of the convolutional layer is still normalized,
        # the values need to be divided by this factor.
        # In the case of the hex_convolution, this factor gets reduced to
        # the number of non zero elements in the hex kernel.
        if self.repair_std_deviation:
            if self.method.lower() == 'hex_convolution':

                num_filter_vars = hx.get_num_hex_points(self.filter_size[0])
                if len(self.filter_size) > 2:
                    # This should probably be *= , but empirically this
                    # provides better results...
                    # [At least for IceCube applications]
                    # Possibly because variance is actually a lot lower in
                    # input, since it will be padded with zeros and these will
                    # propagate to later layers.
                    num_filter_vars += np.prod(self.filter_size[2:])

                layer = layer / np.sqrt(num_filter_vars * num_input_channels)
            else:
                layer = layer / np.sqrt(np.prod(self.filter_size)
                                        * num_input_channels)

        # Add the biases to the results of the convolution.
        # A bias-value is added to each filter-channel.
        if self.biases is not None:
            if self.repair_std_deviation:
                layer = (layer + self.biases) / np.sqrt(2.)
            else:
                layer = layer + self.biases

        # Apply activation and batch normalisation if specified
        layer = self.activation(layer, is_training=is_training)

        # Use as Residual
        if self.use_residual:
            layer = self.residual_add(input=inputs, residual=layer)

        # Use pooling to down-sample the image resolution?
        layer = self._apply_pooling(layer)

        if self.use_dropout and is_training:
            layer = tf.nn.dropout(layer, 1 - keep_prob)

        return layer

    def _apply_pooling(self, layer):
        """Apply pooling to layer

        Parameters
        ----------
        layer : tf.Tensor
            The layer on which to apply pooling
        """
        if self.num_dims == 2:
            layer = pooling.pool2d(layer=layer,
                                   ksize=self.pooling_ksize,
                                   strides=self.pooling_strides,
                                   padding=self.pooling_padding,
                                   pooling_type=self.pooling_type,
                                   )
        elif self.num_dims == 3:
            layer = pooling.pool3d(layer=layer,
                                   ksize=self.pooling_ksize,
                                   strides=self.pooling_strides,
                                   padding=self.pooling_padding,
                                   pooling_type=self.pooling_type,
                                   )
        elif self.num_dims == 4:
            if pooling_type == 'max':
                layer = pooling.max_pool4d_stacked(
                                                input=layer,
                                                ksize=self.pooling_ksize,
                                                strides=self.pooling_strides,
                                                padding=self.pooling_padding)
            elif pooling_type == 'avg':
                layer = pooling.avg_pool4d_stacked(
                                                input=layer,
                                                ksize=self.pooling_ksize,
                                                strides=self.pooling_strides,
                                                padding=self.pooling_padding)
            else:
                raise NotImplementedError("Pooling type not supported: "
                                          "{!r}".format(self.pooling_type))
        else:
            raise NotImplementedError('Only supported 2d, 3d, 4d!')

        return layer


class FCLayer(tf.Module):
    """TF Module for creating a new Fully-Connected Layer

        input: 2-dim tensor of shape [batch_size, num_inputs]
        output: 2-dim tensor of shape [batch_size, num_outputs]
    """

    def __init__(self,
                 input_shape,
                 num_outputs,
                 use_dropout=False,
                 activation='elu',
                 use_batch_normalisation=False,
                 use_residual=False,
                 weights=None,
                 biases=None,
                 max_out_size=None,
                 repair_std_deviation=True,
                 float_precision=FLOAT_PRECISION,
                 name=None):
        """Initialize object

        Parameters
        ----------
        input_shape : TensorShape, or list of int
            The shape of the inputs.
        num_outputs : int
            The number of output nodes.
        use_dropout : bool, optional
            If True, dropout will be used.
        activation : str or callable, optional
            The type of activation function to be used.
        use_batch_normalisation : bool, optional
            If True, batch normalisation will be used.
        use_residual : bool, optional
            If True, layer result will be added as a residual to the input
            layer.
        weights : None or tf.Tensor, optional
            Optionally, the weights to be used in the layer can be provided.
            If None, new weights are created.
        biases : None or tf.Tensor, optional
            Optionally, the biases to be used in the layer can be provided.
            If None, new biases are created.
        max_out_size : None or int, optional
            The max_out_size for the layer.
            This must be a factor of number of channels.
            If None, no max_out is used in the layer.
        repair_std_deviation : bool, optional
            If true, the std deviation of the output will be repaired to be
            closer to 1. This helps to maintain a normalized throughput
            throughout the network.
            Note: this is only approximative and assume that the input has
            a std deviation of 1.
        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        name : None, optional
            The name of the tensorflow module.

        """
        super(FCLayer, self).__init__(name=name)

        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        num_inputs = input_shape[-1]

        # Create new weights and biases.
        if weights is None:
            weights = new_weights(
                shape=[num_inputs, num_outputs],
                float_precision=float_precision,
            )
        if biases is None:
            biases = new_biases(
                length=num_outputs, float_precision=float_precision)

        self.biases = biases
        self.weights = weights

        self.activation = core.Activation(
                            activation_type=activation,
                            input_shape=[None, num_outputs],
                            use_batch_normalisation=use_batch_normalisation,
                            float_precision=float_precision)

        # calculate residual strides
        if max_out_size is not None:
            layer_shape = [None, num_outputs]

            msg = "max out needs to match dim"
            assert layer_shape[-1] % max_out_size == 0, msg

            layer_shape[-1] = layer_shape[-1] // max_out_size
            channel_stride = max(1, num_inputs // layer_shape[-1])
            res_strides = [1 for i in input_shape[:-1]]+[channel_stride]

        else:
            res_strides = None

        # calculate output shape
        if max_out_size is not None:
            output_shape = [None, num_outputs // max_out_size]
        else:
            output_shape = [None, num_outputs]

        # Use as Residual
        if use_residual:
            self.residual_add = core.AddResidual(
                                            residual_shape=output_shape,
                                            strides=res_strides,
                                            float_precision=float_precision)

        self.output_shape = output_shape
        self.input_shape = input_shape
        self.max_out_size = max_out_size
        self.use_residual = use_residual
        self.use_dropout = use_dropout
        self.repair_std_deviation = repair_std_deviation
        self.float_precision = float_precision

    def __call__(self, inputs, is_training, keep_prob):
        """Apply Module.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        is_training : None, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        keep_prob : None, optional
            The keep probability to be used for dropout.
            Can either be a float or a scalar float tf.Tensor.

        Returns
        -------
        tf.Tensor
            The output tensor.

        Raises
        ------
        NotImplementedError
            Description
        ValueError
            Description
        """

        inputs = tf.convert_to_tensor(inputs)

        # check shape of input tensor
        if inputs.shape[1:].as_list() != self.input_shape[1:]:
            raise ValueError('Shape mismatch: {!r} != {!r}'.format(
                inputs.shape[1:].as_list(), self.input_shape[1:]))

        num_inputs = inputs.get_shape().as_list()[-1]

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        layer = tf.matmul(inputs, self.weights)

        # repair to get std dev of 1
        if self.repair_std_deviation:
            layer = layer / np.sqrt(num_inputs)

            layer = (layer + self.biases) / np.sqrt(2.)
        else:
            layer = layer + self.biases

        # Apply activation and batch normalisation if specified
        layer = self.activation(layer, is_training=is_training)

        # max out channel dimension if specified
        if self.max_out_size is not None:
            layer = core.maxout(inputs=layer,
                                num_units=self.max_out_size,
                                axis=-1)

        # Use as Residual
        if self.use_residual:
            layer = self.residual_add(input=inputs, residual=layer)

        if self.use_dropout and is_training:
            layer = tf.nn.dropout(layer, 1 - keep_prob)

        return layer


class ChannelWiseFCLayer(tf.Module):
    """TF Module for creating a new channel wise Fully-Connected Layer

       input: 3-dim tensor of shape [batch_size, num_inputs, num_channel]
       output: 3-dim tensor of shape [batch_size, num_outputs, num_channel]
    """

    def __init__(self,
                 input_shape,
                 num_outputs,
                 use_dropout=False,
                 activation='elu',
                 use_batch_normalisation=False,
                 use_residual=False,
                 weights=None,
                 biases=None,
                 max_out_size=None,
                 repair_std_deviation=True,
                 float_precision=FLOAT_PRECISION,
                 name=None):
        """Initialize object

        Parameters
        ----------
        input_shape : TensorShape, or list of int
            The shape of the inputs.
        num_outputs : int
            The number of output nodes.
        use_dropout : bool, optional
            If True, dropout will be used.
        activation : str or callable, optional
            The type of activation function to be used.
        use_batch_normalisation : bool, optional
            If True, batch normalisation will be used.
        use_residual : bool, optional
            If True, layer result will be added as a residual to the input
            layer.
        weights : None or tf.Tensor, optional
            Optionally, the weights to be used in the layer can be provided.
            If None, new weights are created.
        biases : None or tf.Tensor, optional
            Optionally, the biases to be used in the layer can be provided.
            If None, new biases are created.
        max_out_size : None or int, optional
            The max_out_size for the layer.
            This must be a factor of number of channels.
            If None, no max_out is used in the layer.
        repair_std_deviation : bool, optional
            If true, the std deviation of the output will be repaired to be
            closer to 1. This helps to maintain a normalized throughput
            throughout the network.
            Note: this is only approximative and assume that the input has
            a std deviation of 1.
        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        name : None, optional
            The name of the tensorflow module.

        """
        super(ChannelWiseFCLayer, self).__init__(name=name)

        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        # input: [batch, num_inputs, num_channel]
        msg = '{} != [batch, num_inputs, num_channel]'
        assert len(input_shape) == 3, msg.format(input_shape)

        num_inputs = input_shape[1]
        num_channels = input_shape[2]
        transpose_shape = [-1, num_outputs, num_channels]

        # Create new weights and biases.
        if weights is None:
            weights = new_weights(
                shape=[num_channels, num_inputs, num_outputs],
                float_precision=float_precision,
            )
        if biases is None:
            biases = new_weights(
                shape=[num_outputs, num_channels],
                float_precision=float_precision,
            )

        self.biases = biases
        self.weights = weights

        self.activation = core.Activation(
                            activation_type=activation,
                            input_shape=transpose_shape,
                            use_batch_normalisation=use_batch_normalisation,
                            float_precision=float_precision)

        # # calculate residual strides
        # if max_out_size is not None:
        #     layer_shape = [None, num_channels, num_outputs]

        #     msg = "max out needs to match dim"
        #     assert layer_shape[-1] % max_out_size == 0, msg

        #     layer_shape[-1] = layer_shape[-1] // max_out_size
        #     channel_stride = max(1, num_inputs // layer_shape[-1])
        #     res_strides = [1 for i in input_shape[:-1]]+[channel_stride]

        # else:
        #     res_strides = None
        res_strides = None

        # Use as Residual
        if use_residual:
            self.residual_add = core.AddResidual(
                            residual_shape=[None, num_channels, num_outputs],
                            strides=res_strides,
                            float_precision=float_precision)

        # calculate output shape
        if max_out_size is not None:
            output_shape = [None, num_channels, num_outputs // max_out_size]
        else:
            output_shape = [None, num_channels, num_outputs]

        self.output_shape = output_shape
        self.input_shape = input_shape
        self.max_out_size = max_out_size
        self.use_residual = use_residual
        self.use_dropout = use_dropout
        self.repair_std_deviation = repair_std_deviation
        self.float_precision = float_precision

    def __call__(self, inputs, is_training, keep_prob):
        """Apply Module.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        is_training : None, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        keep_prob : None, optional
            The keep probability to be used for dropout.
            Can either be a float or a scalar float tf.Tensor.

        Returns
        -------
        tf.Tensor
            The output tensor.

        Raises
        ------
        NotImplementedError
            Description
        ValueError
            Description
        """

        inputs = tf.convert_to_tensor(inputs)

        # check shape of input tensor
        if inputs.shape[1:].as_list() != self.input_shape[1:]:
            raise ValueError('Shape mismatch: {!r} != {!r}'.format(
                inputs.shape[1:].as_list(), self.input_shape[1:]))

        input_shape = inputs.get_shape().as_list()
        num_inputs = input_shape[1]
        num_channels = input_shape[2]

        # input_transpose: [num_channel, batch, num_inputs]
        input_transpose = tf.transpose(a=inputs, perm=[2, 0, 1])

        # Calculate the layer as the matrix multiplication of
        # the input and weights, and then add the bias-values.
        # output: [num_channel, batch, num_outputs]
        output = tf.matmul(input_transpose, self.weights)
        layer = tf.transpose(a=output, perm=[1, 2, 0])
        # layer: [batch, num_outputs, num_channel]

        # repair to get std dev of 1
        if self.repair_std_deviation:
            layer = layer / np.sqrt(num_inputs)

            layer = (layer + self.biases) / np.sqrt(2.)
        else:
            layer = layer + self.biases

        # Apply activation and batch normalisation if specified
        layer = self.activation(layer, is_training=is_training)

        # Use as Residual
        if self.use_residual:
            # convert to [batch, num_channel, num_outputs]
            layer = tf.transpose(a=layer, perm=[0, 2, 1])
            layer = self.residual_add(
                input=tf.transpose(a=inputs, perm=[0, 2, 1]), residual=layer)
            # convert back to [batch, num_outputs, num_channel]
            layer = tf.transpose(a=layer, perm=[0, 2, 1])

        # max out channel dimension if specified
        if self.max_out_size is not None:
            layer = core.maxout(inputs=layer,
                                num_units=self.max_out_size,
                                axis=-1)

        if self.use_dropout and is_training:
            layer = tf.nn.dropout(layer, 1 - keep_prob)

        return layer


class FCLayers(tf.Module):
    """TF Module for creating new fully connected layers.
    """

    def __init__(self,
                 input_shape,
                 fc_sizes,
                 use_dropout_list=False,
                 activation_list='elu',
                 use_batch_normalisation_list=False,
                 use_residual_list=False,
                 weights_list=None,
                 biases_list=None,
                 max_out_size_list=None,
                 repair_std_deviation_list=True,
                 float_precision=FLOAT_PRECISION,
                 name='fc_layer',
                 verbose=False,
                 ):
        """Initialize object

        Parameters
        ----------
        input_shape : TensorShape, or list of int
            The shape of the inputs.
        fc_sizes : list of int
            The number of nodes for each layer. The ith int denotes the number
            of nodes for the ith layer. The number of layers is inferred from
            the length of 'fc_sizes'.
        use_dropout_list : bool, optional
            Denotes whether to use dropout in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        activation_list : str or callable, optional
            The type of activation function to be used in each layer.
            If only one activation is provided, it will be used for all layers.
        use_batch_normalisation_list : bool or list of bool, optional
            Denotes whether to use batch normalisation in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        use_residual_list : bool or list of bool, optional
            Denotes whether to use residual additions in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        weights_list : None or list of tf.Tensor, optional
            Optionally, the weights to be used in each layer can be provided.
            If None, new weights are created.
        biases_list : None or list of tf.Tensor, optional
            Optionally, the biases to be used in each layer can be provided.
            If None, new biases are created.
        max_out_size_list : None or int or list of int, optional
            The max_out_size for each layer.
            If None, no max_out is used in the corresponding layer.
            If only a single max_out_size is given, it will be used for all
            layers.
        repair_std_deviation_list : bool or list of bool, optional
            If true, the std deviation of the output will be repaired to be
            closer to 1. This helps to maintain a normalized throughput
            throughout the network.
            Note: this is only approximative and assume that the input has
            a std deviation of 1.
            If only a single boolean is provided, it will be used for all
            layers.
        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        name : str, optional
            An optional name for the layers.

        verbose : bool, optional
            If true, more verbose output is printed.

        Raises
        ------
        ValueError
            Description
        """
        super(FCLayers, self).__init__(name=name)

        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        num_layers = len(fc_sizes)
        if isinstance(activation_list, str):
            activation_list = [activation_list for i in range(num_layers)]
        if use_dropout_list is False or use_dropout_list is True:
            use_dropout_list = [use_dropout_list for i in range(num_layers)]
        # create batch normalisation array
        if isinstance(use_batch_normalisation_list, bool):
            use_batch_normalisation_list = [use_batch_normalisation_list
                                            for i in range(num_layers)]
        # create use_residual_list
        if use_residual_list is False or use_residual_list is True:
            use_residual_list = [use_residual_list for i in range(num_layers)]
        # create weights_list
        if weights_list is None:
            weights_list = [None for i in range(num_layers)]
        # create biases_list
        if biases_list is None:
            biases_list = [None for i in range(num_layers)]
        # create max out array
        if max_out_size_list is None:
            max_out_size_list = [None for i in range(num_layers)]
        # create repair std deviation array
        if isinstance(repair_std_deviation_list, bool):
            repair_std_deviation_list = [
                repair_std_deviation_list for i in range(num_layers)]

        # pick which fc layers to build
        if len(input_shape) == 3:
            # input is of form [batch, num_inputs, channel]
            FCLayerClass = ChannelWiseFCLayer

        elif len(input_shape) == 2:
            # input is of form [batch, num_inputs]
            FCLayerClass = FCLayer

        else:
            raise ValueError('Input dimension is wrong: {}'.format(
                                            inputs.get_shape().as_list()))

        # create layers:
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                previous_layer_shape = input_shape
            else:
                previous_layer_shape = self.layers[i-1].output_shape
            layer_i = FCLayerClass(
                    input_shape=previous_layer_shape,
                    num_outputs=fc_sizes[i],
                    activation=activation_list[i],
                    use_dropout=use_dropout_list[i],
                    use_batch_normalisation=use_batch_normalisation_list[i],
                    use_residual=use_residual_list[i],
                    weights=weights_list[i],
                    biases=biases_list[i],
                    max_out_size=max_out_size_list[i],
                    repair_std_deviation=repair_std_deviation_list[i],
                    float_precision=float_precision,
                    name='{}_{:03d}'.format(name, i),
                    )
            if verbose:
                print('{}_{:03d}'.format(name, i), layer_i.output_shape)
            self.layers.append(layer_i)

    def __call__(self, inputs, is_training, keep_prob=None):
        """Apply layers

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        is_training : None, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        keep_prob : None, optional
            The keep probability to be used for dropout.
            Can either be a float or a scalar float tf.Tensor.

        Returns
        -------
        list of tf.Tensor
            The list of output tensors of each layer.
        """
        layer_outputs = []
        x = inputs
        for layer in self.layers:
            x = layer(x, is_training=is_training, keep_prob=keep_prob)
            layer_outputs.append(x)
        return layer_outputs


class ConvNdLayers(tf.Module):
    """TF Module for creating new conv2d, conv3d, and conv4d layers.
    """

    def __init__(self,
                 input_shape,
                 filter_size_list,
                 num_filters_list,
                 pooling_type_list=None,
                 pooling_strides_list=None,
                 pooling_ksize_list=None,
                 pooling_padding_list='SAME',
                 padding_list='SAME',
                 strides_list=None,
                 activation_list='elu',
                 use_batch_normalisation_list=False,
                 dilation_rate_list=None,
                 use_residual_list=False,
                 use_dropout_list=False,
                 method_list='convolution',
                 repair_std_deviation_list=True,
                 weights_list=None,
                 biases_list=None,
                 trafo_list=None,
                 hex_num_rotations_list=1,
                 hex_azimuth_list=None,
                 hex_zero_out_list=False,
                 float_precision=FLOAT_PRECISION,
                 name='conv_{}d_layer',
                 verbose=False,
                 ):
        """Initialize object

        Parameters
        ----------
        input_shape : TensorShape, or list of int
            The shape of the inputs.
        filter_size_list : list of list of int
            A list of filtersizes.
            If a single filtersize is given, this will be used for every layer.
            Otherwise, a filtersize must be given for each layer.
            3D case:
                Single: filtersize = [filter_x, filter_y, filter_z]
                Multiple: [[x1, y1, z1], ... , [xn, yn, zn] ]
                          where [xi, yi, zi] is the filter size of the
                          ith layer.
            2D case:
                Single: filtersize = [filter_x, filter_y]
                Multiple: [[x1, y1], ... , [xn, yn] ]
                          where [xi, yi] is the filter size of the ith layer.
        num_filters_list : list of int
            The number of filters for each layer. The ith int denotes the
            number of filters for the ith layer. The number of layers is
            inferred from the length of 'num_filters_list'.
        pooling_type_list : str, optional
            The pooling method to use, e.g. 'max', 'avg', 'max_avg'
            The ith value denotes the pooling type for the ith layer.
            If only one pooling type is given, this will be used for all
            layers. If None, no pooling is applied.
        pooling_strides_list : None, optional
            The strides to be used for the pooling operation in each layer.
            If only one stride is provided, it will be used for all layers.
            3D example:
                [[1, 1, 1, 2, 1], [1, 2, 2, 2, 1]]
                The strides are provided for two layers.
                In the first layer, a stride of 2 is used along the z axis.
                The second layer uses a stride of 2 along all spatial
                coordinates.
        pooling_ksize_list : None, optional
            The pooling window size to be used.
            If only one pooling size is given, it will be used for all layers.
            2D example:
                [1, 2, 2, 1]
                Will apply a pooling window of 2x2 in the x and y coordinates.
                Since only a single pooling size is given, this will be used
                for all layers.
        pooling_padding_list : str, optional
            The padding method to be used for each pooling operation.
            Options are 'VALID', 'SAME'.
            If a single padding string is provided, it will be used for all
            layers.
        padding_list : str, optional
            The padding method to be used for each convolution operation.
            Options are 'VALID', 'SAME'.
            If a single padding string is provided, it will be used for all
            layers.
        strides_list : None, optional
            The strides to be used for the convolution operation in each layer.
            If only one stride is provided, it will be used for all layers.
            3D example:
                [[1, 1, 1, 1, 1], [1, 2, 2, 2, 1]]
                The strides are provided for two layers.
                In the first layer, a stride of 1 is used along all axes.
                The second layer uses a stride of 2 along all spatial
                coordinates.
        activation_list : str or callable, optional
            The type of activation function to be used in each layer.
            If only one activation is provided, it will be used for all layers.
        use_batch_normalisation_list : bool or list of bool, optional
            Denotes whether to use batch normalisation in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        dilation_rate_list : None, optional
            The dilation rate to be used for each layer.
            If only a single dilation rate is provided,
            it will be used for all layers.
            If the dilation rate is None, no dilation is applied in
            convolution.
            3D example:
                [[1, 1, 2], [2, 2, 2]]
                Will use a dilation of 2 along the z axis for the first layer
                and a dilation of 2 along all spatial axes for the second
                layer.
            2D example:
                [2, 1]
                Since only a single dilation rate is given, it will be used for
                all layers. In this case, a dilation rate of 2 along the x axis
                is used for all layers.
        use_residual_list : bool or list of bool, optional
            Denotes whether to use residual additions in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        use_dropout_list : bool, optional
            Denotes whether to use dropout in the layers.
            If only a single boolean is provided, it will be used for all
            layers.
        method_list : str or list of str, optional
            Which convolution method to use for each layer, e.g.:
                'convolution', 'dynamic_convolution', 'local_trafo'.
                For details, see: tfscripts.conv.trans3d_op
            If only a single method is provided, it will be used for all
            layers.
        repair_std_deviation_list : bool or list of bool, optional
            If true, the std deviation of the output will be repaired to be
            closer to 1. This helps to maintain a normalized throughput
            throughout the network.
            Note: this is only approximative and assume that the input has
            a std deviation of 1.
            If only a single boolean is provided, it will be used for all
            layers.
        weights_list : None or list of tf.Tensor, optional
            Optionally, the weights to be used in each layer can be provided.
            If None, new weights for the convolution kernels are created.
        biases_list : None or list of tf.Tensor, optional
            Optionally, the biases to be used in each layer can be provided.
            If None, new biases are created.
        trafo_list : None or callable or list of callable, optional
            If the convolution method is 'local_trafo', the callable provided
            by the 'trafo_list' will be used as the transformation on the input
            patch.
            If only one trafo method is given, it will be used for all layers.
        hex_num_rotations_list : int or list of int, optional
            Only used if method == 'hex_convolution'.
            If num_rotations >= 1: weights of a kernel will be shared over
            'num_rotations' many rotated versions of that kernel.
            If only one int is give, it will apply to all layers.
        hex_azimuth_list : list of float or list scalar tf.Tensor, optional
            Only used if method == 'hex_convolution'.
            Hexagonal kernel is turned by the angle 'azimuth'
            [given in degrees] in counterclockwise direction.
            If azimuth is None, the kernel will not be rotated dynamically.
            If only one azimuth angle is given, all layers will be turned by
            the same angle.
        hex_zero_out_list : bool or list of bool, optional
            Only used if method == 'hex_convolution'.
            If True, elements in result tensor which are not part of hexagon or
            IceCube strings (if shape in x and y dimensions is 10x10), will be
            set to zero.
            If only one boolean is given, it will apply to all layers.
        float_precision : TYPE, optional
            Description
        name : str, optional
            An optional name for the layers.
        verbose : bool, optional
            If true, more verbose output is printed.

        Raises
        ------
        ValueError
            Description
        """
        num_dims = len(input_shape)
        name = name.format(num_dims - 2)

        super(ConvNdLayers, self).__init__(name=name)

        if isinstance(input_shape, tf.TensorShape):
            input_shape = input_shape.as_list()

        # check dimension of input
        if num_dims == 6:
            # 4D convolution
            if pooling_strides_list is None:
                pooling_strides_list = [1, 2, 2, 2, 2, 1]
            if pooling_ksize_list is None:
                pooling_ksize_list = [1, 2, 2, 2, 2, 1]
            if strides_list is None:
                strides_list = [1, 1, 1, 1, 1, 1]
        elif num_dims == 5:
            # 3D convolution
            if pooling_strides_list is None:
                pooling_strides_list = [1, 2, 2, 2, 1]
            if pooling_ksize_list is None:
                pooling_ksize_list = [1, 2, 2, 2, 1]
            if strides_list is None:
                strides_list = [1, 1, 1, 1, 1]

        elif num_dims == 4:
            # 2D convolution
            if pooling_strides_list is None:
                pooling_strides_list = [1, 2, 2, 1]
            if pooling_ksize_list is None:
                pooling_ksize_list = [1, 2, 2, 1]
            if strides_list is None:
                strides_list = [1, 1, 1, 1]

        else:
            msg = 'Currently only 2D, 3D, or 4D supported {!r}'
            raise ValueError(msg.format(input_shape))

        num_layers = len(num_filters_list)

        # ---------------
        # Fill values
        # ---------------
        # create pooling_type_list
        if pooling_type_list is None or isinstance(pooling_type_list, str):
            pooling_type_list = [pooling_type_list for i in range(num_layers)]

        # create pooling_strides_list
        if (len(np.array(pooling_strides_list).shape) == 1 and
                len(pooling_strides_list) == num_dims):
            pooling_strides_list = [pooling_strides_list
                                    for i in range(num_layers)]

        # create pooling_ksize_list
        if (len(np.array(pooling_ksize_list).shape) == 1 and
                len(pooling_ksize_list) == num_dims):
            pooling_ksize_list = [pooling_ksize_list for i in
                                  range(num_layers)]

        # create pooling_padding_list
        if pooling_padding_list == 'SAME' or pooling_padding_list == 'VALID':
            pooling_padding_list = [pooling_padding_list
                                    for i in range(num_layers)]

        # create padding_list
        if padding_list == 'SAME' or padding_list == 'VALID':
            padding_list = [padding_list for i in range(num_layers)]

        # create strides_list
        if (len(np.array(strides_list).shape) == 1 and
                len(strides_list) == num_dims):
            strides_list = [strides_list for i in range(num_layers)]

        # create activation_list
        if isinstance(activation_list, str):
            activation_list = [activation_list for i in range(num_layers)]

        # create batch normalisation array
        if isinstance(use_batch_normalisation_list, bool):
            use_batch_normalisation_list = [use_batch_normalisation_list
                                            for i in range(num_layers)]

        # create dilation_rate_list
        if dilation_rate_list is None or (
            len(np.asarray(dilation_rate_list).shape) == 1 and
                len(dilation_rate_list) == num_dims - 2):
            dilation_rate_list = [dilation_rate_list for i in
                                  range(num_layers)]

        # create use_residual_list
        if use_residual_list is False or use_residual_list is True:
            use_residual_list = [use_residual_list for i in range(num_layers)]

        # create use_dropout_list
        if use_dropout_list is False or use_dropout_list is True:
            use_dropout_list = [use_dropout_list for i in range(num_layers)]

        # create method_list
        if isinstance(method_list, str):
            method_list = [method_list for i in range(num_layers)]

        # create repair std deviation array
        if isinstance(repair_std_deviation_list, bool):
            repair_std_deviation_list = [
                repair_std_deviation_list for i in range(num_layers)]

        # create weights_list
        if weights_list is None:
            weights_list = [None for i in range(num_layers)]
        # create biases_list
        if biases_list is None:
            biases_list = [None for i in range(num_layers)]

        # create trafo_list
        if trafo_list is None or len(trafo_list) == 1:
            trafo_list = [trafo_list for i in range(num_layers)]

        # create hex_num_rotations_list
        if isinstance(hex_num_rotations_list, int):
            hex_num_rotations_list = [hex_num_rotations_list
                                      for i in range(num_layers)]

        # create hex_azimuth_list
        if (hex_azimuth_list is None or
                tf.is_tensor(hex_azimuth_list)):
            hex_azimuth_list = [hex_azimuth_list for i in range(num_layers)]

        # create hex_zero out array
        if isinstance(hex_zero_out_list, bool):
            hex_zero_out_list = [hex_zero_out_list for i in range(num_layers)]
        # ---------------

        # create layers:
        self.layers = []
        for i in range(num_layers):
            if i == 0:
                previous_layer_shape = input_shape
            else:
                previous_layer_shape = self.layers[i-1].output_shape
            layer_i = ConvNdLayer(
                    input_shape=previous_layer_shape,
                    filter_size=filter_size_list[i],
                    num_filters=num_filters_list[i],
                    pooling_padding=pooling_padding_list[i],
                    pooling_strides=pooling_strides_list[i],
                    pooling_type=pooling_type_list[i],
                    pooling_ksize=pooling_ksize_list[i],
                    strides=strides_list[i],
                    padding=padding_list[i],
                    use_dropout=use_dropout_list[i],
                    activation=activation_list[i],
                    use_batch_normalisation=use_batch_normalisation_list[i],
                    dilation_rate=dilation_rate_list[i],
                    use_residual=use_residual_list[i],
                    method=method_list[i],
                    repair_std_deviation=repair_std_deviation_list[i],
                    weights=weights_list[i],
                    biases=biases_list[i],
                    trafo=trafo_list[i],
                    hex_num_rotations=hex_num_rotations_list[i],
                    hex_azimuth=hex_azimuth_list[i],
                    hex_zero_out=hex_zero_out_list[i],
                    float_precision=float_precision,
                    name='{}_{:03d}'.format(name, i),
                    )
            if verbose:
                print('{}_{:03d}'.format(name, i), layer_i.output_shape)
            self.layers.append(layer_i)

    def __call__(self, inputs, is_training, keep_prob=None):
        """Apply layers

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        is_training : None, optional
            Indicates whether currently in training or inference mode.
            Must be provided if batch normalisation is used.
            True: in training mode
            False: inference mode.
        keep_prob : None, optional
            The keep probability to be used for dropout.
            Can either be a float or a scalar float tf.Tensor.

        Returns
        -------
        list of tf.Tensor
            The list of output tensors of each layer.
        """
        layer_outputs = []
        x = inputs
        for layer in self.layers:
            x = layer(x, is_training=is_training, keep_prob=keep_prob)
            layer_outputs.append(x)
        return layer_outputs
