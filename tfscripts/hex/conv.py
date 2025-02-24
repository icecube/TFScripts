"""
tfscripts hexagonal convolution utility functions
    Hex utility functions
    hex conv 3d and 4d [tf.Modules]

ToDo:
    - Remove duplicate code
"""

from __future__ import division, print_function

import logging
import tensorflow as tf

# tfscripts specific imports
from tfscripts.utils import SeedCounter
from tfscripts.weights import new_weights
from tfscripts.hex.visual import print_hex_data
from tfscripts.hex import rotation
from tfscripts.hex.icecube import IceCubeKernel
from tfscripts.conv import dynamic_conv, conv4d_stacked

# constants
from tfscripts import FLOAT_PRECISION


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
        raise ValueError("get_num_hex_points: expected edge_length >= 0")
    if edge_length == 0:
        return 1
    return (edge_length - 1) * 6 + get_num_hex_points(edge_length - 1)


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
    c1 = -a1 - b1
    c2 = -a2 - b2
    return (abs(a1 - a2) + abs(b1 - b2) + abs(c1 - c2)) / 2


class HexKernel(tf.Module):
    """Hexagonal convolution kernel"""

    def __init__(
        self,
        filter_size,
        get_ones=False,
        float_precision=FLOAT_PRECISION,
        seed=None,
        name="HexKernel",
    ):
        """Get hexagonal convolution kernel

        Create Weights for a hexagonal kernel.
        The Kernel will be of a hexagonal shape in the first two dimensions,
        while the other dimensions are normal.
        The hexagonal kernel is of the shape:
        [kernel_edge_points, kernel_edge_points, *filter_size[2:]]
        But elements with coordinates in the first two dimensions, that don't belong
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

        get_ones : bool, optional
            If True, returns constant ones for elements in hexagon.
            If False, return trainable tf.tensor for elements in hexagon.
            In both cases, constant zeros are returned for elements outside of
            hexagon.
        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        seed : int, optional
            Seed for the random number generator.

        Returns
        -------
        tf.Tensor
            A Tensor with shape: [ s, s, *filter_size[2:] ]
            where s = 2*filter_size[0] -1 if x == o
                                [hexagon is parallel to axis of first dimension]
                    = 2*filter_size[0] +1 if x != o
                            [hexagon is tilted to axis of first dimension]
        list of tf.Variable
            A list of tensorflow variables created in this function

        Raises
        ------
        ValueError
            Description
        """
        # create seed counter
        cnt = SeedCounter(seed)

        k = filter_size[0]
        x = filter_size[1]

        if x >= k:
            raise ValueError(
                "HexKernel: filter_size (k,x,z) must fulfill "
                "x < k: ({}, {}, {})".format(k, x, filter_size[2])
            )
        if x == 0:
            kernel_edge_points = 2 * k - 1
        else:
            kernel_edge_points = 2 * k + 1

        zeros = tf.zeros(filter_size[2:], dtype=float_precision)
        ones = tf.ones(filter_size[2:], dtype=float_precision)

        self.var_list = []
        self.a_list = []
        self.test_hex_dict = {}
        for a in range(kernel_edge_points):
            b_list = []
            for b in range(kernel_edge_points):

                # -------------------------
                # regular aligned hexagons
                # -------------------------
                if x == 0:
                    if a + b < k - 1 or a + b > 3 * k - 3:
                        weights = zeros
                        self.test_hex_dict[(a, b)] = 0
                    else:
                        if get_ones:
                            weights = ones
                        else:
                            weights = new_weights(
                                filter_size[2:],
                                float_precision=float_precision,
                                seed=cnt(),
                                name=name + f"_weights_{a}_{b}",
                            )
                            self.var_list.append(weights)
                        self.test_hex_dict[(a, b)] = 1

                # -------------------------
                # tilted hexagons
                # -------------------------
                else:
                    inHexagon = False
                    # check if inside normal k.0 aligned hexagon
                    #   |----inside normal k.0 rhombus -----------|
                    if (
                        (a > 0 and a < 2 * k)
                        and (b > 0 and b < 2 * k)
                        and
                        #   |--in k.0 aligned hexagon-|
                        (a + b > k and a + b < 3 * k)
                    ):

                        if a + b > k and a + b < 3 * k:
                            inHexagon = True
                    else:
                        # add 6 additional edges outside of k.0 aligned hexagon
                        if a == 2 * k - x and b == 0:  # Edge 1
                            inHexagon = True
                        elif a == k - x and b == x:  # Edge 2
                            inHexagon = True
                        elif a == 0 and b == k + x:  # Edge 3
                            inHexagon = True
                        elif a == x and b == 2 * k:  # Edge 4
                            inHexagon = True
                        elif a == k + x and b == 2 * k - x:  # Edge 5
                            inHexagon = True
                        elif a == 2 * k and b == k - x:  # Edge 6
                            inHexagon = True
                    # get weights or constant 0 depending on if point is in hexagon
                    if inHexagon:
                        if get_ones:
                            weights = ones
                        else:
                            weights = new_weights(
                                filter_size[2:],
                                float_precision=float_precision,
                                seed=cnt(),
                                name=name + f"_weights_{a}_{b}",
                            )
                            self.var_list.append(weights)
                        self.test_hex_dict[(a, b)] = 1
                    else:
                        weights = zeros
                        self.test_hex_dict[(a, b)] = 0

                b_list.append(weights)
            self.a_list.append(b_list)

    def print_kernel(self):
        """Print the hexagonal kernel

        Print first two dimensions of kernel.
            0 represents a const 0 Tensor of shape filter_size[2:]
            1 represents a trainable Tensor of shape filter_size[2:]
            This can be used to verify the shape of the hex kernel
        """
        print_hex_data(self.test_hex_dict)

    def __call__(self):
        """Get the hexagonal kernel"""
        a_list = [tf.stack(b_list) for b_list in self.a_list]
        hex_kernel = tf.stack(a_list)
        return hex_kernel


class ConvHex(tf.Module):
    """Convolve a hex2d or hex3d layer (2d hex + 1d cartesian)"""

    def __init__(
        self,
        input_shape,
        filter_size,
        num_filters,
        padding="SAME",
        strides=[1, 1, 1, 1, 1],
        num_rotations=1,
        dilation_rate=None,
        zero_out=False,
        kernel=None,
        var_list=None,
        turn_azimuth=False,
        float_precision=FLOAT_PRECISION,
        seed=None,
        name=None,
    ):
        """Initialize object

        Parameters
        ----------
        input_shape : TensorShape, or list of int
            The shape of the inputs.
        filter_size : A list of int
            filter_size = [s, o, z]
            s: size of hexagon
            o: orientation of hexagon
            z: size along z axis

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
            If num_rotations > 1: weights of a kernel will be shared over
                'num_rotations' many rotated versions of that kernel.
        dilation_rate : None or list of int, optional
            The dilation rate to be used for the layer.
            Dilation rate is given by: [dilation_x, dilation_y, dilation_z]
            If the dilation rate is None, no dilation is applied in
            convolution.
        zero_out : bool, optional
            If True, elements in result tensor which are not part of hexagon or
            IceCube strings (if shape in x and y dimensions is 10x10), will be
            set to zero.
        kernel : None or tf.Tensor, optional
            Optionally, the weights to be used as the kernel can be provided.
            If a kernel is provided, a list of variables 'var_list' must also
            be provided.
            If None, new kernel weights are created.
        var_list : list of tf.Variables, optional
            A list of Variables of which the kernel is created from. This must
            only be provided (and only if) the parameter 'kernel' is not None.
        turn_azimuth : bool, optional
            If True, the kernel will be turned by the angle 'azimuth' in
            counterclockwise direction. The azimuth is given in degrees
            and must be provided in the __call__ method.
        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        seed : int, optional
            Seed for the random number generator.
        name : None, optional
            The name of the tensorflow module.

        Deleted Parameters
        ------------------
        input_data : tf.Tensor
            Input data.
        """
        super(ConvHex, self).__init__(name=name)
        self.turn_azimuth = turn_azimuth

        # make sure it is a 2d or 3d convolution
        assert len(input_shape) == 4 or len(input_shape) == 5

        # allocate random weights for kernel
        num_channels = input_shape[-1]

        if kernel is None:
            if self.turn_azimuth and filter_size[:2] != [1, 0]:
                kernel_obj = rotation.DynamicRotationHexKernel(
                    filter_size + [num_channels, num_filters],
                    float_precision=float_precision,
                    seed=seed,
                    name=self.name + "_kernel",
                )
                var_list = kernel_obj.var_list
            else:
                if num_rotations > 1:
                    kernel_obj = rotation.RotatedHexKernel(
                        filter_size + [num_channels, num_filters],
                        num_rotations,
                        float_precision=float_precision,
                        seed=seed,
                        name=self.name + "_kernel",
                    )
                    var_list = kernel_obj.var_list
                else:
                    kernel_obj = HexKernel(
                        filter_size + [num_channels, num_filters],
                        float_precision=float_precision,
                        seed=seed,
                        name=self.name + "_kernel",
                    )
                    var_list = kernel_obj.var_list
        else:

            def kernel_obj():
                return kernel

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        self.strides = strides
        self.num_rotations = num_rotations
        self.dilation_rate = dilation_rate
        self.zero_out = zero_out
        self.float_precision = float_precision
        self.seed = seed
        self.kernel_obj = kernel_obj
        self.kernel_var_list = var_list

    def __call__(self, inputs, azimuth=None):
        """Apply ConvHex Module.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        azimuth : float or scalar float tf.Tensor
            Hexagonal kernel is turned by the angle 'azimuth'
            [given in degrees] in counterclockwise direction

        Returns
        -------
        tf.Tensor
            The output tensor.
        """
        inputs = tf.convert_to_tensor(inputs)

        # make sure it is a 2d or 3d convolution
        assert len(inputs.get_shape()) == 4 or len(inputs.get_shape()) == 5

        if self.turn_azimuth and self.filter_size[:2] != [1, 0]:
            result = dynamic_conv(
                inputs=inputs,
                filters=self.kernel_obj(azimuth),
                strides=self.strides[1:-1],
                padding=self.padding,
                dilation_rate=self.dilation_rate,
            )
        else:
            result = tf.nn.convolution(
                input=inputs,
                filters=self.kernel_obj(),
                strides=self.strides[1:-1],
                padding=self.padding,
                dilations=self.dilation_rate,
            )

        # zero out elements that don't belong on hexagon or IceCube Strings
        if self.zero_out:

            if result.get_shape().as_list()[1:3] == [10, 10]:
                # Assuming IceCube shape
                logger = logging.getLogger(__name__)
                logger.warning("Assuming IceCube shape for layer", result)

                kernel_obj = IceCubeKernel(
                    result.get_shape().as_list()[3:],
                    get_ones=True,
                    float_precision=self.float_precision,
                    seed=self.seed,
                    name=self.name,
                )
                zero_out_matrix = kernel_obj()
                var_list = kernel_obj.var_list
                result = result * zero_out_matrix

                # Make sure there were no extra variables created.
                # These would have to be saved to tf.Module, to allow tracking
                assert var_list == [], "No created variables expected!"

            else:
                # Generic hexagonal shape
                kernel_obj = HexKernel(
                    [
                        (result.get_shape().as_list()[1] + 1) // 2,
                        0,
                        result.get_shape().as_list()[3],
                        self.num_filters * self.num_rotations,
                    ],
                    get_ones=True,
                    float_precision=self.float_precision,
                    seed=self.seed,
                )
                zero_out_matrix = kernel_obj()
                var_list = kernel_obj.var_list

                # Make sure there were no extra variables created.
                # These would have to be saved to tf.Module, to allow tracking
                assert var_list == [], "No created variables expected!"

                if result.get_shape()[1:] == zero_out_matrix.get_shape():
                    result = result * zero_out_matrix
                else:
                    raise ValueError(
                        "ConvHex: Shapes do not match for "
                        "zero_out_matrix and result. "
                        " {!r} != {!r}".format(
                            result.get_shape()[1:], zero_out_matrix.get_shape()
                        )
                    )

        return result


class ConvHex4d(tf.Module):
    """Convolve a hex4hex3d layer (2d hex + 1d cartesian)"""

    def __init__(
        self,
        input_shape,
        filter_size,
        num_filters,
        padding="VALID",
        strides=[1, 1, 1, 1, 1, 1],
        num_rotations=1,
        dilation_rate=None,
        kernel=None,
        var_list=None,
        turn_azimuth=False,
        stack_axis=None,
        zero_out=False,
        float_precision=FLOAT_PRECISION,
        seed=None,
        name=None,
    ):
        """Initialize object

        Parameters
        ----------
        input_shape : TensorShape, or list of int
            The shape of the inputs.
        filter_size : A list of int
            filter_size = [s, o, z, t]
            s: size of hexagon
            o: orientation of hexagon
            z: size along z axis
            t: size along t axis

            The hexagonal filter along axis x and y is put together from
            s and o.

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
        padding : str, optional
            The padding method to be used for the convolution operation.
            Options are 'VALID', 'SAME'.
        strides : list, optional
            The strides to be used for the convolution operation.
            Shape: [1, stride_x, stride_y, stride_z, stride_t, stride_channel]
            Examples:
                [1, 1, 1, 1, 1, 1]: a stride of 1 is used along all axes.
                [1, 1, 2, 1, 2, 1]: a stride of 2 is used along the
                y and t axis.
        num_rotations : int, optional
            If num_rotations >= 1: weights of a kernel will be shared over
                'num_rotations' many rotated versions of that kernel.
        dilation_rate : None or list of int, optional
            The dilation rate to be used for the layer.
            Dilation rate is given by: [dilation_x, dilation_y, dilation_z]
            If the dilation rate is None, no dilation is applied in
            convolution.
        kernel : None or tf.Tensor, optional
            Optionally, the weights to be used as the kernel can be provided.
            If a kernel is provided, a list of variables 'var_list' must also
            be provided.
            If None, new kernel weights are created.
        var_list : list of tf.Variables, optional
            A list of Variables of which the kernel is created from. This must
            only be provided (and only if) the parameter 'kernel' is not None.
        turn_azimuth : bool, optional
            If True, the kernel will be turned by the angle 'azimuth' in
            counterclockwise direction. The azimuth is given in degrees
            and must be provided in the __call__ method.
        stack_axis : Int
              Axis along which the convolutions will be stacked.
              By default the axis with the lowest output dimensionality will be
              chosen. This is only an educated guess of the best choice!
        zero_out : bool, optional
            If True, elements in result tensor which are not part of hexagon or
            IceCube strings (if shape in x and y dimensions is 10x10), will be
            set to zero.
        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        seed : int, optional
            Seed for the random number generator.
        name : None, optional
            The name of the tensorflow module.

        Deleted Parameters
        ------------------
        input_data : tf.Tensor
            Input data.
        """
        super(ConvHex4d, self).__init__(name=name)

        self.turn_azimuth = turn_azimuth

        # make sure it is a 4d convolution
        assert len(input_shape) == 6

        # allocate random weights for kernel
        num_channels = input_shape[5]

        if kernel is None:
            if self.turn_azimuth:
                kernel_obj = HexKernel(
                    filter_size + [num_channels, num_filters],
                    float_precision=float_precision,
                    seed=seed,
                    name=self.name + "_kernel",
                )
                var_list = kernel_obj.var_list
            else:
                if num_rotations > 1:
                    kernel_obj = rotation.RotatedHexKernel(
                        filter_size + [num_channels, num_filters],
                        num_rotations,
                        float_precision=float_precision,
                        seed=seed,
                        name=self.name + "_kernel",
                    )
                    var_list = kernel_obj.var_list
                else:
                    kernel_obj = HexKernel(
                        filter_size + [num_channels, num_filters],
                        float_precision=float_precision,
                        seed=seed,
                        name=self.name + "_kernel",
                    )
                    var_list = kernel_obj.var_list
        else:

            def kernel_obj():
                return kernel

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        self.strides = strides
        self.num_rotations = num_rotations
        self.dilation_rate = dilation_rate
        self.stack_axis = stack_axis
        self.zero_out = zero_out
        self.float_precision = float_precision
        self.seed = seed
        self.kernel_obj = kernel_obj
        self.kernel_var_list = var_list

    def __call__(self, inputs, azimuth=None):
        """Apply ConvHex4d Module.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor.
        azimuth : float or scalar float tf.Tensor
            Hexagonal kernel is turned by the angle 'azimuth'
            [given in degrees] in counterclockwise direction

        Returns
        -------
        tf.Tensor
            The output tensor.
        """
        inputs = tf.convert_to_tensor(inputs)

        # make sure it is a 4d convolution
        assert len(inputs.get_shape()) == 6

        if self.turn_azimuth is not None:
            kernel = self.kernel_obj(azimuth)
        else:
            kernel = self.kernel_obj()

        # convolve with tf conv4d_stacked
        result = conv4d_stacked(
            input=inputs,
            filter=kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            stack_axis=self.stack_axis,
        )

        # zero out elements that don't belong on hexagon
        if self.zero_out:
            kernel_obj = HexKernel(
                [
                    int((result.get_shape().as_list()[1] + 1) / 2),
                    0,
                    result.get_shape().as_list()[3],
                    result.get_shape().as_list()[4],
                    self.num_filters * self.num_rotations,
                ],
                get_ones=True,
                float_precision=self.float_precision,
                seed=self.seed,
            )
            zero_out_matrix = kernel_obj()
            var_list = kernel_obj.var_list

            # Make sure there were no extra variables created.
            # These would have to be saved to tf.Module, to allow tracking
            assert var_list == [], "No created variables expected!"

            if result.get_shape()[1:] == zero_out_matrix.get_shape():
                result = result * zero_out_matrix
            else:
                msg = "conv_hex4d: Shapes do not match for "
                msg += "zero_out_matrix and result. {!r} != {!r}"
                raise ValueError(
                    msg.format(
                        result.get_shape()[1:], zero_out_matrix.get_shape()
                    )
                )

        return result
