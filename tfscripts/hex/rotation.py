"""
tfscripts hexagonal rotation utility functions
    Hex rotation utility functions
    Rotated kernels: static and dynamic

ToDo:
    - Remove duplicate code
"""

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

from tfscripts.utils import SeedCounter

# tfscripts specific imports
from tfscripts.weights import new_weights

# constants
from tfscripts import FLOAT_PRECISION


def get_rotated_corner_weights(corner_weights, azimuth):
    """Rotates the points on a given hexagon layer/circle

    Parameters
    ----------
    corner_weights : list of np.ndarray or list of tf.Tensor
        List of weights along the given hexagon layer/circle
    azimuth : float
      Hexagonal kernel is turned by the angle 'azimuth' [given in degrees]
      in counterclockwise direction

    Returns
    -------
    list of np.ndarray or list of tf.Tensor [same shape and type as input]
        A list of the rotated weights along the given hexagon layer/circle.

    """
    size = len(corner_weights)
    degree_steps = 360.0 / size

    a = azimuth % degree_steps
    b = int(azimuth / degree_steps)

    rotatedcorner_weights = []
    for i in range(size):
        newCorner_i = corner_weights[i - b] + a / degree_steps * (
            corner_weights[i - b - 1] - corner_weights[i - b]
        )
        rotatedcorner_weights.append(newCorner_i)
    return rotatedcorner_weights


def tf_get_rotated_corner_weights(corner_weights, azimuth):
    """Rotates the points on a given hexagon layer/circle

    Parameters
    ----------
    corner_weights : list of tf.Tensor
        List of weights along the given hexagon layer/circle
    azimuth : float or scalar float tf.Tensor
      Hexagonal kernel is turned by the angle 'azimuth' [given in degrees]
      in counterclockwise direction

    Returns
    -------
    list of tf.Tensor [same shape and type as input]
        A list of the rotated weights along the given hexagon layer/circle.

    """
    size = corner_weights.get_shape().as_list()[0]
    num_dims = len(corner_weights.get_shape().as_list()[1:])
    degree_steps = 360.0 / size

    a = tf.reshape(azimuth % degree_steps, [1] * num_dims)
    b = tf.cast(azimuth / degree_steps, tf.int32)
    rotatedcorner_weights = []
    for i in range(size):

        index_1 = i - b
        index_2 = i - b - 1

        # correct negative indices
        index_1 = tf.where(index_1 < 0, size + index_1, index_1)
        index_2 = tf.where(index_2 < 0, size + index_2, index_2)

        newCorner_i = tf.gather(corner_weights, index_1) + a / degree_steps * (
            tf.gather(corner_weights, index_2)
            - tf.gather(corner_weights, index_1)
        )

        rotatedcorner_weights.append(newCorner_i)
    return rotatedcorner_weights


class DynamicRotationHexKernel(tf.Module):
    """Dynamically azimuthally rotated hexagonal kernels."""

    def __init__(
        self,
        filter_size,
        float_precision=FLOAT_PRECISION,
        seed=None,
        name="DynamicRotationHexKernel",
    ):
        """Dynamically azimuthally rotated hexagonal kernels.

        Create Weights for a hexagonal kernel.
        The Kernel is dynamically rotated by the 'azimuth' angle.
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
            filter_size[-2:] = [no_in_channels, no_out_channels]
            s: size of hexagon
            o: orientation of hexagon

            Examples:

                    s = 2, o = 0:
                                        1   1   0             1  1

                                    1   1   1             1   1   1

                                0   1   1                 1   1

        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        seed : int, optional
            Seed for the random number generator.
        name : str, optional
            The name of the operation.

        Returns
        -------
        tf.Tensor
            A Tensor with shape: [ s, s, *filter_size[2:]]
            where s = 2*filter_size[0] -1 if x == o
                                [hexagon is parallel to axis of first dimension]
                    = 2*filter_size[0] +1 if x != o
                                [hexagon is tilted to axis of first dimension]

        Raises
        ------
        ValueError
            Description

        """
        self.float_precision = float_precision
        self.filter_size = filter_size

        # create seed counter
        cnt = SeedCounter(seed)

        self.var_list = []
        self.no_of_dims = len(filter_size)

        self.Z = tf.zeros(filter_size[2:], dtype=float_precision)
        self.center_weight = new_weights(
            filter_size[2:],
            float_precision=float_precision,
            seed=cnt(),
            name=name + "_center_weight",
        )
        self.var_list.append(self.center_weight)

        self.corner_weights1 = None
        self.corner_weights2 = None
        self.corner_weights3 = None

        # HARDCODE MAGIC... ToDo: Generalize and clean up
        if filter_size[0:2] == [2, 0]:
            # hexagonal 2,0 Filter
            self.corner_weights1 = new_weights(
                [6] + filter_size[2:],
                float_precision=float_precision,
                seed=cnt(),
                name=name + "_corner_weights1",
            )
            self.var_list.append(self.corner_weights1)
        elif filter_size[0:2] == [2, 1]:
            # hexagonal 2,1 Filter
            self.corner_weights1 = new_weights(
                [6] + filter_size[2:],
                float_precision=float_precision,
                seed=cnt(),
                name=name + "_corner_weights1",
            )
            self.var_list.append(self.corner_weights1)
            self.corner_weights2 = []
            for i in range(6):
                weights = new_weights(
                    filter_size[2:],
                    float_precision=float_precision,
                    seed=cnt(),
                    name=name + "_corner_weights2",
                )
                self.var_list.append(weights)
                self.corner_weights2.extend([self.Z, weights])
        elif filter_size[0:2] == [3, 0]:
            # hexagonal 3,0 Filter
            self.corner_weights1 = new_weights(
                [6] + filter_size[2:],
                float_precision=float_precision,
                seed=cnt(),
                name=name + "_corner_weights1",
            )
            self.var_list.append(self.corner_weights1)
            self.corner_weights2 = new_weights(
                [12] + filter_size[2:],
                float_precision=float_precision,
                seed=cnt(),
                name=name + "_corner_weights2",
            )
            self.var_list.append(self.corner_weights2)
        elif filter_size[0:2] == [3, 1]:
            # hexagonal 3,1 Filter
            self.corner_weights1 = new_weights(
                [6] + filter_size[2:],
                float_precision=float_precision,
                seed=cnt(),
                name=name + "_corner_weights1",
            )
            self.var_list.append(self.corner_weights1)
            self.corner_weights2 = new_weights(
                [12] + filter_size[2:],
                float_precision=float_precision,
                seed=cnt(),
                name=name + "_corner_weights2",
            )
            self.var_list.append(self.corner_weights2)
            self.corner_weights3 = []
            for i in range(6):
                weights = new_weights(
                    filter_size[2:],
                    float_precision=float_precision,
                    seed=cnt(),
                    name=name + f"_corner_weights3_{i}",
                )
                self.var_list.append(weights)
                self.corner_weights3.extend([self.Z, weights, self.Z])
        elif filter_size[0:2] == [3, 2]:
            # hexagonal 3,2 Filter
            self.corner_weights1 = new_weights(
                [6] + filter_size[2:],
                float_precision=float_precision,
                seed=cnt(),
                name=name + "_corner_weights1",
            )
            self.var_list.append(self.corner_weights1)
            self.corner_weights2 = new_weights(
                [12] + filter_size[2:],
                float_precision=float_precision,
                seed=cnt(),
                name=name + "_corner_weights2",
            )
            self.var_list.append(self.corner_weights2)
            self.corner_weights3 = []
            for i in range(6):
                weights = new_weights(
                    filter_size[2:],
                    float_precision=float_precision,
                    seed=cnt(),
                    name=name + f"_corner_weights3_{i}",
                )
                self.var_list.append(weights)
                self.corner_weights3.extend([self.Z, self.Z, weights])
        elif filter_size[0:2] == [4, 0]:
            # hexagonal 4,0 Filter
            self.corner_weights1 = new_weights(
                [6] + filter_size[2:],
                float_precision=float_precision,
                seed=cnt(),
                name=name + "_corner_weights1",
            )
            self.var_list.append(self.corner_weights1)
            self.corner_weights2 = new_weights(
                [12] + filter_size[2:],
                float_precision=float_precision,
                seed=cnt(),
                name=name + "_corner_weights2",
            )
            self.var_list.append(self.corner_weights2)
            self.corner_weights3 = new_weights(
                [18] + filter_size[2:],
                float_precision=float_precision,
                seed=cnt(),
                name=name + "_corner_weights3",
            )
            self.var_list.append(self.corner_weights3)
        else:
            raise ValueError(
                "DynamicRotationHexKernel: Unsupported "
                "hexagonal filter_size: {!r}".format(filter_size[0:2])
            )

    def __call__(self, azimuth):
        """Dynamically azimuthally rotated hexagonal kernels.

        Parameters
        ----------
        azimuth : tf tensor
            A scalar float tf.Tensor denoting the angle by which the kernel will
            be dynamically rotated. Azimuth angle is given in degrees.

        Returns
        -------
        tf.Tensor
            A Tensor with shape: [ s, s, *filter_size[2:]]
            where s = 2*filter_size[0] -1 if x == o
                                [hexagon is parallel to axis of first dimension]
                    = 2*filter_size[0] +1 if x != o
                                [hexagon is tilted to axis of first dimension]
        """
        azimuth = tf.cast(azimuth, self.float_precision)
        assert azimuth.get_shape().as_list() == []

        multiples = [1] * (self.no_of_dims - 2)
        center_weight = tf.tile(self.center_weight, multiples)

        # Get corner weights
        corner_weights1 = self.corner_weights1
        if self.filter_size[0:2] == [2, 1]:
            corner_weights2 = tf.stack(self.corner_weights2)
        else:
            corner_weights2 = self.corner_weights2

        if self.filter_size[0:2] in ([3, 1], [3, 2]):
            corner_weights3 = tf.stack(self.corner_weights3)
        else:
            corner_weights3 = self.corner_weights3

        # combine kernel by rotating corner weights
        rotated_kernel_rows = []
        if self.filter_size[0:2] == [2, 0]:
            # hexagonal 2,0 Filter
            A = tf_get_rotated_corner_weights(corner_weights1, azimuth)
            rotated_kernel_rows.append(tf.stack([self.Z, A[5], A[0]], axis=0))
            rotated_kernel_rows.append(
                tf.stack([A[3], center_weight, A[1]], axis=0)
            )
            rotated_kernel_rows.append(tf.stack([A[3], A[2], self.Z], axis=0))
        elif self.filter_size[0:2] == [2, 1] or self.filter_size[0:2] == [
            3,
            0,
        ]:
            # hexagonal 2,1 and 3,0 Filter
            A = tf_get_rotated_corner_weights(corner_weights1, azimuth)
            B = tf_get_rotated_corner_weights(corner_weights2, azimuth)
            rotated_kernel_rows.append(
                tf.stack([self.Z, self.Z, B[9], B[10], B[11]], axis=0)
            )
            rotated_kernel_rows.append(
                tf.stack([self.Z, B[8], A[5], A[0], B[0]], axis=0)
            )
            rotated_kernel_rows.append(
                tf.stack([B[7], A[4], center_weight, A[1], B[1]], axis=0)
            )
            rotated_kernel_rows.append(
                tf.stack([B[6], A[3], A[2], B[2], self.Z], axis=0)
            )
            rotated_kernel_rows.append(
                tf.stack([B[5], B[4], B[3], self.Z, self.Z], axis=0)
            )
        elif (
            self.filter_size[0:2] == [3, 1]
            or self.filter_size[0:2] == [3, 2]
            or self.filter_size[0:2] == [4, 0]
        ):
            # hexagonal 3,1 3,2 and 4,0 filter
            A = tf_get_rotated_corner_weights(corner_weights1, azimuth)
            B = tf_get_rotated_corner_weights(corner_weights2, azimuth)
            C = tf_get_rotated_corner_weights(corner_weights3, azimuth)
            rotated_kernel_rows.append(
                tf.stack(
                    [self.Z, self.Z, self.Z, C[15], C[16], C[17], C[0]], axis=0
                )
            )
            rotated_kernel_rows.append(
                tf.stack(
                    [self.Z, self.Z, C[14], B[9], B[10], B[11], C[1]], axis=0
                )
            )
            rotated_kernel_rows.append(
                tf.stack([self.Z, C[13], B[8], A[5], A[0], B[0], C[2]], axis=0)
            )
            rotated_kernel_rows.append(
                tf.stack(
                    [C[12], B[7], A[4], center_weight, A[1], B[1], C[3]],
                    axis=0,
                )
            )
            rotated_kernel_rows.append(
                tf.stack([C[11], B[6], A[3], A[2], B[2], C[4], self.Z], axis=0)
            )
            rotated_kernel_rows.append(
                tf.stack(
                    [C[10], B[5], B[4], B[3], C[5], self.Z, self.Z], axis=0
                )
            )
            rotated_kernel_rows.append(
                tf.stack(
                    [C[9], C[8], C[7], C[6], self.Z, self.Z, self.Z], axis=0
                )
            )
        else:
            raise ValueError(
                "DynamicRotationHexKernel: Unsupported "
                "hexagonal filter_size: {!r}".format(self.filter_size[0:2])
            )

        rotated_kernel = tf.stack(rotated_kernel_rows, axis=0)

        return rotated_kernel


# def get_dynamic_rotation_hex_kernel(
#     filter_size,
#     azimuth,
#     float_precision=FLOAT_PRECISION,
#     seed=None,
#     name="DynamicRotationHexKernel",
# ):
#     """Dynamically azimuthally rotated hexagonal kernels.

#     Create Weights for a hexagonal kernel.
#     The Kernel is dynamically rotated by the 'azimuth' angle.
#     The Kernel will be of a hexagonal shape in the first two dimensions,
#     while the other dimensions are normal.
#     The hexagonal kernel is of the shape:
#         [kernel_edge_points, kernel_edge_points, *filter_size[2:]]
#     But elements with coordinates in the first two dimensions, that don't belong
#     to the hexagon are set to a tf.Constant 0.

#     The hexagon is defined by filter_size[0:2].
#     filter_size[0] defines the size of the hexagon and
#     filter_size[1] the orientation.

#     Parameters
#     ----------
#     filter_size : A list of int
#         filter_size = [s, o, 3. dim(e.g. z), 4. dim(e.g. t),...]
#         filter_size[-2:] = [no_in_channels, no_out_channels]
#         s: size of hexagon
#         o: orientation of hexagon

#         Examples:

#                   s = 2, o = 0:
#                                     1   1   0             1  1

#                                  1   1   1             1   1   1

#                                0   1   1                 1   1

#     azimuth : tf tensor
#         A scalar float tf.Tensor denoting the angle by which the kernel will
#         be dynamically rotated. Azimuth angle is given in degrees.
#         Shape: [batch_size]
#     float_precision : tf.dtype, optional
#         The tensorflow dtype describing the float precision to use.
#     seed : int, optional
#         Seed for the random number generator.
#     name : str, optional
#         The name of the operation.

#     Returns
#     -------
#     tf.Tensor
#         A Tensor with shape: [ s, s, *filter_size[2:]]
#         where s = 2*filter_size[0] -1 if x == o
#                             [hexagon is parallel to axis of first dimension]
#                 = 2*filter_size[0] +1 if x != o
#                             [hexagon is tilted to axis of first dimension]

#     Raises
#     ------
#     ValueError
#         Description

#     """
#     # create seed counter
#     cnt = SeedCounter(seed)

#     azimuth = tf.cast(azimuth, float_precision)
#     assert azimuth.get_shape().as_list() == []

#     var_list = []
#     no_of_dims = len(filter_size)

#     Z = tf.zeros(
#         filter_size[2:], dtype=float_precision
#     )
#     center_weight = new_weights(
#         filter_size[2:],
#         float_precision=float_precision,
#         seed=cnt(),
#         name=name + "_center_weight",
#     )
#     var_list.append(center_weight)
#     multiples = [1] * (no_of_dims - 2)
#     center_weight = tf.tile(center_weight, multiples)

#     # HARDCODE MAGIC... ToDo: Generalize and clean up
#     if filter_size[0:2] == [2, 0]:
#         # hexagonal 2,0 Filter
#         corner_weights1 = new_weights(
#             [6] + filter_size[2:],
#             float_precision=float_precision,
#             seed=cnt(),
#             name=name + "_corner_weights1",
#         )
#         var_list.append(corner_weights1)
#     elif filter_size[0:2] == [2, 1]:
#         # hexagonal 2,1 Filter
#         corner_weights1 = new_weights(
#             [6] + filter_size[2:],
#             float_precision=float_precision,
#             seed=cnt(),
#             name=name + "_corner_weights1",
#         )
#         var_list.append(corner_weights1)
#         corner_weights2 = []
#         for i in range(6):
#             weights = new_weights(
#                 filter_size[2:],
#                 float_precision=float_precision,
#                 seed=cnt(),
#                 name=name + "_corner_weights2",
#             )
#             var_list.append(weights)
#             corner_weights2.extend([Z, weights])
#         corner_weights2 = tf.stack(corner_weights2)
#     elif filter_size[0:2] == [3, 0]:
#         # hexagonal 3,0 Filter
#         corner_weights1 = new_weights(
#             [6] + filter_size[2:],
#             float_precision=float_precision,
#             seed=cnt(),
#             name=name + "_corner_weights1",
#         )
#         var_list.append(corner_weights1)
#         corner_weights2 = new_weights(
#             [12] + filter_size[2:],
#             float_precision=float_precision,
#             seed=cnt(),
#             name=name + "_corner_weights2",
#         )
#         var_list.append(corner_weights2)
#     elif filter_size[0:2] == [3, 1]:
#         # hexagonal 3,1 Filter
#         corner_weights1 = new_weights(
#             [6] + filter_size[2:],
#             float_precision=float_precision,
#             seed=cnt(),
#             name=name + "_corner_weights1",
#         )
#         var_list.append(corner_weights1)
#         corner_weights2 = new_weights(
#             [12] + filter_size[2:],
#             float_precision=float_precision,
#             seed=cnt(),
#             name=name + "_corner_weights2",
#         )
#         var_list.append(corner_weights2)
#         corner_weights3 = []
#         for i in range(6):
#             weights = new_weights(
#                 filter_size[2:],
#                 float_precision=float_precision,
#                 seed=cnt(),
#                 name=name + f"_corner_weights3_{i}",
#             )
#             var_list.append(weights)
#             corner_weights3.extend([Z, weights, Z])
#         corner_weights3 = tf.stack(corner_weights3)
#     elif filter_size[0:2] == [3, 2]:
#         # hexagonal 3,2 Filter
#         corner_weights1 = new_weights(
#             [6] + filter_size[2:],
#             float_precision=float_precision,
#             seed=cnt(),
#             name=name + "_corner_weights1",
#         )
#         var_list.append(corner_weights1)
#         corner_weights2 = new_weights(
#             [12] + filter_size[2:],
#             float_precision=float_precision,
#             seed=cnt(),
#             name=name + "_corner_weights2",
#         )
#         var_list.append(corner_weights2)
#         corner_weights3 = []
#         for i in range(6):
#             weights = new_weights(
#                 filter_size[2:],
#                 float_precision=float_precision,
#                 seed=cnt(),
#                 name=name + f"_corner_weights3_{i}",
#             )
#             var_list.append(weights)
#             corner_weights3.extend([Z, Z, weights])
#         corner_weights3 = tf.stack(corner_weights3)
#     elif filter_size[0:2] == [4, 0]:
#         # hexagonal 4,0 Filter
#         corner_weights1 = new_weights(
#             [6] + filter_size[2:],
#             float_precision=float_precision,
#             seed=cnt(),
#             name=name + "_corner_weights1",
#         )
#         var_list.append(corner_weights1)
#         corner_weights2 = new_weights(
#             [12] + filter_size[2:],
#             float_precision=float_precision,
#             seed=cnt(),
#             name=name + "_corner_weights2",
#         )
#         var_list.append(corner_weights2)
#         corner_weights3 = new_weights(
#             [18] + filter_size[2:],
#             float_precision=float_precision,
#             seed=cnt(),
#             name=name + "_corner_weights3",
#         )
#         var_list.append(corner_weights3)
#     else:
#         raise ValueError(
#             "get_dynamic_rotation_hex_kernel: Unsupported "
#             "hexagonal filter_size: {!r}".format(filter_size[0:2])
#         )

#     rotated_kernel_rows = []
#     if filter_size[0:2] == [2, 0]:
#         # hexagonal 2,0 Filter
#         A = tf_get_rotated_corner_weights(corner_weights1, azimuth)
#         rotated_kernel_rows.append(tf.stack([Z, A[5], A[0]], axis=0))
#         rotated_kernel_rows.append(
#             tf.stack([A[3], center_weight, A[1]], axis=0)
#         )
#         rotated_kernel_rows.append(tf.stack([A[3], A[2], Z], axis=0))
#     elif filter_size[0:2] == [2, 1] or filter_size[0:2] == [3, 0]:
#         # hexagonal 2,1 and 3,0 Filter
#         A = tf_get_rotated_corner_weights(corner_weights1, azimuth)
#         B = tf_get_rotated_corner_weights(corner_weights2, azimuth)
#         rotated_kernel_rows.append(
#             tf.stack([Z, Z, B[9], B[10], B[11]], axis=0)
#         )
#         rotated_kernel_rows.append(
#             tf.stack([Z, B[8], A[5], A[0], B[0]], axis=0)
#         )
#         rotated_kernel_rows.append(
#             tf.stack([B[7], A[4], center_weight, A[1], B[1]], axis=0)
#         )
#         rotated_kernel_rows.append(
#             tf.stack([B[6], A[3], A[2], B[2], Z], axis=0)
#         )
#         rotated_kernel_rows.append(tf.stack([B[5], B[4], B[3], Z, Z], axis=0))
#     elif (
#         filter_size[0:2] == [3, 1]
#         or filter_size[0:2] == [3, 2]
#         or filter_size[0:2] == [4, 0]
#     ):
#         # hexagonal 3,1 3,2 and 4,0 filter
#         A = tf_get_rotated_corner_weights(corner_weights1, azimuth)
#         B = tf_get_rotated_corner_weights(corner_weights2, azimuth)
#         C = tf_get_rotated_corner_weights(corner_weights3, azimuth)
#         rotated_kernel_rows.append(
#             tf.stack([Z, Z, Z, C[15], C[16], C[17], C[0]], axis=0)
#         )
#         rotated_kernel_rows.append(
#             tf.stack([Z, Z, C[14], B[9], B[10], B[11], C[1]], axis=0)
#         )
#         rotated_kernel_rows.append(
#             tf.stack([Z, C[13], B[8], A[5], A[0], B[0], C[2]], axis=0)
#         )
#         rotated_kernel_rows.append(
#             tf.stack(
#                 [C[12], B[7], A[4], center_weight, A[1], B[1], C[3]], axis=0
#             )
#         )
#         rotated_kernel_rows.append(
#             tf.stack([C[11], B[6], A[3], A[2], B[2], C[4], Z], axis=0)
#         )
#         rotated_kernel_rows.append(
#             tf.stack([C[10], B[5], B[4], B[3], C[5], Z, Z], axis=0)
#         )
#         rotated_kernel_rows.append(
#             tf.stack([C[9], C[8], C[7], C[6], Z, Z, Z], axis=0)
#         )
#     else:
#         raise ValueError(
#             "get_dynamic_rotation_hex_kernel: Unsupported "
#             "hexagonal filter_size: {!r}".format(filter_size[0:2])
#         )

#     rotated_kernel = tf.stack(rotated_kernel_rows, axis=0)

#     return rotated_kernel, var_list


# -------------------------------------------------------------------------
#       hexagonal azimuth rotated filters
# -------------------------------------------------------------------------
class RotatedHexKernel(tf.Module):
    """Rotated hexagonal kernels."""

    def __init__(
        self,
        filter_size,
        num_rotations,
        float_precision=FLOAT_PRECISION,
        seed=None,
        name="RotatedHexKernel",
    ):
        """
        Create Weights for a hexagonal kernel.
        The kernel is rotated 'num_rotations' many times.
        Weights are shared over rotated versions.
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
            filter_size[-2:] = [no_in_channels, no_out_channels]
            s: size of hexagon
            o: orientation of hexagon

            Examples:

                    s = 2, o = 0:
                                        1   1   0             1  1

                                    1   1   1             1   1   1

                                0   1   1                 1   1

        num_rotations : int.
        number of rotational kernels to create.
        Kernels will be rotated by 360 degrees / num_rotations
        float_precision : tf.dtype, optional
            The tensorflow dtype describing the float precision to use.
        seed : int, optional
            Seed for the random number generator.
        name : str, optional
            The name of the operation.

        Returns
        -------
        tf.Tensor
            A Tensor with shape:
                [ s, s, *filter_size[2:-1], filter_size[-1]*num_rotations ]
                where s = 2*filter_size[0] -1 if x == o
                                [hexagon is parallel to axis of first dimension]
                        = 2*filter_size[0] +1 if x != o
                                [hexagon is tilted to axis of first dimension]

        Raises
        ------
        ValueError
            Description

        """
        self.float_precision = float_precision
        self.filter_size = filter_size

        # create seed counter
        cnt = SeedCounter(seed)

        # define function to get new weights with correct shape
        self.var_list = []

        def get_new_weights():
            weights = new_weights(
                filter_size[2:-2],
                float_precision=float_precision,
                seed=cnt(),
                name=name,
            )
            self.var_list.append(weights)
            return weights

        self.no_of_dims = len(filter_size)
        self.azimuths = np.linspace(0, 360, num_rotations + 1)[:-1]
        self.Z = tf.zeros(filter_size[2:-2], dtype=float_precision)
        self.center_weight = get_new_weights()

        self.corner_weights1 = None
        self.corner_weights2 = None
        self.corner_weights3 = None

        # HARDCODE MAGIC... ToDo: Generalize
        if filter_size[0:2] == [2, 0]:
            # hexagonal 2,0 Filter
            self.corner_weights1 = [get_new_weights() for i in range(6)]

        elif filter_size[0:2] == [2, 1]:
            # hexagonal 2,1 Filter
            self.corner_weights1 = [get_new_weights() for i in range(6)]

            self.corner_weights2 = []
            for i in range(6):
                self.corner_weights2.extend([self.Z, get_new_weights()])

        elif filter_size[0:2] == [3, 0]:
            # hexagonal 3,0 Filter
            self.corner_weights1 = [get_new_weights() for i in range(6)]
            self.corner_weights2 = [get_new_weights() for i in range(12)]

        elif filter_size[0:2] == [3, 1]:
            # hexagonal 3,1 Filter
            self.corner_weights1 = [get_new_weights() for i in range(6)]
            self.corner_weights2 = [get_new_weights() for i in range(12)]

            self.corner_weights3 = []
            for i in range(6):
                self.corner_weights3.extend(
                    [self.Z, get_new_weights(), self.Z]
                )

        elif filter_size[0:2] == [3, 2]:
            # hexagonal 3,2 Filter
            self.corner_weights1 = [get_new_weights() for i in range(6)]
            self.corner_weights2 = [get_new_weights() for i in range(12)]

            self.corner_weights3 = []
            for i in range(6):
                self.corner_weights3.extend(
                    [self.Z, self.Z, get_new_weights()]
                )

        elif filter_size[0:2] == [4, 0]:
            # hexagonal 4,0 Filter
            self.corner_weights1 = [get_new_weights() for i in range(6)]
            self.corner_weights2 = [get_new_weights() for i in range(12)]
            self.corner_weights3 = [get_new_weights() for i in range(18)]

        else:
            raise ValueError(
                "RotatedHexKernel: Unsupported "
                "hexagonal filter_size: {!r}".format(filter_size[0:2])
            )

        self.in_out_channel_weights = new_weights(
            [num_rotations] + filter_size[-2:],
            float_precision=float_precision,
            seed=cnt(),
            name=name + "_in_out_channel_weights",
        )
        self.var_list.append(self.in_out_channel_weights)

    def __call__(self):
        """Rotated hexagonal kernels.

        Returns
        -------
        tf.Tensor
            A Tensor with shape:
                [ s, s, *filter_size[2:-1], filter_size[-1]*num_rotations ]
                where s = 2*filter_size[0] -1 if x == o
                                [hexagon is parallel to axis of first dimension]
                        = 2*filter_size[0] +1 if x != o
                                [hexagon is tilted to axis of first dimension]
        """
        rotated_kernels = []
        for i, azimuth in enumerate(self.azimuths):
            rotated_kernel_rows = []
            if self.filter_size[0:2] == [2, 0]:
                # hexagonal 2,0 Filter
                A = get_rotated_corner_weights(self.corner_weights1, azimuth)
                rotated_kernel_rows.append(tf.stack([self.Z, A[5], A[0]]))
                rotated_kernel_rows.append(
                    tf.stack([A[3], self.center_weight, A[1]])
                )
                rotated_kernel_rows.append(tf.stack([A[3], A[2], self.Z]))
            elif self.filter_size[0:2] == [2, 1] or self.filter_size[0:2] == [
                3,
                0,
            ]:
                # hexagonal 2,1 and 3,0 Filter
                A = get_rotated_corner_weights(self.corner_weights1, azimuth)
                B = get_rotated_corner_weights(self.corner_weights2, azimuth)
                rotated_kernel_rows.append(
                    tf.stack([self.Z, self.Z, B[9], B[10], B[11]])
                )
                rotated_kernel_rows.append(
                    tf.stack([self.Z, B[8], A[5], A[0], B[0]])
                )
                rotated_kernel_rows.append(
                    tf.stack([B[7], A[4], self.center_weight, A[1], B[1]])
                )
                rotated_kernel_rows.append(
                    tf.stack([B[6], A[3], A[2], B[2], self.Z])
                )
                rotated_kernel_rows.append(
                    tf.stack([B[5], B[4], B[3], self.Z, self.Z])
                )
            elif (
                self.filter_size[0:2] == [3, 1]
                or self.filter_size[0:2] == [3, 2]
                or self.filter_size[0:2] == [4, 0]
            ):
                # hexagonal 3,1 3,2 and 4,0 filter
                A = get_rotated_corner_weights(self.corner_weights1, azimuth)
                B = get_rotated_corner_weights(self.corner_weights2, azimuth)
                C = get_rotated_corner_weights(self.corner_weights3, azimuth)
                rotated_kernel_rows.append(
                    tf.stack(
                        [self.Z, self.Z, self.Z, C[15], C[16], C[17], C[0]]
                    )
                )
                rotated_kernel_rows.append(
                    tf.stack([self.Z, self.Z, C[14], B[9], B[10], B[11], C[1]])
                )
                rotated_kernel_rows.append(
                    tf.stack([self.Z, C[13], B[8], A[5], A[0], B[0], C[2]])
                )
                rotated_kernel_rows.append(
                    tf.stack(
                        [
                            C[12],
                            B[7],
                            A[4],
                            self.center_weight,
                            A[1],
                            B[1],
                            C[3],
                        ]
                    )
                )
                rotated_kernel_rows.append(
                    tf.stack([C[11], B[6], A[3], A[2], B[2], C[4], self.Z])
                )
                rotated_kernel_rows.append(
                    tf.stack([C[10], B[5], B[4], B[3], C[5], self.Z, self.Z])
                )
                rotated_kernel_rows.append(
                    tf.stack([C[9], C[8], C[7], C[6], self.Z, self.Z, self.Z])
                )
            else:
                raise ValueError(
                    "RotatedHexKernel: Unsupported hexagonal "
                    "filter_size: {!r}".format(self.filter_size[0:2])
                )
            rotated_kernel_single = tf.stack(rotated_kernel_rows)

            # Add free parameters for in and out channel
            # tile to correct format
            rotated_kernel_single = tf.expand_dims(rotated_kernel_single, -1)
            rotated_kernel_single = tf.expand_dims(rotated_kernel_single, -1)

            multiples = [
                1 for i in range(self.no_of_dims - 2)
            ] + self.filter_size[-2:]
            rotated_kernel_tiled = tf.tile(rotated_kernel_single, multiples)

            # multiply weights to make in and out channels independent
            rotated_kernel = (
                rotated_kernel_tiled * self.in_out_channel_weights[i]
            )

            rotated_kernels.append(rotated_kernel)

        rotated_kernels = tf.concat(
            values=rotated_kernels, axis=len(self.filter_size) - 1
        )
        return rotated_kernels
