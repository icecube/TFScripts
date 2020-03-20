'''
IceCube specific constants and functions
'''

from __future__ import division, print_function

import numpy as np
import tensorflow as tf

# tfscripts specific imports
from tfscripts.weights import new_weights

# constants
from tfscripts import FLOAT_PRECISION

# -----------------------------------------------------------------------------
#                           IceCube Constants
# -----------------------------------------------------------------------------
string_hex_coord_dict = {

    # row 1
    1: (-4, -1), 2: (-4, 0), 3: (-4, 1), 4: (-4, 2), 5: (-4, 3), 6: (-4, 4),

    # row 2
    7: (-3, -2), 8: (-3, -1), 9: (-3, 0), 10: (-3, 1), 11: (-3, 2),
    12: (-3, 3), 13: (-3, 4),

    # row 3
    14: (-2, -3), 15: (-2, -2), 16: (-2, -1), 17: (-2, 0), 18: (-2, 1),
    19: (-2, 2), 20: (-2, 3), 21: (-2, 4),

    # row 4
    22: (-1, -4), 23: (-1, -3), 24: (-1, -2), 25: (-1, -1), 26: (-1, 0),
    27: (-1, 1), 28: (-1, 2), 29: (-1, 3), 30: (-1, 4),

    # row 5
    31: (0, -5), 32: (0, -4), 33: (0, -3), 34: (0, -2), 35: (0, -1),
    36: (0, 0), 37: (0, 1), 38: (0, 2), 39: (0, 3), 40: (0, 4),

    # row 6
    41: (1, -5), 42: (1, -4), 43: (1, -3), 44: (1, -2), 45: (1, -1),
    46: (1, 0), 47: (1, 1), 48: (1, 2), 49: (1, 3), 50: (1, 4),

    # row 7
    51: (2, -5), 52: (2, -4), 53: (2, -3), 54: (2, -2), 55: (2, -1),
    56: (2, 0), 57: (2, 1), 58: (2, 2), 59: (2, 3),

    # row 8
    60: (3, -5), 61: (3, -4), 62: (3, -3), 63: (3, -2), 64: (3, -1),
    65: (3, 0), 66: (3, 1), 67: (3, 2),

    # row 9
    68: (4, -5), 69: (4, -4), 70: (4, -3), 71: (4, -2), 72: (4, -1),
    73: (4, 0), 74: (4, 1),

    # row 10
    5: (5, -5), 76: (5, -4), 77: (5, -3), 78: (5, -2),
}

# first index goes 'up' in hex coord: string 36 to 46
# second index goes 'right' in hex coord: string 36 to 37
hex_string_coord_dict = {
    # row 1
    (-4, -1): 1, (-4, 0): 2, (-4, 1): 3, (-4, 2): 4, (-4, 3): 5, (-4, 4): 6,

    # row 2
    (-3, -2): 7, (-3, -1): 8, (-3, 0): 9, (-3, 1): 10, (-3, 2): 11,
    (-3, 3): 12, (-3, 4): 13,

    # row 3
    (-2, -3): 14, (-2, -2): 15, (-2, -1): 16, (-2, 0): 17, (-2, 1): 18,
    (-2, 2): 19, (-2, 3): 20, (-2, 4): 21,

    # row 4
    (-1, -4): 22, (-1, -3): 23, (-1, -2): 24, (-1, -1): 25, (-1, 0): 26,
    (-1, 1): 27, (-1, 2): 28, (-1, 3): 29, (-1, 4): 30,

    # row 5
    (0, -5): 31, (0, -4): 32, (0, -3): 33, (0, -2): 34, (0, -1): 35,
    (0, 0): 36, (0, 1): 37, (0, 2): 38, (0, 3): 39, (0, 4): 40,

    # row 6
    (1, -5): 41, (1, -4): 42, (1, -3): 43, (1, -2): 44, (1, -1): 45,
    (1, 0): 46, (1, 1): 47, (1, 2): 48, (1, 3): 49, (1, 4): 50,

    # row 7
    (2, -5): 51, (2, -4): 52, (2, -3): 53, (2, -2): 54, (2, -1): 55,
    (2, 0): 56, (2, 1): 57, (2, 2): 58, (2, 3): 59,

    # row 8
    (3, -5): 60, (3, -4): 61, (3, -3): 62, (3, -2): 63, (3, -1): 64,
    (3, 0): 65, (3, 1): 66, (3, 2): 67,

    # row 9
    (4, -5): 68, (4, -4): 69, (4, -3): 70, (4, -2): 71, (4, -1): 72,
    (4, 0): 73, (4, 1): 74,

    # row 10
    (5, -5): 75, (5, -4): 76, (5, -3): 77, (5, -2): 78,
}

# -----------------------------------------------------------------------------
#                           IceCube Functions
# -----------------------------------------------------------------------------


def get_hex_coord_from_icecube_string(icecube_string_num):
    """Get hexagonal coordinates for a given IceCube string number

    Parameters
    ----------
    icecube_string_num : int
        IceCube string number.

    Returns
    -------
    int, int
        Hexagonal coordinates.
    """
    return string_hex_coord_dict[icecube_string_num]


def get_icecube_string_from_hex_coord(a, b):
    """Get string Number from hex-coordinates a and b

    Parameters
    ----------
    a : int
        Hexagonal coordinate along x axis.
    b : int
        Hexagonal coordinate along y axis.

    Returns
    -------
    int
        IceCube string number.
    """
    return hex_string_coord_dict[(a, b)]


def get_icecube_kernel(shape, get_ones=False, float_precision=FLOAT_PRECISION):
    '''
    Get a kernel of shape 'shape' for IceCube where coordinates of no real
    strings are set to constant zeros.

    Parameters
    ----------
    shape : list of int
        The shape of the desired kernel.

    get_ones : bool, optional
        If True, returns constant ones for real DOMs, zeros for virtual DOMs.
        If False, return trainable parameter for real DOMs,
                zeros for virtual DOMs
    float_precision : tf.dtype, optional
        The tensorflow dtype describing the float precision to use.

    Returns
    -------
    tf.Tensor
        The icecube kernel with the desired shape.
    list of tf.Variable
        A list of tensorflow variables created in this function
    '''
    zeros = tf.zeros(shape, dtype=float_precision)
    ones = tf.ones(shape, dtype=float_precision)

    var_list = []
    a_list = []
    for a in xrange(-4, 6):

        b_list = []
        for b in xrange(-5, 5):

            if (a, b) in hex_string_coord_dict.keys():
                # String exists
                if get_ones:
                    weights = ones
                else:
                    weights = new_weights(shape,
                                          float_precision=float_precision)
                    var_list.append(weights)
            else:
                # virtual string, string does not actually exist
                weights = zeros

            b_list.append(weights)
    a_list.append(tf.stack(b_list))
    icecube_kernel = tf.stack(a_list)
    return icecube_kernel, var_list
