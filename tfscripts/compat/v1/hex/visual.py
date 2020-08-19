'''
tfscripts.compat.v1 hexagonal convolution visualization functions
'''

from __future__ import division, print_function

import numpy as np
import tensorflow as tf
import matplotlib
try:
    import _tkinter
except ImportError:
    matplotlib.use("AGG")
import matplotlib.pyplot as plt

# tfscripts.compat.v1 specific imports
from tfscripts.compat.v1.hex.rotation import get_rotated_corner_weights

# constants
from tfscripts.compat.v1 import FLOAT_PRECISION


def print_hex_data(hex_data):
    """Print hexagonal kernel data to console

    Parameters
    ----------
    hex_data : dict
        A dictionary containing the hex kernel information as follows:
            hex_data = {(a,b): value)} and a, b hexagonal-coordinates
    """
    if len(hex_data.keys()) == 1:
        print('HexKernel is of size 1')
        print('\t1')

    else:
        # get range
        max_a = -float("inf")
        min_a = float("inf")
        max_b = -float("inf")
        min_b = float("inf")
        for key in hex_data.keys():
            a, b = key
            if a > max_a:
                max_a = a
            elif a < min_a:
                min_a = a
            if b > max_b:
                max_b = b
            elif b < min_b:
                min_b = b

        # print on hexagonal grid
        for a in range(max_a, min_a-1, -1):
            row = ''
            for i in range(a - min_a):
                row += '  '
            for b in range(min_b, max_b+1, 1):
                if (a, b) in hex_data.keys():
                    row += ' {:3d}'.format(hex_data[(a, b)])
                else:
                    row += '    '
            print(row+'\n')


def plot_hex2D(hex_grid, file=None, hex_grid_spacing=1.0, norm='linear'):
    '''Plots a 2D hex_grid heatmap

    Assumes hex_grid is shaped:
        1   1   0
      1   1   1
    0   1   1

    and that hex_grid[0] is left column
    and hex_grid[1] is right column

    Parameters
    ----------
    hex_grid : np.ndarray
        Quadratic 2D hexagonal heatmap to be plotted
        Shape: [size, size]
    file : string
        Save plot to this file.
        If None, plt.show() is called instead.
    hex_grid_spacing : float
        distance between hex_grid points along x axis
        (assume to x-axis aligned hexagons)
    norm : str, optional
        Color scale. Options: 'linear', 'log'.

    Raises
    ------
    ValueError
        Description
    '''
    fig = plt.figure()
    hex_grid = np.asarray(hex_grid)

    if len(hex_grid.shape) != 2:
        raise ValueError('Expects 2dim hexgrid, but got {!r}'.format(
                                                            hexgrid.shape))

    if hex_grid.shape[0] != hex_grid.shape[1]:
        raise ValueError('Only supports quadratic grids. {!r}'.format(
                                                            hex_grid.shape))

    minValue = np.min(hex_grid)
    maxValue = np.max(hex_grid)

    if norm == 'linear':
        norm = matplotlib.colors.Normalize(vmin=minValue, vmax=maxValue)
    elif norm == 'log':
        norm = matplotlib.colors.LogNorm(vmin=max(0.0001, minValue),
                                         vmax=maxValue)
    else:
        raise ValueError('Wrong value for norm: {!r}'.format(norm))
    cmap = matplotlib.cm.ScalarMappable(norm=norm,
                                        cmap=plt.get_cmap('viridis'))

    size = len(hex_grid)
    half_dim = size//2

    # create list to hold patches
    patch_list = []
    for a in range(-half_dim, half_dim + 1):
        for b in range(-half_dim, half_dim + 1):
            offset_x = hex_grid_spacing/2. * b
            x = offset_x + a*hex_grid_spacing
            y = b*(hex_grid_spacing*np.sqrt(0.75))
            color = cmap.to_rgba(hex_grid[a + half_dim, b + half_dim])
            patch_list.append(
                    matplotlib.patches.RegularPolygon(
                            xy=(x, y),
                            numVertices=6,
                            radius=hex_grid_spacing/np.sqrt(3),
                            orientation=0.,
                            facecolor=color,
                            edgecolor='black'
                    )
            )

    pc = matplotlib.collections.PatchCollection(patch_list,
                                                match_original=True)
    ax = plt.gca()
    ax.add_collection(pc)
    plt.plot(0, 0)
    plt.xlim = [-1.1*size*hex_grid_spacing/2, 1.1*size*hex_grid_spacing/2]
    plt.ylim = [-0.7*size*hex_grid_spacing/2, 0.7*size*hex_grid_spacing/2]

    plt.axes().set_aspect('equal')
    if file is None:
        plt.show()
    else:
        plt.savefig(file)
    plt.close(fig)


def visualize_rotated_hex_kernel(filter_size, num_rotations,
                                 file='Rotation_{azimuth:2.2f}.png'):
    '''Visualize hexagonal azimuth rotated filters

    Create Weights for a hexagonal kernel.
    The Kernel will be of a hexagonal shape in the first two dimensions.
    The hexagonal kernel is off the shape:
    [kernel_edge_points, kernel_edge_points,*filter_size[2:]]
    But elments with coordinates in the first two dimensions, that don't belong
    to the hexagon are set to zero.

    The hexagon is defined by filter_size[0:2].
    filter_size[0] defines the size of the hexagon and
    filter_size[1] the orientation.

    Parameters
    ----------
    filter_size : A list of int
          filter_size = [s, o, 3. dim(e.g. z), 4. dim(e.g. t),...]
          s: size of hexagon
          o: orientation of hexagon

    num_rotations : int.
      number of rotational kernels to create.
      Kernels will be rotated by 360 degrees / num_rotations


    file : str, optional
        A file pattern to which the plots of the rotated kernels will be saved
        to. The file pattern is formatted with a keyword 'azimuth' which holds
        the current azimuth rotation.

    Raises
    ------
    ValueError
        Description
    '''

    azimuths = np.linspace(0, 360, num_rotations+1)[:-1]
    Z = 0
    center_weight = np.random.uniform(1, high=15, size=1)

    # HARDCODE MAGIC... ToDo: Generalize
    if filter_size[0:2] == [2, 0]:
        # hexagonal 2,0 Filter
        corner_weights1 = np.random.uniform(1, high=15, size=6)
    elif filter_size[0:2] == [2, 1]:
        # hexagonal 2,1 Filter
        corner_weights1 = np.random.uniform(1, high=15, size=6)
        corner_weights2 = []
        for i in range(6):
            corner_weights2.extend(
                    [Z, np.random.uniform(1, high=15, size=1)[0]])
    elif filter_size[0:2] == [3, 0]:
        # hexagonal 3,0 Filter
        corner_weights1 = np.random.uniform(1, high=15, size=6)
        corner_weights2 = np.random.uniform(1, high=15, size=12)
    elif filter_size[0:2] == [3, 1]:
        # hexagonal 3,1 Filter
        corner_weights1 = np.random.uniform(1, high=15, size=6)
        corner_weights2 = np.random.uniform(1, high=15, size=12)
        corner_weights3 = []
        for i in range(6):
            corner_weights3.extend(
                    [Z, np.random.uniform(1, high=15, size=1)[0], Z])
    elif filter_size[0:2] == [3, 2]:
        # hexagonal 3,2 Filter
        corner_weights1 = np.random.uniform(1, high=15, size=6)
        corner_weights2 = np.random.uniform(1, high=15, size=12)
        corner_weights3 = []
        for i in range(6):
            corner_weights3.extend(
                    [Z, Z, np.random.uniform(1, high=15, size=1)[0]])
    elif filter_size[0:2] == [4, 0]:
        # hexagonal 4,0 Filter
        corner_weights1 = np.random.uniform(1, high=15, size=6)
        corner_weights2 = np.random.uniform(1, high=15, size=12)
        corner_weights3 = np.random.uniform(1, high=15, size=18)
    else:
        raise ValueError("visualize_rotated_hex_kernel: Unsupported hexagonal "
                         "filter_size: {!r}".formt(filter_size[0:2]))

    rotated_kernels = []
    for azimuth in azimuths:
        rotated_kernel_rows = []
        if filter_size[0:2] == [2, 0]:
            # hexagonal 2,0 Filter
            A = get_rotated_corner_weights(corner_weights1, azimuth)
            rotated_kernel_rows.append([Z, A[5], A[0]])
            rotated_kernel_rows.append([A[4], center_weight, A[1]])
            rotated_kernel_rows.append([A[3], A[2], Z])
        elif filter_size[0:2] == [2, 1] or filter_size[0:2] == [3, 0]:
            # hexagonal 3,0 Filter
            A = get_rotated_corner_weights(corner_weights1, azimuth)
            B = get_rotated_corner_weights(corner_weights2, azimuth)
            rotated_kernel_rows.append([Z, Z, B[9], B[10], B[11]])
            rotated_kernel_rows.append([Z, B[8], A[5], A[0], B[0]])
            rotated_kernel_rows.append([B[7], A[4], center_weight, A[1], B[1]])
            rotated_kernel_rows.append([B[6], A[3], A[2], B[2], Z])
            rotated_kernel_rows.append([B[5], B[4], B[3], Z, Z])
        elif (filter_size[0:2] == [3, 1] or filter_size[0:2] == [3, 2] or
              filter_size[0:2] == [4, 0]):
            # hexagonal 3,1 Filter
            A = get_rotated_corner_weights(corner_weights1, azimuth)
            B = get_rotated_corner_weights(corner_weights2, azimuth)
            C = get_rotated_corner_weights(corner_weights3, azimuth)
            rotated_kernel_rows.append([Z, Z, Z, C[15], C[16], C[17], C[0]])
            rotated_kernel_rows.append([Z, Z, C[14], B[9], B[10], B[11], C[1]])
            rotated_kernel_rows.append(
                        [Z, C[13], B[8], A[5], A[0], B[0], C[2]])
            rotated_kernel_rows.append(
                        [C[12], B[7], A[4], center_weight, A[1], B[1], C[3]])
            rotated_kernel_rows.append(
                        [C[11], B[6], A[3], A[2], B[2], C[4], Z])
            rotated_kernel_rows.append([C[10], B[5], B[4], B[3], C[5], Z, Z])
            rotated_kernel_rows.append([C[9], C[8], C[7], C[6], Z, Z, Z])
        else:
            raise ValueError("visualize_rotated_hex_kernel: Unsupported "
                             "hexagonal filter_size: {!r}".formt(
                                                            filter_size[0:2]))
        rotated_kernel = rotated_kernel_rows
        rotated_kernels.append(rotated_kernel)

    rotated_kernels = np.asarray(rotated_kernels)
    width = 3
    height = int(num_rotations/width)

    num_rows = len(rotated_kernels[0])

    for i, rot in enumerate(rotated_kernels):
        plot_hex2D(rot, file=file.format(azimuth=azimuths[i]))
