#!/usr/local/bin/python
from __future__ import division, print_function

import unittest
import numpy as np
import tensorflow as tf
from itertools import product

from tfscripts import conv


class TestConvModule(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState(42)

    def test_locally_connected_2d(self):
        """Test locally_connected_2d
        """
        data = tf.constant(self.random_state.normal(size=[3, 2, 4, 2]),
                           dtype=tf.float32)
        kernel = tf.constant(self.random_state.normal(size=[8, 18, 5]),
                             dtype=tf.float32)
        num_outputs = 5
        filter_size = [3, 3]

        true_result = [
            [[[-3.0002763, 0.5384941, 0.7630725, 3.2629867, -0.66444194],
              [3.8548074, 0.39627433, -1.8674281, -0.95996654, 0.53664565],
              [-4.909938, 1.2740642, -3.9075174, -1.3829101, -1.2598065],
              [-0.10984063, 0.8664815, 1.9967539, 1.5672208, -2.882845]],

             [[2.9014444, 1.9103334, -1.027496, 1.3560967, -2.6223845],
              [-4.33412, -1.3134818, 3.1853037, 3.4777935, -1.2933552],
              [0.21087825, 8.807638, 3.8791056, -2.6520872, -0.607337],
              [0.20364004, 1.6297944, -3.6308692, 5.099339, -0.74615014]]],


            [[[0.8476062, -0.31244925, 2.3357773, -3.1579304, 1.1329677],
              [-0.5461242, -7.558393, -3.5357292, 1.6523337, 0.07174261],
              [2.671926, 0.0200243, 8.013702, -2.6686308, 5.518297],
              [0.09115195, 1.8910196, 1.4207761, -1.258101, 5.327482]],

             [[-1.8102887, 1.4639647, -3.0855474, -2.871198, 4.173253],
              [1.1287231, 4.458473, -5.9783406, 0.94382215, 0.49254256],
              [-2.6642013, -3.3089342, 3.1150613, -8.594527, -0.5965001],
              [3.1609251, 0.6673362, 0.33882388, -0.89896905, 2.7638702]]],


            [[[1.7869619, 1.4026697, 0.5613424, -3.588409, 0.39702317],
              [2.9465265, 0.35991257, -6.4655128, 3.9436753, -1.9748771],
              [-0.15422773, 4.3497667, 1.1177152, -3.7317548, -2.9032776],
              [-1.547115, 4.481364, 2.3119464, -3.0446053, 4.720777]],

             [[-1.342461, -4.6650257, -2.8820145, -2.1707344, -0.3114946],
              [-1.8312358, 3.0615933, 3.7827466, 2.4145176, 0.16694853],
              [-1.5634956, -5.2658362, 0.12334719, -4.874626, 4.04391],
              [2.1574607, -0.08705652, -2.270496, -0.37243897, 3.500473]]]]

        conv_layer = conv.LocallyConnected2d(input_shape=data.get_shape(),
                                             num_outputs=num_outputs,
                                             filter_size=filter_size,
                                             kernel=kernel)
        result = conv_layer(data)
        self.assertTrue(np.allclose(true_result, result, atol=1e-6))

    def test_locally_connected_3d(self):
        """Test locally_connected_3d
        """
        data = tf.constant(self.random_state.normal(size=[1, 3, 3, 3, 2]),
                           dtype=tf.float32)
        kernel = tf.constant(self.random_state.normal(size=[27, 54, 2]),
                             dtype=tf.float32)
        num_outputs = 2
        filter_size = [3, 3, 3]

        true_result = [
            [[[[2.0361388, 2.8748353],
               [-4.8949614, 3.601464],
               [-0.6368098, -3.6394928]],

              [[-1.6584768, 2.1284351],
               [-2.4174252, -1.4464463],
               [-2.8733394, 1.4183671]],

              [[-0.24431975,  -5.925253],
               [-5.9550424, 6.0112863],
               [-0.5549349, -0.07634258]]],

             [[[-5.9965715, -0.4911141],
               [-1.3716308, -8.968927],
               [5.7334156, 4.173443]],

              [[-11.042645, -8.164853],
               [8.448492, -4.57167],
               [-1.2495683, 3.2970955]],

              [[2.1897097, 1.4297469],
               [1.5857544, 4.076191],
               [0.44746077,  -0.31982255]]],

             [[[4.443694, 3.1021364],
               [-4.425407, -0.5171719],
               [0.19204128,   1.3953105]],

              [[0.6423461, -4.424771],
               [-8.816259, -10.956709],
               [-2.0773346, 6.9159746]],

              [[2.138038, -5.0552483],
               [1.2522638, 8.907527],
               [4.3867416, 1.9319328]]]]]

        conv_layer = conv.LocallyConnected3d(input_shape=data.get_shape(),
                                             num_outputs=num_outputs,
                                             filter_size=filter_size,
                                             kernel=kernel)
        result = conv_layer(data)
        self.assertTrue(np.allclose(true_result, result, atol=1e-6))

    def test_conv3d_stacked(self):
        """Test locally_connected_3d
        """
        data = tf.constant(self.random_state.normal(size=[7, 5, 4, 7, 2]),
                           dtype=tf.float32)
        kernel = tf.constant(self.random_state.normal(size=[3, 3, 3, 2, 4]),
                             dtype=tf.float32)

        strides_list = [[1, 1, 1, 1, 1], [1, 2, 2, 2, 1]]
        padding_list = ['SAME']
        for strides, padding in product(strides_list, padding_list):

            result_tf = tf.nn.convolution(input=data, filters=kernel,
                                          strides=strides[1:-1],
                                          padding=padding)
            result = conv.conv3d_stacked(input=data, filter=kernel,
                                         strides=strides, padding=padding)

            self.assertTrue(np.allclose(result, result_tf, atol=1e-5))

    def test_conv4d_stacked(self):
        """Test locally_connected_3d
        """
        data = tf.constant(self.random_state.normal(size=[1, 2, 2, 2, 2, 1]),
                           dtype=tf.float32)
        kernel = tf.constant(self.random_state.normal(size=[2, 2, 2, 2, 1, 1]),
                             dtype=tf.float32)

        true_result = [[[[[-3.4794035],
                          [0.19396555]],
                         [[3.3645966],
                          [0.17348471]]],
                        [[[-0.9240602],
                          [1.229038]],
                         [[-0.48166633],
                          [-0.47118214]]]],
                       [[[[3.195785],
                          [-2.968795]],
                         [[-2.0781631],
                          [-0.35241044]]],
                        [[[1.5140774],
                          [2.4484003]],
                         [[1.5703532],
                          [0.5695023]]]]]

        result = conv.conv4d_stacked(input=data, filter=kernel)
        self.assertTrue(np.allclose(true_result, result, atol=1e-6))
