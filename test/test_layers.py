#!/usr/local/bin/python
from __future__ import division, print_function

import unittest
import numpy as np
import tensorflow as tf
from itertools import product

from tfscripts import layers


class TestConvModule(unittest.TestCase):

    def setUp(self):
        self.random_state = np.random.RandomState(42)

    def test_new_conv_nd_layers(self):
        """Test new_conv_nd_layers
        """
        data = tf.constant(self.random_state.normal(size=[3, 4, 4, 4, 2]),
                           dtype=tf.float32)

        shapes = [[3, 3, 3, 2, 3], [3, 3, 3, 3, 7], [3, 3, 3, 7, 1]]
        weights_list = [tf.constant(self.random_state.normal(size=shape),
                                    dtype=tf.float32) for shape in shapes]

        shapes = [[3], [7], [1]]
        biases_list = [tf.constant(self.random_state.normal(size=shape),
                                   dtype=tf.float32) for shape in shapes]

        pooling_strides_list = [[1, 1, 1, 1, 1],
                                [1, 2, 2, 2, 1],
                                [1, 2, 2, 2, 1]]
        pooling_ksize_list = [[1, 1, 1, 1, 1],
                              [1, 2, 2, 2, 1],
                              [1, 2, 2, 2, 1]]
        pooling_type_list = [None, 'max', 'max']
        activation_list = ['elu', 'relu', '']
        filter_size_list = [[3, 3, 3], [2, 0, 3], [3, 3, 3]]
        num_filters_list = [3, 7, 1]
        method_list = ['convolution', 'hex_convolution', 'convolution']

        layer, weights, biases = layers.new_conv_nd_layers(
                                data,
                                filter_size_list=filter_size_list,
                                num_filters_list=num_filters_list,
                                pooling_type_list=pooling_type_list,
                                pooling_strides_list=pooling_strides_list,
                                pooling_ksize_list=pooling_ksize_list,
                                activation_list=activation_list,
                                method_list=method_list,
                                weights_list=weights_list,
                                biases_list=biases_list,
                                verbose=False,
                                )

        result_true = [[[[[1.5821583]]]],
                       [[[[1.8506659]]]],
                       [[[[1.6045672]]]]]

        self.assertTrue(np.allclose(result_true, layer[-1].numpy()))

    def test_new_fc_layers(self):
        """Test new_fc_layers
        """
        data = tf.constant(self.random_state.normal(size=[3, 7]),
                           dtype=tf.float32)

        shapes = [[7, 3], [3, 7], [7, 1]]
        weights_list = [tf.constant(self.random_state.normal(size=shape),
                                    dtype=tf.float32) for shape in shapes]

        shapes = [[3], [7], [1]]
        biases_list = [tf.constant(self.random_state.normal(size=shape),
                                   dtype=tf.float32) for shape in shapes]

        activation_list = ['elu', 'relu', '']
        fc_sizes = [3, 7, 1]

        layer, weights, biases = layers.new_fc_layers(
                                data,
                                fc_sizes=fc_sizes,
                                activation_list=activation_list,
                                weights_list=weights_list,
                                biases_list=biases_list,
                                verbose=False,
                                )
        result_true = [[0.06657974],
                       [-0.3617597],
                       [0.00241985]]

        self.assertTrue(np.allclose(result_true, layer[-1].numpy()))
