#!/usr/bin/env python
# -*-coding:utf-8-*-
# @Author : Weiqun Wu
# @Time : 2018-11-23

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import slim


def multi_column_cnn(inputs, scope='mcnn'):
    with tf.variable_scope(scope):
        with tf.variable_scope('large'):
            large_column = slim.conv2d(inputs, 16, [9, 9], padding='SAME', scope='conv1')
            large_column = slim.conv2d(large_column, 32, [7, 7], padding='SAME', scope='conv2')
            large_column = slim.max_pool2d(large_column, [2, 2], 2, scope='pool1')
            large_column = slim.conv2d(large_column, 16, [7, 7], padding='SAME', scope='conv3')
            large_column = slim.max_pool2d(large_column, [2, 2], 2, scope='pool2')
            large_column = slim.conv2d(large_column, 8, [7, 7], padding='SAME', scope='conv4')

        with tf.variable_scope('medium'):
            medium_column = slim.conv2d(inputs, 20, [7, 7], padding='SAME', scope='conv1')
            medium_column = slim.conv2d(medium_column, 40, [5, 5], padding='SAME', scope='conv2')
            medium_column = slim.max_pool2d(medium_column, [2, 2], 2, scope='pool1')
            medium_column = slim.conv2d(medium_column, 20, [5, 5], padding='SAME', scope='conv3')
            medium_column = slim.max_pool2d(medium_column, [2, 2], 2, scope='pool2')
            medium_column = slim.conv2d(medium_column, 10, [5, 5], padding='SAME', scope='conv4')

        with tf.variable_scope('small'):
            small_column = slim.conv2d(inputs, 24, [5, 5], padding='SAME', scope='conv1')
            small_column = slim.conv2d(small_column, 48, [3, 3], padding='SAME', scope='conv2')
            small_column = slim.max_pool2d(small_column, [2, 2], 2, scope='pool1')
            small_column = slim.conv2d(small_column, 24, [3, 3], padding='SAME', scope='conv3')
            small_column = slim.max_pool2d(small_column, [2, 2], 2, scope='pool2')
            small_column = slim.conv2d(small_column, 12, [3, 3], padding='SAME', scope='conv4')

        net = tf.concat([large_column, medium_column, small_column], axis=3)
        dmp = slim.conv2d(net, 1, [1, 1], padding='SAME', scope='dmp_conv1')

    return dmp


if __name__ == '__main__':
    print('run m_model')
