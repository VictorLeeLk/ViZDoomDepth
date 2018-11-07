"""
@author: Shiyu Huang 
@contact: huangsy13@gmail.com
@file: model.py
"""

from dataflow import image_width, image_height
from utils import VGG_MEAN
import tensorflow as tf
import numpy as np

class DepthTest:
    def __init__(self, model_path=None):
        self.data_dict = np.load(model_path, encoding='latin1').item()

    def build(self, rgbs):
        self.img = rgbs
        red, green, blue = tf.split(rgbs, 3, 3)
        bgr = tf.concat([
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ], 3)
        self.conv1_1 = self.conv_layer(bgr, 'conv1_1')
        self.pool1 = self.max_pool(self.conv1_1, 'pool1')
        self.conv2_1 = self.conv_layer(self.pool1, 'conv2_1')
        self.pool2 = self.max_pool(self.conv2_1, 'pool2')
        self.conv3_1 = self.conv_layer(self.pool2, 'conv3_1')
        self.pool3 = self.max_pool(self.conv3_1, 'pool3')
        self.fc1 = self.fc(self.pool3, 'fc1', in_size=4608, use_relu=True)
        self.fc2 = self.fc(self.fc1, 'fc2', in_size=128)
        self.predict = self.fc2
       
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name, use_relu=True):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)
            if use_relu:
                relu = tf.nn.relu(bias)
                return relu
            else:
                return bias

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name='filter')

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name='biases')

    def fc(self, bottom, name, in_size, use_relu=False):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            if use_relu:
                fc = tf.nn.relu(fc)

            return fc

    def get_fc_var(self, name):
        weights = tf.constant(self.data_dict[name][0], name=name + "_weights")
        biases = tf.constant(self.data_dict[name][1], name=name + "_biases")
        return weights, biases
