#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-26 下午1:41
# @Author  : XRDai
# @File    : add_upsample.py
'''
    
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf

def get_conv_filter(c_h, c_w, in_channel, out_channel, wd=5e-4):

    var = tf.get_variable(
        "filter", [c_h, c_w, in_channel, out_channel],
        initializer=tf.truncated_normal_initializer(stddev=0.1))
    if not tf.get_variable_scope().reuse:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd,
                                   name='weight_loss')
        tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
                             weight_decay)

    return var

def get_bias(out_channel):

    bias_wights = tf.get_variable("bias", [out_channel], initializer=tf.constant_initializer(0.0))

    return bias_wights

## 
def _deconv_layer(bottom, shape, name, c_h=2, c_w=2, s_h=2, s_w=2, Relu=False, add_bias=True, padding='SAME'):
    with tf.variable_scope(name) as scope:

        _, _, _, in_channel = bottom.get_shape().as_list()

        out_channel = shape[-1]

        ##
        filt = get_conv_filter(c_h, c_w, in_channel, out_channel)

        dconv = tf.nn.conv2d_transpose(bottom, filt, shape,
                               strides=[1, s_h, s_w, 1], padding=padding)

        out_put = dconv
        ##
        if add_bias:
            conv_biases = get_bias(out_channel)
            out_put = tf.nn.bias_add(out_put, conv_biases)

        ##
        if Relu:
            out_put = tf.nn.relu(out_put)

    return out_put


def upSample(bottom, shape, name, ksize=2, stride=2, activation_fn=None):
    ## ron
    ## upsampling = slim.conv2d_transpose(end_points[from_layer], 256, [2, 2], stride=2, activation_fn=None, \
    ##                                normalizer_fn=None, scope=out_layer)

    return _deconv_layer(bottom, shape, name, ksize, ksize, stride, stride, Relu=activation_fn)
