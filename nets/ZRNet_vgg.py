#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-14 下午2:48
# @Author  : XRDai
# @File    : _train_1.py
'''

'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from collections import namedtuple
import numpy as np
import tensorflow as tf

from nets import custom_layers
from nets import ssd_common
from nets import add_upsample

slim = tf.contrib.slim

ZRNetParams = namedtuple('ZRNetParameters', ['img_shape',
                                         'num_classes',
                                         'no_annotation_label',
                                         'feat_layers',
                                         'fpn_layers',
                                         'rfpn_layers',
                                         'feat_shapes',
                                         'allowed_borders',
                                         'anchor_sizes',
                                         'anchor_ratios',
                                         'anchor_steps',
                                         'normalizations',
                                         'anchor_offset',
                                         'prior_scaling'
                                         ])


class ZRNet(object):
    default_params = ZRNetParams(
        img_shape=(384, 1248),
        num_classes=4,
        no_annotation_label=0,
        ## feat_layers corresponding to C6, C5, C4, C3 of the backbone network
        ## fpn_layers corresponding to  P6, P5, P4, P3 of the top-down module
        ## rfpn_layers corresponding to Q6, Q5, Q4, Q3 of the down-top module
        feat_layers=['block7', 'block6', 'block5', 'block4'],
        fpn_layers=['P6', 'P5', 'P4', 'P3'],
        rfpn_layers=['rP6', 'rP5', 'rP4', 'rP3'],
        feat_shapes=[(6, 20), (12, 39), (24, 78), (48, 156)],
        allowed_borders=[32, 16, 8, 4],
        anchor_sizes=[(224., 256.),
                      (160., 192.),
                      (96., 128.),
                      (32., 64.)],

        anchor_ratios=[[1, 2, 3, 1./2, 1./3],
                       [1, 2, 3, 1./2, 1./3],
                       [1, 2, 3, 1./2, 1./3],
                       [1, 2, 3, 1./2, 1./3]],
        anchor_steps=[64, 32, 16, 8],
        normalizations=[-1, -1, 8, 10],
        anchor_offset=0.5,
        prior_scaling=[0.1, 0.1, 0.2, 0.2]
        )

    def __init__(self, params=None):
        if isinstance(params, ZRNetParams):
            self.params = params
        else:
            self.params = ZRNet.default_params

    # ======================================================================= #
    def net(self, inputs,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='zr_net_vgg'):

        r = ron_net_reducedfc(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    fpn_layers =self.params.fpn_layers,
                    rfpn_layers=self.params.rfpn_layers,
                    anchor_sizes=self.params.anchor_sizes,
                    is_training=is_training,
                    dropout_keep_prob=dropout_keep_prob,
                    prediction_fn=prediction_fn,
                    normalizations=self.params.normalizations,
                    reuse=reuse,
                    scope=scope)
        return r

    def arg_scope(self, weight_decay=0.0005, is_training=True, data_format='NHWC'):
        """Network arg_scope.
        """
        return zr_arg_scope(weight_decay, is_training=is_training, data_format=data_format)

    # ======================================================================= #
    def anchors(self, img_shape, dtype=np.float32):
        """Compute the default anchor boxes, given an image shape.
        """
        return zrnet_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)


    def bboxes_decode(self, feat_localizations, anchors,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode(
            feat_localizations, anchors,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode_fpn(self, feat_localizations, anchors, batch_size,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode_fpn(
            feat_localizations, anchors, batch_size,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode_fpn1(self, feat_localizations, flatten_anchor_yxhw, anchors, batch_size,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode_fpn1(
            feat_localizations, flatten_anchor_yxhw,
            anchors, batch_size,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode_rfpn(self, feat_localizations, flatten_anchor_yxhw, flaten_yxhw, batch_size,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode_rfpn(
            feat_localizations, flatten_anchor_yxhw, flaten_yxhw, batch_size,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

    def bboxes_decode_rfpn1(self, feat_localizations, flatten_anchor_yxhw, flaten_yxhw, batch_size,
                      scope='ssd_bboxes_decode'):
        """Encode labels and bounding boxes.
        """
        return ssd_common.tf_ssd_bboxes_decode_rfpn1(
            feat_localizations, flatten_anchor_yxhw, flaten_yxhw, batch_size,
            prior_scaling=self.params.prior_scaling,
            scope=scope)

def zrnet_anchor_one_layer(img_shape,
                         feat_shape,
                         sizes,
                         step,
                         offset=0.5,
                         dtype=np.float32):

    y, x = np.mgrid[0:feat_shape[0], 0:feat_shape[1]]
    y = ((y.astype(dtype) + offset) * step) / img_shape[0]
    x = ((x.astype(dtype) + offset) * step) / img_shape[1]

    # Expand dims to support easy broadcasting.
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)

    num_anchors = len(sizes)
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)
    w[:] = sizes[:, 0] / img_shape[1]
    h[:] = sizes[:, 1] / img_shape[0]

    return y, x, h, w


def zrnet_anchors_all_layers(img_shape,
                           layers_shape,
                           anchor_sizes,
                           anchor_steps,
                           offset=0.5,
                           dtype=np.float32):
    """Compute anchor boxes for all feature layers.
    """
    layers_anchors = []
    shape_recorder = []
    full_shape_anchors = {}
    for i, s in enumerate(layers_shape):
        anchor_bboxes = zrnet_anchor_one_layer(img_shape, s,
                                             anchor_sizes[(3 - i)*4: (4-i) * 4],
                                             anchor_steps[i],
                                             offset=offset, dtype=dtype)
        layers_anchors.append(anchor_bboxes)

        ## flatten_anchor
        yref, xref, href, wref = anchor_bboxes
        ymin_ = yref - href / 2.
        xmin_ = xref - wref / 2.
        ymax_ = yref + href / 2.
        xmax_ = xref + wref / 2.

        shape_recorder.append(ymin_.shape)
        full_shape_yxhw = [(ymin_ + ymax_) / 2, (xmin_ + xmax_) / 2, (ymax_ - ymin_), (xmax_ - xmin_)]

        full_shape_anchors[i] = [np.reshape(_, (-1)) for _ in full_shape_yxhw]

    remap_anchors = list(zip(*full_shape_anchors.values()))
    for i in range(len(full_shape_anchors)):
        full_shape_anchors[i] = np.concatenate(remap_anchors[i], axis=0)

    flatten_anchor_yxhw = list(full_shape_anchors.values())

    return layers_anchors, flatten_anchor_yxhw, shape_recorder

def tensor_shape(x, rank=3):
    """Returns the dimensions of a tensor.
    Args:
      image: A N-D Tensor of shape.
    Returns:
      A list of dimensions. Dimensions that are statically known are python
        integers,otherwise they are integer scalar tensors.
    """
    if x.get_shape().is_fully_defined():
        return x.get_shape().as_list()
    else:
        static_shape = x.get_shape().with_rank(rank).as_list()
        dynamic_shape = tf.unstack(tf.shape(x), rank)
        return [s if s is not None else d
                for s, d in zip(static_shape, dynamic_shape)]

def kitti_ssd_multibox_layer(inputs,
                       num_classes,
                       sizes):
    """Construct a multibox layer, return a class and localization predictions.
    """
    net = inputs

    # Number of anchors.
    num_anchors = len(sizes)

    # Location.
    num_loc_pred = num_anchors * 4
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None,
                           scope='conv_loc')
    loc_pred = custom_layers.channel_to_last(loc_pred)
    loc_pred = tf.reshape(loc_pred,
                          tensor_shape(loc_pred, 4)[:-1]+[num_anchors, 4])
    # Class prediction.
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None,
                           scope='conv_cls')
    cls_pred = custom_layers.channel_to_last(cls_pred)
    cls_pred = tf.reshape(cls_pred,
                          tensor_shape(cls_pred, 4)[:-1]+[num_anchors, num_classes])
    return cls_pred, loc_pred

def zr_net(inputs,
            num_classes=ZRNet.default_params.num_classes,
            feat_layers=ZRNet.default_params.feat_layers,
            anchor_sizes=ZRNet.default_params.anchor_sizes,
            anchor_ratios=ZRNet.default_params.anchor_ratios,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='zr_net_vgg'):

    end_points = {}
    # Prediction and localisations layers.
    predictions = []
    logits = []
    localisations = []
    objness_pred = []
    objness_logits = []

    return predictions, logits, objness_pred, objness_logits, localisations, end_points


## Add extra layers on top of a "base" network (e.g. VGGNet or ResNet).
def AddExtraLayers(end_points, feat_layers, fpn_layers, normalizations):

    num_p = 6
    for index, layer in enumerate(feat_layers):

        net = end_points[layer]

        ##
        if normalizations[index] != -1:
            norm_name = "{}_norm".format(layer)
            net = custom_layers.l2_normalization(net, scaling=True, normalization=normalizations[index], scope=norm_name)
            end_points[layer] = net

        out_layer = "TL{}_{}".format(num_p, 1)
        net = slim.conv2d(net, 256, [3, 3], scope=out_layer)

        if num_p == 6:
            out_layer = "TL{}_{}".format(num_p, 2)
            net = slim.conv2d(net, 256, [3, 3], scope=out_layer)
        else:
            out_layer = "TL{}_{}".format(num_p, 2)
            net = slim.conv2d(net, 256, [3, 3], activation_fn=None, scope=out_layer)

            ## P_up
            from_layer = "P{}".format(num_p+1)
            out_layer = "P{}-up".format(num_p+1)

            upsampling = add_upsample.upSample(end_points[from_layer], net.get_shape().as_list(), \
                                               name=out_layer, ksize=2, stride=2, activation_fn=None)

            ## Elt_relu
            out_layer = "Elt{}_relu".format(num_p)
            net = tf.nn.relu(net + upsampling, name=out_layer)

        ## fpn
        out_layer = "P{}".format(num_p)
        net = slim.conv2d(net, 256, [3, 3], scope=out_layer)
        end_points[out_layer] = net

        num_p = num_p - 1

    ##
    num_p = 3
    rfpn_layers = fpn_layers[::-1]
    for index, layer in enumerate(rfpn_layers):

        net = end_points[layer]

        ##
        out_layer = "PC{}".format(num_p)
        net = slim.conv2d(net, 256, [3, 3], scope=out_layer)

        if num_p > 3:
            ##
            from_layer = "rP{}".format(num_p-1)
            out_layer = "rP{}-down".format(num_p-1)

            downsampling = slim.conv2d(end_points[from_layer], 256, [3, 3], stride=2, scope=out_layer)

            ## Elt
            net = net + downsampling

        ##
        out_layer = "rP{}".format(num_p)
        net = slim.conv2d(net, 256, [3, 3], scope=out_layer)
        end_points[out_layer] = net

        num_p = num_p + 1

    return end_points

## block1 - block7
def VGGNetBody(inputs, end_points):

    # Original VGG-16 blocks.
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    end_points['block1'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    # Block 2.
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    end_points['block2'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    # Block 3.
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    end_points['block3'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    # Block 4.
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    end_points['block4'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    # Block 5.
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    end_points['block5'] = net

    # different betweent SSD here
    net = slim.max_pool2d(net, [2, 2], scope='pool5')

    # Block 6
    # Use conv2d instead of fully_connected layers.
    net = slim.conv2d(net, 1024, [3, 3], stride=1, rate=3, padding='SAME', scope='fc6')
    net = slim.conv2d(net, 1024, [1, 1], stride=1, rate=1, padding='SAME', scope='fc7')
    end_points['block6'] = net

    ##
    net = slim.conv2d(net, 256, [1, 1], stride=1, scope='conv6_1')
    net = slim.conv2d(net, 512, [2, 2], stride=2, padding='SAME', scope='conv6_2')

    end_points['block7'] = net

    return end_points

def prediction(end_points, feat_layers, num_classes, anchor_sizes, prediction_fn):

    predictions = []
    logits = []
    localisations = []
    for i, layer in enumerate(feat_layers):
        with tf.variable_scope(layer + '_box'):
            p, l = kitti_ssd_multibox_layer(end_points[layer],
                                                        num_classes,
                                                        anchor_sizes[(3 - i) * 4: (4 - i) * 4])
        predictions.append(prediction_fn(p))
        logits.append(p)
        localisations.append(l)

    return predictions, logits, localisations


def ron_net_reducedfc(inputs,
            num_classes=ZRNet.default_params.num_classes,
            feat_layers=ZRNet.default_params.feat_layers,
            fpn_layers =ZRNet.default_params.fpn_layers,
            rfpn_layers =ZRNet.default_params.rfpn_layers,
            anchor_sizes=ZRNet.default_params.anchor_sizes,
            is_training=True,
            dropout_keep_prob=0.5,
            prediction_fn=slim.softmax,
            normalizations=ZRNet.default_params.normalizations,
            reuse=None,
            scope='zr_net_vgg'):
    """RON net definition.
    """
    # if data_format == 'NCHW':
    #     inputs = tf.transpose(inputs, perm=(0, 3, 1, 2))

    end_points = {}
    with tf.variable_scope(scope, 'zr_net_vgg', [inputs], reuse=reuse):

        ##  block1 - block7
        end_points = VGGNetBody(inputs, end_points)

        ##
        end_points = AddExtraLayers(end_points, feat_layers, fpn_layers, normalizations)

        #  block
        # with tf.variable_scope('block'):
        predictions, logits, localisations = prediction(end_points, \
                        feat_layers, num_classes, anchor_sizes, prediction_fn)

        #  fpn
        with tf.variable_scope('fpn'):
            predictions_fpn, logits_fpn, localisations_fpn = prediction(end_points, \
                            fpn_layers, num_classes, anchor_sizes, prediction_fn)

        # rfpn
        with tf.variable_scope('rfpn'):
            predictions_rfpn, logits_rfpn, localisations_rfpn = prediction(end_points, \
                                                                        rfpn_layers, num_classes, anchor_sizes,
                                                                        prediction_fn)

    return predictions, logits, localisations, predictions_fpn, logits_fpn, \
           localisations_fpn, predictions_rfpn, logits_rfpn, localisations_rfpn, end_points


def truncated_normal_001_initializer():
  # pylint: disable=unused-argument
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    """Initializer function."""
    #print(shape)
    if not dtype.is_floating:
      raise TypeError('Cannot create initializer for non-floating point type.')
    return tf.truncated_normal(shape, 0.0, 0.01, dtype, seed=None)
  return _initializer

def zr_arg_scope(weight_decay=0.0005, is_training=True, data_format='NHWC'):
    """Defines the VGG arg scope.

    Args:
      weight_decay: The l2 regularization coefficient.

    Returns:
      An arg_scope.
    """
    with slim.arg_scope([slim.conv2d_transpose, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
      with slim.arg_scope([slim.conv2d], activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),#truncated_normal_001_initializer(),
                        biases_initializer=tf.zeros_initializer()):
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.max_pool2d],
                            padding='SAME',
                            data_format=data_format):
            with slim.arg_scope([slim.batch_norm],
                            # default no activation_fn for BN
                            activation_fn=None,
                            decay=0.997,
                            epsilon=1e-5,
                            scale=True,
                            fused=True,
                            is_training = is_training,
                            data_format=data_format):
                with slim.arg_scope([custom_layers.pad2d,
                                     custom_layers.l2_normalization,
                                     custom_layers.channel_to_last],
                                    data_format=data_format) as sc:
                    return sc