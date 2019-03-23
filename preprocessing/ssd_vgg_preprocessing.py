#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-14 下午2:48
# @Author  : XRDai
# @File    : ssd_vgg_preprocessing.py
'''

'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import numpy as np

import tensorflow as tf
import tf_extended as tfe

from tensorflow.python.ops import control_flow_ops

from preprocessing import tf_image
# from nets import ssd_common

slim = tf.contrib.slim

# VGG mean parameters.
_R_MEAN = 123.
_G_MEAN = 117.
_B_MEAN = 104.

# Some training pre-processing parameters.
BBOX_CROP_OVERLAP = 0.5         # Minimum overlap to keep a bbox after cropping.
MIN_OBJECT_COVERED = 0.25
CROP_RATIO_RANGE = (0.6, 1.67)  # Distortion ratio during cropping.
EVAL_SIZE = (320, 320)


def tf_image_whitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN]):
    """Subtracts the given means from each image channel.

    Returns:
        the centered image.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    mean = tf.constant(means, dtype=image.dtype)
    image = image - mean
    return image


def tf_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary.

    Returns:
      Centered image.
    """
    mean = tf.constant(means, dtype=image.dtype)
    image = image + mean
    if to_int:
        image = tf.cast(image, tf.int32)
    return image


def np_image_unwhitened(image, means=[_R_MEAN, _G_MEAN, _B_MEAN], to_int=True):
    """Re-convert to original image distribution, and convert to int if
    necessary. Numpy version.

    Returns:
      Centered image.
    """
    img = np.copy(image)
    img += np.array(means, dtype=img.dtype)
    if to_int:
        img = img.astype(np.uint8)
    return img

def kitti_preprocess_for_train(image, labels, bboxes, in_shape,
                         out_shape, data_format='NHWC',
                         scope='preprocessing_train'):

    return

def preprocess_for_eval(image, labels, bboxes, in_shape,
                        out_shape=EVAL_SIZE, data_format='NHWC',
                        difficults=None, resize='WARP_RESIZE',
                        scope='ssd_preprocessing_eval'):
    """Preprocess an image for evaluation.

    Args:
        image: A `Tensor` representing an image of arbitrary size.
        out_shape: Output shape after pre-processing (if resize != None)
        resize: Resize strategy.

    Returns:
        A preprocessed image.
    """
    with tf.name_scope(scope):
        if image.get_shape().ndims != 3:
            raise ValueError('Input must be of size [height, width, C>0]')

        image = tf.to_float(image)
        image = tf_image_whitened(image, [_R_MEAN, _G_MEAN, _B_MEAN])

        # Add image rectangle to bboxes.
        bbox_img = tf.constant([[0., 0., 1., 1.]])
        if bboxes is None:
            bboxes = bbox_img
        else:
            bboxes = tf.concat([bbox_img, bboxes], axis=0)

        if resize == 'NONE':
            # No resizing...
            pass
        elif resize == 'CENTRAL_CROP':
            # Central cropping of the image.
            image, bboxes = tf_image.resize_image_bboxes_with_crop_or_pad(
                image, bboxes, out_shape[0], out_shape[1])
        elif resize == 'PAD_AND_RESIZE':
            # Resize image first: find the correct factor...
            shape = tf.shape(image)
            factor = tf.minimum(tf.to_double(1.0),
                                tf.minimum(tf.to_double(out_shape[0] / shape[0]),
                                           tf.to_double(out_shape[1] / shape[1])))
            resize_shape = factor * tf.to_double(shape[0:2])
            resize_shape = tf.cast(tf.floor(resize_shape), tf.int32)

            image = tf_image.resize_image(image, resize_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)
            # Pad to expected size.
            image, bboxes = tf_image.resize_image_bboxes_with_crop_or_pad(
                image, bboxes, out_shape[0], out_shape[1])
        elif resize == 'WARP_RESIZE':
            # Warp resize of the image.
            image = tf_image.resize_image(image, out_shape,
                                          method=tf.image.ResizeMethod.BILINEAR,
                                          align_corners=False)

        # Split back bounding boxes.
        bbox_img = bboxes[0]
        bboxes = bboxes[1:]
        # Remove difficult boxes.
        if difficults is not None:
            mask = tf.logical_not(tf.cast(difficults, tf.bool))
            labels = tf.boolean_mask(labels, mask)
            bboxes = tf.boolean_mask(bboxes, mask)
        # Image data format.
        if data_format == 'NCHW':
            image = tf.transpose(image, perm=(2, 0, 1))
        return image, labels, bboxes, bbox_img


def preprocess_image(image,
                     labels,
                     bboxes,
                     in_shape,
                     out_shape,
                     data_format,
                     is_training=False,
                     **kwargs):
    if is_training:
        return kitti_preprocess_for_train(image, labels, bboxes, in_shape,
                                    out_shape=out_shape,
                                    data_format=data_format)
    else:
        return preprocess_for_eval(image, labels, bboxes,in_shape,
                                   out_shape=out_shape,
                                   data_format=data_format,
                                   **kwargs)
