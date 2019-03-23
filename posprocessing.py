#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-8-17 上午10:08
# @Author  : XRDai
# @File    : posprocessing.py
'''
    
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tf_extended as tfe
import tensorflow as tf

def flaten_predict(predictions, localisations):
    predictions_shape = tfe.get_shape(predictions[0], 5)
    batch_size = predictions_shape[0]
    num_classes = predictions_shape[-1]

    if batch_size > 1:
        raise ValueError('only batch_size 1 is supported.')

    flaten_pred = []
    flaten_labels = []
    flaten_locations = []
    flaten_scores = []

    for i in range(len(predictions)):
        flaten_pred.append(tf.reshape(predictions[i], [batch_size, -1, num_classes]))
        cls_pred = flaten_pred[i]
        flaten_scores.append(tf.reshape(cls_pred, [batch_size, -1, num_classes]))
        #flaten_scores.append(tf.reshape(tf.reduce_max(cls_pred, -1), [batch_size, -1]))
        # flaten_labels.append(tf.reshape(tf.argmax(cls_pred, -1), [batch_size, -1]))

        ## 除了负样本的 最大概率
        ## 0 是 background， 已经去掉， 故 +1 表示真实的
        flaten_labels.append(tf.reshape(tf.argmax(cls_pred[:, :, 1:], -1) + 1, [batch_size, -1]))

        flaten_locations.append(tf.reshape(localisations[i], [batch_size, -1, 4]))

    total_scores = tf.squeeze(tf.concat(flaten_scores, 1), 0)
    total_locations = tf.squeeze(tf.concat(flaten_locations, 1), 0)
    total_labels = tf.squeeze(tf.concat(flaten_labels, 1), 0)
    # remove bboxes that are not foreground
    non_background_mask = tf.greater(total_labels, 0)

    bbox_mask = non_background_mask
    # return tf.boolean_mask(total_scores, bbox_mask), tf.boolean_mask(total_labels, bbox_mask), tf.boolean_mask(total_locations, bbox_mask)
    return total_scores, total_labels, total_locations

def flaten_predict1(predictions, localisations):
    predictions_shape = tfe.get_shape(predictions[0], 5)
    batch_size = predictions_shape[0]
    num_classes = predictions_shape[-1]

    if batch_size > 1:
        raise ValueError('only batch_size 1 is supported.')

    flaten_pred = []
    flaten_labels = []
    flaten_locations = []
    flaten_scores = []

    for i in range(len(predictions)):
        flaten_pred.append(tf.reshape(predictions[i], [batch_size, -1, num_classes]))
        cls_pred = flaten_pred[i]
        flaten_scores.append(tf.reshape(cls_pred, [batch_size, -1, num_classes]))

        ##
        ##
        flaten_labels.append(tf.reshape(tf.argmax(cls_pred[:, :, 1:], -1) + 1, [batch_size, -1]))

    total_scores = tf.squeeze(tf.concat(flaten_scores, 1), 0)
    total_locations = tf.squeeze(localisations, 0)
    total_labels = tf.squeeze(tf.concat(flaten_labels, 1), 0)
    # remove bboxes that are not foreground
    non_background_mask = tf.greater(total_labels, 0)

    bbox_mask = non_background_mask
    # return tf.boolean_mask(total_scores, bbox_mask), tf.boolean_mask(total_labels, bbox_mask), tf.boolean_mask(total_locations, bbox_mask)
    return total_scores, total_labels, total_locations

def tf_bboxes_nms(scores, labels, bboxes, nms_threshold = 0.5, keep_top_k = 20, mode = 'union', scope=None):
    with tf.name_scope(scope, 'tf_bboxes_nms', [scores, labels, bboxes]):
        # get the cls_score for the most-likely class
        scores = tf.reduce_max(scores, -1)
        # apply threshold
        bbox_mask = tf.greater(scores, 0.5)
        scores, labels, bboxes = tf.boolean_mask(scores, bbox_mask), tf.boolean_mask(labels, bbox_mask), tf.boolean_mask(bboxes, bbox_mask)
        num_anchors = tf.shape(scores)[0]
        def nms_proc(scores, labels, bboxes):
            # sort all the bboxes
            scores, idxes = tf.nn.top_k(scores, k=num_anchors, sorted=True)
            labels, bboxes = tf.gather(labels, idxes), tf.gather(bboxes, idxes)

            ymin = bboxes[:, 0]
            xmin = bboxes[:, 1]
            ymax = bboxes[:, 2]
            xmax = bboxes[:, 3]

            vol_anchors = (xmax - xmin) * (ymax - ymin)

            nms_mask = tf.cast(tf.ones_like(scores, dtype=tf.int8), tf.bool)
            keep_mask = tf.cast(tf.zeros_like(scores, dtype=tf.int8), tf.bool)

            def safe_divide(numerator, denominator):
                return tf.where(tf.greater(denominator, 0), tf.divide(numerator, denominator), tf.zeros_like(denominator))

            def get_scores(bbox, nms_mask):
                # the inner square
                inner_ymin = tf.maximum(ymin, bbox[0])
                inner_xmin = tf.maximum(xmin, bbox[1])
                inner_ymax = tf.minimum(ymax, bbox[2])
                inner_xmax = tf.minimum(xmax, bbox[3])
                h = tf.maximum(inner_ymax - inner_ymin, 0.)
                w = tf.maximum(inner_xmax - inner_xmin, 0.)
                inner_vol = h * w
                this_vol = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if mode == 'union':
                    union_vol = vol_anchors - inner_vol  + this_vol
                elif mode == 'min':
                    union_vol = tf.minimum(vol_anchors, this_vol)
                else:
                    raise ValueError('unknown mode to use for nms.')
                return safe_divide(inner_vol, union_vol) * tf.cast(nms_mask, tf.float32)

            def condition(index, nms_mask, keep_mask):
                return tf.logical_and(tf.reduce_sum(tf.cast(nms_mask, tf.int32)) > 0, tf.less(index, keep_top_k))

            def body(index, nms_mask, keep_mask):
                # at least one True in nms_mask
                indices = tf.where(nms_mask)[0][0]
                bbox = bboxes[indices]
                this_mask = tf.one_hot(indices, num_anchors, on_value=False, off_value=True, dtype=tf.bool)
                keep_mask = tf.logical_or(keep_mask, tf.logical_not(this_mask))
                nms_mask = tf.logical_and(nms_mask, this_mask)

                nms_scores = get_scores(bbox, nms_mask)

                nms_mask = tf.logical_and(nms_mask, nms_scores < nms_threshold)
                return [index+1, nms_mask, keep_mask]

            index = 0
            [index, nms_mask, keep_mask] = tf.while_loop(condition, body, [index, nms_mask, keep_mask])
            return tf.boolean_mask(scores, keep_mask), tf.boolean_mask(labels, keep_mask), tf.boolean_mask(bboxes, keep_mask)

        return tf.cond(tf.less(num_anchors, 1), lambda: (scores, labels, bboxes), lambda: nms_proc(scores, labels, bboxes))

def filter_boxes(scores, labels, bboxes, select_threshold=0.5):
    """Only keep boxes with both sides >= min_size and center within the image.
    min_size_ratio is the ratio relative to net input shape
    """
    # Scale min_size to match image scale
    min_size = 0.0001

    ymin = bboxes[:, 0]
    xmin = bboxes[:, 1]
    ymax = bboxes[:, 2]
    xmax = bboxes[:, 3]

    ws = xmax - xmin
    hs = ymax - ymin
    x_ctr = xmin + ws / 2.
    y_ctr = ymin + hs / 2.

    keep_mask = tf.logical_and(tf.greater(ws, min_size), tf.greater(hs, min_size))
    keep_mask = tf.logical_and(keep_mask, tf.greater(x_ctr, 0.))
    keep_mask = tf.logical_and(keep_mask, tf.greater(y_ctr, 0.))
    keep_mask = tf.logical_and(keep_mask, tf.less(x_ctr, 1.))
    keep_mask = tf.logical_and(keep_mask, tf.less(y_ctr, 1.))
    # keep_mask = tf.logical_and(keep_mask, tf.greater(scores, select_threshold))

    return tf.boolean_mask(scores, keep_mask), tf.boolean_mask(labels, keep_mask), tf.boolean_mask(bboxes, keep_mask)
