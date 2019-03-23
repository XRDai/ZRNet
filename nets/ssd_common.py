# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

def tf_ssd_bboxes_decode_layer(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):

    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2])
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3])
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes

def tf_ssd_bboxes_decode_layer_fpn(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):

    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    ##
    cx = feat_localizations[:, :, :, :, 0] * wref * prior_scaling[0] * 0.5 + xref
    cy = feat_localizations[:, :, :, :, 1] * href * prior_scaling[1] * 0.5 + yref
    w = wref * tf.exp(feat_localizations[:, :, :, :, 2] * prior_scaling[2]) * 0.5
    h = href * tf.exp(feat_localizations[:, :, :, :, 3] * prior_scaling[3]) * 0.5
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes, cy, cx, h, w

def tf_ssd_bboxes_decode_layer_fpn1(feat_localizations,
                               anchors_layer,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2]):

    yref, xref, href, wref = anchors_layer

    # Compute center, height and width
    cx = feat_localizations[:, :, 0] * wref * prior_scaling[0] + xref
    cy = feat_localizations[:, :, 1] * href * prior_scaling[1] + yref
    w = feat_localizations[:, :, 2] * wref * prior_scaling[2] + wref
    h = feat_localizations[:, :, 3] * href * prior_scaling[3] + href
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes

def tf_ssd_bboxes_decode_layer_rfpn(decode_yxhw,
                                    flaten_yxhw):

    yref, xref, href, wref = flaten_yxhw

    # Compute center, height and width
    cx = decode_yxhw[1] + xref
    cy = decode_yxhw[0] + yref
    w = decode_yxhw[3] + wref
    h = decode_yxhw[2] + href
    # Boxes coordinates.
    ymin = cy - h / 2.
    xmin = cx - w / 2.
    ymax = cy + h / 2.
    xmax = cx + w / 2.
    bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bboxes, cy, cx, h, w

def tf_ssd_bboxes_decode_layer_fpn_c(feat_localizations,
                                     h_w,
                                     prior_scaling=[0.1, 0.1, 0.2, 0.2]):

    href, wref = h_w

    # Compute center, height and width
    cx = feat_localizations[:, :, 0] * wref * prior_scaling[0] * 0.5
    cy = feat_localizations[:, :, 1] * href * prior_scaling[1] * 0.5
    w = wref * feat_localizations[:, :, 2] * prior_scaling[2] * 0.5
    h = href * feat_localizations[:, :, 3] * prior_scaling[3] * 0.5
    return cy, cx, h, w

def tf_ssd_bboxes_decode_layer_fpn_c1(feat_localizations,
                                     h_w,
                                     prior_scaling=[0.1, 0.1, 0.2, 0.2]):

    href, wref = h_w

    # Compute center, height and width
    cx = feat_localizations[:, :, 0] * wref * prior_scaling[0]
    cy = feat_localizations[:, :, 1] * href * prior_scaling[1]
    w = wref * feat_localizations[:, :, 2] * prior_scaling[2]
    h = href * feat_localizations[:, :, 3] * prior_scaling[3]
    return cy, cx, h, w

def tf_ssd_bboxes_decode(feat_localizations,
                         anchors,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='ssd_bboxes_decode'):
    """Compute the relative bounding boxes from the SSD net features and
    reference anchors bounding boxes.

    Arguments:
      feat_localizations: List of Tensors containing localization features.
      anchors: List of numpy array containing anchor boxes.

    Return:
      List of Tensors Nx4: ymin, xmin, ymax, xmax
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            bboxes.append(
                tf_ssd_bboxes_decode_layer(feat_localizations[i],
                                           anchors_layer,
                                           prior_scaling))
        return bboxes

def tf_ssd_bboxes_decode_fpn(feat_localizations,
                         anchors,
                         batch_size,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         scope='ssd_bboxes_decode'):

    with tf.name_scope(scope):
        bboxes = []
        cys = []
        cxs = []
        hs = []
        ws = []
        for i, anchors_layer in enumerate(anchors):

            bboxe, cy, cx, h, w = tf_ssd_bboxes_decode_layer_fpn(feat_localizations[i],
                                           anchors_layer,
                                           prior_scaling)
            bboxes.append(tf.reshape(bboxe, [batch_size, -1, 4]))
            cys.append(tf.reshape(cy, [batch_size, -1]))
            cxs.append(tf.reshape(cx, [batch_size, -1]))
            hs.append(tf.reshape(h, [batch_size, -1]))
            ws.append(tf.reshape(w, [batch_size, -1]))

        # flaten_bboxes = tf.squeeze(tf.concat(bboxes, 1), 0)
        flaten_bboxes = tf.concat(bboxes, 1)
        flaten_cys = tf.concat(cys, 1)
        flaten_cxs = tf.concat(cxs, 1)
        flaten_hs = tf.concat(hs, 1)
        flaten_ws = tf.concat(ws, 1)

        return flaten_bboxes, flaten_cys, flaten_cxs, flaten_hs, flaten_ws

def tf_ssd_bboxes_decode_fpn1(feat_localizations,
                             flatten_anchor_yxhw,
                             anchors,
                             batch_size,
                             prior_scaling=[0.1, 0.1, 0.2, 0.2],
                             scope='ssd_bboxes_decode'):

    with tf.name_scope(scope):

        bboxe = tf_ssd_bboxes_decode_layer_fpn1(feat_localizations,
                                                  anchors,
                                                  prior_scaling)

        return bboxe

def tf_ssd_bboxes_decode_rfpn(feat_localizations,
                              flatten_anchor_yxhw,
                              flaten_yxhw,
                              batch_size,
                              prior_scaling=[0.1, 0.1, 0.2, 0.2],
                              scope='ssd_bboxes_decode'):
    with tf.name_scope(scope):

        decode_cys, decode_cxs, decode_hs, decode_ws = \
                tf_ssd_bboxes_decode_layer_fpn_c(feat_localizations,
                                                flatten_anchor_yxhw[2:],
                                                prior_scaling)

        bboxe, cy, cx, h, w = tf_ssd_bboxes_decode_layer_rfpn(
                                                [decode_cys, decode_cxs, decode_hs, decode_ws],
                                                flaten_yxhw)

        return bboxe, cy, cx, h, w

def tf_ssd_bboxes_decode_rfpn1(feat_localizations,
                              flatten_anchor_yxhw,
                              flaten_yxhw,
                              batch_size,
                              prior_scaling=[0.1, 0.1, 0.2, 0.2],
                              scope='ssd_bboxes_decode'):
    with tf.name_scope(scope):
        decode_cys, decode_cxs, decode_hs, decode_ws = \
            tf_ssd_bboxes_decode_layer_fpn_c1(feat_localizations,
                                             flatten_anchor_yxhw[2:],
                                             prior_scaling)

        bboxe, cy, cx, h, w = tf_ssd_bboxes_decode_layer_rfpn(
            [decode_cys, decode_cxs, decode_hs, decode_ws],
            flaten_yxhw)

        return bboxe

