#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 18-11-13 上午9:33
# @Author  : XRDai
# @File    : test_time.py
'''
    The demo of ZRNet with vgg-16 on the KITTI benchmark
    Test time
'''
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import tensorflow as tf
import numpy as np
import json
import logging
import os
import time
import argparse
from scipy.misc import imread

import tf_extended as tfe
import posprocessing
from nets import nets_factory
from preprocessing import preprocessing_factory
from nets import np_methods

slim = tf.contrib.slim
DATA_FORMAT = 'NHWC'

def main(args):

    ## load hypes
    with open(args.hypes, 'r') as f:
        logging.info("f: %s", f)
        hypes = json.load(f)

    ## class
    classPath = hypes['kittiClassPath']
    with open(classPath, 'r') as f:
        names = [line.strip() for line in f]

    txt_dir = args.save_dir
    if not os.path.exists(txt_dir):
        os.makedirs(txt_dir)

    ##
    width = hypes['image_width']
    height = hypes['image_height']
    kitti_shape = [height, width]

    anchors = np.load('Anchor.npy')

    ##
    model_name = args.model
    num_classes = 4

    ## Get the ZRNet network and its anchors.
    zrnet_class = nets_factory.get_network(model_name)
    zrnet_params = zrnet_class.default_params._replace(num_classes=num_classes)
    zrnet_params = zrnet_class.default_params._replace(anchor_sizes=anchors)
    zr_net = zrnet_class(zrnet_params)
    zrnet_anchors, flatten_anchor_yxhw, shape_recorder = zr_net.anchors(kitti_shape)

    ##
    image_input = tf.placeholder(tf.int32, shape=(None, None, 3))

    ##
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(False)

    ## image, labels and bboxes.
    image, glabels, gbboxes, bbox_img = image_preprocessing_fn(image_input, None, None, None,
                                                               out_shape=kitti_shape,
                                                               data_format=DATA_FORMAT)

    ## the output of the backbone detection betwork:  predictions, logits, localisations
    ## the output of the top-down detection betwork:  predictions_fpn, logits_fpn, localisations_fpn
    ## the output of the down-top detection betwork:  predictions_rfpn, logits_rfpn, localisations_rfpn
    arg_scope = zr_net.arg_scope(weight_decay=0.0005,
                                  data_format=DATA_FORMAT)
    with slim.arg_scope(arg_scope):
        predictions, logits, localisations, \
        predictions_fpn, logits_fpn, localisations_fpn, \
        predictions_rfpn, logits_rfpn, localisations_rfpn, end_points = \
            zr_net.net(tf.expand_dims(image, axis=0), is_training=False)

        ## decode backbone network
        flaten_bboxes, flaten_cys, flaten_cxs, flaten_hs, flaten_ws = \
            zr_net.bboxes_decode_fpn(localisations, zrnet_anchors, 1)

        ## decode top-down detection network
        localisations_fpn_ = []
        for i in range(len(localisations_fpn)):
            localisations_fpn_.append(tf.reshape(localisations_fpn[i], [1, -1, 4]))
        _localisations_fpn_ = tf.concat(localisations_fpn_, 1)

        flaten_bboxes_fpn, flaten_cys_fpn, \
        flaten_cxs_fpn, flaten_hs_fpn, flaten_ws_fpn = \
            zr_net.bboxes_decode_rfpn(_localisations_fpn_, flatten_anchor_yxhw, (flaten_cys, flaten_cxs, flaten_hs, flaten_ws), 1)

        ## decode down-top detection network
        localisations_rfpn_ = []
        for i in range(len(localisations_rfpn)):
            localisations_rfpn_.append(tf.reshape(localisations_rfpn[i], [1, -1, 4]))

        _localisations_rfpn_ = tf.concat(localisations_rfpn_, 1)
        flaten_bboxes1 = zr_net.bboxes_decode_rfpn1(_localisations_rfpn_, flatten_anchor_yxhw,
                                                    (flaten_cys_fpn, flaten_cxs_fpn, flaten_hs_fpn, flaten_ws_fpn), 1)

        flaten_scores, flaten_labels, flaten_bboxes = posprocessing.flaten_predict1(predictions_rfpn, flaten_bboxes1)

    ## clip
    flaten_bboxes = tfe.bboxes.bboxes_clip(bbox_img, flaten_bboxes)

    ## model
    checkpoint_path = tf.train.latest_checkpoint(args.logdir)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        slim.assign_from_checkpoint_fn(checkpoint_path, tf.global_variables(), True)(sess)

        ## test
        kitti_test_dir = args.test_dir
        files = os.listdir(kitti_test_dir)
        for j, file in enumerate(files):
            print('%d' % (j))
            image_file = os.path.join(kitti_test_dir, file)
            image_name = os.path.basename(image_file)

            img = imread(image_file, mode='RGB')

            ## test time
            start_time = time.time()
            for i in xrange(10):
                rpredictions, r_scores, rlocalisations, rbbox_img = sess.run(
                    [flaten_labels, flaten_scores, flaten_bboxes, bbox_img],
                    feed_dict={image_input: img})
            dt = (time.time() - start_time) / 10

            print(('Speed (msec)', 1000 * dt))
            print(('Speed (fps)', 1 / dt))
    return

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypes', default='HyPara.json')
    parser.add_argument('--model', default='ZRNet_vgg', help='model')
    parser.add_argument('--logdir', default='./logs/')
    parser.add_argument('--kittiClass', default='./my_Class.txt')
    parser.add_argument('--test_dir', default='../KITTI/testing/image_2')
    parser.add_argument('--save_dir', default='./OUT')

    return parser.parse_args()

if __name__ == '__main__':
    args = make_args()
    main(args)

