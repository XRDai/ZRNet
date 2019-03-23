# -*- coding: utf-8 -*-

import inspect
import numpy as np
import tensorflow as tf


def per_image_standardization(image):
    stddev = np.std(image)
    return (image - np.mean(image)) / max(stddev, 1.0 / np.sqrt(np.multiply.reduce(image.shape)))


def random_crop(image, objects_coord, width_height, scale=1):
    assert 0 < scale <= 1
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        xy_min = tf.reduce_min(objects_coord[:, :2], 0)
        xy_max = tf.reduce_max(objects_coord[:, 2:], 0)
        margin = width_height - xy_max
        shrink = tf.random_uniform([4], maxval=scale) * tf.concat([xy_min, margin], 0)
        _xy_min = shrink[:2]
        _wh = width_height - shrink[2:] - _xy_min
        objects_coord = objects_coord - tf.tile(_xy_min, [2])
        _xy_min_ = tf.cast(_xy_min, tf.int32)
        _wh_ = tf.cast(_wh, tf.int32)
        image = tf.image.crop_to_bounding_box(image, _xy_min_[1], _xy_min_[0], _wh_[1], _wh_[0])
    return image, objects_coord, _wh

def small_x0(xy_min, ratio):
    ## y < x / ratio

    offset_max = tf.minimum(xy_min[1], 25) + 1
    offset_y0 = tf.random_uniform([], minval=0, maxval=offset_max)
    offset_x0 = ratio * offset_y0

    return offset_x0, offset_y0

def big_x0(xy_min, ratio):
    ## y > x / ratio
    offset_max = tf.minimum(xy_min[0], 85) + 1
    offset_x0 = tf.random_uniform([], minval=0, maxval=offset_max)
    offset_y0 = offset_x0 / ratio

    return offset_x0, offset_y0

def small_x1(xy_max, ratio, width_height):
    ## y < x / ratio

    offset_max = tf.maximum(xy_max[0], width_height[0] - 85) ## 1.158
    offset_x0 = tf.random_uniform([], minval=offset_max, maxval=width_height[0])
    offset_y0 = offset_x0 / ratio

    return offset_x0, offset_y0

def big_x1(xy_max, ratio, width_height):
    ## y > x / ratio

    offset_max = tf.maximum(xy_max[1], width_height[1] - 25) ## 1.1497
    offset_y0 = tf.random_uniform([], minval=offset_max, maxval=width_height[1])
    offset_x0 = ratio * offset_y0

    return offset_x0, offset_y0

def random_crop_equiaxial(image, all_coor, objects_coord, dontcare_coord, width_height):
    xy_min = tf.reduce_min(all_coor[:, :2], 0)
    xy_max = tf.reduce_max(all_coor[:, 2:], 0)

    ratio = width_height[0] / width_height[1]  ## 宽高比

    ## 等轴平移 缩放到 最大原图的 120%
    ## 原图大小在[1248, 384] 左右， 1248 * 1.2 = 1497
    ## (1497 - 1248) / 2 = 124.5
    ## 384 * 1.2 = 460   （460 - 384）/ 2 = 38.4
    ## 小点以较小的为主
    offset_x0, offset_y0 = tf.cond(xy_min[1] < xy_min[0] / ratio,
                                   lambda: small_x0(xy_min, ratio),
                                   lambda: big_x0(xy_min, ratio))

    ## 大点以较大的为主
    offset_x1, offset_y1 = tf.cond(xy_max[1] < xy_max[0] / ratio,
                                   lambda: small_x1(xy_max, ratio, width_height),
                                   lambda: big_x1(xy_max, ratio, width_height))

    _xy_min = tf.concat([tf.expand_dims(offset_x0, 0), tf.expand_dims(offset_y0, 0)], 0)

    _wh = tf.concat([tf.expand_dims(offset_x1, 0), tf.expand_dims(offset_y1, 0)], 0) - _xy_min
    objects_coord = objects_coord - tf.tile(_xy_min, [2])
    dontcare_coord = dontcare_coord - tf.tile(_xy_min, [2])
    _xy_min_ = tf.cast(_xy_min, tf.int32)
    _wh_ = tf.cast(_wh, tf.int32)
    image = tf.image.crop_to_bounding_box(image, _xy_min_[1], _xy_min_[0], _wh_[1], _wh_[0])

    return image, objects_coord, dontcare_coord, _wh


def flip_horizontally(image, objects_coord, dontcare_coord, width):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        image = tf.image.flip_left_right(image)

        xmin, ymin, xmax, ymax = objects_coord[:, 0:1], objects_coord[:, 1:2], objects_coord[:, 2:3], objects_coord[:, 3:4]
        objects_coord = tf.concat([width - xmax, ymin, width - xmin, ymax], 1)

        xmin_, ymin_, xmax_, ymax_ = dontcare_coord[:, 0:1], dontcare_coord[:, 1:2], dontcare_coord[:, 2:3], dontcare_coord[:, 3:4]
        dontcare_coord = tf.concat([width - xmax_, ymin_, width - xmin_, ymax_], 1)

    return image, objects_coord, dontcare_coord


def random_flip_horizontally(image, objects_coord, dontcare_coord, width, probability=0.5):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        pred = tf.random_uniform([]) < probability
        fn1 = lambda: flip_horizontally(image, objects_coord, dontcare_coord, width)
        fn2 = lambda: (image, objects_coord, dontcare_coord)
        return tf.cond(pred, fn1, fn2)


def random_grayscale(image, probability=0.5):
    if probability <= 0:
        return image
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        pred = tf.random_uniform([]) < probability
        fn1 = lambda: tf.tile(tf.image.rgb_to_grayscale(image), [1] * (len(image.get_shape()) - 1) + [3])
        fn2 = lambda: image
        return tf.cond(pred, fn1, fn2)
