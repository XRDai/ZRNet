# -*- coding: utf-8 -*-

import os
import re
import importlib
import inspect
import numpy as np
import matplotlib.patches as patches
import tensorflow as tf
from .. import preprocess

def decode_image_objects(paths):
    ## 返回 解码的 图片信息
    with tf.name_scope('parse_example'):
        reader = tf.TFRecordReader()
        _, serialized = reader.read(tf.train.string_input_producer([paths]))
        example = tf.parse_single_example(serialized, features={
            'imagepath': tf.FixedLenFeature([], tf.string),
            'imageshape': tf.FixedLenFeature([3], tf.int64),
            'objects': tf.FixedLenFeature([2], tf.string),
            'dontcare': tf.FixedLenFeature([], tf.string),
            'allrects': tf.FixedLenFeature([], tf.string),
        })

    imagepath = example['imagepath']
    objects = example['objects']
    with tf.name_scope('decode_objects'):
        objects_class = tf.decode_raw(objects[0], tf.int64, name='objects_class')
        objects_coord = tf.decode_raw(objects[1], tf.float32)
        objects_coord = tf.reshape(objects_coord, [-1, 4], name='objects_coord')

    with tf.name_scope('decode_dontcare'):
        dontcare_coord = tf.decode_raw(example['dontcare'], tf.float32)
        dontcare_coord = tf.reshape(dontcare_coord, [-1, 4], name='dontcare_coord')

    with tf.name_scope('allrects'):
        all_coord = tf.decode_raw(example['allrects'], tf.float32)
        all_coord = tf.reshape(all_coord, [-1, 4], name='all_coord')

    with tf.name_scope('load_image'):
        imagefile = tf.read_file(imagepath)
        image = tf.image.decode_png(imagefile, channels=3) ## 当原始图片为 jpg 或 png 时，需相应更改

    return image, example['imageshape'], objects_class, objects_coord, dontcare_coord, all_coord


def data_augmentation_full(image, objects_coord, width_height, config):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        random_crop = config.getfloat(section, 'random_crop')
        if random_crop > 0:
            image, objects_coord, width_height = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: preprocess.random_crop(image, objects_coord, width_height, random_crop),
                lambda: (image, objects_coord, width_height)
            )
    return image, objects_coord, width_height

def data_augmentation_full_equiaxial(image, all_coor, objects_coord, dontcare_coord, width_height, config):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        random_crop = config.getfloat(section, 'random_crop')
        if random_crop > 0:
            image, objects_coord, dontcare_coord, width_height = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: preprocess.random_crop_equiaxial(image, all_coor, objects_coord, dontcare_coord, width_height),
                lambda: (image, objects_coord, dontcare_coord, width_height)
            )
    return image, objects_coord, dontcare_coord, width_height


def resize_image_objects(image, objects_coord, dontcare_coord, width_height, width, height):
    with tf.name_scope(inspect.stack()[0][3]):
        image = tf.image.resize_images(image, [height, width])
        factor = [width, height] / width_height
        objects_coord = objects_coord * tf.tile(factor, [2])
        dontcare_coord = dontcare_coord * tf.tile(factor, [2])

    return image, objects_coord, dontcare_coord


def data_augmentation_resized(image, objects_coord, dontcare_coord, width, height, config):
    section = inspect.stack()[0][3]
    with tf.name_scope(section):
        if config.getboolean(section, 'random_flip_horizontally'):
            image, objects_coord, dontcare_coord = preprocess.random_flip_horizontally(image, objects_coord, dontcare_coord, width)
        if config.getboolean(section, 'random_brightness'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_brightness(image, max_delta=63),
                lambda: image
            )
        if config.getboolean(section, 'random_saturation'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_saturation(image, lower=0.5, upper=1.6),
                lambda: image
            )
        if config.getboolean(section, 'random_hue'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_hue(image, max_delta=0.15),
                lambda: image
            )
        if config.getboolean(section, 'random_contrast'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: tf.image.random_contrast(image, lower=0.5, upper=1.5),
                lambda: image
            )
        if config.getboolean(section, 'noise'):
            image = tf.cond(
                tf.random_uniform([]) < config.getfloat(section, 'enable_probability'),
                lambda: image + tf.truncated_normal(tf.shape(image)) * tf.random_uniform([], 5, 15),
                lambda: image
            )
        grayscale_probability = config.getfloat(section, 'grayscale_probability')
        if grayscale_probability > 0:
            image = preprocess.random_grayscale(image, grayscale_probability)
    return image, objects_coord, dontcare_coord


def transform_labels(objects_class, objects_coord, dontcare_coord, classes, cell_width, cell_height, dtype=np.float32):
    cells = cell_height * cell_width
    mask = np.zeros([cells, 1], dtype=dtype)
    prob = np.zeros([cells, 1, classes], dtype=dtype)
    coords = np.zeros([cells, 1, 4], dtype=dtype)
    offset_xy_min = np.zeros([cells, 1, 2], dtype=dtype)
    offset_xy_max = np.zeros([cells, 1, 2], dtype=dtype)
    assert len(objects_class) == len(objects_coord)
    xmin, ymin, xmax, ymax = objects_coord.T ## 此处的 xmin 等是相对原图大小归一化后的 在 [0：1]
    x = cell_width * (xmin + xmax) / 2 ## 此处 x 是 bbox 的在 cell中 的 中心点 坐标
    y = cell_height * (ymin + ymax) / 2
    ix = np.floor(x)
    iy = np.floor(y)
    offset_x = (x - ix) * 32. ## 不再是 18.5-18 = 0.5， 而是转换为绝对距离
    offset_y = (y - iy) * 32.
    w = (xmax - xmin) * cell_width ## 此处的 w 是相对网格大小的 在 [0:39]
    h = (ymax - ymin) * cell_height
    index = (iy * cell_width + ix).astype(np.int) ## 每个 bbox 中心位置的索引值
    mask[index, :] = 1
    prob[index, :, objects_class] = 1 ## 在特定位置 上对应的 类别位置 概率置 1
    coords[index, 0, 0] = offset_x
    coords[index, 0, 1] = offset_y
    coords[index, 0, 2] = np.sqrt(w)
    coords[index, 0, 3] = np.sqrt(h)
    _w = w / 2  ## 此处的 _w 是 原图 对应在 13*13 的网格上 的 宽度的 一半
    _h = h / 2
    offset_xy_min[index, 0, 0] = offset_x / 32. - _w ## offset_xy_min 是在网格上 gt_box 的 最小点 坐标
    offset_xy_min[index, 0, 1] = offset_y / 32. - _h
    offset_xy_max[index, 0, 0] = offset_x / 32. + _w
    offset_xy_max[index, 0, 1] = offset_y / 32. + _h
    wh = offset_xy_max - offset_xy_min
    assert np.all(wh >= 0)
    areas = np.multiply.reduce(wh, -1)

    ## 增加 kitti_box 中所需要的 x y w h
    ## 其中 x y 是相对左上角点的 而不再是 相对中心
    ## 其中 w h 是在 [0:1] 而不再是原本的宽度 和 高度
    ## 其 conf 和上面 mask 相同
    ## 其 dontcare的 mask 和 下面的 dont_mask
    ##测试
    # objects_coord = np.array([[1, 2, 3, 4], [4, 5, 7, 8]])
    # xmin, ymin, xmax, ymax = objects_coord.T
    # cc = np.zeros([5, 1, 4], dtype=np.float32)
    # index = np.array([0, 1])
    # cc[index, 0, 0] = xmin

    kitti_coords = np.zeros([cells, 1, 4], dtype=dtype)
    kitti_coords[index, 0, 0] = offset_x
    kitti_coords[index, 0, 1] = offset_y
    kitti_coords[index, 0, 2] = w
    kitti_coords[index, 0, 3] = h

    ## 下面添加 dontcare的 mask 用于计算 不含 object 部分的 损失
    dont_mask = np.ones([cells, 1], dtype=dtype)

    if len(dontcare_coord) > 1:
        ## 不考虑 lalala部分， 去除[0, 0, 0, 0],所以索引从 1 开始
        xmin, ymin, xmax, ymax = dontcare_coord[1:, :].T  ## 此处的 xmin 等是相对原图大小归一化后的 在 [0：1]

        x0 = (cell_width * xmin).astype(np.int8)
        x1 = (cell_width * xmax).astype(np.int8)
        x1[x1 > cell_width - 1] = cell_width - 1

        y0 = (cell_height * ymin).astype(np.int8)
        y1 = (cell_height * ymax).astype(np.int8)
        y1[y1 > cell_height - 1] = cell_height - 1

        num = len(xmin)
        for i in range(num):
            ## 注意此处的 range 其上限取不到 故 需 ++++111111
            for x in range(x0[i], x1[i] + 1):
                for y in range(y0[i], y1[i] + 1):
                    index = (y * cell_width + x)
                    dont_mask[index] = 0

    return mask, prob, coords, offset_xy_min, offset_xy_max, areas, dont_mask, kitti_coords


def decode_labels(objects_class, objects_coord, dontcare_coord, classes, cell_width, cell_height):
    with tf.name_scope(inspect.stack()[0][3]):
        mask, prob, coords, offset_xy_min, offset_xy_max, areas, dont_mask, kitti_coords = tf.py_func(transform_labels, \
            [objects_class, objects_coord, dontcare_coord, classes, cell_width, cell_height], [tf.float32] * 8)
        cells = cell_height * cell_width
        with tf.name_scope('reshape_labels'):
            mask = tf.reshape(mask, [cells, 1], name='mask')
            prob = tf.reshape(prob, [cells, 1, classes], name='prob')
            coords = tf.reshape(coords, [cells, 1, 4], name='coords')
            offset_xy_min = tf.reshape(offset_xy_min, [cells, 1, 2], name='offset_xy_min')
            offset_xy_max = tf.reshape(offset_xy_max, [cells, 1, 2], name='offset_xy_max')
            areas = tf.reshape(areas, [cells, 1], name='areas')
            dont_mask = tf.reshape(dont_mask, [cells, 1], name='dont_mask')
            kitti_coords = tf.reshape(kitti_coords, [cells, 1, 4], name='kitti_coords')
    return mask, prob, coords, offset_xy_min, offset_xy_max, areas, dont_mask, kitti_coords


def load_image_labels(paths, classes, width, height, cell_width, cell_height, config):
    with tf.name_scope('batch'):
        image, imageshape, objects_class, objects_coord, dontcare_coord, all_coord = decode_image_objects(paths[0])
        image = tf.cast(image, tf.float32)
        width_height = tf.cast(imageshape[1::-1], tf.float32)

        if config.getboolean('data_augmentation_full', 'enable'):
            # image, objects_coord, width_height = data_augmentation_full(image, objects_coord, width_height, config)
            image, objects_coord, dontcare_coord, width_height = \
                data_augmentation_full_equiaxial(image, all_coord, objects_coord, dontcare_coord, width_height, config)

        image, objects_coord, dontcare_coord = resize_image_objects(image, objects_coord, dontcare_coord, width_height, width, height)

        if config.getboolean('data_augmentation_resized', 'enable'):
            image, objects_coord, dontcare_coord = data_augmentation_resized(image, objects_coord, dontcare_coord, width, height, config)

        image = tf.clip_by_value(image, 0, 255)

        objects_coord = objects_coord / [width, height, width, height]
        dontcare_coord = dontcare_coord / [width, height, width, height]

        labels = decode_labels(objects_class, objects_coord, dontcare_coord, classes, cell_width, cell_height)
    return image, labels
