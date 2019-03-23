# -*- coding: utf-8 -*-

import numpy as np
import copy

def iou(xy_min1, xy_max1, xy_min2, xy_max2):
    assert(not np.isnan(xy_min1).any())
    assert(not np.isnan(xy_max1).any())
    assert(not np.isnan(xy_min2).any())
    assert(not np.isnan(xy_max2).any())
    assert np.all(xy_min1 <= xy_max1)
    assert np.all(xy_min2 <= xy_max2)
    areas1 = np.multiply.reduce(xy_max1 - xy_min1)
    areas2 = np.multiply.reduce(xy_max2 - xy_min2)
    _xy_min = np.maximum(xy_min1, xy_min2) 
    _xy_max = np.minimum(xy_max1, xy_max2)
    _wh = np.maximum(_xy_max - _xy_min, 0)
    _areas = np.multiply.reduce(_wh)
    assert _areas <= areas1
    assert _areas <= areas2
    return _areas / np.maximum(areas1 + areas2 - _areas, 1e-10)


def non_max_suppress(conf, xy_min, xy_max, threshold, threshold_iou):
    _, _, classes = conf.shape

    ## 新增
    conf_reshape = conf.reshape(-1, classes)
    xy_min_reshape = xy_min.reshape(-1, 2)
    xy_max_reshape = xy_max.reshape(-1, 2)

    max_conf_row = np.max(conf_reshape, axis=1)

    ## 返回 每行中 最大值 大于一定阈值的下标 索引
    index = np.where(max_conf_row > threshold)

    boxes = [(_conf, _xy_min, _xy_max) for _conf, _xy_min, _xy_max in zip(conf_reshape[index], xy_min_reshape[index], xy_max_reshape[index])]

    ## 对每一类 进行 nms
    for c in range(classes):
        boxes.sort(key=lambda box: box[0][c], reverse=True) ## 分数由高到低排列
        deep_box = []
        deep_box = copy.deepcopy(boxes)  ## 深拷贝
        for i in range(len(boxes) - 1):
            box = deep_box[i]
            if box[0][c] <= threshold: ## 小于阈值不做处理
                break

            for j in range(i+1, len(boxes)):
                box_j = deep_box[j]
                if box_j[0][c] <= threshold:
                    break
                _box = boxes[j]
                if iou(box[1], box[2], _box[1], _box[2]) >= threshold_iou:
                    _box[0][c] = 0  ## _box 指向 boxes[i + 1:] 中的同一个地方， 当_box改变时， boxes 中相应的位置也改变

    return boxes
