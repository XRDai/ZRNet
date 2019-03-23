#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image, ImageDraw

def run(one_image, one_imageshape, one_label, objects_coord, image_name):

    images = Image.fromarray(one_image.astype(np.uint8))
    colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
    thickness = 3

    for i, c in enumerate(objects_coord):

        box = objects_coord[i]

        draw = ImageDraw.Draw(images)

        top, left, bottom, right = box * ([one_imageshape[0], one_imageshape[1]] * 2)
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(images.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(images.size[0], np.floor(right + 0.5).astype('int32'))

        for j in range(thickness):
            draw.rectangle(
                [left + j, top + j, right - j, bottom - j], outline=colors[one_label[i]])
        del draw

    images.save("%s" % image_name)

def run1(one_image, one_imageshape, one_label, objects_coord, image_name):

    # images = Image.fromarray((one_image*255).astype(np.uint8))
    images = Image.fromarray(one_image.astype(np.uint8))
    colors = [(0, 0, 0), (0, 0, 255), (0, 255, 0), (255, 0, 0)]
    thickness = 3

    for i, c in enumerate(objects_coord):

        box = objects_coord[i]

        draw = ImageDraw.Draw(images)

        top, left, bottom, right = box * ([one_imageshape[0], one_imageshape[1]] * 2)
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(images.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(images.size[0], np.floor(right + 0.5).astype('int32'))

        for j in range(thickness):
            draw.rectangle(
                [left + j, top + j, right - j, bottom - j], outline=colors[one_label[i]])
        del draw

    images.save("%s" % image_name)

