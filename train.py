#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 19-3-23 上午9:33
# @Author  : XRDai
# @File    : train.py
'''
    The training code will be appear soon.  
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
