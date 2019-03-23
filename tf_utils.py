from __future__ import print_function
import os
from pprint import pprint

import tensorflow as tf
from tensorflow.contrib.slim.python.slim.data import parallel_reader

slim = tf.contrib.slim


# =========================================================================== #
# General tools.
# =========================================================================== #
def reshape_list(l, shape=None):
    """Reshape list of (list): 1D to 2D or the other way around.

    Args:
      l: List or List of list.
      shape: 1D or 2D shape.
    Return
      Reshaped list.
    """
    r = []
    if shape is None:
        # Flatten everything.
        for a in l:
            if isinstance(a, (list, tuple)):
                r = r + list(a)
            else:
                r.append(a)
    else:
        # Reshape to list of list.
        i = 0
        for s in shape:
            if s == 1:
                r.append(l[i])
            else:
                r.append(l[i:i+s])
            i += s
    return r

def reshape_listN(gclasses, glocalisations, gscores, gbboxes, batch_shape, batch_size):
    n_gclasses = []
    n_glocalisations = []
    n_gscores = []
    n_gbboxes = []

    for s in range(batch_shape):
        temp_gclasses = tf.stack(gclasses[s:batch_size*batch_shape:batch_shape], axis=0)
        temp_glocalisations = tf.stack(glocalisations[s:batch_size*batch_shape:batch_shape], axis=0)
        temp_gscores = tf.stack(gscores[s:batch_size*batch_shape:batch_shape], axis=0)
        temp_gbboxes = tf.stack(gbboxes[s:batch_size*batch_shape:batch_shape], axis=0)

        n_gclasses.append(temp_gclasses)
        n_glocalisations.append(temp_glocalisations)
        n_gscores.append(temp_gscores)
        n_gbboxes.append(temp_gbboxes)

    return n_gclasses, n_glocalisations, n_gscores, n_gbboxes

