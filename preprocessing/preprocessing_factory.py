
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from preprocessing import ssd_vgg_preprocessing

slim = tf.contrib.slim


def get_preprocessing(is_training=False):
    """Returns preprocessing_fn(image, height, width, **kwargs).

    Args:
      name: The name of the preprocessing function.
      is_training: `True` if the model is being used for training.

    Returns:
      preprocessing_fn: A function that preprocessing a single image (pre-batch).
        It has the following signature:
          image = preprocessing_fn(image, output_height, output_width, ...).

    Raises:
      ValueError: If Preprocessing `name` is not recognized.
    """

    def preprocessing_fn(image, labels, bboxes, in_shape,
                         out_shape, data_format='NHWC', **kwargs):
        return ssd_vgg_preprocessing.preprocess_image(
            image, labels, bboxes, in_shape, out_shape, data_format=data_format,
            is_training=is_training, **kwargs)
    return preprocessing_fn
