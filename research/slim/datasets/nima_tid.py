"""Provides data for the NIMA dataset.

The dataset scripts used to create the dataset can be found at:
AVA:
TID2013: tensorflow/models/research/slim/datasets/download_and_convert_nima_tid.py

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'nima_tid_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 3320, 'validation': 350}

_CONVERSION_DIR = 'TFRecords'

_ITEMS_TO_DESCRIPTIONS = {
  'image': 'A color image of varying size.',
  'id': 'the id of the image',
  'mean': 'mean opinion score',
  'stddev': 'standard deviation of mean opinion score',
}

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
  """Gets a dataset tuple with instructions for reading from TID2013.

  Args:
    split_name: A train/validation split name.
    dataset_dir: The base directory of the dataset sources.
    file_pattern: The file pattern to use when matching the dataset sources.
      It is assumed that the pattern contains a '%s' string so that the split
      name can be inserted.
    reader: The TensorFlow reader type.

  Returns:
    A `Dataset` namedtuple.

  Raises:
    ValueError: if `split_name` is not a valid train/validation split.
  """
  if split_name not in SPLITS_TO_SIZES:
    raise ValueError('split name %s was not recognized.' % split_name)

  if not file_pattern:
    file_pattern = _FILE_PATTERN
  tfrecord_dir = dataset_dir if dataset_dir.endswith(_CONVERSION_DIR) else os.path.join(dataset_dir, _CONVERSION_DIR)
  
  file_pattern = os.path.join(tfrecord_dir, file_pattern % split_name)

  # Allowing None in the signature so that dataset_factory can use the default.
  if reader is None:
    reader = tf.TFRecordReader

  keys_to_features = {
    'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/format': tf.FixedLenFeature((), tf.string, default_value='bmp'),
    'image/id': tf.FixedLenFeature((), tf.string, default_value=''),
    'image/target/mean': tf.FixedLenFeature((), tf.float32),
    'image/target/stddev': tf.FixedLenFeature((), tf.float32),        
    'image/height': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    'image/width': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
  }

  items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'id': slim.tfexample_decoder.Tensor('image/id'),
    'mean': slim.tfexample_decoder.Tensor('image/target/mean'),
    'stddev': slim.tfexample_decoder.Tensor('image/target/stddev'),
    'height': slim.tfexample_decoder.Tensor('image/height'),
    'width': slim.tfexample_decoder.Tensor('image/width'),
  }

  decoder = slim.tfexample_decoder.TFExampleDecoder(
    keys_to_features, items_to_handlers)

  return slim.dataset.Dataset(
      data_sources=file_pattern,
      reader=reader,
      decoder=decoder,
      num_samples=SPLITS_TO_SIZES[split_name],
      items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
      # num_classes=_NUM_CLASSES,
      # labels_to_names=labels_to_names
    )