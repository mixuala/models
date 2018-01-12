"""Downloads and converts TID2013 data to TFRecords of TF-Example protos.

This module assumes the TID data has been downloaded and uncompressed.

This module reads the files 
that make up the TID2013 data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a minute to run.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import re
import sys


import tensorflow as tf

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
_DATA_URL = 'http://www.ponomarenko.info/tid2013/tid2013.rar'


# The number of images in the validation set.
_NUM_VALIDATION = 350

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 4

# targets: 
_TARGETS_FILENAME = 'targets.txt'

# output dir for converted records
_ARCHIVE_DIR = 'archive'
_CONVERSION_DIR = 'TFRecords'

# sub-dirs for inputs, converted TFRecords, and label data
#   all dirs should exist under `dataset_dir`
_SOURCE_DIR = 'images'
_TARGET_DIR = 'TFRecords'
# _SOURCE_DIR = 'images-256'
# _TARGET_DIR = 'TFRecords_resized'
_LABEL_DIR = 'labels'

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities for BMPs."""

  def __init__(self):
    # Initializes function that decodes RGB BMP data.
    self._decode_bmp_data = tf.placeholder(dtype=tf.string)
    self._decode_bmp = tf.image.decode_bmp(self._decode_bmp_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_bmp(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_bmp(self, sess, image_data):
    image = sess.run(self._decode_bmp,
                     feed_dict={self._decode_bmp_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'nima_tid_%s_%04d-of-%04d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)
 
def _convert_dataset(split_name, filenames, targets, dataset_dir):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to bmp images.
    targets: A list of tuples [(filename, mean, stddev),('I01_01_1.bmp', 5.51429, 0.13013),...]
    dataset_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      # sys.stdout.write('\r>> split=%s count filenames=%d, target rows=%d' % ( split_name, len(filenames), len(targets)))
      # sys.stdout.flush()

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(filenames), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            imageId, mean, stddev = targets[i]

            example = _image_to_tfexample(
                image_data, b'bmp', height, width, imageId, mean, stddev)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()


def _get_filenames_and_targets(image_dir):
  """Returns a list of filenames and target of mean opinion scores and stddev.

  Args:
    image_dir: A directory containing TID2013 `BMP` encoded images.

  Returns:
    A list of image file paths, relative to `image_dir` and 
    list of targets
  """

  global _NUM_VALIDATION

  # tid_root = os.path.join(image_dir, 'raw_archive')

  image_paths = []
  ids = []
  targets = []
  for filename in sorted(os.listdir(image_dir)):
    # if filename.endswith(".bmp"):
    if re.match('.*\.bmp$', filename, re.I):
      ids.append(filename)
      path = os.path.join(image_dir, filename)
      image_paths.append(path)


  # targets
  # mos_with_names.txt: e.g. `5.51429 I01_01_1.bmp`
 
  # target_mean_file = os.path.join(image_dir, 'mos_with_names.txt')
  target_mean_file = os.path.join(os.path.dirname(image_dir), _LABEL_DIR, 'mos_with_names.txt')
  target_mean = []

  with tf.gfile.Open(target_mean_file, 'r') as f:
    # for line in f.readlines():
    #   target_mean.append( line.strip().split(' '))
    target_mean = [line.strip().split(' ') for line in f.readlines()]

  # mos_std.txt: e.g. `0.13013`
  # target_stddev_file = os.path.join(image_dir, 'mos_std.txt')
  target_stddev_file = os.path.join(os.path.dirname(image_dir), _LABEL_DIR, 'mos_std.txt')
  target_stddev = []
  with tf.gfile.Open(target_stddev_file, 'r') as f:
    target_stddev = [line.strip() for line in f.readlines()]
  
  # list of tuples, e.g. [(filename, mean, stddev),('I01_01_1.bmp', 5.51429, 0.13013),...]
  targets = []
  for i in range(len(target_mean)):
    mean, filename = target_mean[i]
    if filename not in ids: break 
    targets.append( (filename, float(mean), float(target_stddev[i]) ) )

  return image_paths, sorted(targets, key=lambda v: v[0]  )





def _image_to_tfexample(image_data, image_format, height, width, imageId, mean, stddev):

  def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


  def float_feature(values):
    """Returns a TF-Feature of floats.

    Args:
      values: A scalar of list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
      values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

  def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
      values: A scalar or list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
      values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))



  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/id': bytes_feature(  tf.compat.as_bytes(imageId)  ),
      'image/target/mean': float_feature(mean),
      'image/target/stddev': float_feature(stddev),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

def _write_targets_file(targets, dataset_dir,
                     filename=_TARGETS_FILENAME):
  """Writes a file with the targets tuple.

  Args:
    targets: A list of tuples [(filename, mean, stddev),('I01_01_1.bmp', 5.51429, 0.13013),...]    
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  targets_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(targets_filename, 'w') as f:
    for target in targets:
      fname, mean, stddev = target
      f.write('%s: %f, %f\n' % (target))





  """ to resize in OSX
  export SOURCE=/snappi.ai/tensorflow/nima/data/tid/images
  export TARGET=/snappi.ai/tensorflow/nima/data/tid/images-256
  mkdir -p $TARGET
  for f in $SOURCE/*.bmp; do
    sips -z 256 256 $f --out $TARGET
  done
  for f in $SOURCE/*.BMP; do
    sips -z 256 256 $f --out $TARGET
  done
  """


def run(dataset_dir):
  """Runs the TFRecord conversion operation.

  Args:
    dataset_dir: The dataset directory where the dataset is stored.
  """
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  # dataset_utils.download_and_uncompress_tarball(_DATA_URL, dataset_dir)
  global _SOURCE_DIR
  image_dir = os.path.join(dataset_dir, _SOURCE_DIR)
  photo_filenames, targets = _get_filenames_and_targets(image_dir)
  # print('\r>> count filenames=%d, target rows=%d, validation rows=%d' % ( len(photo_filenames), len(targets), _NUM_VALIDATION))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[_NUM_VALIDATION:]
  validation_filenames = photo_filenames[:_NUM_VALIDATION]

  # save conversion output to a subfolder
  conversion_dir = os.path.join(dataset_dir, _TARGET_DIR)
  if not tf.gfile.Exists(conversion_dir):
    tf.gfile.MakeDirs(conversion_dir)

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, targets, conversion_dir)
  _convert_dataset('validation', validation_filenames, targets, conversion_dir)

  
  # Finally, write the targets file:
  _write_targets_file(targets, conversion_dir)

  # _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the TID2013 dataset!')