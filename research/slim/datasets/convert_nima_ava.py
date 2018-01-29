"""Downloads and converts AVA data to TFRecords of TF-Example protos.

This module assumes the AVA data has been downloaded and uncompressed.

This module reads the 255510 JPG files 
that make up the AVA data and creates two TFRecord datasets: one for train
and one for test. Each TFRecord dataset is comprised of a set of TF-Example
protocol buffers, each of which contain a single image and label.

The script should take about a 4 hours to run on the full AVA dataset.
or 2.5 hours on a resized dataset or 3.5 hours on the full dataset
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import re
import sys
import numpy as np

import preprocessing.nima_preprocessing as nima_pre


import tensorflow as tf

from datasets import dataset_utils

# The URL where the Flowers data can be downloaded.
_DATA_URL = 'http://www.ponomarenko.info/tid2013/tid2013.rar'


# The number of images in the validation set. the AVA dataset contains 255510 JPGs
_NUM_VALIDATION = 51102  # 20% test, reset on each run()

# Seed for repeatability.
_RANDOM_SEED = 0

# The number of shards per dataset split.
_NUM_SHARDS = 5

# targets: 
_TFRECORD_TEMPLATE = 'nima_ava_%s_%04d-of-%04d.tfrecord'
_TARGETS_FILENAME = 'targets.txt'

# sub-dirs for inputs, converted TFRecords, and label data
#   all dirs should exist under `dataset_dir`
_SOURCE_DIR = 'images'
_RESIZED_SOURCE_DIR = _SOURCE_DIR +'-256'
_TARGET_DIR = 'TFRecords'
_LABEL_DIR = 'labels'

class ImageBasepathByType( object ):
  """Helper class to derive original and resized file paths for images in dataset """
  basepath=""
  resized_basepath=""

  @staticmethod
  def set_basepaths(basepath, resized_basepath):
    ImageBasepathByType.basepath = basepath
    ImageBasepathByType.resized_basepath = resized_basepath

  @staticmethod
  def original(filename=None):
    if filename is None:
      return ImageBasepathByType.basepath
    return os.path.join(ImageBasepathByType.basepath, os.path.basename(filename))

  @staticmethod
  def resized(filename=None):
    if filename is None:
      return ImageBasepathByType.resized_basepath
    return os.path.join(ImageBasepathByType.resized_basepath, os.path.basename(filename))


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(self._decode_jpeg,
                     feed_dict={self._decode_jpeg_data: image_data})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename( dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True


def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = _TFRECORD_TEMPLATE % (split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)
 
def _convert_dataset(split_name, filenames, targets, target_dir, resized=False):
  """Converts the given filenames to a TFRecord dataset.

  Args:
    split_name: The name of the dataset, either 'train' or 'validation'.
    filenames: A list of absolute paths to jpg images.
    targets: A list of tuples [(id, [ratings], [tags]),('1234.jpg', 5.51429, 0.13013),...]
    target_dir: The directory where the converted datasets are stored.
  """
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename( target_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
          for i in range(start_ndx, end_ndx):
            try:
              sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                  i+1, len(filenames), shard_id))
              sys.stdout.flush()

              # Read the filename:
              image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
              if split_name=="train" and resized==True:
                # get original image dimensions
                original_filename = ImageBasepathByType.original(filenames[i])
                orig_image_data = tf.gfile.FastGFile(original_filename, 'rb').read()
                height, width = image_reader.read_image_dims(sess, orig_image_data)
                # sys.stdout.write('\n     dim=%dx%d size=%d, %d file=%s' % (height, width, 
                #   len(image_data), len(orig_image_data), original_filename))
                # sys.stdout.flush()
              else: 
                height, width = image_reader.read_image_dims(sess, image_data)

              imageId, ratings, tags = targets[i]

              example = _image_to_tfexample(
                  image_data, b'jpg', height, width, imageId, ratings, tags)
              tfrecord_writer.write(example.SerializeToString())
            except:
              sys.stdout.write('\n>> Converting image %d/%d shard %d, EXCEPTION: invalid jpg, file=...%s' % (
                  i+1, len(filenames), shard_id, filenames[i][-20:]))
              sys.stdout.flush()
              # raise


  sys.stdout.write('\n')
  sys.stdout.flush()


def _get_filenames_and_targets(image_dir):
  """Returns a list of filenames and target of mean opinion scores and stddev.

  Args:
    image_dir: A directory containing AVA `JPG` encoded images, either full-res or resized (256,256,3).

  Returns:
    A list of image file paths, relative to `image_dir` and 
    list of targets
  """
  global _NUM_VALIDATION

  ids = []
  image_paths = []
  targets = []
  for filename in sorted(os.listdir(image_dir)):
    # if filename.endswith(".jpg"):
    if re.match('.*\.jpg$', filename, re.I):
      ids.append( int(filename[:-1*len(".jpg")]) ) 
      path = os.path.join(image_dir, filename)
      image_paths.append(path)


  # targets
  # AVA.txt: e.g. `1 953619 0 1 5 17 38 36 15 6 5 1 1 22 1396`
  # e.g. ./images/../labels/AVA.txt
  target_file = os.path.join(os.path.dirname(image_dir), _LABEL_DIR, 'AVA.txt')
  np_data = np.loadtxt(target_file, dtype=np.int64)
  data_id = np_data[:,1]
  data_ratings = np_data[:,2:12].tolist()
  data_tags = np_data[:,12:14].tolist()
  

  # list of tuples, e.g. [(id, array(10) ratings, array(2) tags),('1234.jpg', 5.51429, 0.13013),...]
  targets = list(zip(data_id, data_ratings, data_tags))
  # print("e.g. targets", targets[:3])

  # scale validation if we find < 90% fewer filenames
  if len(ids)/len(data_id) < 0.9 or True:
    # set validation = 20% of total dataset
    _NUM_VALIDATION = math.ceil( 0.2 * len(ids) )
    id_lookup = {}
    for key in ids: id_lookup[key] = True
    targets = [ v for v in targets if v[0] in id_lookup ]   

    print('\r>> count found filenames=%d, original target rows=%d, validation rows=%d' % (
                len(ids), len(data_id), _NUM_VALIDATION))

  return image_paths, sorted(targets, key=lambda v: v[0])





def _image_to_tfexample(image_data, image_format, height, width, imageId, ratings, tags):

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


  # ratings: type(ratings)=<class 'list'>, e.g. ratings=[1, 3, 3, 3, 10, 10, 13, 20, 8, 15]
  # expand rating counts into scores
  scores = []
  for i in range(len(ratings)):
    scores.extend(  [i+1 for _ in range(ratings[i])]  )
  mean = np.mean(scores) 
  stddev = np.std(scores)
  # sys.stdout.write('\n   >  imageId=%s, type(ratings)=%s, ratings=%s, mean=%f' % (imageId, type(ratings), str(ratings), mean))
  # sys.stdout.flush() 
  return tf.train.Example(features=tf.train.Features(feature={
      'image/encoded': bytes_feature(image_data),
      'image/format': bytes_feature(image_format),
      'image/id': bytes_feature(  tf.compat.as_bytes(str(imageId))  ),
      'image/ratings': int64_feature(ratings),
      'image/target/mean': float_feature(mean),
      'image/target/stddev': float_feature(stddev),
      'image/tags': int64_feature(tags),
      'image/height': int64_feature(height),
      'image/width': int64_feature(width),
  }))

def _write_targets_file(targets, dataset_dir,
                     filename=_TARGETS_FILENAME):
  """Writes a file with the targets tuple.

  Args:
    targets: A list of tuples [(filename, ratings shape=[10], tags shape=[2]),('1234.jpg', [...], [...]),...]    
    dataset_dir: The directory in which the labels file should be written.
    filename: The filename where the class names are written.
  """
  targets_filename = os.path.join(dataset_dir, filename)
  with tf.gfile.Open(targets_filename, 'w') as f:
    for target in targets:
      id, ratings, tags = target
      f.write('%s: %s, %s\n' % (id, str(ratings), str(tags)))

def run(dataset_dir, resized=False, shards=None):
  """Runs the TFRecord conversion operation.

  Args:
    dataset_dir: The dataset root directory where the dataset is stored.
    resized: boolean, if True, use resized JPGs (256,256,3) for conversion
    shards: override TFRecord shard count
  """

  global _NUM_VALIDATION
  _NUM_VALIDATION = 51102

  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  if shards is not None:
    global _NUM_SHARDS
    _NUM_SHARDS = shards

  # prepare for train/validation path manipulation
  global _SOURCE_DIR, _RESIZED_SOURCE_DIR
  original_path = os.path.join(dataset_dir, _SOURCE_DIR)
  resized_path = os.path.join(dataset_dir, _RESIZED_SOURCE_DIR)  # e.g. /images-256/
  ImageBasepathByType.set_basepaths( original_path, resized_path )

  basepath = ImageBasepathByType.resized() if resized else ImageBasepathByType.original()
  photo_filenames, targets = _get_filenames_and_targets( basepath )
  print(">>> converting images from basepath=", basepath)


  # Shuffle, then Divide into train and validation:
  random.seed(_RANDOM_SEED)
  random.shuffle(photo_filenames)
  training_filenames = photo_filenames[_NUM_VALIDATION:]
  validation_filenames = photo_filenames[:_NUM_VALIDATION]

  if resized:
    # after shuffle and train/validation split, we need to recover full size images for validation set
    validation_filenames = [ImageBasepathByType.original( f ) for f in validation_filenames]

  # save conversion output to a subfolder
  target_dir = os.path.join(dataset_dir, _TARGET_DIR)
  if resized and not target_dir.endswith('_resized'):
    #  e.g. target_dir = 'TFRecords_resized'
    target_dir += '_resized'
  if not tf.gfile.Exists(target_dir):
    tf.gfile.MakeDirs(target_dir)

  # First, convert the training and validation sets.
  _convert_dataset('train', training_filenames, targets, target_dir, resized=resized)
  _convert_dataset('validation', validation_filenames, targets, target_dir)

  
  # Finally, write the targets file:
  _write_targets_file(targets, target_dir)

  # _clean_up_temporary_files(dataset_dir)
  print('\nFinished converting the AVA dataset!')