
# from slim_walkthrough
import tensorflow as tf
from preprocessing import preprocessing_factory
from preprocessing import vgg_preprocessing, inception_preprocessing
from tensorflow.contrib import slim

def load_batch(dataset, batch_size=32, height=224, width=224, 
            is_training=False, 
            model="vgg16",
            resized=True,
            include_raw_images=True,
            label_name="ratings"):
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
      model: [vgg16 | inception]
      resized: Whether the TFRecords were converted with images already resized to (256,256,3)
      include_raw_images: Boolean, include resized raw images. 
      ## TODO: figure out how to return an empty set for images_raw
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: (optional) A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization. 
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label, orig_height, orig_width = data_provider.get(['image', label_name, 'height', 'width'])
    
    # Preprocess image for usage by the appropriate model.
    # we assume the AVA ratings are from the full, undistorted image, so it does
    # not make sense to distort or randomly crop the images to learn ratings
    preprocessing = preprocessing_factory.get_preprocessing(model)
        
    if 'vgg_preprocessing' in preprocessing.lib.__name__:
      image = preprocessing(image_raw, height, width, 
                              is_training=is_training,
                              resized=resized,
                              resize_side_min=256,
                              resize_side_max=256)

    elif 'inception_preprocessing' in preprocessing.lib.__name__:
      image = preprocessing(image_raw, height, width, 
                              bbox=None, 
                              fast_mode=False,
                              is_training=is_training,
                              resized=resized,
                              resize_side_min=256,
                              resize_side_max=256,
                              distort_color=False,
                              add_image_summaries=False,
                              central_fraction=None,
                              )
    else:
      raise RuntimeError("preprocessing is not configured")

    # Preprocess the image for display purposes.

    # if not resize_raw:
    #   height, width = [orig_height, orig_width]  # error 'size' must be a 1-D int32 Tensor
    if include_raw_images:
      image_raw = tf.expand_dims(image_raw, 0)
      image_raw = tf.image.resize_images(image_raw, [height, width])
      image_raw = tf.squeeze(image_raw)  

      # Batch it up.
      images, images_raw, labels = tf.train.batch(
            [image, image_raw, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=2 * batch_size)
      return images, images_raw, labels

    else:
      # Batch it up. Save resources by ignoring image_raw
      images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=1,
            capacity=2 * batch_size)
      return images, labels
    