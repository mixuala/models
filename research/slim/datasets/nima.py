
# from slim_walkthrough
import tensorflow as tf
from preprocessing import preprocessing_factory
from tensorflow.contrib import slim

nima_preprocessing = preprocessing_factory.get_preprocessing('nima')

def load_batch(dataset, batch_size=32, height=224, width=224, 
            is_training=False, 
            resized=True,
            model="vgg16",
            label_name="ratings"):
    """Loads a single batch of data.
    
    Args:
      dataset: The dataset to load.
      batch_size: The number of images in the batch.
      height: The size of each image after preprocessing.
      width: The size of each image after preprocessing.
      is_training: Whether or not we're currently training or evaluating.
      resized: Whether the TFRecords were converted with images already resized to (256,256,3)
    
    Returns:
      images: A Tensor of size [batch_size, height, width, 3], image samples that have been preprocessed.
      images_raw: A Tensor of size [batch_size, height, width, 3], image samples that can be used for visualization.
      labels: A Tensor of size [batch_size], whose values range between 0 and dataset.num_classes.
    """
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, common_queue_capacity=32,
        common_queue_min=8)
    image_raw, label = data_provider.get(['image', label_name])
    
    # Preprocess image for usage by the appropriate model.
    image = {
      'vgg16': nima_preprocessing(image_raw, height, width, is_training=is_training,
                              resized=resized),
      'inception': None,
      'mobilenet': None,
    }[model]

        
    # Preprocess the image for display purposes.
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