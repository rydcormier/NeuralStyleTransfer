# nst_utils.py
#
# Utilities for Nerural Style Transfer
 
import numpy as np
import PIL.Image
import tensorflow as tf
import matplotlib.pyplot as plt



OUTPUT_PATH     = 'output/'
STYLE_IMAGE     = 'images/drop-of-water.jpg'
CONTENT_IMAGE   = 'images/persian_cat.jpg'
MAX_DIM         = 512
CHANNEL_COUNT   = 3
NOISE_RATIO     = 0.3

# Layers for computing loss
CONTENT_LAYERS  = ['block5_conv2']
STYLE_LAYERS    = ['block1_conv1',
                   'block2_conv1',
                   'block3_conv1',
                   'block4_conv1',
                   'block5_conv1']
STYLE_WEIGHTS   = [ 1.0, 1.0, 1.0, 1.0, 1.0 ]
# content cost weight
ALPHA           = 1e-2
# style cost weight
BETA            = 1e4

# Adam parameters
LEARNING_RATE   = 0.02
BETA_1          = 0.99
EPSILON         = 1e-1


def tensor_to_image(tensor):
    """Transform a tensor to an image and return."""
    # convert from float image
    tensor = tensor * 255
    tensor = np.array(tensor, dtype='uint8')
    # remove extra dimension if necessary
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

def generate_noise_image(content_image, noise_ratio = NOISE_RATIO):
    """
    Generates a noisy image by adding random noise to the content_image
    """
    # get shape
    shape = content_image.get_shape().as_list()

    # Generate a random noise_image
    noise_image = np.random.uniform(-20, 20, shape).astype('float32')

    # Set the input_image to be a weighted average of the content_image and a noise_image
    input_image = noise_image * noise_ratio + content_image * (1 - noise_ratio)

    return input_image



def clip_0_1(image):
    """Keep the values between 0 and 1."""
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def load_img(img_path, max_dim=MAX_DIM):
    """Load an image from the given path and format for processing."""
    # read the file and convert to float image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=CHANNEL_COUNT)
    img = tf.image.convert_image_dtype(img, tf.float32)

    # scale to fit
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    # resize and make dims compatible for the model
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]

    return img


def save_img(path, image_data):
    """Save the image data (float image) to path."""
    # check dimensions
    if len(image_data.get_shape()) > 3:
        assert image_data.get_shape()[0] == 1
        image_data = tf.squeeze(image_data, axis=0)
    # cast to uint8 on [0, 255]
    image = tf.cast(image_data * 255, dtype='uint8')
    # encode jpg
    image = tf.io.encode_jpeg(image)
    # write to path
    tf.io.write_file(path, image)


def imshow(image, title=None):
    """Display an image."""
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)
