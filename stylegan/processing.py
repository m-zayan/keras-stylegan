import tensorflow as tf

from . import helpers


def normalize_in_range(tensor, low=0.0, high=255.0, axis=None):

    min_value = tf.math.reduce_min(tensor, axis=axis, keepdims=True)
    max_value = tf.math.reduce_max(tensor, axis=axis, keepdims=True)

    normalized = (tensor - min_value) / max_value

    output = high * normalized + low

    return output


def normalize(images):

    images = tf.clip_by_value(images, clip_value_min=0.0, clip_value_max=255.0)
    images = (images - 127.5) / 127.5

    return images


def denormalize(images):

    images = (images * 127.5) + 127.5
    images = tf.clip_by_value(images, clip_value_min=0.0, clip_value_max=255.0)

    return images


def sample_latent(size, latent_shape):

    shape = (size, *latent_shape)
    values = tf.random.normal(shape=shape, mean=0.0, stddev=1.0)

    return values


def sample_pseudo_labels(size, num_classes):

    n = tf.maximum(1, num_classes)
    shape = (size, )

    values = tf.random.uniform(shape=shape, minval=0, maxval=n, dtype=tf.int32)
    values = tf.one_hot(values, depth=n)

    return values


def sample_adversarial_labels(size, num_classes):

    n = tf.maximum(1, num_classes)
    shape = (size, n)

    values = tf.ones(shape=shape, dtype=tf.float32)

    return values


def random_uniform_state(shape=None, precision=2):

    if shape is None:

        shape = ()

    probability = tf.random.uniform(shape=shape, minval=0, maxval=1.0, dtype=tf.float32)

    return helpers.set_precision(probability, precision)
