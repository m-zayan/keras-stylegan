import tensorflow as tf
from tensorflow.keras import backend


def set_precision(a, precision):

    factor = tf.cast(10 ** precision, dtype=tf.float32)

    return tf.round(a * factor) / factor

# -------------------------------------------------------------------------------------------------------------------


def log(a, base):

    base = tf.convert_to_tensor(base, dtype=tf.float32)

    return tf.math.log(a) / tf.math.log(base)

# -------------------------------------------------------------------------------------------------------------------


def check_frequency(boolean_tensor, threshold):

    a = tf.cast(boolean_tensor, dtype=tf.float32)
    b = tf.math.reduce_mean(a)
    c = tf.greater_equal(b, threshold)

    return tf.reduce_all(c)

# -------------------------------------------------------------------------------------------------------------------


def check_bounds(a, l, r, threshold=0.8):

    x = tf.greater_equal(a, l)
    y = tf.less_equal(a, r)
    z = tf.logical_and(x, y)

    return check_frequency(z, threshold=threshold)

# -------------------------------------------------------------------------------------------------------------------


def compare(a, l, r, threshold=0.5):

    x = tf.math.abs(a - l)
    y = tf.math.abs(r - a)

    a = tf.greater_equal(y, x)
    a = check_frequency(a, threshold=threshold)

    ret = tf.cond(a, lambda: l, lambda: r)

    return ret

# -------------------------------------------------------------------------------------------------------------------


def estimate_local_level(x):

    epsilon = backend.epsilon()

    a = tf.maximum(epsilon, tf.math.abs(x))
    a = tf.math.ceil(log(a, 10))
    a = tf.cast(a, dtype=tf.int32)

    return a

# -------------------------------------------------------------------------------------------------------------------


def get_level_scale(level):

    e = tf.cast(level, dtype=tf.float32)
    x = tf.cast(10.0, dtype=tf.float32)

    a = tf.pow(x, e)
    b = tf.cast(1.0, dtype=tf.float32)

    scale = tf.divide(b, a)

    return scale


# -------------------------------------------------------------------------------------------------------------------

def get_dynamic_level(x, min_level=-1, max_level=1):

    local_level = estimate_local_level(x)

    level = compare(local_level, min_level, max_level)
    level = tf.cast(local_level - level, dtype=tf.float32)

    def inner():

        scale = get_level_scale(tf.zeros_like(level))

        return scale

    def outer():

        scale = get_level_scale(level)

        return scale

    inside = check_bounds(local_level, min_level, max_level)

    scale = tf.cond(inside, inner, outer)
    scale = tf.stop_gradient(scale)

    return scale


# -------------------------------------------------------------------------------------------------------------------

def set_dynamic_level(x, min_level=-1, max_level=1):

    scale = get_dynamic_level(x, min_level, max_level)

    return scale * x
