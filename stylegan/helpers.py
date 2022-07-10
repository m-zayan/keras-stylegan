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


def check_bounds(a, l, r):

    x = tf.greater_equal(a, l)
    y = tf.less_equal(a, r)
    z = tf.logical_and(x, y)

    return tf.math.reduce_all(z)

# -------------------------------------------------------------------------------------------------------------------


def compare(a, l, r):

    x = tf.math.abs(a - l)
    y = tf.math.abs(r - a)

    ret = tf.cond(tf.greater_equal(y, x), lambda: l, lambda: r)

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

    scale = 1.0 / 10 ** level

    return scale


# -------------------------------------------------------------------------------------------------------------------

def set_dynamic_level(x, min_level=-1, max_level=1):

    local_level = estimate_local_level(x)

    def inner():

        scale = tf.cast(1.0, dtype=tf.float32)

        return scale

    def outer():

        level = compare(local_level, min_level, max_level)
        level = tf.cast(local_level - level, dtype=tf.float32)
        scale = get_level_scale(level)

        return scale

    inside = check_bounds(local_level, min_level, max_level)

    scale = tf.cond(inside, inner, outer)
    scale = tf.stop_gradient(scale)

    return scale * x
