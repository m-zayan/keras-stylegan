import tensorflow as tf
from tensorflow.keras import backend

from . import helpers


# -------------------------------------------------------------------------------------------------------------------

def kl_divergence_loss(target, predicted):

    epsilon = backend.epsilon()

    a = tf.math.log(tf.minimum(1.0, target + epsilon))
    b = tf.math.log(tf.minimum(1.0, predicted + epsilon))

    loss = target * (a - b)

    return loss

# -------------------------------------------------------------------------------------------------------------------


# noinspection PyUnusedLocal
def generator_logistic_loss(real_labels, fake_labels, real_scores, fake_scores,
                            penalty=0.0, min_level=-2, max_level=0):

    p1 = tf.nn.sigmoid(fake_labels * fake_scores + penalty)
    p2 = tf.nn.sigmoid(fake_labels * fake_scores - penalty)

    loss = kl_divergence_loss(1.0 - p1, p2)
    loss = tf.math.reduce_mean(loss, axis=0)

    return helpers.set_dynamic_level(loss, min_level=min_level, max_level=max_level)


# noinspection PyUnusedLocal
def discriminator_logistic_loss(real_labels, fake_labels, real_scores, fake_scores,
                                penalty=0.0, min_level=-2, max_level=0):

    epsilon = backend.epsilon()

    p_real = tf.nn.sigmoid(real_labels * real_scores - penalty)
    p_fake = tf.nn.sigmoid(fake_labels * fake_scores + penalty)

    min_term = -tf.math.log((tf.minimum(1.0, (1.0 - p_fake) + epsilon)))
    max_term = -tf.math.log(tf.minimum(1.0, p_real + epsilon))

    loss = min_term + max_term
    loss = tf.math.reduce_mean(loss, axis=0)

    return helpers.set_dynamic_level(loss, min_level=min_level, max_level=max_level)

# -------------------------------------------------------------------------------------------------------------------


# noinspection PyUnusedLocal
def generator_wgan_loss(real_labels, fake_labels, real_scores, fake_scores, min_level=-2, max_level=0):

    max_term = -(fake_labels * fake_scores)

    loss = tf.math.reduce_mean(max_term, axis=0)

    return helpers.set_dynamic_level(loss, min_level=min_level, max_level=max_level)


# noinspection PyUnusedLocal
def discriminator_wgan_loss(real_labels, fake_labels, real_scores, fake_scores,
                            wgan_epsilon=1e-4, min_level=-2, max_level=0):

    min_term = fake_labels * fake_scores
    max_term = -(real_labels * real_scores)

    epsilon_penalty = wgan_epsilon * ((real_labels * real_scores) ** 2.0)

    loss = (min_term + max_term) + epsilon_penalty
    loss = tf.math.reduce_mean(loss, axis=0)

    return helpers.set_dynamic_level(loss, min_level=min_level, max_level=max_level)


# -------------------------------------------------------------------------------------------------------------------

def wgan_gradient_penalty(mixed_gradients, wgan_target=1.0, wgan_lambda=10.0, min_level=-2, max_level=0):

    slopes_norms = tf.math.reduce_sum(mixed_gradients ** 2.0, axis=[1, 2, 3], keepdims=True)
    slopes_norms = tf.sqrt(slopes_norms)

    penalty = (slopes_norms - wgan_target) ** 2.0
    penalty *= (wgan_lambda / wgan_target ** 2.0)

    penalty = tf.math.reduce_mean(penalty)

    return helpers.set_dynamic_level(penalty, min_level=min_level, max_level=max_level)


# -------------------------------------------------------------------------------------------------------------------

# noinspection PyUnusedLocal
def ada_state_estimator(real_labels, fake_labels, real_scores, fake_scores):

    return tf.math.reduce_min(real_labels * real_scores)
