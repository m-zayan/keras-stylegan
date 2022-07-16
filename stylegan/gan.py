import copy
import math

import numpy as np

import tensorflow as tf
from tensorflow.keras import layers, models

from . import layers as gan_layers
from . import models

# --------------------------------------------------------------------------------------------------------------------

MIN_FILTERS = 1
MAX_FILTERS = 256

# --------------------------------------------------------------------------------------------------------------------

MAPPING_CFGS = {'latent_dim': 256, 'num_clusters': None, 'normalized': True, 'depth': 4,
                'disentangled_latent_dim': 256, 'activation': layers.LeakyReLU(alpha=0.2),
                'gain': np.sqrt(2.0), 'learning_rate_multiplier': 1.0}

# --------------------------------------------------------------------------------------------------------------------

SYNTHESIS_CFGS = {'resolution': 256, 'low_resolution': 4, 'num_channels': 3, 'in_filters': 8192, 'in_decay': 1.0,
                  'activation': layers.LeakyReLU(alpha=0.2), 'gain': np.sqrt(2.0), 'learning_rate_multiplier': 1.0,
                  'constant_input_dim': 256, 'use_skip': True, 'fused': True}

# --------------------------------------------------------------------------------------------------------------------

DISCRIMINATOR_CFGS = {'resolution': 256, 'low_resolution': 4, 'num_channels': 3, 'num_clusters': 1,
                      'in_filters': 4096, 'in_decay': 1.0, 'activation': layers.LeakyReLU(alpha=0.2),
                      'gain': np.sqrt(2.0), 'learning_rate_multiplier': 1.0,  'std_group_size': 4, 'use_skip': True}

# --------------------------------------------------------------------------------------------------------------------


def set_scope(prefix):

    def get_block_name(level):

        block_name = f'{2**level}x{2**level}'

        return f'{prefix}/{block_name}' if prefix else block_name

    return get_block_name

# --------------------------------------------------------------------------------------------------------------------


def get_num_filters(in_filters, in_decay, level):

    num_filters = in_filters / (2 ** (in_decay * level))

    return int(np.clip(num_filters, a_min=MIN_FILTERS, a_max=MAX_FILTERS))


# --------------------------------------------------------------------------------------------------------------------


def upsampling2d(inputs, factor, name=None):

    x = gan_layers.Resample(factor=factor, name=name + '/upsample' if name else None)(inputs)

    return x

# --------------------------------------------------------------------------------------------------------------------


def downsampling2d(inputs, factor, name=None):

    fan_in = inputs.shape[-1]

    x = gan_layers.PixelShuffle(factor=-factor, filters=fan_in, name=name + '/downsample' if name else None)(inputs)

    return x

# --------------------------------------------------------------------------------------------------------------------


def make_nd(dense_output, size=(1, 1), name=None):

    k = np.prod(size)

    assert dense_output.shape[-1] % k == 0, 'Invalid size, make_nd(...)'

    return layers.Reshape(target_shape=(*size, dense_output.shape[-1] // k),
                          name=name + f'/{len(size)}d' if name else None)(dense_output)

# --------------------------------------------------------------------------------------------------------------------


def gather_dim1(inputs, index, keepdims=False, name=None):

    """ Note:
        tf.gather(...) is not stable in TPU for tensorflow older versions.
    """

    def _gather(x):

        xi = x[:, index, ...]

        if keepdims:

            xi = tf.expand_dims(xi, axis=1)

        return xi

    yi = layers.Lambda(_gather, name=f'{name}/w{index}' if name else None)(inputs)

    return yi


# --------------------------------------------------------------------------------------------------------------------


def mlp_block(inputs, units, activation, gain, learning_rate_multiplier, name=None):

    x = gan_layers.EqualizedDense(units, gain=gain, learning_rate_multiplier=learning_rate_multiplier,
                                  kernel_initializer='random_normal', use_bias=True,
                                  name=name + '/z' if name else None)(inputs)

    if (activation is not None) and (activation != 'linear'):

        x = layers.Activation(activation, name=name + '/act' if name else None)(x)

    return x

# --------------------------------------------------------------------------------------------------------------------


def conv2d_block(inputs, filters, kernel_size, strides, gain, learning_rate_multiplier, use_bias=True,
                 activation=None, factor=None, normalized=False, transposed=False, name=None):

    x = inputs

    # [Upsampling]
    if (factor is not None) and (factor > 0):

        x = upsampling2d(x, factor, name=name)

    # =================================================================================================================

    # [Resampled]
    x = gan_layers.EqualizedConv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same', gain=gain,
                                   learning_rate_multiplier=learning_rate_multiplier,
                                   kernel_initializer='random_normal', use_bias=use_bias,
                                   normalized=normalized, transposed=transposed,
                                   name=name + '/econv2d' if name else None)(x)

    if (activation is not None) and (activation != 'linear'):

        x = layers.Activation(activation, name=name + '/conv2d/act' if name else None)(x)

    # =================================================================================================================

    # [Downsampling]
    if (factor is not None) and (factor < 0):

        x = downsampling2d(x, abs(factor), name=name)

    return x

# --------------------------------------------------------------------------------------------------------------------


def style_modulation_block(inputs, disentangled_latent_inputs, index, filters, kernel_size, strides, gain,
                           learning_rate_multiplier, normalized=False, transposed=False, demodulate=True,
                           fused=False, use_bias=False, name=None):

    wi = gather_dim1(disentangled_latent_inputs, index=index, keepdims=False, name=name)

    x = gan_layers.StyleModulation2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same',
                                     gain=gain, learning_rate_multiplier=learning_rate_multiplier,
                                     kernel_initializer='random_normal', use_bias=use_bias,
                                     normalized=normalized, transposed=transposed, demodulate=demodulate,
                                     fused=fused, name=name + '/stylemod' if name else None)(inputs, wi)

    return x

# --------------------------------------------------------------------------------------------------------------------


def to_rgb_block(inputs, num_channels, gain, learning_rate_multiplier, name=None):

    x = conv2d_block(inputs, filters=num_channels, kernel_size=(1, 1), strides=(1, 1), gain=gain,
                     activation=None, learning_rate_multiplier=learning_rate_multiplier, use_bias=True,
                     normalized=False, transposed=False, name=name + '/torgb' if name else None)
    return x


# --------------------------------------------------------------------------------------------------------------------

def from_rgb_block(inputs, dim, gain, learning_rate_multiplier, activation=None, name=None):

    x = conv2d_block(inputs, filters=dim, kernel_size=(1, 1), strides=(1, 1), activation=activation, gain=gain,
                     learning_rate_multiplier=learning_rate_multiplier, use_bias=False, factor=None,
                     normalized=False, transposed=False, name=name + '/fromrgb' if name else None)

    return x


# --------------------------------------------------------------------------------------------------------------------


def noise_block(inputs, activation, name=None):

    x = gan_layers.LearnableNoise(stddev=1.0, axis=-1, name=name + '/noise' if name else None)(inputs)
    x = gan_layers.AddBias(axis=-1, name=name + '/bias' if name else None)(x)
    x = layers.Activation(activation, name=name + '/act' if name else None)(x)

    return x

# --------------------------------------------------------------------------------------------------------------------


def synthesis_block(inputs, filters, kernel_size, gain, learning_rate_multiplier, disentangled_latent_inputs, index,
                    activation, level, fused=False):

    name = set_scope('g')(level) + '/'

    # [Upsampling]
    x = upsampling2d(inputs, factor=2, name=name)

    x = style_modulation_block(x, disentangled_latent_inputs, index, filters, kernel_size, (1, 1), gain,
                               learning_rate_multiplier, normalized=False, transposed=False, demodulate=True,
                               fused=fused, use_bias=False, name=name + str(1))

    x = noise_block(x, activation, name + str(1))

    x = style_modulation_block(x, disentangled_latent_inputs, index + 1, filters, kernel_size, (1, 1), gain,
                               learning_rate_multiplier, normalized=False, transposed=False, demodulate=True,
                               fused=fused, use_bias=False, name=name + str(2))

    x = noise_block(x, activation, name + str(2))

    return x

# --------------------------------------------------------------------------------------------------------------------


def encoding_block(inputs, in_filters, out_filters, kernel_size, gain,
                   learning_rate_multiplier, activation, level):

    name = set_scope('d')(level) + '/'

    x = conv2d_block(inputs, in_filters, kernel_size, (1, 1), gain, learning_rate_multiplier, use_bias=False,
                     activation=activation, factor=None, normalized=False, transposed=False, name=name + str(1))

    x = conv2d_block(x, out_filters, kernel_size, (1, 1), gain, learning_rate_multiplier, use_bias=False,
                     activation=activation, factor=None, normalized=False, transposed=False, name=name + str(2))

    # [Downsampling]
    x = downsampling2d(x, factor=2, name=name)

    return x

# --------------------------------------------------------------------------------------------------------------------


def build_mapping_network(latent_dim, num_clusters, normalized, depth,
                          disentangled_latent_dim, activation, gain,
                          learning_rate_multiplier, broadcast, name='mapping'):

    latent_inputs = layers.Input(shape=(latent_dim, ), name='latent_inputs')

    inputs = [latent_inputs]

    w = latent_inputs

    if normalized:

        w = gan_layers.RMSENormalization(axis=-1, name='rmsenorm')(w)

    if num_clusters is not None:

        scores_input = layers.Input(shape=(num_clusters, ), name='label_inputs', dtype=tf.float32)
        labels_embedding = layers.Dense(units=latent_dim, name='labels_embedding')(scores_input)

        w = layers.Concatenate(axis=-1, name='concat')([w, labels_embedding])

        inputs.append(scores_input)

    for i in range(depth):

        w = mlp_block(w, units=disentangled_latent_dim, activation=activation, gain=gain,
                      learning_rate_multiplier=learning_rate_multiplier, name=f'block{i + 1}')

    # [Broadcast]
    w = layers.RepeatVector(n=broadcast, name='w')(w)

    net = models.Model(inputs, w, name=name)

    return net


# --------------------------------------------------------------------------------------------------------------------


def build_synthesis_network(resolution, low_resolution, num_channels, in_filters, in_decay, activation, gain,
                            learning_rate_multiplier, mapping_broadcast, disentangled_latent_dim,
                            constant_input_dim, use_skip, fused, name='synthesis'):

    # ----------------------------------------------------------------------------------------------------------------

    get_block_name = set_scope('g')

    # ----------------------------------------------------------------------------------------------------------------

    begin_level = int(math.log(low_resolution, 2))
    end_level = int(math.log(resolution, 2))

    assert 2 ** begin_level == low_resolution, 'Invalid low_resolution, build_synthesis_network(...)'
    assert 2 ** end_level == resolution and resolution >= 4, 'Invalid resolution, build_synthesis_network(...)'

    assert mapping_broadcast == 2 * (end_level - begin_level + 1), \
        'Invalid mapping_broadcast, build_synthesis_network(...)'

    # =================================================================================================================

    disentangled_latent_inputs = layers.Input(shape=(mapping_broadcast, disentangled_latent_dim),
                                              name='disentangled_latent_inputs')

    # =================================================================================================================

    learnable_inputs = gan_layers.LearnableConstant(shape=(low_resolution, low_resolution, constant_input_dim),
                                                    name='constant_inputs')(disentangled_latent_inputs)

    # =================================================================================================================

    x = learnable_inputs
    y = None
    r = None

    # =================================================================================================================

    out_filters = get_num_filters(in_filters, in_decay, 0)

    # base synthesis block
    x = style_modulation_block(x, disentangled_latent_inputs, index=0, filters=out_filters, kernel_size=(3, 3),
                               strides=(1, 1), gain=gain, learning_rate_multiplier=learning_rate_multiplier,
                               normalized=False, transposed=False, fused=fused, name=get_block_name(begin_level))

    x = noise_block(x, activation, get_block_name(begin_level))

    if use_skip:

        # projection / intermediate-reconstruction
        y = to_rgb_block(x, num_channels, 1.0, learning_rate_multiplier, name=get_block_name(begin_level))

    # =================================================================================================================

    sub_block_index = 1

    # =================================================================================================================

    for i in range(begin_level, end_level):

        out_filters = get_num_filters(in_filters, in_decay, i)

        # =============================================================================================================

        x = synthesis_block(x, out_filters, (3, 3), gain, learning_rate_multiplier, disentangled_latent_inputs,
                            sub_block_index, activation, level=i+1, fused=fused)

        sub_block_index += 2

        # =============================================================================================================

        if use_skip:

            # resample
            r = conv2d_block(y, num_channels, (3, 3), (1, 1), gain, learning_rate_multiplier, use_bias=False,
                             activation=None, factor=2, normalized=False, transposed=False,
                             name=get_block_name(i + 1) + '/yi/r')

        if use_skip or i == end_level - 1:

            # projection / intermediate-reconstruction
            y = to_rgb_block(x, num_channels, 1.0, learning_rate_multiplier, name=get_block_name(i + 1))

        if use_skip:

            y = gan_layers.Residual(name=get_block_name(i + 1) + '/out')(y, r)

    # =================================================================================================================

    net = models.Model([disentangled_latent_inputs], [y], name=name)

    return net


# --------------------------------------------------------------------------------------------------------------------

def build_discriminator(resolution, low_resolution, num_channels, num_clusters, in_filters, in_decay,
                        activation, gain, learning_rate_multiplier, std_group_size, use_skip, name='discriminator'):

    # ----------------------------------------------------------------------------------------------------------------

    get_block_name = set_scope('d')

    # ----------------------------------------------------------------------------------------------------------------

    begin_level = int(math.log(low_resolution, 2))
    end_level = int(math.log(resolution, 2))

    assert 2 ** begin_level == low_resolution, 'Invalid low_resolution, build_discriminator(...)'
    assert 2 ** end_level == resolution and resolution >= 4, 'Invalid resolution, build_discriminator(...)'

    # =================================================================================================================

    image_inputs = layers.Input(shape=(resolution, resolution, num_channels), name='image_inputs')

    # =================================================================================================================

    x = None
    y = image_inputs
    r = None

    # =================================================================================================================

    for i in reversed(range(begin_level, end_level)):

        out_filters1 = get_num_filters(in_filters, in_decay, i - 1)
        out_filters2 = get_num_filters(in_filters, in_decay, i - 2)

        # =============================================================================================================

        if use_skip or i == end_level - 1:

            # projection / shallow-encoding
            r = from_rgb_block(y, out_filters1, gain, learning_rate_multiplier, activation=activation,
                               name=get_block_name(i + 1))

        elif not use_skip:

            r = x

        if use_skip and i != end_level - 1:

            r = gan_layers.Residual(name=get_block_name(i + 1) + '/fmap')(x, r)

        # =============================================================================================================

        x = encoding_block(r, out_filters1, out_filters2, (3, 3), gain, learning_rate_multiplier,
                           activation, level=i + 1)

        # =============================================================================================================

        if use_skip:

            # resample
            y = conv2d_block(y, filters=out_filters1, kernel_size=(3, 3), strides=(1, 1),
                             activation=activation, gain=1.0, learning_rate_multiplier=learning_rate_multiplier,
                             factor=-2, normalized=False, transposed=False, name=get_block_name(i + 1) + '/fmap/r')

    # =================================================================================================================

    out_filters1 = get_num_filters(in_filters, in_decay, 1)
    out_filters2 = get_num_filters(in_filters, in_decay, 0)

    if use_skip:

        # projection / shallow-encoding
        r = from_rgb_block(y, out_filters1, gain, learning_rate_multiplier, activation=activation,
                           name=get_block_name(begin_level))

        x = gan_layers.Residual(name=get_block_name(begin_level) + '/out')(x, r)

    x = gan_layers.MinibatchSTD(group_size=std_group_size, reduce_axes=[1, 2, 3])(x)

    x = conv2d_block(x, filters=out_filters1, kernel_size=(1, 1), strides=(1, 1), activation=activation, gain=gain,
                     learning_rate_multiplier=learning_rate_multiplier, factor=None,
                     normalized=False, transposed=False, name=get_block_name(begin_level) + '/fmap')

    # =================================================================================================================

    features = layers.Flatten(name='fmap_flatten')(x)

    features = mlp_block(features, units=out_filters2, activation=activation, gain=gain,
                         learning_rate_multiplier=learning_rate_multiplier, name='features')

    scores_output = mlp_block(features, units=num_clusters, activation=None, gain=gain,
                              learning_rate_multiplier=learning_rate_multiplier, name='scores_output')

    # =================================================================================================================

    net = models.Model(image_inputs, [scores_output], name=name)

    return net

# --------------------------------------------------------------------------------------------------------------------


def build_networks(mapping_cfgs=None, synthesis_cfgs=None, discriminator_cfgs=None):

    _mapping_cfgs = copy.deepcopy(MAPPING_CFGS)
    _synthesis_cfgs = copy.deepcopy(SYNTHESIS_CFGS)
    _discriminator_cfgs = copy.deepcopy(DISCRIMINATOR_CFGS)

    # =================================================================================================================

    if mapping_cfgs is not None:

        _mapping_cfgs.update(mapping_cfgs)

    if synthesis_cfgs is not None:

        _synthesis_cfgs.update(synthesis_cfgs)

    if discriminator_cfgs is not None:

        _discriminator_cfgs.update(discriminator_cfgs)

    # =================================================================================================================

    begin_level = int(math.log(_synthesis_cfgs['low_resolution'], 2))
    end_level = int(math.log(_synthesis_cfgs['resolution'], 2))

    mapping_broadcast = 2 * (end_level - begin_level + 1)

    # =================================================================================================================

    mapping_network = build_mapping_network(**_mapping_cfgs, broadcast=mapping_broadcast)

    synthesis_network = build_synthesis_network(**_synthesis_cfgs, mapping_broadcast=mapping_broadcast,
                                                disentangled_latent_dim=_mapping_cfgs['disentangled_latent_dim'])

    discriminator_network = build_discriminator(**_discriminator_cfgs)

    # =================================================================================================================

    return (mapping_network, synthesis_network, discriminator_network),\
           (_mapping_cfgs, _synthesis_cfgs, _discriminator_cfgs)


def build_stylegan(mapping_cfgs=None, synthesis_cfgs=None, discriminator_cfgs=None, **kwargs):

    (mapping, synthesis, discriminator), cfgs = build_networks(mapping_cfgs=mapping_cfgs, synthesis_cfgs=synthesis_cfgs,
                                                               discriminator_cfgs=discriminator_cfgs)

    latent_shape = (cfgs[0]['latent_dim'], 1)
    num_clusters = cfgs[-1]['num_clusters']

    model = models.StyleGAN(latent_shape=latent_shape, num_clusters=num_clusters, mapping=mapping,
                            synthesis=synthesis, discriminator=discriminator, **kwargs)

    return model
