import numpy as np

import tensorflow as tf
from tensorflow.keras import activations, backend
from tensorflow.keras.layers import Layer

from tensorflow.python.keras.utils import conv_utils

from . import processing


# --------------------------------------------------------------------------------------------------------------------

def get_num_gpus():

    return len(tf.config.list_physical_devices('GPU'))

# --------------------------------------------------------------------------------------------------------------------


def get_broadcastable_shape(input_shape, axis):

    axis = list(sorted(axis))

    rank = len(input_shape)
    shape = []

    last_axis = 0

    for i in range(len(axis)):

        if axis[i] < 0:

            axis[i] = rank + axis[i]

        # check if axis mismatches the inputs' rank
        assert 0 <= axis[i] < rank, f'axis out of range: [-{rank}, {rank}), axis={axis[i]}'

        for j in range(last_axis, axis[i] - 1):

            shape.append(1)

        shape.append(input_shape[axis[i]])
        last_axis = axis[i]

    for i in range(last_axis, rank - 1):

        shape.append(1)

    return axis, shape


# --------------------------------------------------------------------------------------------------------------------

def get_weights_scale(shape, gain=2.0, learning_rate_multiplier=1.0):

    # [..., out_dim]
    fan_in = np.prod(shape[:-1])

    # estimate stddev (i.e., HeNormal)
    he_stddev = gain / np.sqrt(fan_in)

    scale = he_stddev * learning_rate_multiplier

    return scale


# --------------------------------------------------------------------------------------------------------------------

def conv2d_output_shape(input_shape, filters, kernel_size, padding, strides):

    batch_size = input_shape[0]

    h, w = input_shape[1], input_shape[2]
    kh, kw = kernel_size[0], kernel_size[1]

    out_height = conv_utils.conv_output_length(h, kh, padding=padding.lower(), stride=strides[0], dilation=1)
    out_width = conv_utils.conv_output_length(w, kw, padding=padding.lower(), stride=strides[1], dilation=1)

    output_shape = (batch_size, out_height, out_width, filters)

    return output_shape

# --------------------------------------------------------------------------------------------------------------------


def deconv2d_output_shape(input_shape, filters, kernel_size, padding, strides):

    batch_size = input_shape[0]

    h, w = input_shape[1], input_shape[2]
    kh, kw = kernel_size[0], kernel_size[1]

    out_height = conv_utils.deconv_output_length(h, kh, padding=padding.lower(), output_padding=None,
                                                 stride=strides[0], dilation=1)

    out_width = conv_utils.deconv_output_length(w, kw, padding=padding.lower(), output_padding=None,
                                                stride=strides[1], dilation=1)

    output_shape = (batch_size, out_height, out_width, filters)

    return output_shape

# --------------------------------------------------------------------------------------------------------------------


class AddBias(Layer):

    def __init__(self, axis=None, **kwargs):

        super().__init__(**kwargs)

        if axis is None:

            axis = [-3, -2]

        elif isinstance(axis, int):

            axis = [axis]

        self.axis = axis

    def get_config(self):

        cfgs = super().get_config()
        cfgs.update({'axis': self.axis})

        return cfgs

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        super().build(input_shape)

        self.axis, shape = get_broadcastable_shape(input_shape, axis=self.axis)

        # initialize bias
        self.bias = self.add_weight(shape=shape, initializer='zeros', trainable=True, name='bias')

    def set_weights(self, weights):

        bias = tf.reshape(weights[0], shape=tf.shape(self.bias))

        self.bias.assign(bias)

    def call(self, inputs):

        return inputs + self.bias


# --------------------------------------------------------------------------------------------------------------------

class LearnableConstant(Layer):

    def __init__(self, shape, trainable=True, **kwargs):

        super().__init__(**kwargs)

        # [batch_size, ...]
        self.shape = (1, *shape)

        self.trainable = trainable

    def get_config(self):

        cfgs = super().get_config()
        cfgs.update({'shape': self.shape})

        return cfgs

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        super().build(input_shape)

        initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=1.0)

        # initialize
        self.constant = tf.Variable(initializer(self.shape), trainable=True, name='constant')

    def set_weights(self, weights):

        self.constant.assign(weights[0])

    def call(self, inputs, training=None):

        batch_size = tf.shape(inputs)[0]

        return tf.repeat(self.constant, batch_size, axis=0)

# --------------------------------------------------------------------------------------------------------------------


class LearnableNoise(Layer):

    def __init__(self, stddev=1.0, axis=None, training_only=False, **kwargs):

        super().__init__(**kwargs)

        if isinstance(axis, int):

            axis = [axis]

        self.stddev = stddev
        self.axis = axis

        self.training_only = training_only

    def get_config(self):

        cfgs = super().get_config()
        cfgs.update({'axis': self.axis, 'stddev': self.stddev, 'training_only': self.training_only})

        return cfgs

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        super().build(input_shape)

        if self.axis is None:

            shape = ()

        else:

            self.axis, shape = get_broadcastable_shape(input_shape, axis=self.axis)

        # initialize
        self.strength = self.add_weight(shape=shape, initializer='zeros', trainable=True, name='strength')

    def set_weights(self, weights):

        self.strength.assign(weights[0])

    def call(self, inputs, training=None):

        shape = tf.shape(inputs)[:-1]

        def add_noise():

            # [batch_size, h, w, 1]
            noise = backend.random_normal(shape=shape, mean=0.0, stddev=self.stddev, dtype=inputs.dtype)
            noise = tf.expand_dims(noise, axis=-1)

            return inputs + self.strength * noise

        if self.training_only:

            return backend.in_train_phase(add_noise, inputs, training=training)

        else:

            return add_noise()


# --------------------------------------------------------------------------------------------------------------------


class AdaIN(Layer):

    """
    Adaptive Instance Normalization (AdaIN):
        (https://arxiv.org/pdf/1703.06868v2.pdf)
    """

    def __init__(self, axis=None, **kwargs):

        super().__init__(**kwargs)

        if axis is None:

            axis = [1, 2]

        elif isinstance(axis, int):

            axis = [axis]

        self.axis = axis

    def get_config(self):

        cfgs = super().get_config()
        cfgs.update({'axis': self.axis})

        return cfgs

    def eval(self, x):

        epsilon = backend.epsilon()

        mean, variance = tf.nn.moments(x, axes=self.axis, keepdims=True)
        std = tf.sqrt(variance + epsilon)

        return mean, std

    def call(self, reference_features, target_features):

        """
        - reference_features = content_features
        - target_features = style_features
        """

        target_mean, target_std = self.eval(target_features)
        reference_mean, reference_std = self.eval(reference_features)

        features = (reference_features - reference_mean) / reference_std

        outputs = target_std * features + target_mean

        return outputs

# --------------------------------------------------------------------------------------------------------------------


class PixelMixer(Layer):

    def __init__(self, axis=None, values_range=(-1.0, 1.0), *args, **kwargs):

        super().__init__(*args, **kwargs)

        if axis is None:

            axis = [3]

        elif isinstance(axis, int):

            axis = [axis]

        self.axis = axis
        self.values_range = values_range

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        super().build(input_shape)

        self.axis, shape = get_broadcastable_shape(input_shape, axis=self.axis)
        self.mixture_shape = [1, ] * len(input_shape)

        # initialize rho
        self.rho: tf.Variable = self.add_weight(shape=shape, initializer='ones', trainable=self.trainable, name='rho')

    def get_config(self):

        cfgs = super().get_config()
        cfgs.update({'axis': self.axis, 'mixture_shape': self.mixture_shape})

        return cfgs

    def update_rho(self):

        self.rho.assign(tf.clip_by_value(self.rho, clip_value_min=-2.0, clip_value_max=2.0))

    def get_factors(self, reference_images):

        self.update_rho()

        batch_size = tf.shape(reference_images)[0]

        self.mixture_shape[0] = batch_size

        a = tf.random.uniform(shape=self.mixture_shape, minval=0.0, maxval=1.0)

        return a * tf.nn.sigmoid(self.rho)

    def call(self, reference_images, target_images):

        a = self.get_factors(reference_images)

        mixture = reference_images + a * (target_images - reference_images)
        mixture = tf.clip_by_value(mixture, clip_value_min=self.values_range[0], clip_value_max=self.values_range[1])

        return mixture

# --------------------------------------------------------------------------------------------------------------------


class RMSENormalization(Layer):

    """ Similar to local-response-normalization:
        (https://arxiv.org/abs/1710.10196)
    """

    def __init__(self, axis=-1, **kwargs):

        super().__init__(**kwargs)

        self.axis = axis

    def get_config(self):

        cfgs = super().get_config()
        cfgs.update({'axis': self.axis})

        return cfgs

    def call(self, inputs):

        epsilon = backend.epsilon()

        squared_mean = tf.math.reduce_mean(inputs ** 2.0, axis=self.axis, keepdims=True)
        outputs = inputs * tf.math.rsqrt(squared_mean + epsilon)

        return outputs

# --------------------------------------------------------------------------------------------------------------------


class InstantLayerNormalization(Layer):

    """
    InstantLayerNormalization:
        (https://arxiv.org/pdf/1607.08022.pdf)
    """

    def __init__(self, axis=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if axis is None:

            axis = [-1]

        elif isinstance(axis, int):

            axis = [axis]

        self.axis = axis

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        super().build(input_shape)

        self.axis, shape = get_broadcastable_shape(input_shape, axis=self.axis)

        # initialize gamma
        self.gamma = self.add_weight(shape=shape, initializer='ones', trainable=True, name='gamma')

        # initialize beta
        self.beta = self.add_weight(shape=shape, initializer='zeros', trainable=True, name='beta')

    def get_config(self):

        cfgs = super().get_config()
        cfgs.update({'axis': self.axis})

        return cfgs

    def call(self, inputs):

        epsilon = backend.epsilon()

        mean, variance = tf.nn.moments(inputs, axes=self.axis, keepdims=True)
        std = tf.math.sqrt(variance + epsilon)

        outputs = (inputs - mean) / std

        outputs = self.gamma * outputs + self.beta

        return outputs

# --------------------------------------------------------------------------------------------------------------------


class Truncation(Layer):

    def __init__(self, axis=None, threshold=0.7, beta=0.995, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if axis is None:

            axis = [1]

        elif isinstance(axis, int):

            axis = [axis]

        self.axis = axis

        self.threshold = threshold
        self.beta = beta

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        super().build(input_shape)

        self.axis, shape = get_broadcastable_shape(input_shape, axis=self.axis)

        # initialize
        self.average = tf.Variable(tf.zeros(shape=shape, dtype=tf.float32), trainable=False, name='average')

    def get_config(self):

        cfgs = super().get_config()
        cfgs.update({'axis': self.axis, 'threshold': self.threshold, 'beta': self.beta})

        return cfgs

    def update_average(self, current_average):

        self.average.assign(self.beta * tf.identity(self.average) + (1.0 - self.beta) * current_average)

    def call(self, inputs):

        average = tf.identity(self.average)
        outputs = average + (inputs - average) * self.threshold

        return outputs


# --------------------------------------------------------------------------------------------------------------------

class Blur(Layer):

    def __init__(self, kernel=None, strides=1, padding='same', normalized=True, **kwargs):

        super().__init__(**kwargs)

        if kernel is None:

            kernel = [1, 2, 1]

        if isinstance(strides, int):

            strides = (strides, strides)

        self.kernel = kernel
        self.strides = strides
        self.padding = padding.upper()

        self.normalized = normalized

    def get_config(self):

        cfgs = super().get_config()
        cfgs.update({'strides': self.strides, 'padding': self.padding, 'normalized': self.normalized})

        return cfgs

    def build(self, input_shape):

        super().build(input_shape)

        self.kernel = tf.cast(self.kernel, dtype=tf.float32)
        self.kernel = self.kernel[:, None] * self.kernel[None, :]
        self.kernel = self.kernel[:, :, None, None]

        if self.normalized:

            self.kernel = self.kernel / tf.reduce_sum(self.kernel)

        self.kernel = tf.tile(self.kernel, multiples=[1, 1, input_shape[-1], 1])

        self.kernel = tf.Variable(self.kernel, dtype=tf.float32, trainable=False, name='kernel')

    def call(self, inputs):

        outputs = tf.nn.depthwise_conv2d(inputs, self.kernel, strides=[1, 1, self.strides[0], self.strides[1]],
                                         padding=self.padding, data_format='NHWC')

        return outputs

# --------------------------------------------------------------------------------------------------------------------


class Group(Layer):

    def __init__(self, group_size, axis=0, *args, **kwargs):

        super(Group, self).__init__(*args, **kwargs)

        assert isinstance(axis, int), 'Invalid axis, GroupNormalization(...)'

        self.group_size = group_size
        self.axis = axis

    def get_config(self):

        cfgs = super(Group, self).get_config()
        cfgs.update({'group_size': self.group_size, 'axis': self.axis})

        return cfgs

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        super(Group, self).build(input_shape)

        self._axes = []

        rank = len(input_shape)

        if self.axis < 0:
            self.axis += rank

        k = 0

        for j in range(rank):

            if j == self.axis:

                k = 1
                self._axes.append(j)

            else:

                self._axes.append(j + k)

        self.shape_indicator = [0] * (rank + 1)

    def update_indicator(self, indices, values):

        for i in range(len(indices)):

            idx, val = indices[i], values[i]
            self.shape_indicator[idx] = val

    def update_group_shape(self, inputs):

        shape: tf.TensorShape = tf.shape(inputs)

        self.update_indicator(self._axes, shape)

        size = shape[self.axis]

        k = tf.minimum(size, self.group_size)
        self.update_indicator([self.axis, self.axis + 1], [k, size // k])

    def read_shape(self, axes=None):

        if axes is None:
            axes = self._axes

        indicator_slice = []

        for i in axes:
            indicator_slice.append(self.shape_indicator[i])

        return indicator_slice

    def call(self, inputs):

        self.update_group_shape(inputs)

        inputs = tf.reshape(inputs, shape=tf.identity(self.shape_indicator))

        return inputs


# --------------------------------------------------------------------------------------------------------------------


class MinibatchSTD(Layer):

    def __init__(self, group_size, reduce_axes=None, *args, **kwargs):

        super().__init__(*args, **kwargs)

        if reduce_axes is None:

            reduce_axes = [-3, -2, -1]

        self.group_size = group_size
        self.reduce_axes = reduce_axes

    def get_config(self):

        cfgs = super().get_config()
        cfgs.update({'axis': self.axis})

        return cfgs

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        super().build(input_shape)

        rank = len(input_shape)

        self._axes = [0] + [j + 1 for j in self.reduce_axes]

        self._split = Group(group_size=self.group_size, axis=0)
        self._split.build(input_shape)

        self.shape_indicator = [1] * (rank + 1)

    def update_indicator(self, indices, values):

        for i in range(len(indices)):

            idx, val = indices[i], values[i]
            self.shape_indicator[idx] = val

    def eval(self, x):

        epsilon = backend.epsilon()

        mean, variance = tf.nn.moments(x, axes=[0], keepdims=False)
        std = tf.sqrt(variance + epsilon)
        average_std = tf.reduce_mean(std, axis=self.reduce_axes, keepdims=True)
        average_std = tf.expand_dims(average_std, axis=0)

        # noinspection PyProtectedMember
        self.update_indicator(self._axes[:-1], self._split.read_shape(self._axes)[:-1])

        return average_std

    def call(self, inputs):

        groups = self._split(inputs)

        average_std = self.eval(groups)

        outputs = tf.tile(average_std, self.shape_indicator)
        outputs = tf.reshape(outputs, tf.shape(inputs)[:-1])
        outputs = tf.expand_dims(outputs, axis=-1)
        outputs = tf.concat([inputs, outputs], axis=-1)

        return outputs

# --------------------------------------------------------------------------------------------------------------------


class EqualizedConv2D(Layer):

    def __init__(self, filters, kernel_size, strides=(1, 1), padding='same', activation='linear',
                 kernel_initializer='random_normal', bias_initializer='zeros', use_bias=True, gain=1.0,
                 learning_rate_multiplier=1.0, normalized=False, transposed=False, **kwargs):

        super(EqualizedConv2D, self).__init__(**kwargs)

        assert kernel_size[0] >= 1 and kernel_size[0] % 2 == 1, f'Invalid kernel_size for EqualizedConv2D(...)'
        assert kernel_size[1] >= 1 and kernel_size[1] % 2 == 1, f'Invalid kernel_size for EqualizedConv2D(...)'

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = (strides, strides) if isinstance(strides, int) else strides
        self.padding = padding.upper()
        self.activation = activations.get(activation)

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.use_bias = use_bias

        self.gain = gain
        self.learning_rate_multiplier = learning_rate_multiplier

        self.normalized = normalized
        self.transposed = transposed

    def get_config(self):

        cfgs = super(EqualizedConv2D, self).get_config()

        cfgs.update({'filters': self.filters, 'kernel_size': self.kernel_size, 'strides': self.strides,
                     'padding': self.padding, 'activation': self.activation.__name__,
                     'kernel_initializer': self.kernel_initializer, 'bias_initializer': self.bias_initializer,
                     'use_bias': self.use_bias, 'gain': self.gain,
                     'learning_rate_multiplier': self.learning_rate_multiplier,
                     'normalized': self.normalized, 'transposed': self.transposed})

        return cfgs

    # noinspection PyAttributeOutsideInit
    def init_kernel(self, input_shape):

        in_dim, out_dim = input_shape[-1], self.filters

        kernel_shape = [*self.kernel_size, in_dim, out_dim]

        a = tf.math.reduce_max(tf.cast(self.strides, dtype=tf.float32)) ** 2.0

        self.kernel_scale = tf.Variable(a * get_weights_scale(kernel_shape, self.gain, self.learning_rate_multiplier),
                                        dtype=tf.float32, trainable=False, name='kernel_scale')

        self.bias_scale = tf.Variable(self.learning_rate_multiplier, dtype=tf.float32,
                                      trainable=False, name='bias_scale')

        if self.kernel_initializer is not None:

            if self.kernel_initializer == 'random_normal':

                stddev = (1.0 / self.learning_rate_multiplier)

                initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=stddev)
                self.kernel = initializer(kernel_shape)

            else:

                initializer = tf.keras.initializers.get(self.kernel_initializer)
                self.kernel = initializer(kernel_shape)

            self.kernel = tf.Variable(self.kernel, trainable=True, name='kernel')

            return

        # number of rows (fan_in) should be greater than or equal to the number of columns (filters)
        shape = (np.prod(kernel_shape[:-1]), out_dim)

        assert shape[0] >= shape[1], 'Invalid filters_dim for EqualizedConv2D(...)'

        initializer = tf.initializers.Orthogonal(gain=1.0 / self.learning_rate_multiplier)

        self.kernel = tf.reshape(initializer(shape=shape), shape=kernel_shape)
        self.kernel = tf.Variable(self.kernel, trainable=True, name='kernel')

    # noinspection PyAttributeOutsideInit
    def init_bias(self):

        shape = (self.filters, )

        self.bias = self.add_weight(shape=shape, initializer=self.bias_initializer, trainable=True, name='bias')

    def build(self, input_shape):

        super(EqualizedConv2D, self).build(input_shape)

        self.init_kernel(input_shape=input_shape)

        if self.use_bias:

            self.init_bias()

    def compute_output_shape(self, input_shape):

        if self.transposed:

            output_shape = deconv2d_output_shape(input_shape, self.filters, self.kernel_size,
                                                 self.padding, self.strides)

        else:

            output_shape = conv2d_output_shape(input_shape, self.filters, self.kernel_size,
                                               self.padding, self.strides)

        return output_shape

    def set_weights(self, weights):

        self.kernel.assign(weights[0])

        if self.use_bias and len(weights) > 1:

            self.bias.assign(weights[1])

    def adjusted_kernel(self):

        kernel = tf.identity(self.kernel)

        if self.normalized:

            kernel /= tf.reduce_sum(kernel)

        kernel *= self.kernel_scale

        return kernel

    def call(self, inputs, kernel=None):

        if kernel is None:

            kernel = self.adjusted_kernel()

        if self.transposed:

            output_shape = self.compute_output_shape(tf.shape(inputs))

            kernel = tf.transpose(kernel, perm=[0, 1, 3, 2])

            outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape=output_shape,
                                             strides=self.strides, padding=self.padding,
                                             data_format='NHWC', dilations=[1, 1, 1, 1], name=None)

            output_shape = self.compute_output_shape(inputs.shape)

            outputs = tf.ensure_shape(outputs, shape=output_shape)

        else:

            outputs = tf.nn.conv2d(inputs, kernel,
                                   strides=self.strides, padding=self.padding,
                                   data_format='NHWC', dilations=[1, 1, 1, 1], name=None)

        if self.use_bias:

            outputs += self.bias * tf.identity(self.bias_scale)

        if self.activation is not None:

            outputs = self.activation(outputs)

        return outputs

# --------------------------------------------------------------------------------------------------------------------


class EqualizedDense(Layer):

    def __init__(self, units, activation='linear', kernel_initializer='random_normal',
                 bias_initializer='zeros', use_bias=True, gain=1.0, learning_rate_multiplier=1.0,
                 normalized=False, **kwargs):

        super(EqualizedDense, self).__init__(**kwargs)

        self.units = units
        self.activation = activations.get(activation)

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

        self.use_bias = use_bias

        self.gain = gain
        self.learning_rate_multiplier = learning_rate_multiplier

        self.normalized = normalized

    def get_config(self):

        cfgs = super(EqualizedDense, self).get_config()

        cfgs.update({'units': self.units, 'activation': self.activation.__name__,
                     'kernel_initializer': self.kernel_initializer, 'bias_initializer': self.bias_initializer,
                     'use_bias': self.use_bias, 'gain': self.gain,
                     'learning_rate_multiplier': self.learning_rate_multiplier, 'normalized': self.normalized})

        return cfgs

    # noinspection PyAttributeOutsideInit
    def init_kernel(self, input_shape):

        in_dim, out_dim = input_shape[-1], self.units

        kernel_shape = [in_dim, out_dim]

        self.kernel_scale = tf.Variable(get_weights_scale(kernel_shape, self.gain, self.learning_rate_multiplier),
                                        dtype=tf.float32, trainable=False, name='kernel_scale')

        self.bias_scale = tf.Variable(self.learning_rate_multiplier, dtype=tf.float32,
                                      trainable=False, name='bias_scale')

        if self.kernel_initializer is not None:

            if self.kernel_initializer == 'random_normal':

                stddev = (1.0 / self.learning_rate_multiplier)

                initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=stddev)
                self.kernel = initializer(kernel_shape)

            else:

                initializer = tf.keras.initializers.get(self.kernel_initializer)
                self.kernel = initializer(kernel_shape)

            self.kernel = tf.Variable(self.kernel, trainable=True, name='kernel')

            return

        # number of rows (fan_in) should be greater than or equal to the number of columns (units)
        assert kernel_shape[0] >= kernel_shape[1], 'Invalid filters_dim for EqualizedDense(...)'

        initializer = tf.initializers.Orthogonal(gain=1.0 / self.learning_rate_multiplier)

        self.kernel = initializer(shape=kernel_shape)
        self.kernel = tf.Variable(self.kernel, trainable=True, name='kernel')

    # noinspection PyAttributeOutsideInit
    def init_bias(self):

        shape = (self.units, )

        self.bias = self.add_weight(shape=shape, initializer=self.bias_initializer, trainable=True, name='bias')

    def build(self, input_shape):

        super(EqualizedDense, self).build(input_shape)

        self.init_kernel(input_shape=input_shape)

        if self.use_bias:

            self.init_bias()

    def set_weights(self, weights):

        self.kernel.assign(weights[0])

        if self.use_bias and len(weights) > 1:

            self.bias.assign(weights[1])

    def adjusted_kernel(self):

        kernel = tf.identity(self.kernel)

        if self.normalized:

            kernel /= tf.reduce_sum(kernel)

        kernel *= self.kernel_scale

        return kernel

    def call(self, inputs):

        outputs = tf.matmul(inputs, self.kernel * tf.identity(self.kernel_scale))

        if self.use_bias:

            outputs += self.bias * tf.identity(self.bias_scale)

        if self.activation is not None:

            outputs = self.activation(outputs)

        return outputs

# --------------------------------------------------------------------------------------------------------------------


class StyleModulation2D(EqualizedConv2D):

    def __init__(self, filters, kernel_size, strides=1, padding='same', activation='linear',
                 kernel_initializer='random_normal', bias_initializer='zeros', use_bias=False, gain=1.0,
                 learning_rate_multiplier=1.0, normalized=False, transposed=False,
                 demodulate=True, fused=False, **kwargs):

        super(StyleModulation2D, self).__init__(filters=filters, kernel_size=kernel_size, strides=strides,
                                                padding=padding, activation=activation,
                                                kernel_initializer=kernel_initializer,
                                                bias_initializer=bias_initializer,
                                                use_bias=use_bias, gain=gain,
                                                learning_rate_multiplier=learning_rate_multiplier,
                                                normalized=normalized,
                                                transposed=transposed, **kwargs)

        self.demodulate = demodulate
        self.fused = fused

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        super(StyleModulation2D, self).build(input_shape)

        self.transformer = EqualizedDense(units=input_shape[-1], use_bias=True, activation='linear',
                                          bias_initializer='ones', name=self.name + '_affine')

    def compute(self, inputs, kernel):

        return super().call(inputs, kernel=kernel * self.kernel_scale)

    def fusion(self, inputs, kernel):

        output_shape = self.compute_output_shape(inputs.shape)

        def step(elems):

            yi = self.compute(tf.expand_dims(elems[0], axis=0), elems[1])

            return tf.squeeze(yi, axis=0)

        spec = tf.TensorSpec(shape=output_shape[1:], dtype=inputs.dtype)
        outputs = tf.map_fn(step, (inputs, kernel), fn_output_signature=spec)

        return outputs

    def modulate(self, inputs, style):

        style = self.transformer(style)

        if not self.fused:

            inputs *= style[:, None, None, :]

        return inputs, style

    # noinspection PyUnboundLocalVariable
    def call(self, inputs, style):

        epsilon = backend.epsilon()

        inputs, style = self.modulate(inputs, style)

        kernel = tf.identity(self.kernel)

        if self.fused or self.demodulate:

            modulation_kernel = kernel[None] * style[:, None, None, :, None]

        if self.demodulate:

            norm = tf.math.reduce_sum(modulation_kernel ** 2.0, axis=[1, 2, 3], keepdims=True)
            inverse_scale = tf.math.rsqrt(norm + epsilon)

            modulation_kernel = modulation_kernel * inverse_scale if self.fused else modulation_kernel

        if self.fused:

            outputs = self.fusion(inputs, modulation_kernel)

        else:

            outputs = self.compute(inputs, kernel)
            outputs = outputs * tf.squeeze(inverse_scale, axis=3) if self.demodulate else outputs

        return outputs

# --------------------------------------------------------------------------------------------------------------------


class AdaptiveRandomState(Layer):

    def __init__(self, fn, alter=None, initial_state=0.0, cycle_length=4,
                 state_gain=4.0, shape=None, on_batch=False, num_cycles=8, **kwargs):

        super(AdaptiveRandomState, self).__init__(**kwargs)

        self.fn = fn
        self.alter = alter if alter else lambda x: x

        self.initial_state = initial_state
        self.cycle_length = cycle_length
        self.state_gain = state_gain
        self.shape = shape
        self.on_batch = on_batch
        self.num_cycles = num_cycles

    def get_config(self):

        cfgs = super(AdaptiveRandomState, self).get_config()

        cfgs.update({'fn': self.fn, 'initial_state': self.initial_state,
                     'cycles': self.cycles, 'state_gain': self.state_gain})

        return cfgs

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        super(AdaptiveRandomState, self).build(input_shape)

        if self.shape is None:

            self.shape = input_shape[1:]

        self.current_state = tf.clip_by_value(self.initial_state, -1.0, 1.0)
        self.current_state = tf.Variable(self.current_state, dtype=tf.float32, trainable=True, name='state')

        self.current_step = tf.Variable(0, dtype=tf.int32, trainable=True, name='step')

        self.status = tf.Variable(0, dtype=tf.int32, trainable=True, name='status')

        # idle
        if self.fn is None:

            self.status.assign(-1)

    def reset(self):

        self.current_state.assign(tf.clip_by_value(self.initial_state, -1.0, 1.0))
        self.current_step.assign(0)

    def set_status(self, status):

        self.status.assign(status)

    def is_idle(self):

        return tf.equal(self.status, -1)

    def update_state(self, state, step_factor=1):

        step = tf.math.floormod(self.current_step, self.num_cycles * self.cycle_length)
        step += 1

        is_next = tf.reduce_all(tf.equal(0, tf.math.floormod(step, self.cycle_length)))
        is_next = tf.logical_and(tf.logical_not(self.is_idle()), is_next)

        def get_next_state(state):

            current_state = tf.identity(self.current_state)

            a = 1.0 / tf.cast(step * step_factor, dtype=tf.float32)

            next_state = (1.0 - a) * current_state + a * tf.nn.tanh(state)

            return next_state

        next_state = tf.cond(is_next, lambda: get_next_state(state), lambda: tf.identity(self.current_state))

        self.current_state.assign(next_state)
        self.current_step.assign(step)

    def state(self):

        return tf.nn.sigmoid(self.state_gain * self.current_state)

    def apply_fn(self, inputs):

        rnd_state = processing.random_uniform_state(precision=2)

        if tf.greater(self.state(), rnd_state):

            return self.fn(inputs)

        else:

            return self.alter(inputs)

    def call(self, inputs, training=None):

        if self.fn is None:

            return inputs

        if training:

            if self.on_batch:

                return self.apply_fn(inputs)

            spec = tf.TensorSpec(shape=self.shape, dtype=inputs.dtype)
            outputs = tf.map_fn(self.apply_fn, inputs, fn_output_signature=spec)

            return outputs

        else:

            if self.on_batch:

                return self.alter(inputs)

            spec = tf.TensorSpec(shape=self.shape, dtype=inputs.dtype)
            outputs = tf.map_fn(self.alter, inputs, fn_output_signature=spec)

            return outputs


# ------------------------------------------------------------------------------------------------------------------


class Residual(Layer):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

    def get_config(self):

        cfgs = super().get_config()

        return cfgs

    def call(self, x, y):

        # TODO: consider using helpers.set_dynamic_level(...) instead of tf.clip_by_value(...)
        outputs = tf.clip_by_value(x + y, clip_value_min=-1.0, clip_value_max=1.0)

        return outputs


# ------------------------------------------------------------------------------------------------------------------


class Resample(Layer):

    def __init__(self, factor, **kwargs):

        super().__init__(**kwargs)

        if isinstance(factor, int):

            factor = (factor, factor)

        assert (0 not in factor), f'Invalid factor={factor}, Resample(...)'

        self.factor = factor

    def get_config(self):

        cfgs = super().get_config()
        cfgs.update({'factor': self.factor})

        return cfgs

    # noinspection PyAttributeOutsideInit
    def build(self, input_shape):

        super().build(input_shape)

        height_scale = abs(self.factor[0])
        width_scale = abs(self.factor[1])

        if self.factor[0] < 0:

            height_scale = 1 / height_scale

        if self.factor[1] < 0:

            width_scale = 1 / width_scale

        self.in_height = input_shape[1]
        self.in_width = input_shape[2]

        self.out_height = int(height_scale * self.in_height)
        self.out_width = int(width_scale * self.in_width)

        self.num_channels = input_shape[-1]

    def resize_nearest(self, inputs):

        y1 = abs(min(1, self.factor[0]))
        x1 = abs(min(1, self.factor[1]))

        y2 = max(1, self.factor[0])
        x2 = max(1, self.factor[1])

        outputs = inputs[:, 0::y1, None, 0::x1, None, :]
        outputs = tf.tile(outputs, [1, 1, y2, 1, x2, 1])
        outputs = tf.reshape(outputs, (-1, self.out_height, self.out_width, self.num_channels))

        return outputs

    def call(self, inputs):

        outputs = self.resize_nearest(inputs)

        return outputs
