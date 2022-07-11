import numpy as np

import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.python.keras.engine import base_layer

from . import functional, processing
from . import layers as gan_layers


class StyleGAN(Model):
    
    def __init__(self, latent_shape, num_clusters, mapping: Model, synthesis: Model,
                 discriminator: Model, batch_size=None, use_pseudo_labels=False, augmenter=None,
                 ada_target=1.0, ada_step=100, on_batch_ada=False, ada_state_estimator=None,
                 recursive_lookup=False, **kwargs):

        super(StyleGAN, self).__init__(**kwargs)

        self.latent_shape = latent_shape
        self.num_clusters = num_clusters

        self.mapping: Model = mapping
        self.generator: Model = synthesis
        self.discriminator: Model = discriminator

        self.batch_size = batch_size
        self.use_pseudo_labels = use_pseudo_labels

        self._use_clusters_embedding = (len(self.mapping.inputs) == 2)

        self.truncation = gan_layers.Truncation(axis=[1], name='truncation')
        self.mixer = gan_layers.PixelMixer(axis=3, name='pixel_mixer')

        self.augmenter = gan_layers.AdaptiveRandomState(fn=augmenter, alter=None, initial_state=-1.0,
                                                        cycle_length=4, num_cycles=8, on_batch=on_batch_ada, name='ada')

        self.ada_target = ada_target
        self.ada_step = ada_step

        self.ada_state_estimator = ada_state_estimator if ada_state_estimator else functional.ada_state_estimator

        self.recursive_lookup = recursive_lookup

        self.build(input_shape=[])

    def build(self, input_shape):

        super().build(input_shape=input_shape)

        self.truncation.build([None, self.mapping.output_shape[-1]])
        self.mixer.build(self.generator.output_shape)
        self.augmenter.build(self.generator.output_shape)

        self.built = True

    @property
    def layers(self):

        return list(self._flatten_layers(include_self=False, recursive=self.recursive_lookup))

    def summary(self, line_length=None, positions=None, print_fn=None):

        super(StyleGAN, self).summary()

    # noinspection PyAttributeOutsideInit
    def compile(self, generator_optimizer='rmsprop', discriminator_optimizer='rmsprop',
                generator_loss=None, discriminator_loss=None, gradient_penalty='wgan', clip_min=None, clip_max=None,
                use_clip=False, run_eagerly=None, steps_per_execution=None, **kwargs):

        base_layer.keras_api_gauge.get_cell('compile').set(True)

        with self.distribute_strategy.scope():

            self.steps_per_execution = steps_per_execution

        self._run_eagerly = run_eagerly

        self.generator_optimizer = optimizers.get(generator_optimizer)
        self.discriminator_optimizer = optimizers.get(discriminator_optimizer)

        self.generator_loss = generator_loss if generator_loss else \
            functional.generator_wgan_loss

        self.discriminator_loss = discriminator_loss if discriminator_loss else \
            functional.discriminator_wgan_loss

        self.gradient_penalty = functional.wgan_gradient_penalty if (gradient_penalty == 'wgan') else \
            gradient_penalty

        self.clip_min = clip_min if (clip_min is not None) else -np.inf
        self.clip_max = clip_max if (clip_max is not None) else np.inf

        self.use_clip = use_clip

        self._configure_steps_per_execution(steps_per_execution or 1)

        # Initializes attrs that are reset each time `compile` is called.
        self._reset_compile_cache()
        self._is_compiled = True

    def call(self, inputs, training=None, mask=None):

        w = self.mapping(inputs, training=training)
        w = self.truncation(w)
        outputs = self.generator([w], training=training)

        return outputs

    def get_config(self):

        cfgs = super(StyleGAN, self).get_config()

        return cfgs

    def sample_latent(self, size):

        return processing.sample_latent(size, latent_shape=self.latent_shape)

    def sample_labels(self, size):

        if self.use_pseudo_labels:

            return processing.sample_pseudo_labels(size, num_classes=self.num_clusters)

        else:

            return processing.sample_adversarial_labels(size, num_classes=self.num_clusters)

    def train_step(self, data):

        real_images, real_labels = data

        batch_size = self.batch_size if self.batch_size else tf.shape(real_images)[0]

        fake_latent = self.sample_latent(batch_size)
        fake_labels = self.sample_labels(batch_size)

        real_images = self.augmenter(real_images, training=True)

        with tf.GradientTape(persistent=True) as tape, tf.GradientTape(persistent=False) as inner_tape:

            # [mapping] ===========================================================================================

            if not self._use_clusters_embedding:

                w = self.mapping([fake_latent], training=True)

            else:

                w = self.mapping([fake_latent, fake_labels], training=True)

            self.truncation.update_average(tf.math.reduce_mean(w[:, 0, :], axis=0))

            w = self.truncation(w)

            # [generator] ==========================================================================================

            fake_images = self.generator([w], training=True)
            fake_images = self.augmenter(fake_images, training=True)

            # [discriminator] ======================================================================================

            fake_scores = self.discriminator([fake_images], training=True)
            real_scores = self.discriminator([real_images], training=True)

            # [update states] ======================================================================================

            state = self.ada_state_estimator(real_labels, fake_labels, real_scores, fake_scores)
            self.augmenter.update_state(state - self.ada_target, step_factor=batch_size*self.ada_step)

            # [losses] =============================================================================================

            gloss = self.generator_loss(real_labels, fake_labels, real_scores, fake_scores)
            dloss = self.discriminator_loss(real_labels, fake_labels, real_scores, fake_scores)

            # [gradient penalty] ===================================================================================

            if self.gradient_penalty is not None:

                # interpolated images
                mixed_images = self.mixer(real_images, fake_images)

                mixed_scores = self.discriminator([mixed_images], training=True)
                mixed_gradients = inner_tape.gradient(tf.reduce_sum(mixed_scores), mixed_images)

                gradient_penalty = self.gradient_penalty(mixed_gradients)
                total_dloss = dloss + gradient_penalty

            else:

                gradient_penalty = tf.constant(0.0, dtype=tf.float32)
                total_dloss = dloss

        # [Update Generator] ======================================================================================

        trainable_vars = self.mapping.trainable_variables + self.generator.trainable_variables
        gradients = tape.gradient(gloss, trainable_vars)

        self.generator_optimizer.apply_gradients(zip(gradients, trainable_vars))

        # [Update Discriminator] ===================================================================================

        trainable_vars = self.discriminator.trainable_variables
        gradients = tape.gradient(total_dloss, trainable_vars)

        self.discriminator_optimizer.apply_gradients(zip(gradients, trainable_vars))

        # ==========================================================================================================

        if self.use_clip:

            for var in trainable_vars:

                var.assign(tf.clip_by_value(var, self.clip_min, self.clip_max))

        # ==========================================================================================================

        ret = {'loss': gloss + total_dloss, 'generator_loss': gloss,
               'discriminator_loss': dloss}

        if self.gradient_penalty is not None:

            ret.update({'gp': gradient_penalty})

        if self.augmenter.fn:

            ret.update({'augmenter_state': self.augmenter.state()})

        return ret

    def test_step(self, data):

        raise NotImplementedError('test_step(...) is not yet supported')
