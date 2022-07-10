import sys

from tensorflow.keras import backend
from tensorflow.keras.callbacks import Callback


class LearningRateScheduler(Callback):

    def __init__(self, schedule, verbose=0):

        super(LearningRateScheduler, self).__init__()

        self.schedule = schedule
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):

        generator_lr = float(backend.get_value(self.model.generator_optimizer.lr))
        discriminator_lr = float(backend.get_value(self.model.discriminator_optimizer.lr))

        generator_lr = self.schedule(epoch, generator_lr)
        discriminator_lr = self.schedule(epoch, discriminator_lr)

        backend.set_value(self.model.generator_optimizer.lr, backend.get_value(generator_lr))
        backend.set_value(self.model.discriminator_optimizer.lr, backend.get_value(discriminator_lr))

        if self.verbose > 0:

            sys.stdout.write(
                f'\nEpoch {epoch + 1}: LearningRateScheduler setting learning ' f'generator_rate to {generator_lr} '
                f'| discriminator_rate to {discriminator_lr}.\n')

    def on_epoch_end(self, epoch, logs=None):

        logs = logs or {}

        logs['generator_lr'] = backend.get_value(self.model.generator_optimizer.lr)
        logs['discriminator_lr'] = backend.get_value(self.model.discriminator_optimizer.lr)
