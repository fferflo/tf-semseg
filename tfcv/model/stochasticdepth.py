import tensorflow as tf
from . import config
from .util import *

# https://arxiv.org/abs/1603.09382
class DropPath(tf.keras.layers.Layer):
    # scale_at_train_time=True: implemented as in https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/drop.py
    # scale_at_train_time=False: implemented as in tfa.layers.StochasticDepth
    def __init__(self, drop_probability, scale_at_train_time=True, **kwargs):
        super().__init__(**kwargs)
        self.scale_at_train_time = scale_at_train_time
        self.drop_probability = drop_probability
        if drop_probability >= 1.0 or drop_probability < 0.0:
            raise ValueError(f"Invalid drop_probability, must be in range [0.0, 1.0)")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=False):
        if self.scale_at_train_time and training:
            x = x / (1 - self.drop_probability)
        elif not self.scale_at_train_time and not training:
            x = x * (1 - self.drop_probability)

        if training and self.drop_probability > 0:
            shape = tf.concat([[tf.shape(x)[0]], [1] * (len(x.shape) - 1)], axis=0) # Drop per instance in batch
            drop = tf.cast(tf.where(tf.random.uniform(shape=shape, minval=0.0, maxval=1.0) < self.drop_probability, 0.0, 1.0), x.dtype)
            x = x * drop

        return x

    def get_config(self):
        config = super().get_config()
        config["scale_at_train_time"] = self.scale_at_train_time
        config["drop_probability"] = self.drop_probability
        return config

def shortcut(shortcut, residual, drop_probability, scale_at_train_time=True, name=None, config=config.Config()):
    if shortcut.shape[-1] != residual.shape[-1]:
        raise ValueError("Tensors must have same number of channels")

    return shortcut + DropPath(drop_probability=drop_probability, scale_at_train_time=scale_at_train_time, name=join(name, "drop"))(residual)
