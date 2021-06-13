from .util import *
from . import config
import numpy as np
import tensorflow as tf

def interpolate_block(x, level, resize_method, filters, name=None, config=config.Config()):
    orig_x = x

    x = tf.image.resize(x, tf.stack([level, level]), method="area")
    x = config.conv(x, filters, kernel_size=1, strides=1, name=join(name, "conv"), use_bias=False)
    x = config.norm(x, name=join(name, "norm"))
    x = config.act(x)
    x = tf.image.resize(x, tf.shape(orig_x)[1:-1], method=resize_method)

    return x

def psp(x, resize_method="bilinear", filters=None, name="psp", bin_sizes=[6, 3, 2, 1], config=config.Config()):
    if filters is None:
        filters = x.shape[-1] // len(bin_sizes)

    x = tf.keras.layers.Concatenate()([x] + [interpolate_block(x, bin_size, resize_method, filters=filters, name=join(name, f"pool{bin_size}"), config=config) for bin_size in bin_sizes])

    return x
