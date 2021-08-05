from .util import *
from . import config
import numpy as np
import tensorflow as tf

def interpolate_block(x, level, resize_method, filters, name=None, config=config.Config()):
    orig_x = x

    # Variant 1: Does not have gradients
    # x = resize(x, tf.stack([level, level]), method="area", config=config)

    # Variant 2: Not working, since pooling currently does not allow dynamic kernel sizes and strides
    # kernel_size = tf.cast(tf.math.ceil(tf.shape(x)[1:-1] / level), "int32")
    # x = pool(x, kernel_size=kernel_size, stride=kernel_size)

    # Variant 3: Might be slower than other variants
    def pool(x):
        for axis in range(1, len(x.shape) - 1):
            split_bounds = tf.cast(tf.range(0, level + 1, dtype="float32") / tf.cast(level, "float32") * tf.cast(tf.shape(x)[axis], "float32"), "int32")
            size_splits = split_bounds[1:] - split_bounds[:-1]
            splits = tf.split(x, size_splits, axis=axis, num=level)
            splits = [tf.reduce_mean(split, axis=axis) for split in splits]
            x = tf.stack(splits, axis=axis)
        return x
    x = tf.keras.layers.Lambda(pool, output_shape=tuple([None] + [level] * (len(x.shape) - 2) + [x.shape[-1]]))(x)

    x = conv_norm_act(x, filters=filters, kernel_size=1, stride=1, name=name, config=config)
    x = resize(x, tf.shape(orig_x)[1:-1], method=resize_method, config=config)

    return x

def psp(x, resize_method="bilinear", filters=None, name="psp", bin_sizes=[6, 3, 2, 1], config=config.Config()):
    if filters is None:
        filters = x.shape[-1] // len(bin_sizes)

    x = tf.concat([x] + [interpolate_block(x, bin_size, resize_method, filters=filters, name=join(name, f"pool{i + 1}"), config=config) for i, bin_size in enumerate(bin_sizes)], axis=-1)

    return x
