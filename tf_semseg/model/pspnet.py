from .util import *
from . import resnet

import numpy as np
import tensorflow as tf
import sys, os, re, math

def interpolate_block(x, level, norm=default_norm, name=None):
    orig_x = x

    pool_size = tuple([math.ceil(x.shape[d] / level) for d in range(1, len(x.shape) - 1)])

    x = AveragePool(pool_size=pool_size, strides=pool_size, padding="same")(x)
    x = Conv(512, kernel_size=1, strides=1, name=join(name, "conv"), use_bias=False)(x)
    x = norm(x, name=join(name, "norm"))
    x = tf.keras.layers.ReLU()(x)

    x = tf.image.resize(x, orig_x.shape[1:-1])

    return x

def pspnet(x, name="psp", bin_sizes=[6, 3, 2, 1], norm=default_norm):
    if len(set(x.shape[1:-1])) != 1:
        print("WARNING: Got non-square input shape " + str(x.shape) + " for PSPNet")

    x = tf.keras.layers.Concatenate()([x] + [interpolate_block(x, bin_size, name=join(name, "pool" + str(bin_size))) for bin_size in bin_sizes])

    x = Conv(512, kernel_size=3, strides=1, padding="same", name=join(name, "final_conv"), use_bias=False)(x)
    x = norm(x, name=join(name, "final_norm"))
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.Dropout(0.1)(x) # TODO: where to set dropout parameter

    return x
