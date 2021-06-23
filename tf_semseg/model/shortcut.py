import tensorflow as tf
from . import config
from .util import *

def add(dest, src, stride=1, activation=True, name=None, config=config.Config()):
    src_channels = src.get_shape()[-1]
    dest_channels = dest.get_shape()[-1]

    if src_channels != dest_channels or stride > 1:
        src = config.conv(src, dest_channels, kernel_size=1, strides=stride, dilation_rate=1, use_bias=False, padding="same", name=join(name, "conv"))
        src = config.norm(src, name=join(name, "norm"))
        if activation:
            src = config.act(src)

    return src + dest

def concat(dest, src, stride=1, activation=True, name=None, config=config.Config()):
    if stride > 1:
        src = config.conv(src, kernel_size=1, strides=stride, dilation_rate=1, use_bias=False, padding="same", name=join(name, "conv"))
        src = config.norm(src, name=join(name, "norm"))
        if activation:
            src = config.act(src)

    return tf.concat([dest, src], axis=-1)
