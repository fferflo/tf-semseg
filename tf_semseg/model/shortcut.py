import tensorflow as tf
from . import config
from .util import *

def add(dest, src, stride=1, activation=True, name=None, config=config.Config()):
    src_channels = src.get_shape()[-1]
    dest_channels = dest.get_shape()[-1]

    if src_channels != dest_channels or stride > 1:
        src = conv(src, dest_channels, kernel_size=1, stride=stride, bias=False, name=join(name, "conv"), config=config)
        src = norm(src, name=join(name, "norm"), config=config)
        if activation:
            src = act(src, config=config)

    return src + dest

def concat(dest, src, stride=1, activation=True, name=None, config=config.Config()):
    if stride > 1:
        src = conv(src, kernel_size=1, stride=stride, bias=False, name=join(name, "conv"), config=config)
        src = norm(src, name=join(name, "norm"), config=config)
        if activation:
            src = act(src, config=config)

    return tf.concat([dest, src], axis=-1)
