import tensorflow as tf
from .util import *
from . import config

def decode(x, filters, name="decode", config=config.Config()):
    x = config.conv(x, filters, kernel_size=1, strides=1, name=join(name, "conv"), use_bias=True)
    return x

def decode_resize(x, filters, shape, name="decode", config=config.Config()):
    x = config.conv(x, filters, kernel_size=1, strides=1, name=join(name, "conv"), use_bias=True)
    x = tf.image.resize(x, shape)
    return x
