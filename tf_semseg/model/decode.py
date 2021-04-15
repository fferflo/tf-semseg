import tensorflow as tf
from .util import *

def decode(x, filters, name="decode"):
    x = Conv(filters, kernel_size=1, strides=1, name=join(name, "conv"), use_bias=True)(x)
    return x

def decode_resize(x, filters, shape, name="decode"):
    x = Conv(filters, kernel_size=1, strides=1, name=join(name, "conv"), use_bias=True)(x)
    x = tf.image.resize(x, shape[1:-1])
    return x
