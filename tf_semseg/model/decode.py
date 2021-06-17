import tensorflow as tf
from .util import *
from . import config

def decode(x, filters, shape=None, name="decode", use_bias=True, config=config.Config()):
    x = config.conv(x, filters, kernel_size=1, strides=1, name=join(name, "conv"), use_bias=use_bias)
    if not shape is None:
        x = config.resize(x, shape, method="bilinear")
    return x
