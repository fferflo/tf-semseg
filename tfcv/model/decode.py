import tensorflow as tf
from .util import *
from . import config

def decode(x, filters, shape=None, dropout=None, name="decode", bias=True, config=config.Config()):
    if not dropout is None:
        x = tf.keras.layers.Dropout(dropout)(x)
    x = conv(x, filters, kernel_size=1, stride=1, bias=bias, name=join(name, "conv"), config=config)
    if not shape is None:
        x = resize(x, shape, method="bilinear", config=config)
    return x
