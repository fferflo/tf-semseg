import tensorflow as tf
from . import config
from .util import *

def squeeze_excite(x, reduction=16, filters=None, name="squeeze_excite", config=config.Config()):
    x_orig = x
    if filters is None:
        filters = x.shape[-1]

    x = tf.reduce_mean(x, axis=list(range(len(x.shape)))[1:-1], keepdims=True)
    x = tf.keras.layers.Reshape([1] * (len(x_orig.shape) - 2) + [x.shape[-1]])(x)
    x = config.conv(x, filters // reduction, kernel_size=1, use_bias=True, name=join(name, "conv1"))
    x = config.act(x)
    x = config.conv(x, filters, kernel_size=1, use_bias=True, name=join(name, "conv2"))
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = x_orig * x
    return x
