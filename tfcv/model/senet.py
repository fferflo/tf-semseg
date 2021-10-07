import tensorflow as tf
from . import config
from .util import *

def squeeze_excite_channel(x, reduction, filters=None, name="squeeze_excite_channel", config=config.Config()):
    x_orig = x
    if filters is None:
        filters = x.shape[-1]

    x = tf.reduce_mean(x, axis=list(range(len(x.shape)))[1:-1], keepdims=True)
    x = conv(x, filters // reduction, kernel_size=1, bias=True, name=join(name, "conv1"), config=config)
    x = act(x, config=config)
    x = conv(x, filters, kernel_size=1, bias=True, name=join(name, "conv2"), config=config)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = x_orig * x
    return x

def squeeze_excite_spatial(x, name="squeeze_excite_spatial", config=config.Config()):
    x_orig = x

    x = conv(x, 1, kernel_size=1, bias=True, name=join(name, "conv"), config=config)
    x = tf.keras.layers.Activation("sigmoid")(x)

    x = x_orig * x
    return x

def squeeze_excite_concurrent(x, reduction, name="squeeze_excite_concurrent", config=config.Config()):
    x_channel = squeeze_excite_channel(x, reduction=reduction, name=join(name, "channel"), config=config)
    x_spatial = squeeze_excite_spatial(x, name=join(name, "spatial"), config=config)

    x = tf.math.maximum(x_channel, x_spatial)
    return x
