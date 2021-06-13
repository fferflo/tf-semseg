import tensorflow as tf
from .util import *
from .resnet import shortcut
from . import config

def non_bottleneck_block_1d(x, filters=None, stride=1, dilation_rate=1, name="non-bottleneck-1d", config=config.Config()):
    orig_x = x

    if filters is None:
        filters = x.shape[-1]

    x = config.conv(x, filters, kernel_size=(3, 1), strides=(stride, 1), dilation_rate=1, use_bias=True, padding="same", name=join(name, "1", "dim0", "conv"))
    x = config.act(x)
    x = config.conv(x, filters, kernel_size=(1, 3), strides=(1, stride), dilation_rate=1, use_bias=True, padding="same", name=join(name, "1", "dim1", "conv"))
    x = config.norm(x, name=join(name, "1", "norm"))
    x = config.act(x)

    x = config.conv(x, filters, kernel_size=(3, 1), strides=1, dilation_rate=(dilation_rate, 1), use_bias=True, padding="same", name=join(name, "2", "dim0", "conv"))
    x = config.act(x)
    x = config.conv(x, filters, kernel_size=(1, 3), strides=1, dilation_rate=(1, dilation_rate), use_bias=True, padding="same", name=join(name, "2", "dim1", "conv"))
    x = config.norm(x, name=join(name, "2", "norm"))

    # TODO: dropout here https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py#L25

    x = shortcut(x, orig_x, stride=stride, name=join(name, "shortcut"), config=config)

    x = config.act(x)
    return x
