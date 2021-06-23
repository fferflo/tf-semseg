import tensorflow as tf
from functools import partial

def join(*args):
    result = None
    for a in args:
        if result is None:
            result = a
        else:
            result = result + "/" + a
    return result

from . import config

# TODO: conv_norm_act has param stride, config.conv has param strides
# TODO: move to config?
def conv_norm_act(x, filters=None, stride=1, name=None, kernel_size=3, dilation_rate=1, groups=1, use_bias=False, config=config.Config()):
    if filters is None:
        filters = x.shape[-1]
    x = config.conv(x, filters, kernel_size=kernel_size, strides=stride, groups=groups, dilation_rate=dilation_rate, use_bias=use_bias, padding="same", name=join(name, "conv"))
    x = config.norm(x, name=join(name, "norm"))
    x = config.act(x)
    return x

def repeat(x, n, block, name=None, **kwargs):
    for i in range(n):
        x = block(x, name=join(name, str(i + 1)), **kwargs)
    return x
