import tensorflow as tf
from .util import *
from . import config, pspnet

def head(xs, filters=512, psp_bin_sizes=[1, 2, 3, 6], name=None, config=config.Config()):
    for i in range(len(xs) - 1):
        xs[i] = conv_norm_act(xs[i], filters=filters, kernel_size=1, stride=1, name=join(name, f"initial{i + 1}"), config=config)
    xs[-1] = pspnet.psp(xs[-1], filters=filters, resize_method="bilinear", bin_sizes=psp_bin_sizes, name=join(name, "psp"), config=config)
    xs[-1] = conv_norm_act(xs[-1], filters=filters, kernel_size=3, stride=1, name=join(name, f"initial{len(xs)}"), config=config)

    for i in reversed(list(range(len(xs) - 1))):
        xs[i] = xs[i] + resize(xs[i + 1], tf.shape(xs[i])[1:-1], method="bilinear", config=config)

    for i in range(len(xs) - 1):
        xs[i] = conv_norm_act(xs[i], filters=filters, kernel_size=3, stride=1, name=join(name, f"fpn{i + 1}"), config=config)

    for i in range(1, len(xs)):
        xs[i] = resize(xs[i], tf.shape(xs[0])[1:-1], method="bilinear", config=config)
    x = tf.concat(xs, axis=-1)
    x = conv_norm_act(x, filters=filters, kernel_size=3, stride=1, name=join(name, f"final"), config=config)

    return x
