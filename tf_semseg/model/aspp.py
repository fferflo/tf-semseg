from .util import *
from . import config
import tensorflow as tf

def aspp(x, filters=256, atrous_rates=[12, 24, 36], name="aspp", config=config.Config()):
    # 1x1 conv
    x0 = conv_norm_act(x, filters=filters, kernel_size=1, stride=1, name=join(name, f"1x1"), config=config)

    # Atrous convs
    xs = [conv_norm_act(x, filters=filters, kernel_size=3, stride=1, dilation_rate=d, name=join(name, f"atrous{i + 1}"), config=config) for i, d in enumerate(atrous_rates)]

    # Global pooling
    x1 = tf.reduce_mean(x, axis=list(range(len(x.shape)))[1:-1], keepdims=True)
    x1 = conv_norm_act(x1, filters=filters, kernel_size=1, stride=1, name=join(name, f"global"), config=config)
    x1 = tf.broadcast_to(x1, tf.shape(x0))

    x = tf.concat([x0] + xs + [x1], axis=-1)

    return x
