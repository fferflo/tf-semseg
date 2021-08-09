from .util import *
from . import config, shortcut, resnet
import tensorflow as tf

def block(x, num_units, filters=48, bottleneck_factor=4, stride=1, dilation_rate=1, name="densenet-block", config=config.Config()):
    # Layers
    for unit_index in range(num_units):
        x_orig = x
        x = norm_act_conv(x, filters=bottleneck_factor * filters, kernel_size=1, stride=1, name=join(name, f"unit{unit_index + 1}", "1"), config=config)
        x = norm_act_conv(x, filters=filters, kernel_size=3, stride=1, dilation_rate=dilation_rate, name=join(name, f"unit{unit_index + 1}", "2"), config=config)
        # TODO: dropout here
        x = tf.concat([x_orig, x], axis=-1)

    # Transition
    x = norm_act_conv(x, filters=x.shape[-1] // 2, kernel_size=1, stride=1, name=join(name, "transition"), config=config)
    if stride > 1:
        x = tf.nn.avg_pool(x, stride, stride, padding="VALID")

    return x

def densenet(x, filters=48, num_units=[6, 12, 36, 24], strides=[2, 1, 1, 1], dilation_rates=[1, 1, 2, 4], bottleneck_factor=4, stem=True, name="densenet", config=config.Config()):
    if stem: # TODO: new stem.py file, combine with resnet and others
        x = conv_norm_act(x, filters=96, kernel_size=7, stride=2, name=join(name, "stem"), config=config)
        x = pool(x, kernel_size=3, stride=2, mode="max", config=config)

    for block_index in range(len(num_units)):
        x = block(x, num_units[block_index], filters=filters, bottleneck_factor=bottleneck_factor, stride=strides[block_index],
                dilation_rate=dilation_rates[block_index], name=join(name, f"block{block_index + 1}"), config=config)
    x = norm(x, name=join(name, "final", "norm"), config=config)

    return x
