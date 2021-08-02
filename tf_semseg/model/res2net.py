import tensorflow as tf
from .util import *
from . import resnet, config

def hierarchical_conv(x, block=conv_norm_act, scales=4, stride=1, kernel_size=3, name="res2net-conv", config=config.Config(), **kwargs):
    assert scales >= 2

    splits = tf.split(x, num_or_size_splits=scales, axis=-1)

    # Split 1
    if stride != 1:
        splits[0] = pool(splits[0], kernel_size=kernel_size, stride=stride, mode="avg", name=join(name, "scale1"), config=config)

    # Split 2
    splits[1] = block(splits[1], filters=splits[1].shape[-1], stride=stride, kernel_size=kernel_size, name=join(name, "scale2"), config=config, **kwargs)

    # Splits 3...
    for s in range(2, scales):
        splits[s] = splits[s] + splits[s - 1]
        splits[s] = block(splits[s], filters=splits[1].shape[-1], stride=stride, kernel_size=kernel_size, name=join(name, "scale" + str(s + 1)), config=config)

    x = tf.keras.layers.Concatenate()(splits)

    return x

def bottleneck_block_v1(x, block=hierarchical_conv, bottleneck_factor=2, **kwargs):
    return resnet.bottleneck_block_v1(x, block=block, bottleneck_factor=bottleneck_factor, **kwargs)

def res2net_v1_50(x, block=bottleneck_block_v1, **kwargs):
    return resnet.resnet_v1_50(x, block=block, **kwargs)

def res2net_v1_101(x, block=bottleneck_block_v1, **kwargs):
    return resnet.resnet_v1_101(x, block=block, **kwargs)

def res2net_v1_152(x, block=bottleneck_block_v1, **kwargs):
    return resnet.resnet_v1_152(x, block=block, **kwargs)
