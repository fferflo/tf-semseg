# Concerning bias in convolutional layer: https://github.com/KaimingHe/deep-residual-networks/issues/10#issuecomment-194037195

import tensorflow as tf
import numpy as np
import sys, os, pyunpack
from .util import *

def shortcut(x, orig_x, stride, name, norm=default_norm):
    in_channels = orig_x.get_shape()[-1]
    out_channels = x.get_shape()[-1]

    if in_channels != out_channels or stride > 1:
        orig_x = Conv(out_channels, kernel_size=1, strides=stride, dilation_rate=1, use_bias=False, padding="same", name=join(name, "conv"))(orig_x)
        orig_x = norm(orig_x, name=join(name, "norm"))

    return x + orig_x

def stem(x, type, name, norm=default_norm): # For variants, see: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py#L482
    if type == "b":
        x = Conv(64, kernel_size=7, strides=2, dilation_rate=1, use_bias=False, padding="same", name=join(name, "conv"))(x)
        x = norm(x, name=join(name, "norm"))
        x = tf.keras.layers.ReLU()(x)
        pool = MaxPool
    else:
        if type == "s":
            filters = [64, 64, 128]
            pool = MaxPool
        else:
            print("Unknown stem " + type)
            sys.exit(-1)

        x = Conv(filters[0], kernel_size=3, strides=2, dilation_rate=1, use_bias=False, padding="same", name=join(name, "conv1"))(x)
        x = norm(x, name=join(name, "norm1"))
        x = tf.keras.layers.ReLU()(x)

        x = Conv(filters[1], kernel_size=3, strides=1, dilation_rate=1, use_bias=False, padding="same", name=join(name, "conv2"))(x)
        x = norm(x, name=join(name, "norm2"))
        x = tf.keras.layers.ReLU()(x)

        x = Conv(filters[2], kernel_size=3, strides=1, dilation_rate=1, use_bias=False, padding="same", name=join(name, "conv3"))(x)
        x = norm(x, name=join(name, "norm3"))
        x = tf.keras.layers.ReLU()(x)

    x = pool(pool_size=3, strides=2)(x)
    return x



def basic_block_v1(x, filters=None, stride=1, dilation_rate=1, name="resnet-basic-v1", block=conv_norm_relu, norm=default_norm, **kwargs):
    orig_x = x

    if filters is None:
        filters = x.shape[-1]

    x = block(x, filters=filters, stride=stride, dilation_rate=dilation_rate, name=join(name, "1"), **kwargs)

    x = Conv(filters, kernel_size=3, strides=1, dilation_rate=1, use_bias=False, padding="same", name=join(name, "2", "conv"))(x)
    x = norm(x, name=join(name, "2", "norm"))

    x = shortcut(x, orig_x, stride=stride, name=join(name, "shortcut"), norm=norm)

    x = tf.keras.layers.ReLU()(x)
    return x

def bottleneck_block_v1(x, filters, stride=1, dilation_rate=1, name="resnet-bottleneck-v1", block=conv_norm_relu, bottleneck_factor=4, norm=default_norm, **kwargs):
    orig_x = x

    x = Conv(filters, kernel_size=1, strides=1, dilation_rate=1, use_bias=False, padding="same", name=join(name, "reduce", "conv"))(x)
    x = norm(x, name=join(name, "reduce", "norm"))
    x = tf.keras.layers.ReLU()(x)

    x = block(x, stride=stride, dilation_rate=dilation_rate, name=join(name, "center"), **kwargs)

    x = Conv(filters * bottleneck_factor, kernel_size=1, strides=1, dilation_rate=1, use_bias=False, padding="same", name=join(name, "expand", "conv"))(x)
    x = norm(x, name=join(name, "expand", "norm"))

    x = shortcut(x, orig_x, stride=stride, name=join(name, "shortcut"), norm=norm)

    x = tf.keras.layers.ReLU()(x)
    return x

def resnet(x, block, num_residual_units, filters, dilation_rates, strides, name=None, stem="b", norm=default_norm, **kwargs):
    if stem != None:
        stem_func = globals()["stem"] # Variable has the same name as the function
        x = stem_func(x, stem, name=join(name, "stem_" + stem), norm=norm)

    # Residual blocks
    for block_index in range(len(num_residual_units)):
        for unit_index in range(num_residual_units[block_index]):
            x = block(x,
                    filters=filters[block_index],
                    stride=strides[block_index] if unit_index == 0 else 1,
                    dilation_rate=dilation_rates[block_index],
                    name=join(name, f"block{block_index + 1}", f"unit{unit_index + 1}"),
                    norm=norm,
                    **kwargs)
    return x

def resnet_v1_50(x, block=bottleneck_block_v1, name="resnet_v1_50", stem="b", dilated=False, **kwargs):
    x = resnet(x,
        block=block,
        num_residual_units=[3, 4, 6, 3],
        filters=[64, 128, 256, 512],
        dilation_rates=[1, 1, 2, 4] if dilated else [1, 1, 1, 1],
        strides=[1, 2, 1, 1] if dilated else [1, 2, 2, 2],
        stem=stem,
        name=name,
        **kwargs)
    return x

def resnet_v1_101(x, block=bottleneck_block_v1, name="resnet_v1_101", stem="b", dilated=False, **kwargs):
    x = resnet(x,
        block=block,
        num_residual_units=[3, 4, 23, 3],
        filters=[64, 128, 256, 512],
        dilation_rates=[1, 1, 2, 4] if dilated else [1, 1, 1, 1],
        strides=[1, 2, 1, 1] if dilated else [1, 2, 2, 2],
        stem=stem,
        name=name,
        **kwargs)
    return x

def resnet_v1_152(x, block=bottleneck_block_v1, name="resnet_v1_152", stem="b", dilated=False, **kwargs):
    x = resnet(x,
        block=block,
        num_residual_units=[3, 8, 36, 3],
        filters=[64, 128, 256, 512],
        dilation_rates=[1, 1, 2, 4] if dilated else [1, 1, 1, 1],
        strides=[1, 2, 1, 1] if dilated else [1, 2, 2, 2],
        stem=stem,
        name=name,
        **kwargs)
    return x
