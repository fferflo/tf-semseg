# Concerning bias in convolutional layer: https://github.com/KaimingHe/deep-residual-networks/issues/10#issuecomment-194037195

import tensorflow as tf
import numpy as np
import sys
from .util import *
from . import config, shortcut

def stem(x, type, name, config=config.Config()): # For variants, see: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py#L482
    if type == "b":
        x = conv_norm_act(x, filters=64, kernel_size=7, stride=2, name=name, config=config)
        pool_mode = "max"
    else:
        if type == "s":
            filters = [64, 64, 128]
            pool_mode = "max"
        else:
            raise ValueError("Unknown stem " + type)

        x = conv_norm_act(x, filters=filters[0], kernel_size=3, stride=2, name=join(name, "1"), config=config)
        x = conv_norm_act(x, filters=filters[1], kernel_size=3, stride=1, name=join(name, "2"), config=config)
        x = conv_norm_act(x, filters=filters[2], kernel_size=3, stride=1, name=join(name, "3"), config=config)

    x = pool(x, kernel_size=3, stride=2, mode=pool_mode, config=config)

    return x

def basic_block_v1(x, filters=None, stride=1, dilation_rate=1, name="resnet-basic-v1", block=conv_norm_act, config=config.Config(), **kwargs):
    orig_x = x

    if filters is None:
        filters = x.shape[-1]

    x = block(x, filters=filters, stride=stride, dilation_rate=dilation_rate, name=join(name, "1"), config=config, **kwargs)

    x = conv(x, filters, kernel_size=3, stride=1, bias=False, name=join(name, "2", "conv"), config=config)
    x = norm(x, name=join(name, "2", "norm"), config=config)

    x = shortcut.add(x, orig_x, stride=stride, activation=False, name=join(name, "shortcut"), config=config)
    # TODO: dropout?
    x = act(x, config=config)
    return x

def bottleneck_block_v1(x, filters, stride=1, dilation_rate=1, name="resnet-bottleneck-v1", block=conv_norm_act, bottleneck_factor=4, config=config.Config(), **kwargs):
    orig_x = x

    x = conv_norm_act(x, filters, kernel_size=1, stride=1, name=join(name, "reduce"), config=config)

    x = block(x, stride=stride, dilation_rate=dilation_rate, name=join(name, "center"), config=config, **kwargs)

    x = conv(x, filters * bottleneck_factor, kernel_size=1, stride=1, bias=False, name=join(name, "expand", "conv"), config=config)
    x = norm(x, name=join(name, "expand", "norm"), config=config)

    x = shortcut.add(x, orig_x, stride=stride, activation=False, name=join(name, "shortcut"), config=config)
    # TODO: dropout?
    x = act(x, config=config)
    return x

def resnet(x, block, num_residual_units, filters, strides, dilation_rates, name=None, stem="b", config=config.Config(), **kwargs):
    if stem != None:
        x = globals()["stem"](x, stem, name=join(name, "stem_" + stem), config=config)

    # Residual blocks
    for block_index in range(len(num_residual_units)):
        for unit_index in range(num_residual_units[block_index]):
            x = block(x,
                    filters=filters[block_index],
                    stride=strides[block_index] if unit_index == 0 else 1,
                    dilation_rate=(dilation_rates[block_index - 1] if block_index > 0 else 1) if unit_index == 0 else dilation_rates[block_index],
                    name=join(name, f"block{block_index + 1}", f"unit{unit_index + 1}"),
                    config=config,
                    **kwargs)
        x = set_name(x, join(name, f"block{block_index + 1}"))

    return x

def strides_and_dilation_rates(strides, dilate):
    if isinstance(dilate, bool):
        dilate = [dilate, dilate, dilate, dilate]
    dilation_rates = [1, 1, 1, 1]
    strides = [s for s in strides]
    for i, d in enumerate(dilate):
        if d:
            for j in range(i, len(dilate)):
                dilation_rates[j] *= strides[i]
            strides[i] = 1
    return strides, dilation_rates

def resnet_v1_50(x, block=bottleneck_block_v1, name="resnet_v1_50", stem="b", strides=[1, 2, 2, 2], dilate=False, **kwargs):
    strides, dilation_rates = strides_and_dilation_rates(strides, dilate)
    x = resnet(x,
        block=block,
        num_residual_units=[3, 4, 6, 3],
        filters=[64, 128, 256, 512],
        dilation_rates=dilation_rates,
        strides=strides,
        stem=stem,
        name=name,
        **kwargs)
    return x

def resnet_v1_101(x, block=bottleneck_block_v1, name="resnet_v1_101", stem="b", strides=[1, 2, 2, 2], dilate=False, **kwargs):
    strides, dilation_rates = strides_and_dilation_rates(strides, dilate)
    x = resnet(x,
        block=block,
        num_residual_units=[3, 4, 23, 3],
        filters=[64, 128, 256, 512],
        dilation_rates=dilation_rates,
        strides=strides,
        stem=stem,
        name=name,
        **kwargs)
    return x

def resnet_v1_152(x, block=bottleneck_block_v1, name="resnet_v1_152", stem="b", strides=[1, 2, 2, 2], dilate=False, **kwargs):
    strides, dilation_rates = strides_and_dilation_rates(strides, dilate)
    x = resnet(x,
        block=block,
        num_residual_units=[3, 8, 36, 3],
        filters=[64, 128, 256, 512],
        dilation_rates=dilation_rates,
        strides=strides,
        stem=stem,
        name=name,
        **kwargs)
    return x
