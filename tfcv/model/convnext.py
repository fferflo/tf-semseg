import tensorflow as tf
import numpy as np
from . import config, shortcut
from .util import *
from functools import partial

def stem(x, filters, name=None, config=config.Config()):
    x = conv_norm(x, filters=filters, kernel_size=4, stride=4, bias=True, padding=0, name=name, config=config)
    return x

def block(x, filters=None, dilation_rate=1, factor=4, name="convnext-block", config=config.Config()):
    orig_x = x

    if filters is None:
        filters = x.shape[-1]

    x = conv(x, filters=filters, kernel_size=7, stride=1, dilation_rate=dilation_rate, groups=filters, bias=True, name=join(name, "depthwise"), config=config)
    x = norm(x, name=join(name, "norm"), config=config)
    x = conv(x, filters=filters * factor, kernel_size=1, stride=1, bias=True, name=join(name, "pointwise", "1"), config=config)
    x = act(x, name=join(name, "act"), config=config)
    x = conv(x, filters=filters, kernel_size=1, stride=1, bias=True, name=join(name, "pointwise", "2"), config=config)

    x = ScaleLayer(name=join(name, "scale"))(x)

    x = shortcut.add(x, orig_x, stride=1, activation=False, name=join(name, "shortcut"), config=config) # TODO: tfa.layers.StochasticDepth()([x_orig, x])

    return x

def convnext(x, block, num_units, filters, strides, dilation_rates, skip_stride_one_downsampling=True, stem=True, name=None, config=config.Config()):
    if stem:
        x = globals()["stem"](x, filters=filters[0], name=join(name, "stem"), config=config)

    for block_index in range(len(num_units)): # TODO: rename (block, unit) to (stage, block)
        # Downsample
        if strides[block_index] > 1:
            if dilation_rates[block_index] != 1:
                raise ValueError("Stride and dilation_rate cannot both be > 1")
            x = norm_conv(
                x,
                filters=filters[block_index],
                kernel_size=strides[block_index],
                stride=strides[block_index],
                bias=True,
                name=join(name, f"downsample{block_index + 1}"),
                config=config,
            )

        for unit_index in range(num_units[block_index]):
            x = block(
                x,
                filters=filters[block_index],
                dilation_rate=dilation_rates[block_index],
                name=join(name, f"block{block_index + 1}", f"unit{unit_index + 1}"),
                config=config,
            )
        x = set_name(x, join(name, f"block{block_index + 1}"))

    return x

def convnext_tiny(x, strides=[1, 2, 2, 2], dilate=False, skip_stride_one_downsampling=True, name="convnext_tiny", config=config.Config()):
    strides, dilation_rates = strides_and_dilation_rates(strides, dilate)
    return convnext(
        x,
        block=partial(block, factor=4),
        num_units=[3, 3, 9, 3],
        filters=[96, 192, 384, 768],
        strides=strides,
        dilation_rates=dilation_rates,
        skip_stride_one_downsampling=skip_stride_one_downsampling,
        stem=True,
        name=name,
        config=config,
    )

def convnext_small(x, strides=[1, 2, 2, 2], dilate=False, skip_stride_one_downsampling=True, name="convnext_small", config=config.Config()):
    strides, dilation_rates = strides_and_dilation_rates(strides, dilate)
    return convnext(
        x,
        block=partial(block, factor=4),
        num_units=[3, 3, 27, 3],
        filters=[96, 192, 384, 768],
        strides=strides,
        dilation_rates=dilation_rates,
        skip_stride_one_downsampling=skip_stride_one_downsampling,
        stem=True,
        name=name,
        config=config,
    )

def convnext_base(x, strides=[1, 2, 2, 2], dilate=False, skip_stride_one_downsampling=True, name="convnext_base", config=config.Config()):
    strides, dilation_rates = strides_and_dilation_rates(strides, dilate)
    return convnext(
        x,
        block=partial(block, factor=4),
        num_units=[3, 3, 27, 3],
        filters=[128, 256, 512, 1024],
        strides=strides,
        dilation_rates=dilation_rates,
        skip_stride_one_downsampling=skip_stride_one_downsampling,
        stem=True,
        name=name,
        config=config,
    )

def convnext_large(x, strides=[1, 2, 2, 2], dilate=False, skip_stride_one_downsampling=True, name="convnext_large", config=config.Config()):
    strides, dilation_rates = strides_and_dilation_rates(strides, dilate)
    return convnext(
        x,
        block=partial(block, factor=4),
        num_units=[3, 3, 27, 3],
        filters=[192, 384, 768, 1536],
        strides=strides,
        dilation_rates=dilation_rates,
        skip_stride_one_downsampling=skip_stride_one_downsampling,
        stem=True,
        name=name,
        config=config,
    )

def convnext_xlarge(x, strides=[1, 2, 2, 2], dilate=False, skip_stride_one_downsampling=True, name="convnext_xlarge", config=config.Config()):
    strides, dilation_rates = strides_and_dilation_rates(strides, dilate)
    return convnext(
        x,
        block=partial(block, factor=4),
        num_units=[3, 3, 27, 3],
        filters=[256, 512, 1024, 2048],
        strides=strides,
        dilation_rates=dilation_rates,
        skip_stride_one_downsampling=skip_stride_one_downsampling,
        stem=True,
        name=name,
        config=config,
    )
