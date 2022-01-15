import tensorflow as tf
import numpy as np
import sys
from .util import *
from . import config, shortcut
from functools import partial

def stem(x, filters, name=None, config=config.Config()):
    x = conv_norm(x, filters=filters, kernel_size=4, stride=4, bias=True, padding=0, name=name, config=config)
    return x

def block(x, filters=None, factor=4, name="convnext-block", config=config.Config()):
    orig_x = x

    if filters is None:
        filters = x.shape[-1]

    x = conv(x, filters=filters, kernel_size=7, stride=1, groups=filters, bias=True, name=join(name, "depthwise"), config=config)
    x = norm(x, name=join(name, "norm"), config=config)
    x = conv(x, filters=filters * factor, kernel_size=1, stride=1, bias=True, name=join(name, "pointwise", "1"), config=config)
    x = act(x, name=join(name, "act"), config=config)
    x = conv(x, filters=filters, kernel_size=1, stride=1, bias=True, name=join(name, "pointwise", "2"), config=config)

    x = ScaleLayer(name=join(name, "scale"))(x)

    x = shortcut.add(x, orig_x, stride=1, activation=False, name=join(name, "shortcut"), config=config) # TODO: tfa.layers.StochasticDepth()([x_orig, x])

    return x

def convnext(x, block, num_units, filters, stem=True, name=None, config=config.Config(), **kwargs):
    if stem:
        x = globals()["stem"](x, filters=filters[0], name=join(name, "stem"), config=config)

    for block_index in range(len(num_units)): # TODO: rename (block, unit) to (stage, block)
    # Downsample
        if block_index > 0:
            x = norm_conv(x, filters=filters[block_index], kernel_size=2, stride=2, bias=True, padding=0, name=join(name, f"downsample{(block_index - 1) + 1}"), config=config)

        for unit_index in range(num_units[block_index]):
            x = block(x,
                    filters=filters[block_index],
                    name=join(name, f"block{block_index + 1}", f"unit{unit_index + 1}"),
                    config=config,
                )
        x = set_name(x, join(name, f"block{block_index + 1}"))

    return x

def convnext_tiny(x, name="convnext_tiny", config=config.Config()):
    return convnext(
        x,
        block=partial(block, factor=4),
        num_units=[3, 3, 9, 3],
        filters=[96, 192, 384, 768],
        stem=True,
        name=name,
        config=config,
    )

def convnext_small(x, name="convnext_small", config=config.Config()):
    return convnext(
        x,
        block=partial(block, factor=4),
        num_units=[3, 3, 27, 3],
        filters=[96, 192, 384, 768],
        stem=True,
        name=name,
        config=config,
    )

def convnext_base(x, name="convnext_base", config=config.Config()):
    return convnext(
        x,
        block=partial(block, factor=4),
        num_units=[3, 3, 27, 3],
        filters=[128, 256, 512, 1024],
        stem=True,
        name=name,
        config=config,
    )

def convnext_large(x, name="convnext_large", config=config.Config()):
    return convnext(
        x,
        block=partial(block, factor=4),
        num_units=[3, 3, 27, 3],
        filters=[192, 384, 768, 1536],
        stem=True,
        name=name,
        config=config,
    )

def convnext_xlarge(x, name="convnext_xlarge", config=config.Config()):
    return convnext(
        x,
        block=partial(block, factor=4),
        num_units=[3, 3, 27, 3],
        filters=[256, 512, 1024, 2048],
        stem=True,
        name=name,
        config=config,
    )
