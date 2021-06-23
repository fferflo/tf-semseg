import tensorflow as tf
from .util import *
from . import config, shortcut

def non_bottleneck_block_1d(x, filters=None, stride=1, dilation_rate=1, name="non-bottleneck-1d", config=config.Config()):
    orig_x = x

    if filters is None:
        filters = x.shape[-1]

    dims = len(x.shape) - 2

    for dim in range(dims):
        kernel_size = [1] * dims
        kernel_size[dim] = 3
        strides = [1] * dims
        strides[dim] = stride
        x = config.conv(x, filters, kernel_size=tuple(kernel_size), strides=tuple(strides), dilation_rate=1, use_bias=True, padding="same", name=join(name, "1", f"dim{dim}", "conv"))
        if dim == dims - 1:
            x = config.norm(x, name=join(name, "1", "norm"))
        x = config.act(x)

    for dim in range(dims):
        kernel_size = [1] * dims
        kernel_size[dim] = 3
        dilation_rates = [1] * dims
        dilation_rates[dim] = dilation_rate
        x = config.conv(x, filters, kernel_size=tuple(kernel_size), strides=1, dilation_rate=tuple(dilation_rates), use_bias=True, padding="same", name=join(name, "2", f"dim{dim}", "conv"))
        if dim == dims - 1:
            x = config.norm(x, name=join(name, "2", "norm"))
            # TODO: dropout here https://github.com/Eromera/erfnet_pytorch/blob/master/train/erfnet.py#L25
            x = shortcut.add(x, orig_x, stride=stride, activation=False, name=join(name, "shortcut"), config=config)
        x = config.act(x)

    return x
