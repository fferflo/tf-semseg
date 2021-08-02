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

def conv(*args, config=config.Config(), **kwargs):
    return config.conv(*args, **kwargs)

def norm(*args, config=config.Config(), **kwargs):
    return config.norm(*args, **kwargs)

def act(*args, config=config.Config(), **kwargs):
    return config.act(*args, **kwargs)

def resize(*args, config=config.Config(), **kwargs):
    return config.resize(*args, **kwargs)

def upsample(*args, config=config.Config(), **kwargs):
    return config.upsample(*args, **kwargs)

def pool(*args, config=config.Config(), **kwargs):
    return config.pool(*args, **kwargs)



def conv_norm_act(x, filters=None, stride=1, kernel_size=3, dilation_rate=1, groups=1, use_bias=False, name=None, config=config.Config()):
    x = conv(x, filters=filters, kernel_size=kernel_size, stride=stride, groups=groups, dilation_rate=dilation_rate, use_bias=use_bias, name=join(name, "conv"), config=config)
    x = norm(x, name=join(name, "norm"), config=config)
    x = act(x, config=config)
    return x

def repeat(x, n, block, name=None, **kwargs):
    for i in range(n):
        x = block(x, name=join(name, str(i + 1)), **kwargs)
    return x

def get_predecessor(input, output, predicate):
    result = list(filter(lambda x: predicate(x.name), tf.keras.Model(inputs=[input], outputs=[output]).layers))
    if len(result) > 1:
        raise ValueError("Tensor has more than one predecessor matching the given predicate")
    if len(result) == 0:
        raise ValueError("Tensor has no predecessor matching the given predicate")
    return result[0].output
