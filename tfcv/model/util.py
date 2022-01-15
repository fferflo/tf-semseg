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

def dropout(*args, config=config.Config(), **kwargs):
    return config.dropout(*args, **kwargs)



def conv_norm_act(x, *args, bias=False, name=None, config=config.Config(), **kwargs):
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    x = norm(x, name=join(name, "norm"), config=config)
    x = act(x, config=config)
    return x

def conv_norm(x, *args, bias=False, name=None, config=config.Config(), **kwargs):
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    x = norm(x, name=join(name, "norm"), config=config)
    return x

def conv_act(x, *args, bias=True, name=None, config=config.Config(), **kwargs):
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    x = act(x, config=config)
    return x

def norm_act_conv(x, *args, bias=False, name=None, config=config.Config(), **kwargs):
    x = norm(x, name=join(name, "norm"), config=config)
    x = act(x, config=config)
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    return x

def norm_conv(x, *args, bias=False, name=None, config=config.Config(), **kwargs):
    x = norm(x, name=join(name, "norm"), config=config)
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    return x

def act_conv(x, *args, bias=False, name=None, config=config.Config(), **kwargs):
    x = act(x, config=config)
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    return x



def pad_to_size(x, shape, mode="center"):
    pad_width = shape - tf.shape(x)[1:-1]
    if mode == "center":
        pad_width_front = pad_width // 2
        pad_width_back = pad_width - pad_width_front
    elif mode == "back":
        pad_width_front = pad_width * 0
        pad_width_back = pad_width
    elif mode == "front":
        pad_width_front = pad_width
        pad_width_back = pad_width * 0
    else:
        raise ValueError(f"Invalid padding mode {mode}")
    pad_width = [(0, 0)] + [(pad_width_front[i], pad_width_back[i]) for i in range(len(x.shape) - 2)] + [(0, 0)]
    x = tf.pad(x, pad_width)
    return x

def set_name(x, name):
    return tf.keras.layers.Lambda(lambda x: x, name=name)(x)

def repeat(x, n, block, name=None, **kwargs):
    for i in range(n):
        x = block(x, name=join(name, str(i + 1)), **kwargs)
    return x

def get_predecessor(x, predicate):
    done = set()
    result = []
    def recurse(layer):
        if not layer.name in done:
            if predicate(layer.name):
                result.append(layer)
            done.add(layer.name)
            for node in layer.inbound_nodes:
                try:
                    if isinstance(node.inbound_layers, list):
                        for layer in node.inbound_layers:
                            recurse(layer)
                    else:
                        recurse(node.inbound_layers)
                except AttributeError as e:
                    continue
    if "_keras_history" in vars(x):
        recurse(x._keras_history.layer)
    if len(result) > 1:
        raise ValueError(f"Node has more than one predecessor matching the given predicate: {[n.name for n in result]}")
    if len(result) == 0:
        raise ValueError("Node has no predecessor matching the given predicate")
    return result[0].output
