import tensorflow as tf
import numpy as np

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

def pool(*args, config=config.Config(), **kwargs):
    return config.pool(*args, **kwargs)

def resize(*args, config=config.Config(), **kwargs):
    return config.resize(*args, **kwargs)



def conv_norm_act(x, *args, bias=False, name=None, config=config.Config(), **kwargs):
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    x = norm(x, name=join(name, "norm"), config=config)
    x = act(x, config=config)
    x = set_name(x, name=name)
    return x

def conv_norm(x, *args, bias=False, name=None, config=config.Config(), **kwargs):
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    x = norm(x, name=join(name, "norm"), config=config)
    x = set_name(x, name=name)
    return x

def conv_act(x, *args, bias=True, name=None, config=config.Config(), **kwargs):
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    x = act(x, config=config)
    x = set_name(x, name=name)
    return x

def norm_act_conv(x, *args, bias=False, name=None, config=config.Config(), **kwargs):
    x = norm(x, name=join(name, "norm"), config=config)
    x = act(x, config=config)
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    x = set_name(x, name=name)
    return x

def norm_conv(x, *args, bias=False, name=None, config=config.Config(), **kwargs):
    x = norm(x, name=join(name, "norm"), config=config)
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    x = set_name(x, name=name)
    return x

def act_conv(x, *args, bias=False, name=None, config=config.Config(), **kwargs):
    x = act(x, config=config)
    x = conv(x, *args, bias=bias, name=join(name, "conv"), config=config, **kwargs)
    x = set_name(x, name=name)
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

# https://arxiv.org/pdf/2103.17239.pdf
class ScaleLayer(tf.keras.layers.Layer): # TODO: move somewhere else
    def __init__(self, initial_value=1e-6, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = initial_value
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis

    def build(self, input_shape):
        input_shape = input_shape[1:]
        shape = np.ones(len(input_shape) if isinstance(input_shape, tuple) else input_shape.rank, dtype="int32")
        for axis in self.axis:
            assert axis != 0
            axis = axis - 1 if axis > 0 else axis
            shape[axis] = input_shape[axis]
        self.scale = self.add_weight("scale", shape=shape, initializer="zeros", trainable=True)
        initial_value = tf.cast(self.initial_value, self.scale.dtype)
        while len(initial_value.shape) < len(self.scale.shape): # TODO: tfcv util function for this type of broadcasting
            initial_value = initial_value[tf.newaxis]
        initial_value = tf.broadcast_to(initial_value, tf.shape(self.scale))
        self.scale.assign(initial_value)

    def call(self, x):
        return x * self.scale[tf.newaxis, ...]

    def get_config(self):
        config = super().get_config()
        config["initial_value"] = self.initial_value
        config["axis"] = self.axis
        return config

class BiasLayer(tf.keras.layers.Layer): # TODO: move somewhere else
    def __init__(self, initial_value=0, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.initial_value = initial_value
        if isinstance(axis, int):
            self.axis = [axis]
        else:
            self.axis = axis

    def build(self, input_shape):
        input_shape = input_shape[1:]
        shape = np.ones(len(input_shape) if isinstance(input_shape, tuple) else input_shape.rank, dtype="int32")
        for axis in self.axis:
            assert axis != 0
            axis = axis - 1 if axis > 0 else axis
            shape[axis] = input_shape[axis]
        self.bias = self.add_weight("bias", shape=shape, initializer="zeros", trainable=True)
        initial_value = tf.cast(self.initial_value, self.bias.dtype)
        while len(initial_value.shape) < len(self.bias.shape): # TODO: tfcv util function for this type of broadcasting
            initial_value = initial_value[tf.newaxis]
        initial_value = tf.broadcast_to(initial_value, tf.shape(self.bias))
        self.bias.assign(initial_value)

    def call(self, x):
        return x + self.bias[tf.newaxis, ...]

    def get_config(self):
        config = super().get_config()
        config["initial_value"] = self.initial_value
        config["axis"] = self.axis
        return config

def strides_and_dilation_rates(strides, dilate):
    if isinstance(dilate, bool):
        dilate = [dilate] * len(strides)
    dilation_rates = [1] * len(strides)
    strides = [s for s in strides]
    for i, d in enumerate(dilate):
        if d:
            for j in range(i, len(dilate)):
                dilation_rates[j] *= strides[i]
            strides[i] = 1
    return strides, dilation_rates
