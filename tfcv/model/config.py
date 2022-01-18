import tensorflow as tf
import numpy as np
import tfcv

def get_pytorch_same_padding(dims, kernel_size, dilation_rate=1):
    kernel_size = np.asarray(kernel_size)
    dilation_rate = np.asarray(dilation_rate)
    kernel_size = kernel_size + (kernel_size - 1) * (dilation_rate - 1)
    padding = np.floor((kernel_size - 1) / 2).astype("int32")
    padding = np.broadcast_to(padding, (dims,))
    return tuple(padding.tolist())

def add_padding(x, padding, kernel_size, dilation_rate=1):
    if padding == "tensorflow":
        keras_padding = "SAME"
    elif padding == "pytorch" or padding == "caffe":
        x = tfcv.model.keras.ZeroPaddingND(get_pytorch_same_padding(len(x.get_shape()) - 2, kernel_size, dilation_rate))(x)
        keras_padding = "VALID"
    elif isinstance(padding, int) and padding == 0:
        keras_padding = "VALID"
    else:
        raise ValueError(f"Invalid padding {padding}")
    return x, keras_padding

def conv(x, filters=None, stride=1, kernel_size=3, dilation_rate=1, groups=1, bias=True, padding="tensorflow", name=None):
    if not isinstance(bias, bool):
        raise ValueError(f"Invalid bias value {bool}")
    if filters is None:
        filters = x.shape[-1]
    x, keras_padding = add_padding(x, padding=padding, kernel_size=kernel_size, dilation_rate=dilation_rate)
    return tfcv.model.keras.ConvND(filters=filters, strides=stride, kernel_size=kernel_size, dilation_rate=dilation_rate, groups=groups, use_bias=bias, padding=keras_padding, name=name)(x)

def norm(x, epsilon=1e-5, momentum=0.997, name=None): # TODO: remove parameters, use BN default params
    return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, name=name)(x)

def act(x, name=None):
    return tf.keras.layers.ReLU(name=name)(x)

def pool(x, mode, stride, kernel_size, padding="tensorflow", name=None):
    x, keras_padding = add_padding(x, padding=padding, kernel_size=kernel_size)
    if mode.lower() == "max":
        return tfcv.model.keras.MaxPoolND(pool_size=kernel_size, strides=stride, padding=keras_padding, name=name)(x)
    elif mode.lower() == "avg":
        return tfcv.model.keras.AveragePoolingND(pool_size=kernel_size, strides=stride, padding=keras_padding, name=name)(x)
    else:
        raise ValueError(f"Invalid pooling mode {mode}")

# This matches pytorch behavior, e.g. of torch.nn.functional.interpolate and torch.nn.Upsample
def resize(x, shape, method, align_corners=False, name=None):
    if align_corners:
        return tf.compat.v1.image.resize(x, shape, method=method, align_corners=True, name=name)
    else:
        return tf.image.resize(x, shape, method=method, name=name) # Has align_corners=True by default

class Config:
    def __init__(self, conv=conv, norm=norm, act=act, pool=pool, resize=resize):
        self.conv = conv
        self.norm = norm
        self.act = act
        self.pool = pool
        self.resize = resize



def partial_with_default_args(func, **default_kwargs):
    def wrapper(*args, **passed_kwargs):
        new_kwargs = {}
        for k, v in default_kwargs.items():
            new_kwargs[k] = v
        for k, v in passed_kwargs.items():
            new_kwargs[k] = v
        return func(*args, **new_kwargs)
    return wrapper
pwda = partial_with_default_args

def TensorflowConfig(conv=pwda(conv, padding="tensorflow"), pool=pwda(pool, padding="tensorflow"), **kwargs):
    return Config(conv=conv, pool=pool, **kwargs)

def PytorchConfig(conv=pwda(conv, padding="pytorch"), pool=pwda(pool, padding="pytorch"), **kwargs):
    return Config(conv=conv, pool=pool, **kwargs)

def CaffeConfig(conv=pwda(conv, padding="caffe"), pool=pwda(pool, padding="caffe"), **kwargs):
    return Config(conv=conv, pool=pool, **kwargs)
