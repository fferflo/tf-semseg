import tensorflow as tf
import numpy as np
import math
from .util import *

def default_norm(x, *args, epsilon=1e-5, momentum=0.997, **kwargs):
    return tf.keras.layers.BatchNormalization(*args, momentum=momentum, epsilon=epsilon, **kwargs)(x)

def default_act(x, **kwargs):
    return tf.keras.layers.ReLU(**kwargs)(x)

default_mode = "tensorflow"

class Config:
    def __init__(self, norm=default_norm, act=default_act, mode=default_mode, resize_align_corners=False, upsample_mode="resize"):
        assert mode in ["tensorflow", "pytorch"]

        def conv(x, *args, **kwargs):
            if mode == "tensorflow":
                return Conv(*args, **kwargs)(x)
            elif mode == "pytorch":
                return PyTorchConv(*args, **kwargs)(x)
            else:
                assert False
        self.conv = conv

        if resize_align_corners:
            self.resize = lambda x, shape, method, name=None: tf.compat.v1.image.resize(x, shape, method=method, align_corners=True, name=name)
        else:
            self.resize = lambda x, shape, method, name=None: tf.image.resize(x, shape, method=method, name=name)

        def maxpool(x, *args, **kwargs):
            if mode == "tensorflow":
                return MaxPool(*args, **kwargs)(x)
            elif mode == "pytorch":
                return PyTorchMaxPool(*args, **kwargs)(x)
            else:
                assert False
        self.maxpool = maxpool

        def avgpool(x, *args, **kwargs):
            if mode == "tensorflow":
                return AveragePool(*args, **kwargs)(x)
            elif mode == "pytorch":
                return PyTorchAveragePool(*args, **kwargs)(x)
            else:
                assert False
        self.avgpool = avgpool

        if upsample_mode == "resize":
            def upsample(x, factor, method="nearest", name=None):
                return self.resize(x, factor * tf.shape(x)[1:-1], method=method, name=name)
        elif upsample_mode == "upsample-pool":
            def upsample(x, factor, method="nearest", name=None):
                if method == "nearest":
                    return UpSample(factor)(x)
                elif method == "bilinear":
                    index = 0
                    while factor > 1: # Simple prime factorization
                        for k in range(2, factor + 1):
                            if factor % k == 0:
                                break
                        x = UpSample(k, name=join(name, str(index), "upsample"))(x)
                        x = tf.pad(x, [[0, 0]] + [[1, 1] for _ in range(len(x.shape) - 2)] + [[0, 0]], mode="SYMMETRIC", name=join(name, str(index), "pad"))
                        x = tf.nn.avg_pool(x, ksize=k + 1, strides=1, padding="VALID", name=join(name, str(index), "avgpool"))
                        factor = factor // k
                        index += 1
                    return x
                else:
                    raise ValueError("Invalid method")
        else:
            raise ValueError("Invalid upsample mode")
        self.upsample = upsample

        self.norm = norm
        self.act = act



########## Layers ##########

def get_pytorch_same_padding(dims, kernel_size, dilation=1):
    kernel_size = np.asarray(kernel_size)
    dilation = np.asarray(dilation)
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = np.ceil((kernel_size - 1) / 2).astype("int32")
    padding = np.broadcast_to(padding, (dims,))
    return tuple(padding.tolist())

def ZeroPad(padding, *args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            keras_padding = padding
            if isinstance(keras_padding, tuple):
                keras_padding = keras_padding[0]
            return tf.keras.layers.ZeroPadding1D(padding=keras_padding, *args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.ZeroPadding2D(padding=padding, *args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.ZeroPadding3D(padding=padding, *args, **kwargs)(x)
        else:
            assert False, "Unsupported number of dimensions"
    return constructor

def Conv(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.Conv1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.Conv2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.Conv3D(*args, **kwargs)(x)
        else:
            assert False, "Unsupported number of dimensions"
    return constructor

def PyTorchConv(filters, kernel_size, padding="same", dilation_rate=1, **kwargs):
    def constructor(x):
        keras_padding = padding
        if padding.lower() == "same":
            x = ZeroPad(get_pytorch_same_padding(len(x.get_shape()) - 2, kernel_size, dilation_rate))(x)
            keras_padding = "VALID"
        return Conv(filters, kernel_size, padding=keras_padding, dilation_rate=dilation_rate, **kwargs)(x)
    return constructor

def MaxPool(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.MaxPool1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.MaxPool2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.MaxPool3D(*args, **kwargs)(x)
        else:
            assert False, "Unsupported number of dimensions"
    return constructor

def PyTorchMaxPool(pool_size, padding="same", **kwargs):
    def constructor(x):
        keras_padding = padding
        if padding.lower() == "same":
            x = ZeroPad(get_pytorch_same_padding(len(x.get_shape()) - 2, pool_size))(x)
            keras_padding = "VALID"
        return MaxPool(pool_size, padding=keras_padding, **kwargs)(x)
    return constructor

def AveragePool(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.AveragePooling1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.AveragePooling2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.AveragePooling3D(*args, **kwargs)(x)
        else:
            assert False, "Unsupported number of dimensions"
    return constructor

def PyTorchAveragePool(pool_size, padding="same", **kwargs):
    def constructor(x):
        keras_padding = padding
        if padding.lower() == "same":
            x = ZeroPad(get_pytorch_same_padding(len(x.get_shape()) - 2, pool_size))(x)
            keras_padding = "VALID"
        return AveragePool(pool_size, padding=keras_padding, **kwargs)(x)
    return constructor

def UpSample(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.UpSampling1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.UpSampling2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.UpSampling3D(*args, **kwargs)(x)
        else:
            assert False, "Unsupported number of dimensions"
    return constructor
