import tensorflow as tf
import numpy as np
import math
from .util import *

def ConvND(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.Conv1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.Conv2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.Conv3D(*args, **kwargs)(x)
        else:
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor

def ZeroPaddingND(padding, *args, **kwargs):
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
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor

def MaxPoolND(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.MaxPool1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.MaxPool2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.MaxPool3D(*args, **kwargs)(x)
        else:
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor

def AveragePoolingND(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.AveragePooling1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.AveragePooling2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.AveragePooling3D(*args, **kwargs)(x)
        else:
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor

def UpSamplingND(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.UpSampling1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.UpSampling2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.UpSampling3D(*args, **kwargs)(x)
        else:
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor

def SpatialDropoutND(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.SpatialDropout1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.SpatialDropout2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.SpatialDropout3D(*args, **kwargs)(x)
        else:
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor

def get_pytorch_same_padding(dims, kernel_size, dilation=1):
    kernel_size = np.asarray(kernel_size)
    dilation = np.asarray(dilation)
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = np.ceil((kernel_size - 1) / 2).astype("int32")
    padding = np.broadcast_to(padding, (dims,))
    return tuple(padding.tolist())



def default_norm(x, epsilon=1e-5, momentum=0.997, **kwargs):
    return tf.keras.layers.BatchNormalization(momentum=momentum, epsilon=epsilon, **kwargs)(x)

def default_act(x, **kwargs):
    return tf.keras.layers.ReLU(**kwargs)(x)

def default_dropout(x, rate, **kwargs):
    return tf.keras.layers.Dropout(rate=rate, **kwargs)(x)

default_mode = "tensorflow"

class Config:
    def __init__(self, norm=default_norm, act=default_act, mode=default_mode, resize_align_corners=False, upsample_mode="resize", dropout=default_dropout):
        self.mode = mode
        if not mode in ["tensorflow", "pytorch"]:
            raise ValueError(f"Invalid config mode {mode}")

        def conv(x, filters=None, stride=1, kernel_size=3, dilation_rate=1, groups=1, bias=True, name=None):
            if filters is None:
                filters = x.shape[-1]
            if mode == "tensorflow":
                return ConvND(filters=filters, strides=stride, kernel_size=kernel_size, dilation_rate=dilation_rate, groups=groups, use_bias=bias, padding="SAME", name=name)(x)
            elif mode == "pytorch":
                x = ZeroPaddingND(get_pytorch_same_padding(len(x.get_shape()) - 2, kernel_size, dilation_rate))(x)
                return ConvND(filters=filters, strides=stride, kernel_size=kernel_size, dilation_rate=dilation_rate, groups=groups, use_bias=bias, padding="VALID", name=name)(x)
            else:
                assert False
        self.conv = conv

        if resize_align_corners:
            self.resize = lambda x, shape, method, name=None: tf.compat.v1.image.resize(x, shape, method=method, align_corners=True, name=name)
        else:
            self.resize = lambda x, shape, method, name=None: tf.image.resize(x, shape, method=method, name=name)

        def maxpool(x, stride, kernel_size, name):
            if mode == "tensorflow":
                return MaxPoolND(pool_size=kernel_size, strides=stride, padding="SAME", name=name)(x)
            elif mode == "pytorch":
                x = ZeroPaddingND(get_pytorch_same_padding(len(x.get_shape()) - 2, kernel_size))(x)
                return MaxPoolND(pool_size=kernel_size, strides=stride, padding="VALID", name=name)(x)
            else:
                assert False
        def avgpool(x, stride, kernel_size, name):
            if mode == "tensorflow":
                return AveragePoolingND(pool_size=kernel_size, strides=stride, padding="SAME", name=name)(x)
            elif mode == "pytorch":
                x = ZeroPaddingND(get_pytorch_same_padding(len(x.get_shape()) - 2, kernel_size))(x)
                return AveragePoolingND(pool_size=kernel_size, strides=stride, padding="VALID", name=name)(x)
            else:
                assert False
        def pool(x, mode, kernel_size, stride=1, name=None):
            if mode.lower() == "max":
                return maxpool(x, kernel_size=kernel_size, stride=stride, name=name)
            elif mode.lower() == "avg":
                return avgpool(x, kernel_size=kernel_size, stride=stride, name=name)
            else:
                raise ValueError(f"Invalid pooling mode {mode}")
        self.pool = pool

        if upsample_mode == "resize":
            def upsample(x, factor, method="nearest", name=None):
                return self.resize(x, factor * tf.shape(x)[1:-1], method=method, name=name)
        elif upsample_mode == "upsample-pool":
            def upsample(x, factor, method="nearest", name=None):
                if method == "nearest":
                    return UpSamplingND(factor)(x)
                elif method == "bilinear":
                    index = 0
                    while factor > 1: # Simple prime factorization
                        for k in range(2, factor + 1):
                            if factor % k == 0:
                                break
                        x = UpSamplingND(k, name=join(name, str(index), "upsample"))(x)
                        x = tf.pad(x, [[0, 0]] + [[1, 1] for _ in range(len(x.shape) - 2)] + [[0, 0]], mode="SYMMETRIC", name=join(name, str(index), "pad"))
                        x = tf.nn.avg_pool(x, ksize=k + 1, strides=1, padding="VALID", name=join(name, str(index), "avgpool"))
                        factor = factor // k
                        index += 1
                    return x
                else:
                    raise ValueError(f"Invalid method {method}")
        else:
            raise ValueError(f"Invalid upsample mode {upsample_mode}")
        self.upsample = upsample

        if isinstance(dropout, str):
            if dropout == "spatial":
                dropout = lambda x, rate, name=None: SpatialDropoutND(rate, name=name)(x)
            else:
                raise ValueError(f"Invalid dropout mode {dropout}")
        self.dropout = dropout # TODO: this should be named spatial dropout

        self.norm = norm
        self.act = act
