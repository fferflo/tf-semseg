import tensorflow as tf
import numpy as np

def default_norm(x, *args, epsilon=1e-5, momentum=0.997, **kwargs):
    return tf.keras.layers.BatchNormalization(*args, momentum=momentum, epsilon=epsilon, **kwargs)(x)

def default_act(x, **kwargs):
    return tf.keras.layers.ReLU(**kwargs)(x)

default_mode = "tensorflow"

class Config:
    def __init__(self, norm=default_norm, act=default_act, mode=default_mode, resize_align_corners=False):
        assert mode in ["tensorflow", "pytorch"]

        def conv(x, *args, **kwargs):
            if mode == "tensorflow":
                return Conv(*args, **kwargs)(x)
            elif mode == "pytorch":
                return PyTorchConv(*args, **kwargs)(x)
            else:
                assert False
        self.conv = conv

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
                return Conv(*args, **kwargs)(x)
            elif mode == "pytorch":
                return PyTorchAveragePool(*args, **kwargs)(x)
            else:
                assert False
        self.avgpool = avgpool

        self.upsample = lambda x, *args, **kwargs: UpSample(*args, **kwargs)(x)

        if resize_align_corners:
            self.resize = lambda x, shape, method: tf.compat.v1.image.resize(x, shape, method=method, align_corners=True)
        else:
            self.resize = lambda x, shape, method: tf.image.resize(x, shape, method=method)

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

def ZeroPad(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.ZeroPadding1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.ZeroPadding2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.ZeroPadding3D(*args, **kwargs)(x)
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
