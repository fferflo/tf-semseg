import tensorflow as tf
from .util import *
from . import config
from . import upsample as upsample_

def decode(x, filters, shape=None, stride=None, dropout=None, name="decode", upsample="bilinear", bias=True, config=config.Config()):
    if not dropout is None:
        x = tf.keras.layers.Dropout(dropout)(x)

    if not shape is None or not stride is None:
        if not shape is None and not stride is None:
            raise ValueError("Can only pass either stride or shape")
        if shape is None:
            shape = tf.shape(x)[1:-1] * stride
        if stride is None:
            stride = shape // tf.shape(x)[1:-1]
        elif not tf.is_tensor(stride):
            assert isinstance(stride, int)
            stride = [stride] * len(x.shape[1:-1])

        if upsample == "bilinear":
            x = conv(x, filters, kernel_size=1, stride=1, bias=bias, name=join(name, "conv"), config=config)
            x = resize(x, shape, method="bilinear", name=join(name, "resize"), config=config)
        elif upsample == "subpixel":
            if stride[0] != stride[1]:
                raise ValueError("Subpixel upsample method can only be used with equal strides per dimension")
            x = upsample_.subpixel_conv(x, stride=stride[0], filters=filters, name=join(name, "subpixel-conv"), config=config)
        else:
            raise ValueError(f"Invalid upsample type {upsample}")
    else:
        x = conv(x, filters, kernel_size=1, stride=1, bias=bias, name=join(name, "conv"), config=config)
    return x
