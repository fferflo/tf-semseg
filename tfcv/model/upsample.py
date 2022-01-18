import tensorflow as tf
from . import config
from .util import *
import tfcv

# def upsample_pool(x):
#     index = 0
#     while stride > 1: # Simple prime factorization
#         for k in range(2, stride + 1):
#             if stride % k == 0:
#                 break
#         x = tfcv.model.keras.UpSamplingND(k, name=join(name, str(index), "upsample"))(x)
#         x = tf.pad(x, [[0, 0]] + [[1, 1] for _ in range(len(x.shape) - 2)] + [[0, 0]], mode="SYMMETRIC", name=join(name, str(index), "pad"))
#         x = tf.nn.avg_pool(x, ksize=k + 1, strides=1, padding="VALID", name=join(name, str(index), "avgpool"))
#         stride = stride // k
#         index += 1
#     return x

def resize(x, stride, method="bilinear", name=None, config=config.Config()):
    return tfcv.model.util.resize(x, shape=tf.shape(x)[1:-1] * stride, method=method, name=name)

def subpixel_conv(x, stride, filters=None, bias=True, name=None, config=config.Config()):
    if len(x.shape) != 4:
        raise ValueError("Subpixel upsample method can only be used on 2d input")
    if filters is None:
        filters = x.shape[-1]
    x = conv(x, filters=filters * stride * stride, kernel_size=1, stride=1, bias=bias, name=join(name, "conv"), config=config)
    x = tf.nn.depth_to_space(x, stride)
    return x

def repeat(x, stride, name=None, config=config.Config()):
    return tfcv.model.keras.UpSamplingND(stride)(x)

# TODO: transposed convolution here
