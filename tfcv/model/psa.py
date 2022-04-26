import tensorflow as tf
from . import config, transformer, resnet, einops
from .util import *

def spatial_softmax_pool(weight, value):
    weight = tf.keras.activations.softmax(weight, axis=list(range(len(weight.shape)))[1:-1])
    return einops.apply("b s... fq, b s... fv -> b fq fv", weight, value)

def spatial_avg_pool(x):
    return einops.apply("b s... f -> b 1... f", x, output_ndims=len(x.shape), reduction="mean")

def channel_softmax_pool(weight, value):
    weight = tf.keras.activations.softmax(weight, axis=-1)
    return einops.apply("b s... f, b 1... f -> b s... 1", weight, value)




def spatial_attention(x, sequential, reduction=4, fix_bias=True, filters=None, name="spatial_attention", config=config.Config()):
    if filters is None:
        filters = x.shape[-1]
    filters_intermediate = filters // 2

    weights = spatial_softmax_pool(
        weight=conv(x, 1, kernel_size=1, bias=False, name=join(name, "query", "conv"), config=config), # [batch, dims..., 1]
        value=conv(x, filters_intermediate, kernel_size=1, bias=False, name=join(name, "value", "conv"), config=config), # [batch, dims..., filters_intermediate]
    ) # [batch, 1, filters_intermediate]
    weights = einops.apply("b 1 f -> b 1... f", weights, output_ndims=len(x.shape))

    if sequential:
        weights = conv(weights, filters_intermediate // reduction, kernel_size=1, bias=not fix_bias, name=join(name, "weight", "1", "conv"), config=config) # [batch, 1..., channels]
        weights = tf.keras.layers.LayerNormalization(name=join(name, "weight", "1", "norm"))(weights)
        weights = act(weights, config=config)
        weights = conv(weights, filters, kernel_size=1, bias=True, name=join(name, "weight", "2", "conv"), config=config) # [batch, 1..., channels]
    else:
        weights = conv(weights, filters, kernel_size=1, bias=fix_bias, name=join(name, "weight", "conv"), config=config) # [batch, 1..., channels]

    weights = tf.math.sigmoid(weights)

    x = x * weights
    return x

def channel_attention(x, filters=None, name="channel_attention", config=config.Config()):
    if filters is None:
        filters = x.shape[-1]
    filters_intermediate = filters // 2

    weights = channel_softmax_pool(
        weight=conv(x, filters_intermediate, kernel_size=1, bias=False, name=join(name, "query", "conv"), config=config), # [batch, dims..., filters_intermediate]
        value=spatial_avg_pool(conv(x, filters_intermediate, kernel_size=1, bias=False, name=join(name, "value", "conv"), config=config)) # [batch, 1..., filters_intermediate]
    ) # [batch, dims..., 1]

    weights = tf.math.sigmoid(weights)

    x = x * weights
    return x




def sequential(x, reduction=4, filters=None, fix_bias=True, name="psa", config=config.Config()):
    x = spatial_attention(x, sequential=True, reduction=reduction, filters=filters, fix_bias=fix_bias, name=join(name, "spatial"), config=config)
    x = channel_attention(x, filters=filters, name=join(name, "channel"), config=config)
    return x

def parallel(x, reduction=4, filters=None, fix_bias=True, name="psa", config=config.Config()):
    x_s = spatial_attention(x, sequential=False, reduction=reduction, filters=filters, fix_bias=fix_bias, name=join(name, "spatial"), config=config)
    x_c = channel_attention(x, filters=filters, name=join(name, "channel"), config=config)
    return x_s + x_c
