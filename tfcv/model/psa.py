import tensorflow as tf
from . import config, transformer, resnet
from .util import *

def spatial_softmax_pool(value, query=None):
    if query is None:
        query = value
    # query: # [batch, dims..., filters_q]
    # value: # [batch, dims..., filters_v]

    query = transformer.tokenize(query) # [batch, dims, filters_q]
    query = tf.nn.softmax(query, axis=1)

    value = transformer.tokenize(value) # [batch, dims, filters_v]

    x = tf.linalg.matmul(query, value, transpose_a=True) # [batch, filters_q, filters_v]

    return x

def channel_softmax_pool(value, query=None):
    if query is None:
        query = value
    # query: # [batch, dims..., filters]
    # value: # [batch, 1..., filters]

    query = transformer.tokenize(query) # [batch, dims, filters]
    query = tf.nn.softmax(query, axis=-1)

    value = transformer.tokenize(value) # [batch, 1, filters]

    x = tf.linalg.matmul(query, value, transpose_b=True) # [batch, dims, 1]

    return x



def spatial_attention(x, sequential, reduction=4, fix_bias=True, filters=None, name="spatial_attention", config=config.Config()):
    x_orig = x
    if filters is None:
        filters = x.shape[-1]
    filters_intermediate = filters // 2

    weights = spatial_softmax_pool(
        query=conv(x, 1, kernel_size=1, bias=False, name=join(name, "query", "conv"), config=config), # [batch, dims..., 1]
        value=conv(x, filters_intermediate, kernel_size=1, bias=False, name=join(name, "value", "conv"), config=config), # [batch, dims..., filters_intermediate]
    ) # [batch, 1, filters_intermediate]
    weights = transformer.detokenize(weights, shape=[1] * (len(x.shape) - 2)) # [batch, 1..., filters_intermediate]

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




def spatial_avg_pool(x):
    return tf.reduce_mean(x, axis=list(range(len(x.shape)))[1:-1], keepdims=True)

def channel_attention(x, filters=None, name="channel_attention", config=config.Config()):
    x_orig = x
    if filters is None:
        filters = x.shape[-1]
    filters_intermediate = filters // 2

    weights = channel_softmax_pool(
        query=conv(x, filters_intermediate, kernel_size=1, bias=False, name=join(name, "query", "conv"), config=config), # [batch, dims..., filters_intermediate]
        value=spatial_avg_pool(conv(x, filters_intermediate, kernel_size=1, bias=False, name=join(name, "value", "conv"), config=config)) # [batch, 1..., filters_intermediate]
    ) # [batch, dims, 1]
    weights = transformer.detokenize(weights, shape=tf.shape(x)[1:-1]) # [batch, dims..., filters_intermediate]

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
