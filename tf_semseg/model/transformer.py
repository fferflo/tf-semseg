import tensorflow as tf
from .util import *
from . import config

def split_heads(x, heads):
    assert x.shape[-1] % heads == 0
    filters_per_head = tf.shape(x)[:-1] // heads
    new_shape = tf.stack([tf.shape(x)[:-1] + [heads, filters_per_head]], axis=0)
    return tf.transpose(tf.reshape(x, new_shape), (0, 3, 1, 2))

def combine_heads(x):
    new_shape = tf.stack([tf.shape(x)[:-2] + [x.shape[-2] * x.shape[-1]]], axis=0)
    return tf.transpose(tf.reshape(x, new_shape), (0, 2, 3, 1))

def tokenize(x):
    return tf.reshape(x, tf.stack([tf.shape(x)[0], -1, x.shape[-1]], axis=0))

def detokenize(x, shape):
    return tf.reshape(x, tf.concat([tf.shape(x)[:1], shape, tf.shape(x)[-1:]], axis=0))

def multi_head_attention(query, key, value, heads=1, config=config.Config()):
    # Reduce to single spatial dimension
    if len(query.shape) > 3:
        result_shape = tf.shape(query)[1:-1]
        query = tokenize(query) # [batch, tokens_q, filters_qk]
        key = tokenize(key) # [batch, tokens_kv, filters_qk]
        value = tokenize(value) # [batch, tokens_kv, filters_v]
    else:
        result_shape = None

    # Split heads
    if heads > 1:
        query = split_heads(query, heads) # [batch, head, tokens_q, filters_qk // heads]
        key = split_heads(key, heads) # [batch, head, tokens_kv, filters_qk // heads]
        value = split_heads(value, heads) # [batch, head, tokens_kv, filters_v // heads]

    # Compute attention weights
    query *= (query.shape[-1] // heads) ** -0.5
    weights = tf.matmul(query, key, transpose_b=True) # [batch, head, tokens_q, tokens_kv]
    # TODO: Add bias?
    weights = tf.nn.softmax(weights, axis=-1) # [batch, head, tokens_q, tokens_kv]
    # TODO: dropout here?

    # Apply attention weights
    result = tf.matmul(weights, value) # [batch, head, tokens_q, filters_v // heads]

    # Combine heads
    if heads > 1:
        result = combine_heads(result) # [batch, tokens_q, filters_v]

    # Reshape spatial dimensions
    if not result_shape is None:
        result = detokenize(result, result_shape)

    return result
