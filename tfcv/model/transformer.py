import tensorflow as tf
from .util import *
from . import config, einops
from functools import partial

def positional_embedding_learned(x, train_patch_nums=None, new_patch_nums=None, has_class_token=False, name=None, config=config.Config()):
    if (train_patch_nums is None) != (new_patch_nums is None):
        raise ValueError("Must give as arguments either both of train_patch_nums and new_patch_nums, or neither")
    if train_patch_nums is None:
        num_tokens = tf.shape(x)[1]
    else:
        num_tokens = tf.math.reduce_prod(train_patch_nums) + (1 if has_class_token else 0)

    positions = tf.range(0, num_tokens * tf.math.minimum(tf.shape(x)[0], 1)) # Makes embedding output depend on x
    positional_embedding = tf.keras.layers.Embedding(input_dim=num_tokens, output_dim=x.shape[-1], name=name)(positions)
    positional_embedding = positional_embedding[tf.newaxis]

    if not train_patch_nums is None:
        if has_class_token:
            class_token = positional_embedding[:, :1]
            positional_embedding = positional_embedding[:, 1:]

        positional_embedding = einops.apply("b (s...) f -> b s... f", positional_embedding, s=train_patch_nums)
        positional_embedding = resize(positional_embedding, new_patch_nums, method="bicubic", config=config) # TODO: bicubic should be in vit config?
        positional_embedding = einops.apply("b s... f -> b (s...) f", positional_embedding)

        if has_class_token:
            positional_embedding = tf.concat([class_token, positional_embedding], axis=1)

    x = x + positional_embedding
    # TODO: dropout here
    return x

def class_token(x, name=None, config=config.Config()):
    positions = tf.convert_to_tensor([tf.math.minimum(tf.shape(x)[-1], 0)]) # Makes embedding output depend on x
    class_token = tf.keras.layers.Embedding(input_dim=1, output_dim=x.shape[2], name=name)(positions)
    x = tf.concat([class_token[tf.newaxis], x], axis=1)
    return x

def split_windows(x, window_size, pad_mode="center"):
    patch_nums = (tf.shape(x)[1:-1] + window_size - 1) // window_size
    x = pad_to_size(x, patch_nums * window_size, mode=pad_mode)
    return einops.apply("b (window_num window_size)... c -> b (window_num...) (window_size... c)", x, window_size=window_size)

def merge_windows(x, window_size, shape):
    return einops.apply("b (window_num...) (window_size... c) -> b (window_num window_size)... c", x, window_size=window_size)

def split_heads(x, heads, config=config.Config()):
    if x.shape[-1] % heads != 0:
        raise ValueError(f"Channel dimension {x.shape[-1]} must be divisible by number of heads {heads}")
    return einops.apply("b tokens (heads filters_per_head) -> b heads tokens filters_per_head", x, heads=heads)

def merge_heads(x, config=config.Config()):
    return einops.apply("b heads tokens filters_per_head -> b tokens (heads filters_per_head)", x)

def multihead_attention_function(func):
    def outer(query, key, value, heads=1, name=None, config=config.Config()):
        # Reduce to single spatial dimension
        result_shape = tf.shape(query)[1:-1]
        query = einops.apply("b s... f -> b (s...) f", query)
        key = einops.apply("b s... f -> b (s...) f", key)
        value = einops.apply("b s... f -> b (s...) f", value)

        # Split heads
        query = split_heads(query, heads, config=config) # [batch, head, tokens_q, filters_qk // heads]
        key = split_heads(key, heads, config=config) # [batch, head, tokens_kv, filters_qk // heads]
        value = split_heads(value, heads, config=config) # [batch, head, tokens_kv, filters_v // heads]

        result = func(query, key, value, name=name, config=config)

        # Combine heads
        result = merge_heads(result, config=config) # [batch, tokens_q, filters_v]

        # Reshape spatial dimensions
        result = einops.apply("b (s...) f -> b s... f", result, s=result_shape)

        return result
    return outer

# TODO: attention types: https://github.com/idiap/fast-transformers/tree/master/fast_transformers/attention

# https://proceedings.mlr.press/v119/katharopoulos20a.html
@multihead_attention_function
def linear_attention(query, key, value, epsilon=1e-6, name=None, config=config.Config()):
    query = tf.nn.elu(query) + 1
    key = tf.nn.elu(key) + 1

    factor = tf.cast(tf.shape(value)[1], value.dtype)
    value = value / factor # Prevent fp16 overflow

    kv = tf.matmul(key, value, transpose_a=True) # [batch, head, filters_qk // heads, filters_v // heads]
    z = 1 / (einops.apply("b h tokens_q filters_qk, b h tokens_kv filters_qk -> b h tokens_q", query, key, reduction="sum") + epsilon)

    return einops.apply("b h tokens_q filters_qk, b h filters_qk filters_v, b h tokens_q -> b h tokens_q filters_v", query, kv, z) * factor

# https://arxiv.org/abs/1706.03762
@multihead_attention_function
def full_attention(query, key, value, name=None, config=config.Config()):
    query *= query.shape[-1] ** -0.5
    weights = tf.matmul(query, key, transpose_b=True) # [batch, head, tokens_q, tokens_kv]
    # TODO: Add bias?
    weights = tf.nn.softmax(weights, axis=-1) # [batch, head, tokens_q, tokens_kv]
    # TODO: dropout here?

    return tf.matmul(weights, value) # [batch, head, tokens_q, filters_v // heads]
