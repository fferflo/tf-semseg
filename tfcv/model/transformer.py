import tensorflow as tf
from .util import *
from . import config, resnet, stochasticdepth, einops
from functools import partial

shortcut = partial(stochasticdepth.shortcut, drop_probability=0.0, scale_at_train_time=True)

def encode(x, filters=None, mlp_filters=None, shortcut=shortcut, mlp_layers=2, heads=1, qkv_bias=True, name=None, config=config.Config()):
    if mlp_layers < 2:
        raise ValueError(f"Must have at least 2 MLP layers, got {mlp_layers}")
    if filters is None:
        filters = x.shape[-1]
    if mlp_filters is None:
        mlp_filters = x.shape[-1]

    # Self-attention
    x_orig = x
    x = norm(x, name=join(name, "mha", "norm"), config=config)
    x = conv(x, filters=3 * filters, kernel_size=1, stride=1, bias=qkv_bias, name=join(name, "mha", "in_proj"), config=config)
    query, key, value = tf.split(x, num_or_size_splits=3, axis=-1)
    x = multihead_attention(query, key, value, heads=heads, name=join(name, "mha"), config=config)
    x = conv(x, filters=filters, kernel_size=1, stride=1, bias=qkv_bias, name=join(name, "mha", "out_proj"), config=config)
    x = shortcut(x_orig, x, name=join(name, "mha", "shortcut"), config=config)

    # MLP
    x_orig = x
    x = norm(x, name=join(name, "mlp", "norm"), config=config)
    for i in range(mlp_layers):
        x = conv(x, filters=mlp_filters if i < mlp_layers - 1 else filters, kernel_size=1, stride=1, bias=True, name=join(name, "mlp", f"{i + 1}", "conv"), config=config)
        if i < mlp_layers - 1:
            x = act(x, config=config)
        # x = tf.keras.layers.Dropout(0.1)(x) # TODO: dropout
    x = shortcut(x_orig, x, name=join(name, "mlp", "shortcut"), config=config)

    return x

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

def multihead_attention(query, key, value, heads=1, name=None, config=config.Config()):
    # Reduce to single spatial dimension
    if len(query.shape) > 3:
        result_shape = tf.shape(query)[1:-1]
        query = einops.apply("b s... f -> b (s...) f", query) # [batch, tokens_q, filters_qk]
        key = einops.apply("b s... f -> b (s...) f", key) # [batch, tokens_kv, filters_qk]
        value = einops.apply("b s... f -> b (s...) f", value) # [batch, tokens_kv, filters_v]
    else:
        result_shape = None

    # Split heads
    if heads > 1:
        query = split_heads(query, heads, config=config) # [batch, head, tokens_q, filters_qk // heads]
        key = split_heads(key, heads, config=config) # [batch, head, tokens_kv, filters_qk // heads]
        value = split_heads(value, heads, config=config) # [batch, head, tokens_kv, filters_v // heads]

    # Compute attention weights
    query *= query.shape[-1] ** -0.5
    weights = tf.matmul(query, key, transpose_b=True) # [batch, head, tokens_q, tokens_kv]
    # TODO: Add bias?
    weights = tf.nn.softmax(weights, axis=-1) # [batch, head, tokens_q, tokens_kv]
    # TODO: dropout here?

    # Apply attention weights
    result = tf.matmul(weights, value) # [batch, head, tokens_q, filters_v // heads]

    # Combine heads
    if heads > 1:
        result = merge_heads(result, config=config) # [batch, tokens_q, filters_v]

    # Reshape spatial dimensions
    if not result_shape is None:
        result = einops.apply("b (s...) f -> b s... f", result, s=result_shape)

    return result
