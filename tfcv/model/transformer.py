import tensorflow as tf
from .util import *
from . import config, shortcut

def tokenize(x):
    return tf.reshape(x, tf.stack([tf.shape(x)[0], tf.math.reduce_prod(tf.shape(x)[1:-1]), x.shape[-1]], axis=0))
# TODO: should this be named flatten and unflatten?
def detokenize(x, shape):
    return tf.reshape(x, tf.concat([tf.shape(x)[:1], shape, tf.shape(x)[-1:]], axis=0))

def encode(x, filters=None, mlp_filters=None, mlp_layers=2, heads=1, qkv_bias=True, name=None, config=config.Config()):
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
    x = shortcut.add(x, x_orig, name=join(name, "mha", "shortcut")) # TODO: x = tfa.layers.StochasticDepth()([x_orig, x]) as shortcut

    # MLP
    x_orig = x
    x = norm(x, name=join(name, "mlp", "norm"), config=config)
    for i in range(mlp_layers):
        x = conv(x, filters=mlp_filters if i < mlp_layers - 1 else filters, kernel_size=1, stride=1, bias=True, name=join(name, "mlp", f"{i + 1}", "conv"), config=config)
        if i < mlp_layers - 1:
            x = act(x, config=config)
        # x = tf.keras.layers.Dropout(0.1)(x) # TODO: dropout
    x = shortcut.add(x, x_orig, name=join(name, "mlp", "shortcut")) # TODO: x = tfa.layers.StochasticDepth()([x_orig, x]) as shortcut

    return x

def positional_embedding(x, train_patch_nums=None, new_patch_nums=None, has_class_token=False, name=None, config=config.Config()):
    if (train_patch_nums is None) != (new_patch_nums is None):
        raise ValueError("Must give as arguments either both of train_patch_nums and new_patch_nums, or neither")
    if train_patch_nums is None:
        num_tokens = tf.shape(x)[1]
    else:
        num_tokens = tf.math.reduce_prod(train_patch_nums) + (1 if has_class_token else 0)

    positions = tf.range(0, num_tokens * tf.math.minimum(tf.shape(x)[0], 1)) # Makes embedding output depend on x
    positional_embedding = tf.keras.layers.Embedding(input_dim=num_tokens, output_dim=x.shape[-1], name=name)(positions)
    positional_embedding = tf.expand_dims(positional_embedding, axis=0)

    if not train_patch_nums is None:
        if has_class_token:
            class_token = positional_embedding[:, :1]
            positional_embedding = positional_embedding[:, 1:]

        positional_embedding = detokenize(positional_embedding, train_patch_nums)
        positional_embedding = resize(positional_embedding, new_patch_nums, method="bicubic", config=config)
        positional_embedding = tokenize(positional_embedding)

        if has_class_token:
            positional_embedding = tf.concat([class_token, positional_embedding], axis=1)

    x = x + positional_embedding
    # TODO: dropout here
    return x

def class_token(x, name=None, config=config.Config()):
    positions = tf.convert_to_tensor([tf.math.minimum(tf.shape(x)[-1], 0)]) # Makes embedding output depend on x
    class_token = tf.keras.layers.Embedding(input_dim=1, output_dim=x.shape[2], name=name)(positions)
    x = tf.concat([tf.expand_dims(class_token, axis=0), x], axis=1)
    return x

def split_windows(x, window_size, pad_mode="center"):
    if isinstance(window_size, int):
        window_size = tf.convert_to_tensor([window_size] * (len(x.shape) - 2))
    patch_nums = (tf.shape(x)[1:-1] + window_size - 1) // window_size
    x = pad_to_size(x, patch_nums * window_size, mode=pad_mode)
    axes = list(range(len(window_size)))

    # Split each axis into patch and window_size
    shape = tf.stack([tf.shape(x)[0]] + [d for i in axes for d in [patch_nums[i], window_size[i]]] + [x.shape[-1]])
    x = tf.reshape(x, shape=shape) # [batch, (patch_num, window_size)..., pixel_filters]

    # Move window_size axes to end
    perm = [0] + [(1 + 2 * i) for i in axes] + [(2 + 2 * i) for i in axes] + [len(x.shape) - 1]
    x = tf.transpose(x, perm) # [batch, patch_num..., window_size..., pixel_filters]

    # Reshape to window sequence
    shape = tf.stack([tf.shape(x)[0], tf.math.reduce_prod(patch_nums), tf.math.reduce_prod(window_size) * x.shape[-1]])
    x = tf.reshape(x, shape=shape) # [batch, patch_num, patch_filters]

    return x

def merge_windows(x, window_size, shape):
    patch_nums = shape // window_size
    axes = list(range(len(window_size)))

    # Reshape to restore patch_num and window_size axes
    shape = tf.stack([tf.shape(x)[0]] + [d for d in patch_nums] + [d for d in window_size] + [x.shape[-1]])
    x = tf.reshape(x, shape=shape) # [batch, patch_num..., window_size..., pixel_filters]

    # Interleave patch_num axes with window_size axes
    perm = [0] + [d for i in axes for d in [1 + i, 1 + i + len(axes)]] + [len(x.shape) - 1]
    x = tf.transpose(x, perm) # [batch, patch_num..., window_size..., pixel_filters]

    # Merge patches per axis
    x = tf.reshape(x, shape=patch_nums * window_size)

    return x

def split_heads(x, heads, config=config.Config()):
    if x.shape[-1] % heads != 0:
        raise ValueError(f"Channel dimension {x.shape[-1]} must be divisible by number of heads {heads}")
    filters_per_head = x.shape[-1] // heads

    if config.mode == "pytorch":
        x = tf.transpose(x, (1, 0, 2)) # [tokens, batch, filters]
        new_shape = tf.concat([tf.shape(x)[:-1], [heads, filters_per_head]], axis=0)
        x = tf.reshape(x, new_shape) # [tokens, batch, head, filters // heads]
        x = tf.transpose(x, (1, 2, 0, 3)) # [batch, head, tokens, filters // heads]
    else:
        new_shape = tf.concat([tf.shape(x)[:-1], [heads, filters_per_head]], axis=0)
        x = tf.reshape(x, new_shape) # [batch, tokens, head, filters // heads]
        x = tf.transpose(x, (0, 2, 1, 3)) # [batch, head, tokens, filters // heads]
    return x

def merge_heads(x, config=config.Config()):
    if config.mode == "pytorch":
        x = tf.transpose(x, (2, 0, 1, 3)) # [tokens, batch, head, filters // heads]
        new_shape = tf.concat([tf.shape(x)[:-2], [x.shape[-2] * x.shape[-1]]], axis=0)
        x =  tf.reshape(x, new_shape) # [tokens, batch, filters]
        x = tf.transpose(x, (1, 0, 2)) # [batch, tokens, filters]
    else:
        x = tf.transpose(x, (0, 2, 1, 3)) # [batch, tokens, head, filters // heads]
        new_shape = tf.concat([tf.shape(x)[:-2], [x.shape[-2] * x.shape[-1]]], axis=0)
        x =  tf.reshape(x, new_shape) # [batch, tokens, filters]
    return x

def multihead_attention(query, key, value, heads=1, name=None, config=config.Config()):
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
        result = detokenize(result, result_shape)

    return result
