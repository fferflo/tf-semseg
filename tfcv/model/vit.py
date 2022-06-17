import tensorflow as tf
from .util import *
from . import config, transformer, einops, stochasticdepth
from functools import partial

shortcut = partial(stochasticdepth.shortcut, drop_probability=0.0, scale_at_train_time=True)

def block(x, filters=None, mlp_filters=None, shortcut=shortcut, mlp_layers=2, heads=1, qkv_bias=True, name=None, config=config.Config()):
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
    x = transformer.full_attention(query, key, value, heads=heads, name=join(name, "mha"), config=config)
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

def vit(x, window_size, filters, num_blocks, block=block, pad_mode="center", positional_embedding_patch_nums=None, name=None, config=config.Config()):
    # Create windows
    patch_nums = (tf.shape(x)[1:-1] + window_size - 1) // window_size
    x = transformer.split_windows(x, window_size, pad_mode=pad_mode) # [batch, patch, filters]

    # Embed
    x = conv(x, filters=filters, kernel_size=1, stride=1, bias=True, name=join(name, "embed", "conv"), config=config)
    x = transformer.class_token(x, name=join(name, "embed", "class_token"), config=config)
    x = transformer.positional_embedding_learned(
        x,
        train_patch_nums=positional_embedding_patch_nums,
        new_patch_nums=patch_nums if not positional_embedding_patch_nums is None else None,
        has_class_token=True,
        name=join(name, "embed", "positional_embedding"),
        config=config,
    )

    # Encoder blocks
    for block_index in range(num_blocks):
        x = block(x, name=join(name, f"block{block_index + 1}"), config=config)
        x = set_name(x, join(name, f"block{block_index + 1}"))

    return x, patch_nums

def neck(x, patch_nums, scale, filters=None, resize_method="bilinear", name=None, config=config.Config()):
    # x: [batch, tokens, filters]
    if filters is None:
        filters = x.shape[-1]
    scale = tf.convert_to_tensor(scale)

    x = einops.apply("b (patch_nums...) f -> b patch_nums... f", x[:, 1:], patch_nums=patch_nums)
    x = conv(x, filters=filters, kernel_size=1, stride=1, bias=True, name=join(name, "conv1"), config=config)
    x = resize(x, tf.cast(tf.cast(tf.shape(x)[1:-1], scale.dtype) * scale, "int32"), method=resize_method, config=config)
    x = conv(x, filters=filters, kernel_size=3, stride=1, bias=True, name=join(name, "conv2"), config=config)

    return x
