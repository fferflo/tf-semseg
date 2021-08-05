import tensorflow as tf
from .util import *
from . import config, transformer

def vit(x, window_size, filters, num_blocks, block, pad_mode="center", positional_embedding_patch_nums=None, name=None, config=config.Config()):
    # Create windows
    patch_nums = (tf.shape(x)[1:-1] + window_size - 1) // window_size
    x = transformer.split_windows(x, window_size, pad_mode=pad_mode) # [batch, patch, filters]

    # Embed
    x = conv(x, filters=filters, kernel_size=1, stride=1, bias=True, name=join(name, "embed", "conv"), config=config)
    x = transformer.class_token(x, name=join(name, "embed", "class_token"), config=config)
    x = transformer.positional_embedding(
        x,
        train_patch_nums=positional_embedding_patch_nums,
        new_patch_nums=patch_nums if not positional_embedding_patch_nums is None else None,
        has_class_token=True,
        name=join(name, "embed", "positional_embedding"),
        config=config)

    # Encoder blocks
    for block_index in range(num_blocks):
        x = block(x, name=join(name, f"block{block_index + 1}"), config=config)
        x = set_name(x, join(name, f"block{block_index + 1}"))

    return x, patch_nums

def neck(x, patch_nums, scale, filters=None, resize_method="bilinear", name=None, config=config.Config()):
    if filters is None:
        filters = x.shape[-1]
    scale = tf.convert_to_tensor(scale)

    x = tf.reshape(x[:, 1:], tf.concat([[tf.shape(x)[0]], patch_nums, [x.shape[-1]]], axis=0))
    x = conv(x, filters=filters, kernel_size=1, stride=1, bias=True, name=join(name, "conv1"), config=config)
    x = resize(x, tf.cast(tf.cast(tf.shape(x)[1:-1], scale.dtype) * scale, "int32"), method=resize_method, config=config)
    x = conv(x, filters=filters, kernel_size=3, stride=1, bias=True, name=join(name, "conv2"), config=config)

    return x
