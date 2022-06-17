import tensorflow as tf
import numpy as np
from .util import *
from . import config, transformer, einops, stochasticdepth, graph
from functools import partial

shortcut = partial(stochasticdepth.shortcut, drop_probability=0.0, scale_at_train_time=True)

def patch_embed(x, filters, patch_size=7, stride=4, bias=True, name=None, config=config.Config()):
    x = conv(x, filters=filters, kernel_size=patch_size, stride=stride, bias=bias, name=join(name, "conv"), config=config)
    x = norm(x, name=join(name, "norm"), config=config)
    return x

# Spatial reduction attention: https://openaccess.thecvf.com/content/ICCV2021/papers/Wang_Pyramid_Vision_Transformer_A_Versatile_Backbone_for_Dense_Prediction_Without_ICCV_2021_paper.pdf
def block(x, filters=None, mlp_ratio=4, shortcut=shortcut, sr_ratio=1, mlp_layers=2, heads=1, kernel_size=3, qkv_bias=True, name=None, config=config.Config()):
    if mlp_layers < 2:
        raise ValueError(f"Must have at least 2 MLP layers, got {mlp_layers}")
    if filters is None:
        filters = x.shape[-1]
    mlp_filters = filters * mlp_ratio

    # Self-attention
    x_orig = x
    x = norm(x, name=join(name, "mha", "norm"), config=config)
    query = conv(x, filters=1 * filters, kernel_size=1, stride=1, bias=qkv_bias, name=join(name, "mha", "query"), config=config)
    if sr_ratio > 1:
        x = conv(x, filters=filters, kernel_size=sr_ratio, stride=sr_ratio, bias=qkv_bias, name=join(name, "mha", "spatial-reduction", "conv"), config=config)
        x = norm(x, name=join(name, "mha", "spatial-reduction", "norm"), config=config)
    x = conv(x, filters=2 * filters, kernel_size=1, stride=1, bias=qkv_bias, name=join(name, "mha", "key-value"), config=config)
    key, value = tf.split(x, num_or_size_splits=2, axis=-1)
    x = transformer.full_attention(query, key, value, heads=heads, name=join(name, "mha"), config=config)
    x = conv(x, filters=filters, kernel_size=1, stride=1, bias=qkv_bias, name=join(name, "mha", "out_proj"), config=config)
    x = shortcut(x_orig, x, name=join(name, "mha", "shortcut"), config=config)

    # MLP
    x_orig = x
    x = norm(x, name=join(name, "mlp", "norm"), config=config)
    for i in range(mlp_layers):
        x = conv(x, filters=mlp_filters if i < mlp_layers - 1 else filters, kernel_size=1, stride=1, bias=True, name=join(name, "mlp", f"{i + 1}", "pointwise"), config=config)
        if i < mlp_layers - 1:
            x = conv(x, filters=mlp_filters, groups=mlp_filters, kernel_size=kernel_size, stride=1, bias=True, name=join(name, "mlp", f"{i + 1}", "depthwise"), config=config)
            x = act(x, config=config)
        # x = tf.keras.layers.Dropout(0.1)(x) # TODO: dropout
    x = shortcut(x_orig, x, name=join(name, "mlp", "shortcut"), config=config)

    x = set_name(x, name) # TODO: add set_name in all blocks

    return x

def encode(x, num_units, filters, patch_sizes, strides, sr_ratios, heads, last_drop_path_rate=0.1, block=block, name=None, config=config.Config()):
    total_unit_index = 0
    drop_path_rates = np.linspace(0.0, last_drop_path_rate, sum(num_units))
    for block_index in range(len(num_units)):
        b_name = join(name, f"block{block_index + 1}")
        x = patch_embed(x, filters=filters[block_index], patch_size=patch_sizes[block_index], stride=strides[block_index], name=join(b_name, "patch-embed"), config=config)
        for unit_index in range(num_units[block_index]):
            u_name = join(b_name, f"unit{unit_index + 1}")
            x = block(
                x,
                filters=filters[block_index],
                sr_ratio=sr_ratios[block_index],
                heads=heads[block_index],
                shortcut=partial(stochasticdepth.shortcut, drop_probability=drop_path_rates[total_unit_index], scale_at_train_time=True),
                name=u_name,
                config=config,
            )
            total_unit_index += 1
        x = norm(x, name=join(b_name, "norm"), config=config)
        x = set_name(x, b_name)
    return x

def decode(xs, filters, name=None, config=config.Config()):
    for i in range(len(xs)):
        xs[i] = conv(xs[i], filters=filters, kernel_size=1, stride=1, name=join(name, f"in{i + 1}"), config=config)

    for i in range(1, len(xs)):
        xs[i] = resize(xs[i], tf.shape(xs[0])[1:-1], method="bilinear", config=config)
    x = tf.concat(xs[::-1], axis=-1)

    x = conv_norm_act(x, filters=filters, kernel_size=1, stride=1, name=join(name, f"fuse"), config=config)
    return x

def mit_b0(x, block=block, name="mit_b0", config=config.Config()):
    return encode(
        x,
        block=block,
        num_units=[2, 2, 2, 2],
        filters=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        name=name,
        config=config,
    )

def mit_b1(x, block=block, name="mit_b1", config=config.Config()):
    return encode(
        x,
        block=block,
        num_units=[2, 2, 2, 2],
        filters=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        name=name,
        config=config,
    )

def mit_b2(x, block=block, name="mit_b2", config=config.Config()):
    return encode(
        x,
        block=block,
        num_units=[3, 4, 6, 3],
        filters=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        name=name,
        config=config,
    )

def mit_b3(x, block=block, name="mit_b3", config=config.Config()):
    return encode(
        x,
        block=block,
        num_units=[3, 4, 18, 3],
        filters=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        name=name,
        config=config,
    )

def mit_b4(x, block=block, name="mit_b4", config=config.Config()):
    return encode(
        x,
        block=block,
        num_units=[3, 8, 27, 3],
        filters=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        name=name,
        config=config,
    )

def mit_b5(x, block=block, name="mit_b5", config=config.Config()):
    return encode(
        x,
        block=block,
        num_units=[3, 6, 40, 3],
        filters=[64, 128, 320, 512],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        heads=[1, 2, 5, 8],
        name=name,
        config=config,
    )

def segformer_b0(x, block=block, name="segformer_b0", config=config.Config(), decoder_config=None):
    if decoder_config is None:
        decoder_config = config

    x = mit_b0(x, name=name, config=config)
    xs = [graph.get_unique(x, pred=lambda layer: layer.name.endswith(f"block{i}")) for i in [1, 2, 3, 4]]
    x = decode(xs, filters=256, name=join(name, "head"), config=decoder_config)

    return x

def segformer_b1(x, block=block, name="segformer_b1", config=config.Config(), decoder_config=None):
    if decoder_config is None:
        decoder_config = config

    x = mit_b1(x, name=name, config=config)
    xs = [graph.get_unique(x, pred=lambda layer: layer.name.endswith(f"block{i}")) for i in [1, 2, 3, 4]]
    x = decode(xs, filters=256, name=join(name, "head"), config=decoder_config)

    return x

def segformer_b2(x, block=block, name="segformer_b2", config=config.Config(), decoder_config=None):
    if decoder_config is None:
        decoder_config = config

    x = mit_b2(x, name=name, config=config)
    xs = [graph.get_unique(x, pred=lambda layer: layer.name.endswith(f"block{i}")) for i in [1, 2, 3, 4]]
    x = decode(xs, filters=768, name=join(name, "head"), config=decoder_config)

    return x

def segformer_b3(x, block=block, name="segformer_b3", config=config.Config(), decoder_config=None):
    if decoder_config is None:
        decoder_config = config

    x = mit_b3(x, name=name, config=config)
    xs = [graph.get_unique(x, pred=lambda layer: layer.name.endswith(f"block{i}")) for i in [1, 2, 3, 4]]
    x = decode(xs, filters=768, name=join(name, "head"), config=decoder_config)

    return x

def segformer_b4(x, block=block, name="segformer_b4", config=config.Config(), decoder_config=None):
    if decoder_config is None:
        decoder_config = config

    x = mit_b4(x, name=name, config=config)
    xs = [graph.get_unique(x, pred=lambda layer: layer.name.endswith(f"block{i}")) for i in [1, 2, 3, 4]]
    x = decode(xs, filters=768, name=join(name, "head"), config=decoder_config)

    return x

def segformer_b5(x, block=block, name="segformer_b5", config=config.Config(), decoder_config=None):
    if decoder_config is None:
        decoder_config = config

    x = mit_b5(x, name=name, config=config)
    xs = [graph.get_unique(x, pred=lambda layer: layer.name.endswith(f"block{i}")) for i in [1, 2, 3, 4]]
    x = decode(xs, filters=768, name=join(name, "head"), config=decoder_config)

    return x
