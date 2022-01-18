import tensorflow as tf
import numpy as np
import re,  tfcv
from ... import vit, upernet, transformer, decode
from ... import config as config_
from ...util import *
from .util import convert_name_upernet
from functools import partial

color_mean = np.asarray([123.675, 116.28, 103.53])
color_std = np.asarray([58.395, 57.12, 57.375])

def preprocess(color):
    color = (color - color_mean) / color_std
    return color

def convert_name(key):
    key = key.replace("vit/embed/conv", "backbone.patch_embed.projection")
    key = key.replace("vit/embed/positional_embedding", "backbone.pos_embed")
    key = key.replace("vit/embed/class_token", "backbone.cls_token")

    key = re.sub("^vit/block([0-9]*)/mha/norm$", lambda m: f"backbone.layers.{int(m.group(1)) - 1}.ln1", key)
    key = re.sub("^vit/block([0-9]*)/mha/([a-z]*)_proj$", lambda m: f"backbone.layers.{int(m.group(1)) - 1}.attn.attn.{m.group(2)}_proj", key)
    key = re.sub("^vit/block([0-9]*)/mlp/norm$", lambda m: f"backbone.layers.{int(m.group(1)) - 1}.ln2", key)
    key = re.sub("^vit/block([0-9]*)/mlp/1/conv$", lambda m: f"backbone.layers.{int(m.group(1)) - 1}.ffn.layers.0.0", key)
    key = re.sub("^vit/block([0-9]*)/mlp/2/conv$", lambda m: f"backbone.layers.{int(m.group(1)) - 1}.ffn.layers.1", key)

    key = re.sub("^neck([0-9]*)/conv1$", lambda m: f"neck.lateral_convs.{int(m.group(1)) - 1}.conv", key)
    key = re.sub("^neck([0-9]*)/conv2$", lambda m: f"neck.convs.{int(m.group(1)) - 1}.conv", key)

    key = convert_name_upernet(key)

    return key

encoder_config = config_.PytorchConfig(
    norm=lambda x, *args, **kwargs: tf.keras.layers.LayerNormalization(*args, epsilon=1e-6, **kwargs)(x),
    act=tf.keras.activations.gelu,
    resize=config_.partial_with_default_args(config_.resize, align_corners=False),
)

decoder_config = config_.PytorchConfig(
    norm=lambda x, *args, **kwargs: tf.keras.layers.BatchNormalization(*args, momentum=0.9, epsilon=1e-5, **kwargs)(x),
    resize=config_.partial_with_default_args(config_.resize, align_corners=False),
)

def create(input=None):
    return_model = input is None
    if input is None:
        input = tf.keras.layers.Input((None, None, 3))

    x = input
    vit_block = partial(transformer.encode, filters=768, mlp_filters=4 * 768, mlp_layers=2, heads=12, qkv_bias=True)
    x, patch_nums = vit.vit(x, window_size=16, filters=768, num_blocks=12, block=vit_block, pad_mode="back",
            positional_embedding_patch_nums=[32, 32], name="vit", config=encoder_config)

    xs = [get_predecessor(x, lambda name: name.endswith(f"block{i}")) for i in [3, 6, 9, 12]]
    xs = [vit.neck(x, patch_nums, scale=s, name=f"neck{i + 1}", config=decoder_config) for i, x, s in zip(range(4), xs, [4, 2, 1, 0.5])]
    x = upernet.head(xs, filters=512, psp_bin_sizes=[1, 2, 3, 6], name="head", config=decoder_config)

    x = decode.decode(x, filters=150, shape=tf.shape(input)[1:-1], dropout=0.1, name="decode", config=decoder_config)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=[input], outputs=[x])

    # TODO: weight initialization from:
    # https://github.com/open-mmlab/mmsegmentation/tree/master/configs/vit
    url = "https://download.openmmlab.com/mmsegmentation/v0.5/vit/upernet_vit-b16_mln_512x512_80k_ade20k/upernet_vit-b16_mln_512x512_80k_ade20k-0403cee1.pth"
    weights = tf.keras.utils.get_file("upernet_vit-b16_mln_512x512_80k_ade20k-0403cee1.pth", url)

    tfcv.model.pretrained.weights.load_pth(weights, model, convert_name, ignore=lambda n: n.startswith("auxiliary"), map={
        "backbone.patch_embed.projection.weight": lambda w: np.expand_dims(np.reshape(np.transpose(w, (2, 3, 1, 0)), [-1, w.shape[0]]), axis=0)
    })

    return model if return_model else x
