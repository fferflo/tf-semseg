import tensorflow as tf
from ... import hrnet, decode
from ...config import Config
from ...util import *
import re, os
import numpy as np

color_mean = np.asarray([0.485, 0.456, 0.406])
color_std = np.asarray([0.229, 0.224, 0.225])

def preprocess(color):
    color = color / 255.0
    color = (color - color_mean) / color_std
    return color

def convert_name(layer):
    key = layer
    key = key.replace("norm", "bn")
    key = re.sub("^stem/([0-9]*)/([a-z]*)$", "\\2\\1", key)

    key = re.sub("^block1/module1/branch1/unit([0-9]*)/", lambda m: f"layer1.{int(m.group(1)) - 1}.", key)
    key = re.sub("reduce/([a-z]*)$", lambda m: m.group(1) + "1", key)
    key = re.sub("center/([a-z]*)$", lambda m: m.group(1) + "2", key)
    key = re.sub("expand/([a-z]*)$", lambda m: m.group(1) + "3", key)

    key = re.sub("^block([0-9]*)/module([0-9]*)/branch([0-9]*)/unit([0-9]*)/([0-9]*)/([a-z]*)$",
        lambda m: f"stage{m.group(1)}.{int(m.group(2)) - 1}.branches.{int(m.group(3)) - 1}.{int(m.group(4)) - 1}.{m.group(6)}{m.group(5)}", key)

    key = re.sub("shortcut/conv$", "downsample.0", key)
    key = re.sub("shortcut/bn$", "downsample.1", key)

    if "transition" in key:
        key = re.sub("/conv$", "/0", key)
        key = re.sub("/bn$", "/1", key)
        key = re.sub("^block([0-9]*)/transition/branch([0-9]*)/([0-9]*)$", lambda m: f"transition{m.group(1)}.{int(m.group(2)) - 1}.{m.group(3)}", key)
        key = re.sub("^block([0-9]*)/transition/branch([0-9]*)/([0-9]*)/([0-9]*)$", lambda m: f"transition{m.group(1)}.{int(m.group(2)) - 1}.{m.group(3)}.{m.group(4)}", key)

    if "fuse" in key:
        key = re.sub("/conv$", "/0", key)
        key = re.sub("/bn$", "/1", key)
        key = re.sub("^block([0-9]*)/module([0-9]*)/fuse_branch([0-9]*)to([0-9]*)/",
            lambda m: f"stage{m.group(1)}.{int(m.group(2)) - 1}.fuse_layers.{m.group(4)}.{m.group(3)}.", key)
        key = re.sub("upsample/([0-9]*)$", "\\1", key)
        key = re.sub("downsample/([0-9]*)/([0-9]*)$", "\\1.\\2", key)

    key = key.replace("last_layer/conv", "last_layer.0")
    key = key.replace("last_layer/bn", "last_layer.1")
    key = key.replace("decode/conv", "last_layer.3")

    return key

def hrnet_v2_w48_cityscapes():
    input = tf.keras.layers.Input((None, None, 3))

    config = Config(
        mode="pytorch",
        norm=lambda x, *args, **kwargs: tf.keras.layers.BatchNormalization(*args, momentum=0.9, epsilon=1e-5, **kwargs)(x),
        resize_align_corners=True
    )

    x = input
    x = hrnet.hrnet_v2_w48(x, config=config)
    x = config.conv(x, filters=x.shape[-1], kernel_size=1, strides=1, use_bias=True, padding="same", name=join("last_layer", "conv")) # Yes, this has bias in pretrained weights
    x = config.norm(x, name=join("last_layer", "norm"))
    x = config.act(x)
    x = decode.decode_resize(x, 19, tf.shape(input)[1:-1], config=config)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=[input], outputs=[x])

    url = "https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_cs_8090_torch11.pth"
    weights = tf.keras.utils.get_file("hrnet_cs_8090_torch11.pth", url)

    import torch

    all_weights = dict(torch.load(weights, map_location=torch.device("cpu")))
    def get_weight(key):
        if not key in all_weights:
            print(f"Variable {key} not found in hrnet_cs_8090_torch11.pth")
            os._exit(-1)
        result = all_weights[key]
        del all_weights[key]
        return np.asarray(result)
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            key = convert_name(layer.name)

            if "conv" in layer.name:
                weights = get_weight(key + ".weight")
                weights = np.transpose(weights, (2, 3, 1, 0))
                if not layer.bias is None:
                    bias = get_weight(key + ".bias")
                    layer.set_weights([weights, bias])
                else:
                    assert (key + ".bias") not in all_weights
                    layer.set_weights([weights])
            elif "norm" in layer.name:
                weights = get_weight(key + ".weight")
                bias = get_weight(key + ".bias")
                running_mean = get_weight(key + ".running_mean")
                running_var = get_weight(key + ".running_var")
                layer.set_weights([weights, bias, running_mean, running_var])
            else:
                print(f"Invalid type of layer {layer.name}")
                os._exit(-1)
    for key in list(all_weights.keys()):
        if "num_batches_tracked" in key:
            del all_weights[key]
    keys = list(all_weights.keys())
    if len(keys) > 0:
        print(f"Failed to find layer for torch variables: {keys}")
        os._exit(-1)

    return model
