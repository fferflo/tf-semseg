import tensorflow as tf
from ... import hrnet, decode, ocr
from ...config import Config
from ...util import *
import numpy as np
import tf_semseg, re, os

color_mean = np.asarray([0.485, 0.456, 0.406])
color_std = np.asarray([0.229, 0.224, 0.225])

def preprocess(color):
    color = color / 255.0
    color = (color - color_mean) / color_std
    return color

def convert_name_hrnet(layer):
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

    return key

def hrnet_v2_w48_cityscapes():
    input = tf.keras.layers.Input((None, None, 3))

    def convert_name(name):
        name = convert_name_hrnet(name)
        name = name.replace("last_layer/conv", "last_layer.0")
        name = name.replace("last_layer/bn", "last_layer.1")
        name = name.replace("decode/conv", "last_layer.3")
        return name

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
    tf_semseg.model.pretrained.weights.load_pth(weights, model, convert_name)

    return model

def convert_name_ocr(key):
    key = key.replace("norm", "bn")

    key = re.sub("^ocr/regions/conv$", "aux_head.0", key)
    key = re.sub("^ocr/regions/bn$", "aux_head.1", key)
    key = re.sub("^ocr/regions/decode/conv$", "aux_head.3", key)
    key = re.sub("ocr/initial/conv", "conv3x3_ocr.0", key)
    key = re.sub("ocr/initial/bn", "conv3x3_ocr.1", key)
    key = re.sub("decode/conv", "cls_head", key)

    if "distribute" in key:
        def map1(g1, g2, g3):
            if g3 == "bn":
                g2 += 1
            g1 = g1.replace("query", "f_pixel")
            g1 = g1.replace("key", "f_object")
            g1 = g1.replace("down", "f_down")
            g1 = g1.replace("up", "f_up")
            result = f"ocr_distri_head.object_context_block.{g1}.{g2}"
            if g3 == "bn":
                result = result + ".0"
            return result
        key = re.sub("ocr/distribute/([a-z]*)/([0-9]*)/([a-z]*)", lambda m: map1(m.group(1), 2 * (int(m.group(2)) - 1), m.group(3)), key)
        key = re.sub("ocr/distribute/([a-z]*)/([a-z]*)", lambda m: map1(m.group(1), 0, m.group(2)), key)

    key = key.replace("final/conv", "ocr_distri_head.conv_bn_dropout.0")
    key = key.replace("final/bn", "ocr_distri_head.conv_bn_dropout.1.0")

    return key

def hrnet_v2_w48_ocr_cityscapes():
    input = tf.keras.layers.Input((None, None, 3))

    def convert_name(name):
        name = convert_name_hrnet(name)
        name = convert_name_ocr(name)
        name = "model." + name
        return name

    config = Config(
        mode="pytorch",
        norm=lambda x, *args, **kwargs: tf.keras.layers.BatchNormalization(*args, momentum=0.9, epsilon=1e-5, **kwargs)(x),
        resize_align_corners=True
    )

    x = input
    x = hrnet.hrnet_v2_w48(x, config=config)
    x = ocr.ocr(x, regions=19, filters=512, filters_qkv=256, fix_bias_before_norm=False, config=config)
    x = conv_norm_act(x, filters=512, kernel_size=1, stride=1, use_bias=False, name=join("final"), config=config)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = decode.decode_resize(x, 19, tf.shape(input)[1:-1], config=config)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=[input], outputs=[x])

    url = "https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_cs_8162_torch11.pth"
    weights = tf.keras.utils.get_file("hrnet_ocr_cs_8162_torch11.pth", url)
    tf_semseg.model.pretrained.weights.load_pth(weights, model, convert_name, ignore=lambda name: name.startswith("loss."))

    return model
