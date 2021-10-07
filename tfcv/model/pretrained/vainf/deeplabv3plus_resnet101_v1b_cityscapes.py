import tensorflow as tf
import numpy as np
import sys, re, os, tfcv
from google_drive_downloader import GoogleDriveDownloader as gdd
from ... import resnet, aspp, decode
from ...config import Config
from ...util import *

color_mean = np.asarray([0.485, 0.456, 0.406])
color_std = np.asarray([0.229, 0.224, 0.225])

def preprocess(color):
    color = color / 255.0
    color = (color - color_mean) / color_std
    return color

def convert_name(key):
    key = key.replace("norm", "bn")

    key = re.sub("^resnet_v1_101/stem_b/([a-z]*)$", lambda m: f"backbone.{m.group(1)}1", key)
    key = re.sub("^resnet_v1_101/block([0-9]*)/unit([0-9]*)/", lambda m: f"backbone.layer{m.group(1)}.{int(m.group(2)) - 1}.", key)

    key = key.replace(".shortcut/conv", ".downsample.0")
    key = key.replace(".shortcut/bn", ".downsample.1")

    key = re.sub("reduce/([a-z]*)$", lambda m: m.group(1) + "1", key)
    key = re.sub("center/([a-z]*)$", lambda m: m.group(1) + "2", key)
    key = re.sub("expand/([a-z]*)$", lambda m: m.group(1) + "3", key)

    key = key.replace("shortcut/conv", "classifier.project.0")
    key = key.replace("shortcut/bn", "classifier.project.1")

    key = key.replace("aspp/1x1/conv", "classifier.aspp.convs.0.0")
    key = key.replace("aspp/1x1/bn", "classifier.aspp.convs.0.1")

    key = re.sub("^aspp/atrous([0-9]*)/conv", lambda m: f"classifier.aspp.convs.{int(m.group(1))}.0", key)
    key = re.sub("^aspp/atrous([0-9]*)/bn", lambda m: f"classifier.aspp.convs.{int(m.group(1))}.1", key)

    key = key.replace("aspp/global/conv", "classifier.aspp.convs.4.1")
    key = key.replace("aspp/global/bn", "classifier.aspp.convs.4.2")

    key = key.replace("aspp/final/conv", "classifier.aspp.project.0")
    key = key.replace("aspp/final/bn", "classifier.aspp.project.1")

    key = key.replace("final/conv", "classifier.classifier.0")
    key = key.replace("final/bn", "classifier.classifier.1")
    key = key.replace("decode/conv", "classifier.classifier.3")

    return key

config = Config(
    mode="pytorch",
    norm=lambda x, *args, **kwargs: tf.keras.layers.BatchNormalization(*args, momentum=0.99, epsilon=1e-5, **kwargs)(x),
    resize_align_corners=False
)

def create(input=None):
    return_model = input is None
    if input is None:
        input = tf.keras.layers.Input((None, None, 3))

    x = input
    x = resnet.resnet_v1_101(x, stem="b", dilate=[False, False, False, True], config=config)

    x_skip = get_predecessor(input, x, lambda x: x.endswith("block1"))
    x_skip = conv_norm_act(x_skip, filters=48, kernel_size=1, stride=1, name="shortcut", config=config)

    x = aspp.aspp(x, filters=256, atrous_rates=[6, 12, 18], config=config)
    x = conv_norm_act(x, filters=256, kernel_size=1, stride=1, name=join("aspp", "final"), config=config)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = resize(x, tf.shape(x_skip)[1:-1], method="bilinear", config=config)
    x = tf.concat([x_skip, x], axis=-1) # TODO: shortcut with resize

    x = conv_norm_act(x, filters=256, kernel_size=3, stride=1, name="final", config=config)
    x = decode.decode(x, 19, shape=tf.shape(input)[1:-1], config=config)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=[input], outputs=[x])

    # TODO: weight initialization from:
    # https://github.com/VainF/DeepLabV3Plus-Pytorch
    download_file = os.path.join(os.path.expanduser("~"), ".keras", "best_deeplabv3plus_resnet101_cityscapes_os16.pth")
    gdd.download_file_from_google_drive(file_id="1t7TC8mxQaFECt4jutdq_NMnWxdm6B-Nb", dest_path=download_file)

    tfcv.model.pretrained.weights.load_pth(download_file, model, convert_name)

    return model if return_model else x
