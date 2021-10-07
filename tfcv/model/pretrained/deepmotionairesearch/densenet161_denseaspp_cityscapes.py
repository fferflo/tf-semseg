import tensorflow as tf
import numpy as np
import sys, re, os, tfcv
from google_drive_downloader import GoogleDriveDownloader as gdd
from ... import densenet, aspp, decode
from ...config import Config
from ...util import *

color_mean = np.asarray([0.290101, 0.328081, 0.286964])
color_std = np.asarray([0.182954, 0.186566, 0.184475])
atrous_rates = [3, 6, 12, 18, 24]

def preprocess(color):
    color = color / 255.0
    color = (color - color_mean) / color_std
    return color

def convert_name(key):
    key = re.sub("^densenet/stem/([a-z]*)$", lambda m: f"module.features.{m.group(1)}0", key)

    key = re.sub("^densenet/block([0-9]*)/unit([0-9]*)/([0-9]*)/([a-z]*)$", lambda m:
                f"module.features.denseblock{m.group(1)}.denselayer{m.group(2)}.{m.group(4)}.{m.group(3)}", key)
    key = re.sub("^densenet/block([0-9]*)/transition/([a-z]*)$", lambda m: f"module.features.transition{m.group(1)}.{m.group(2)}", key)

    key = re.sub("^densenet/final/norm$", lambda m: f"module.features.norm5", key)

    key = re.sub("^dense-aspp/atrous([0-9]*)/([0-9]*)/([a-z]*)$", lambda m: f"module.ASPP_{atrous_rates[int(m.group(1)) - 1]}.{m.group(3)}.{m.group(2)}", key)

    key = re.sub("^decode/conv$", lambda m: f"module.classification.1", key)

    return key

config = Config(
    mode="pytorch",
    norm=lambda x, *args, **kwargs: tf.keras.layers.BatchNormalization(*args, momentum=0.9, epsilon=1e-5, **kwargs)(x),
    resize_align_corners=False
)

def create(input=None):
    return_model = input is None
    if input is None:
        input = tf.keras.layers.Input((None, None, 3))

    x = input
    x = densenet.densenet(x, filters=48, num_units=[6, 12, 36, 24], strides=[2, 1, 1, 1], dilation_rates=[1, 1, 2, 4], bottleneck_factor=4, stem=True, config=config)
    x = aspp.denseaspp(x, filters=128, bottleneck_factor=4, atrous_rates=atrous_rates, dropout=0.1, bias=True, name="dense-aspp", config=config)
    x = decode.decode(x, 19, dropout=0.1, shape=tf.shape(input)[1:-1], config=config)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=[input], outputs=[x])

    # TODO: weight initialization from:
    # https://github.com/DeepMotionAIResearch/DenseASPP
    download_file = os.path.join(os.path.expanduser("~"), ".keras", "denseASPP161_795.pkl")
    gdd.download_file_from_google_drive(file_id="1kMKyboVGWlBxgYRYYnOXiA1mj_ufAXNJ", dest_path=download_file)

    tfcv.model.pretrained.weights.load_pth(download_file, model, convert_name) # TODO: rename load_pth to load_torch

    return model if return_model else x
