import tensorflow as tf
import pyunpack, os, tf_semseg
import numpy as np
from ...config import Config

def preprocess(color):
    color = color - np.array([123.68, 116.779, 103.939])
    color = color[:, :, ::-1]
    return color

def convert_name(name):
    name = name.replace("stem_b/conv", "conv1")
    name = name.replace("stem_b/norm", "conv1/BatchNorm")
    name = name.replace("kernel", "weights")
    name = name.replace("unit", "unit_")
    name = name.replace("reduce/conv", "bottleneck_v1/conv1")
    name = name.replace("reduce/norm", "bottleneck_v1/conv1/BatchNorm")
    name = name.replace("center/conv", "bottleneck_v1/conv2")
    name = name.replace("center/norm", "bottleneck_v1/conv2/BatchNorm")
    name = name.replace("expand/conv", "bottleneck_v1/conv3")
    name = name.replace("expand/norm", "bottleneck_v1/conv3/BatchNorm")
    name = name.replace("shortcut/conv", "bottleneck_v1/shortcut")
    name = name.replace("shortcut/norm", "bottleneck_v1/shortcut/BatchNorm")
    return name

config = Config()

def create_x(dilated, resnet_v1_x, url):
    input = tf.keras.layers.Input((None, None, 3))
    x = input
    x = resnet_v1_x(x, dilated=dilated, stem="b", config=config)
    model = tf.keras.Model(inputs=[input], outputs=[x])

    weights_compressed = tf.keras.utils.get_file(url.split("/")[-1], url)
    weights_uncompressed = weights_compressed[:-len("_2016_08_28.tar.gz")] + ".ckpt"
    if not os.path.isfile(weights_uncompressed):
        pyunpack.Archive(weights_compressed).extractall(os.path.dirname(weights_compressed))
    tf_semseg.model.pretrained.weights.load_ckpt(weights_uncompressed, model, convert_name)

    return model
