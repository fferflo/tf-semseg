import tensorflow as tf
import numpy as np
import sys, re, h5py
from ... import resnet, pspnet, decode
from .. import weights
from ...config import Config
from ...util import *

# Expects BGR input, trained for size (713, 713)

def preprocess(color):
    color = color - np.array([123.68, 116.779, 103.939])
    color = color[:, :, ::-1]
    return color

config = Config(
    mode="pytorch", # Same as caffe
    norm=lambda x, *args, **kwargs: tf.keras.layers.BatchNormalization(*args, momentum=0.9, epsilon=1e-5, **kwargs)(x),
    resize_align_corners=True
)

def create(input=None):
    return_model = input is None
    if input is None:
        input = tf.keras.layers.Input((None, None, 3))

    x = input
    x = resnet.resnet_v1_101(x, dilate=[False, False, True, True], stem="s", config=config)
    x = pspnet.psp(x, bin_sizes=[6, 3, 2, 1], resize_method="bilinear", config=config)
    x = conv_norm_act(x, filters=512, kernel_size=3, stride=1, name="final", config=config)
    x = decode.decode(x, 19, shape=tf.shape(input)[1:-1], dropout=0.1, config=config)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=[input], outputs=[x])

    # https://github.com/Vladkryvoruchko/PSPNet-Keras-tensorflow
    url = "https://www.dropbox.com/s/c17g94n946tpalb/pspnet101_cityscapes.h5?dl=1"
    weights = tf.keras.utils.get_file("pspnet101_cityscapes.h5", url)

    with h5py.File(weights, "r") as f:
        weights = {name: f[name][name] for name in f.keys() if name in f[name]}

        weights_left = set(weights.keys())

        def set_bn_weights(layer, weights):
            mean = np.array(weights["moving_mean:0"]).reshape(-1)
            variance = np.array(weights["moving_variance:0"]).reshape(-1)
            scale = np.array(weights["gamma:0"]).reshape(-1)
            offset = np.array(weights["beta:0"]).reshape(-1)
            layer.set_weights([scale, offset, mean, variance])
        for layer in model.layers:
            if len(layer.get_weights()) > 0:
                # Resnet: Stem
                match = re.match("resnet_v1_101/stem_s/(.*)/(.*)", layer.name)
                if match:
                    index = int(match.group(1))
                    name = "conv1_" + str(index) + "_3x3"
                    if index == 1:
                        name = name + "_s2"

                    if match.group(2).startswith("norm"):
                        name = name + "_bn"
                        set_bn_weights(layer, weights[name])
                        weights_left.remove(name)
                    else:
                        assert match.group(2).startswith("conv")
                        assert "bias" not in weights[name]
                        layer.set_weights([weights[name]["kernel:0"]])
                        weights_left.remove(name)
                    continue

                # Resnet: Residual blocks
                match = re.match(re.escape("resnet_v1_101/block") + "(.*?)" + re.escape("/unit") + "(.*?)" + re.escape("/") + "(.*)", layer.name)
                if match:
                    block = int(match.group(1))
                    unit = int(match.group(2))
                    name = "conv" + str(block + 1) + "_" + str(unit) + "_"

                    if match.group(3).startswith("reduce"):
                        name = name + "1x1_reduce"
                    elif match.group(3).startswith("center"):
                        name = name + "3x3"
                    elif match.group(3).startswith("expand"):
                        name = name + "1x1_increase"
                    else:
                        assert match.group(3).startswith("shortcut")
                        name = name + "1x1_proj"

                    if layer.name.endswith("conv"):
                        assert "bias" not in weights[name]
                        layer.set_weights([weights[name]["kernel:0"]])
                        weights_left.remove(name)
                    else:
                        assert layer.name.endswith("norm")
                        name = name + "_bn"
                        set_bn_weights(layer, weights[name])
                        weights_left.remove(name)
                    continue

                # Pspnet
                match = re.match(re.escape("psp/pool") + "(.*?)" + re.escape("/") + "(.*)", layer.name)
                if match:
                    pool_size = [6, 3, 2, 1][int(match.group(1)) - 1]
                    name = "conv5_3_pool" + str(pool_size) + "_conv"

                    if layer.name.endswith("conv"):
                        assert "bias" not in weights[name]
                        layer.set_weights([weights[name]["kernel:0"]])
                        weights_left.remove(name)
                    else:
                        assert layer.name.endswith("norm")
                        name = name + "_bn"
                        set_bn_weights(layer, weights[name])
                        weights_left.remove(name)
                    continue

                if layer.name.startswith("final"):
                    name = "conv5_4"
                    if layer.name.endswith("conv"):
                        assert "bias" not in weights[name]
                        layer.set_weights([weights[name]["kernel:0"]])
                        weights_left.remove(name)
                    else:
                        assert layer.name.endswith("norm")
                        name = name + "_bn"
                        set_bn_weights(layer, weights[name])
                        weights_left.remove(name)
                    continue

                assert layer.name == "decode/conv"
                name = "conv6"
                layer.set_weights([weights[name]["kernel:0"], weights[name]["bias:0"]])
                weights_left.remove(name)
        if len(weights_left) > 0:
            raise weights.LoadWeightsException("Failed to load weights for layers " + str(weights_left))

    return model if return_model else x
