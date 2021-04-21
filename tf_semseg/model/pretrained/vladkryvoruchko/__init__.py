import tensorflow as tf
import numpy as np
import sys, re, h5py
from ... import resnet, pspnet, decode

def pspnet_resnet_v1s_101_cityscapes():
    input = tf.keras.layers.Input((713, 713, 3))

    batchnorm = lambda x, *args, **kwargs: tf.keras.layers.BatchNormalization(*args, **kwargs)(x)

    x = input
    x = resnet.resnet_v1_101(x, dilated=True, stem="s", norm=batchnorm)
    x = pspnet.pspnet(x, bin_sizes=[6, 3, 2, 1], norm=batchnorm)
    x = decode.decode(x, 19)
    x = tf.compat.v1.image.resize(x, input.shape[1:-1], align_corners=True) # tf.image.resize is not compatible with trained weights
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
                match = re.match(re.escape("resnet_v1_101/stem_s/") + "(.*)", layer.name)
                if match:
                    index = int(match.group(1)[4:])
                    name = "conv1_" + str(index) + "_3x3"
                    if index == 1:
                        name = name + "_s2"

                    if match.group(1).startswith("norm"):
                        name = name + "_bn"
                        set_bn_weights(layer, weights[name])
                        weights_left.remove(name)
                    else:
                        assert match.group(1).startswith("conv")
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
                    pool_size = int(match.group(1))
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

                if layer.name.startswith("psp/final_"):
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
            print("Failed to load weights for layers " + str(weights_left))
            sys.exit(-1)

    return model
