import tensorflow as tf
import pyunpack, os, sys
from ... import resnet
# TODO: refactor this, add preprocess and config
def load_weights(input, output, url):
    weights_compressed = tf.keras.utils.get_file(url.split("/")[-1], url)
    weights_uncompressed = weights_compressed[:-len("_2016_08_28.tar.gz")] + ".ckpt"
    if not os.path.isfile(weights_uncompressed):
        pyunpack.Archive(weights_compressed).extractall(os.path.dirname(weights_compressed))

    model = tf.keras.Model(inputs=[input], outputs=[output])
    for v in model.variables:
        name = v.name
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
        new_var = tf.train.load_variable(weights_uncompressed, name)
        v.assign(new_var)
    return model

def resnet_v1b_50_imagenet(dilated=False):
    input = tf.keras.layers.Input((None, None, 3))
    x = input
    x = resnet.resnet_v1_50(x, dilated=dilated, stem="b")
    return load_weights(input, x, "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz")

def resnet_v1b_101_imagenet(dilated=False):
    input = tf.keras.layers.Input((None, None, 3))
    x = input
    x = resnet.resnet_v1_101(x, dilated=dilated, stem="b")
    return load_weights(input, x, "http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz")

def resnet_v1b_152_imagenet(dilated=False):
    input = tf.keras.layers.Input((None, None, 3))
    x = input
    x = resnet.resnet_v1_152(input, dilated=dilated, stem="b")
    return load_weights(input, x, "http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz")
