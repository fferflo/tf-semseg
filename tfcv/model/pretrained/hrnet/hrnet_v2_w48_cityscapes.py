import tensorflow as tf
from ... import hrnet, decode
from ...util import *
import tfcv

from .util import preprocess, config, convert_name_hrnet

def convert_name(name):
    name = convert_name_hrnet(name)
    name = name.replace("last_layer/conv", "last_layer.0")
    name = name.replace("last_layer/bn", "last_layer.1")
    name = name.replace("decode/conv", "last_layer.3")
    return name

def create(input=None):
    return_model = input is None
    if input is None:
        input = tf.keras.layers.Input((None, None, 3))

    x = input
    x = hrnet.hrnet_v2_w48(x, config=config)
    x = conv_norm_act(x, filters=x.shape[-1], kernel_size=1, stride=1, bias=True, name="last_layer", config=config) # Yes, this has bias in pretrained weights
    x = decode.decode(x, 19, shape=tf.shape(input)[1:-1], config=config)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=[input], outputs=[x])

    url = "https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_cs_8090_torch11.pth"
    weights = tf.keras.utils.get_file("hrnet_cs_8090_torch11.pth", url)
    tfcv.model.pretrained.weights.load_pth(weights, model, convert_name)

    return model if return_model else x
