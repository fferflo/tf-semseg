import tensorflow as tf
from ... import hrnet, decode, ocr
from ...util import *
import tfcv

from .util import preprocess, config, convert_name_hrnet, convert_name_ocr

def convert_name(name):
    name = convert_name_hrnet(name)
    name = convert_name_ocr(name)
    name = "model." + name
    return name

def create(input=None):
    return_model = input is None
    if input is None:
        input = tf.keras.layers.Input((None, None, 3))

    x = input
    x = hrnet.hrnet_v2_w48(x, config=config)
    x = ocr.ocr(x, regions=19, filters=512, filters_qkv=256, fix_bias_before_norm=False, config=config)
    x = conv_norm_act(x, filters=512, kernel_size=1, stride=1, bias=False, name=join("final"), config=config)
    x = tf.keras.layers.Dropout(0.05)(x)
    x = decode.decode(x, 19, shape=tf.shape(input)[1:-1], config=config)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=[input], outputs=[x])

    url = "https://github.com/hsfzxjy/models.storage/releases/download/HRNet-OCR/hrnet_ocr_cs_8162_torch11.pth"
    weights = tf.keras.utils.get_file("hrnet_ocr_cs_8162_torch11.pth", url)
    tfcv.model.pretrained.weights.load_pth(weights, model, convert_name, ignore=lambda name: name.startswith("loss."))

    return model if return_model else x
