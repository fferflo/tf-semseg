import tensorflow as tf
from ... import hrnet, decode, ocr, mscale
from ...util import *
from ...config import Config
import tfcv, os, re
from google_drive_downloader import GoogleDriveDownloader as gdd

from ..hrnet.util import preprocess, convert_name_hrnet, convert_name_ocr # This model is based on hrnet_v2_w48_ocr from the hrnet package

config = Config(
    mode="pytorch",
    norm=lambda x, *args, **kwargs: tf.keras.layers.BatchNormalization(*args, momentum=0.9, epsilon=1e-5, **kwargs)(x),
    resize_align_corners=False
)

def convert_name(name):
    if "stem" in name or "block" in name:
        name = "module.backbone." + convert_name_hrnet(name)
    elif "ocr" in name or name.startswith("final"):
        name = "module.ocr." + convert_name_ocr(name)
        name = name.replace("module.ocr.aux_head.1", "module.ocr.aux_head.1.0")
        name = name.replace("module.ocr.aux_head.3", "module.ocr.aux_head.2")
        name = name.replace("module.ocr.conv3x3_ocr.1", "module.ocr.conv3x3_ocr.1.0")
    else:
        name = name.replace("norm", "bn")
        name = re.sub("^mscale/output/decode/conv", "module.ocr.cls_head", name)
        name = re.sub("^mscale/attention/([0-9]*)/([a-z]*)", lambda m: f"module.scale_attn.{m.group(2)}{int(m.group(1)) - 1}", name)
        name = re.sub("^mscale/attention/decode/conv", "module.scale_attn.conv2", name)
    return name

def create(input=None):
    return_model = input is None
    if input is None:
        input = tf.keras.layers.Input((None, None, 3))

    x = input
    x = hrnet.hrnet_v2_w48(x, config=config)
    x = ocr.ocr(x, regions=19, filters=512, filters_qkv=256, fix_bias_before_norm=False, config=config)
    x = conv_norm_act(x, filters=512, kernel_size=1, stride=1, bias=False, name="final", config=config)
    output, weights = mscale.mscale_decode(x, filters=19, filters_mid=256, shape=tf.shape(input)[1:-1], dropout=0.05, name="mscale", config=config)

    model = tf.keras.Model(inputs=[input], outputs=[output, weights])

    weights = os.path.join(os.path.expanduser("~"), ".keras", "cityscapes_ocrnet.HRNet_Mscale_outstanding-turtle.pth")
    gdd.download_file_from_google_drive(file_id="1lse0Mqf7ny5qqV99nGQ3ccXTKJ6kNGoH", dest_path=weights)
    tfcv.model.pretrained.weights.load_pth(weights, model, convert_name)

    return model if return_model else x
