import tensorflow as tf
import tfcv, re, os, gdown
from ... import decode, segformer
from ... import config as config_
from ...util import *
from functools import partial
from google_drive_downloader import GoogleDriveDownloader as gdd

color_mean = np.asarray([123.675, 116.28, 103.53])
color_std = np.asarray([58.395, 57.12, 57.375])

def preprocess(color):
    color = (color - color_mean) / color_std
    return color

def convert_name(name):
    name = "/".join(name.split("/")[1:])

    name = re.sub("block([0-9]*)/patch-embed/conv", lambda m: f"backbone.patch_embed{m.group(1)}.proj", name)
    name = re.sub("block([0-9]*)/patch-embed/norm", lambda m: f"backbone.patch_embed{m.group(1)}.norm", name)

    name = re.sub("block([0-9]*)/unit([0-9]*)/mha/norm", lambda m: f"backbone.block{m.group(1)}.{int(m.group(2)) - 1}.norm1", name)
    name = re.sub("block([0-9]*)/unit([0-9]*)/mha/query", lambda m: f"backbone.block{m.group(1)}.{int(m.group(2)) - 1}.attn.q", name)
    name = re.sub("block([0-9]*)/unit([0-9]*)/mha/key-value", lambda m: f"backbone.block{m.group(1)}.{int(m.group(2)) - 1}.attn.kv", name)
    name = re.sub("block([0-9]*)/unit([0-9]*)/mha/out_proj", lambda m: f"backbone.block{m.group(1)}.{int(m.group(2)) - 1}.attn.proj", name)
    name = re.sub("block([0-9]*)/unit([0-9]*)/mha/spatial-reduction/conv", lambda m: f"backbone.block{m.group(1)}.{int(m.group(2)) - 1}.attn.sr", name)
    name = re.sub("block([0-9]*)/unit([0-9]*)/mha/spatial-reduction/norm", lambda m: f"backbone.block{m.group(1)}.{int(m.group(2)) - 1}.attn.norm", name)
    name = re.sub("block([0-9]*)/unit([0-9]*)/mlp/norm", lambda m: f"backbone.block{m.group(1)}.{int(m.group(2)) - 1}.norm2", name)

    name = re.sub("block([0-9]*)/unit([0-9]*)/mlp/1/pointwise", lambda m: f"backbone.block{m.group(1)}.{int(m.group(2)) - 1}.mlp.fc1", name)
    name = re.sub("block([0-9]*)/unit([0-9]*)/mlp/1/depthwise", lambda m: f"backbone.block{m.group(1)}.{int(m.group(2)) - 1}.mlp.dwconv.dwconv", name)
    name = re.sub("block([0-9]*)/unit([0-9]*)/mlp/2/pointwise", lambda m: f"backbone.block{m.group(1)}.{int(m.group(2)) - 1}.mlp.fc2", name)

    name = re.sub("block([0-9]*)/norm", lambda m: f"backbone.norm{m.group(1)}", name)

    name = re.sub("head/in([0-9]*)", lambda m: f"decode_head.linear_c{m.group(1)}.proj", name)
    name = re.sub("head/fuse/conv", lambda m: f"decode_head.linear_fuse.conv", name)
    name = re.sub("head/fuse/norm", lambda m: f"decode_head.linear_fuse.bn", name)

    name = re.sub("decode/conv", lambda m: f"decode_head.linear_pred", name)

    return name

config = config_.PytorchConfig(
    norm=lambda x, *args, **kwargs: tf.keras.layers.LayerNormalization(*args, epsilon=1e-6, **kwargs)(x),
    resize=config_.partial_with_default_args(config_.resize, align_corners=False),
    act=lambda x, **kwargs: tf.keras.layers.Activation(tf.keras.activations.gelu, **kwargs)(x),
)

decoder_config = config_.PytorchConfig(
    norm=lambda x, *args, **kwargs: tf.keras.layers.BatchNormalization(*args, momentum=0.9, epsilon=1e-5, **kwargs)(x),
    resize=config_.partial_with_default_args(config_.resize, align_corners=False),
    act=lambda x, **kwargs: tf.keras.layers.Activation(tf.keras.activations.relu, **kwargs)(x),
)

def create_x(input, segformer_variant, num_classes, gdd_id, gdd_name, name=None):
    return_model = input is None
    if input is None:
        input = tf.keras.layers.Input((None, None, 3))

    x = input

    x = segformer_variant(x, name=name, config=config, decoder_config=decoder_config)

    x = decode.decode(x, filters=num_classes, shape=tf.shape(input)[1:-1], dropout=0.1, name=join(name, "decode"), config=decoder_config)
    x = tf.keras.layers.Softmax()(x)

    model = tf.keras.Model(inputs=[input], outputs=[x])

    # https://github.com/NVlabs/SegFormer
    weights_file = os.path.join(os.path.expanduser("~"), ".keras", gdd_name)
    if not os.path.isfile(weights_file):
        # gdd.download_file_from_google_drive(file_id=gdd_id, dest_path=weights_file) # TODO: fix gdd vs gdown
        gdown.download(id=gdd_id, output=weights_file, quiet=False)

    tfcv.model.pretrained.weights.load_pth(weights_file, model, convert_name, ignore=lambda name: "decode_head.conv_seg" in name)

    return model if return_model else x


def make_builder(variant, num_classes, gdd_id, gdd_name):
    class builder:
        @staticmethod
        def create(input=None, name=f"segformer_{variant}"):
            return create_x(
                input=input,
                num_classes=num_classes,
                segformer_variant=vars(segformer)[f"segformer_{variant}"],
                gdd_id=gdd_id,
                gdd_name=gdd_name,
                name=name,
            )

        preprocess = preprocess
        config = config
    return builder

class segformer_b0_ade20k_512(make_builder("b0", 150, "1je1GL6TXU3U-cZZsUv08ITUkVW4mBPYy", "segformer.b0.512x512.ade.160k.pth")): pass
class segformer_b0_cityscapes_512x1024(make_builder("b0", 19, "1yjPTULZCGAYpK0XCg1SNHXhMga4_w03r", "segformer.b0.512x1024.city.160k.pth")): pass
class segformer_b0_cityscapes_640x1024(make_builder("b0", 19, "1t4fOtwJqpnUJvMZhZNEcq7bbFgyAQO7P", "segformer.b0.640x1280.city.160k.pth")): pass
class segformer_b0_cityscapes_768(make_builder("b0", 19, "1hMrg7e3z7iPHzb-jAKLwe5KD6jpzfXbY", "segformer.b0.768x768.city.160k.pth")): pass
class segformer_b0_cityscapes_1024(make_builder("b0", 19, "10lD5u0xVDJDKkIYxJDWkSeA2mfK_tgh9", "segformer.b0.1024x1024.city.160k.pth")): pass
# low accuracy: class segformer_b1_ade20k_512(make_builder("b1", 150, "1PNaxIg3gAqtxrqTNsYPriR2c9j68umuj", "segformer.b1.512x512.ade.160k.pth")): pass
class segformer_b1_cityscapes_1024(make_builder("b1", 19, "1sSdiqRsRMhLJCfs0SydF7iKgeQNcXDZj", "segformer.b1.1024x1024.city.160k.pth")): pass
class segformer_b2_ade20k_512(make_builder("b2", 150, "13AMcdZYePbrTtwVzdJwZP5PF8PKehGhU", "segformer.b2.512x512.ade.160k.pth")): pass
class segformer_b2_cityscapes_1024(make_builder("b2", 19, "1MZhqvWDOKdo5rBPC2sL6kWL25JpxOg38", "segformer.b2.1024x1024.city.160k.pth")): pass
class segformer_b3_ade20k_512(make_builder("b3", 150, "16ILNDrZrQRJrXsIcSjUC56ueR72Rlant", "segformer.b3.512x512.ade.160k.pth")): pass
class segformer_b3_cityscapes_1024(make_builder("b3", 19, "1dc1YM2b3844-dLKq0qe77qb9_7brReIF", "segformer.b3.1024x1024.city.160k.pth")): pass
class segformer_b4_ade20k_512(make_builder("b4", 150, "171YHhri1rT5lwxmfPW76eU9DPP9OR27n", "segformer.b4.512x512.ade.160k.pth")): pass
class segformer_b4_cityscapes_1024(make_builder("b4", 19, "1F9QqGFzhr5wdX-FWax1xE2l7B8lqs42s", "segformer.b4.1024x1024.city.160k.pth")): pass
class segformer_b5_ade20k_640(make_builder("b5", 150, "11F7GHP6F8S9nUOf_KDvg8pouDEFEBGYz", "segformer.b5.640x640.ade.160k.pth")): pass
class segformer_b5_cityscapes_1024(make_builder("b5", 19, "1z3eFf-xVMkcb1Nmcibv6Ut-lTh81RLgO", "segformer.b5.1024x1024.city.160k.pth")): pass
