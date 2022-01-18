import tensorflow as tf
import tfcv, re
import numpy as np
from ... import config as config_

color_mean = np.asarray([0.485, 0.456, 0.406])
color_std = np.asarray([0.229, 0.224, 0.225])

def preprocess(color):
    color = color / 255.0
    color = (color - color_mean) / color_std
    return color

def convert_name(name):
    name = "/".join(name.split("/")[1:])

    name = name.replace("stem/conv", "downsample_layers.0.0")
    name = name.replace("stem/norm", "downsample_layers.0.1")

    name = re.sub("^block([0-9]*)/unit([0-9]*)", lambda m: f"stages.{int(m.group(1)) - 1}.{int(m.group(2)) - 1}", name)

    name = name.replace("scale", "gamma")
    name = name.replace("depthwise", "dwconv")
    name = re.sub("pointwise/([0-9]*)", lambda m: f"pwconv{int(m.group(1))}", name)

    def func(n):
        return 1 if n == "conv" else 0
    name = re.sub("downsample([0-9]*)/([a-z]*)", lambda m: f"downsample_layers.{int(m.group(1)) - 1}.{func(m.group(2))}", name)

    name = name.replace("/", ".")

    return name

config = config_.PytorchConfig(
    norm=lambda x, *args, **kwargs: tf.keras.layers.LayerNormalization(*args, epsilon=1e-6, **kwargs)(x),
    resize=config_.partial_with_default_args(config_.resize, align_corners=False),
    act=lambda x, **kwargs: tf.keras.layers.Activation(tf.keras.activations.gelu, **kwargs)(x),
)

def create_x(input, convnext, url, name):
    return_model = input is None
    if input is None:
        input = tf.keras.layers.Input((None, None, 3))

    x = input
    x = convnext(x, name=name, config=config)

    model = tf.keras.Model(inputs=[input], outputs=[x])

    weights = tf.keras.utils.get_file(url.split("/")[-1], url)
    tfcv.model.pretrained.weights.load_pth(weights, model, convert_name, ignore=lambda n: n.startswith("norm.") or n.startswith("head."))

    return model if return_model else x

def make_builder(variant, url):
    class builder:
        @staticmethod
        def create(input=None, name=f"convnext_{variant}"):
            return create_x(
                input=input,
                convnext=vars(tfcv.model.convnext)[f"convnext_{variant}"],
                url=url,
                name=name,
            )

        preprocess = preprocess
        config = config
    return builder

class convnext_tiny_imagenet1k_224(make_builder("tiny", f"https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth")): pass
class convnext_small_imagenet1k_224(make_builder("small", f"https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth")): pass
class convnext_base_imagenet1k_224(make_builder("base", f"https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth")): pass
class convnext_large_imagenet1k_224(make_builder("large", f"https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth")): pass

class convnext_base_imagenet1k_384(make_builder("base", f"https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_384.pth")): pass
class convnext_large_imagenet1k_384(make_builder("large", f"https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth")): pass

class convnext_base_imagenet22k_224(make_builder("base", f"https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth")): pass
class convnext_large_imagenet22k_224(make_builder("large", f"https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth")): pass
class convnext_xlarge_imagenet22k_224(make_builder("xlarge", f"https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth")): pass
