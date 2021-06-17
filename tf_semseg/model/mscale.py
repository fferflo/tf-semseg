import tensorflow as tf
from .util import *
from . import config, decode

def mscale_decode(x, filters, filters_mid, shape=None, name="mscale", config=config.Config()):
    output = x
    output = decode.decode(output, filters, name=join(name, "output", "decode"), config=config)
    if not shape is None:
        output = config.resize(output, shape=shape, method="bilinear")

    weights = x
    weights = conv_norm_act(weights, filters=filters_mid, kernel_size=3, stride=1, name=join(name, "attention", "1"), config=config)
    weights = conv_norm_act(weights, filters=filters_mid, kernel_size=3, stride=1, name=join(name, "attention", "2"), config=config)
    # TODO: dropout here, or move dropout into decode?
    weights = decode.decode(weights, 1, name=join(name, "attention", "decode"), use_bias=False)
    weights = tf.keras.layers.Activation("sigmoid")(weights)
    if not shape is None:
        weights = config.resize(weights, shape=shape, method="bilinear")

    return output, weights

def predictor_multiscale(predictor, scales, resize_method="bilinear", config=config.Config()):
    def predict(x):
        x_orig = x
        for index, scale in enumerate(sorted(scales, reverse=True)): # High resolution to low resolution
            x = x_orig
            x = config.resize(x, tf.cast(scale * tf.cast(tf.shape(x)[1:-1], "float32"), "int32"), method=resize_method)
            output, weights = predictor(x)

            if index == 0:
                fused = output
            elif scale >= 1.0:
                # Downscale previous prediction
                fused = config.resize(fused, tf.shape(output)[1:-1], method=resize_method)
                fused = weights * output + (1 - weights) * fused
            else:
                # Upscale current prediction
                output = config.resize(output, tf.shape(fused)[1:-1], method=resize_method)
                weights = config.resize(weights, tf.shape(fused)[1:-1], method=resize_method)
                fused = weights * output + (1 - weights) * fused
        x = fused
        return x
    return predict

def predictor_singlescale(predictor, resize_method="bilinear", config=config.Config()):
    def predict(x):
        output, _ = predictor(x)
        x = config.resize(output, tf.shape(x)[1:-1], method=resize_method)
        return x
    return predict
