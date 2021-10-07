import tensorflow as tf
from .util import *
from . import config, decode

def mscale_decode(x, filters, filters_mid, shape=None, dropout=None, name="mscale", config=config.Config()):
    if not dropout is None:
        x = tf.keras.layers.Dropout(dropout)(x)

    output = x
    output = decode.decode(output, filters, name=join(name, "output", "decode"), config=config)
    if not shape is None:
        output = resize(output, shape=shape, method="bilinear", config=config)

    weights = x
    weights = conv_norm_act(weights, filters=filters_mid, kernel_size=3, stride=1, name=join(name, "attention", "1"), config=config)
    weights = conv_norm_act(weights, filters=filters_mid, kernel_size=3, stride=1, name=join(name, "attention", "2"), config=config)
    weights = decode.decode(weights, 1, name=join(name, "attention", "decode"), bias=False)
    weights = tf.keras.layers.Activation("sigmoid")(weights)
    if not shape is None:
        weights = resize(weights, shape=shape, method="bilinear", config=config)

    return output, weights

def predictor_multiscale(predictor, scales, resize_method="bilinear", config=config.Config()):
    def predict(x):
        x_orig = x
        for index, scale in enumerate(sorted(scales, reverse=True)): # High resolution to low resolution
            x = x_orig
            x = resize(x, tf.cast(scale * tf.cast(tf.shape(x)[1:-1], "float32"), "int32"), method=resize_method, config=config)
            output, weights = predictor(x)

            if index == 0:
                fused = output
            elif scale >= 1.0:
                # Downscale previous prediction
                fused = resize(fused, tf.shape(output)[1:-1], method=resize_method, config=config)
                fused = weights * output + (1 - weights) * fused
            else:
                # Upscale current prediction
                output = resize(output, tf.shape(fused)[1:-1], method=resize_method, config=config)
                weights = resize(weights, tf.shape(fused)[1:-1], method=resize_method, config=config)
                fused = weights * output + (1 - weights) * fused
        x = fused
        return x
    return predict

def predictor_singlescale(predictor, resize_method="bilinear", config=config.Config()):
    def predict(x):
        output, _ = predictor(x)
        x = resize(output, tf.shape(x)[1:-1], method=resize_method, config=config)
        return x
    return predict
