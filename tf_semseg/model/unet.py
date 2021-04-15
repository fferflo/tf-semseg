import tensorflow as tf
from .util import *

default_downsample = lambda x, pool_size, name=None: MaxPool(pool_size=pool_size, name=name)(x)
default_upsample = lambda x, pool_size, name=None: UpSample(size=pool_size, name=name)(x)
default_block = partial(repeat, n=2, block=conv_norm_relu)

def unet(x, filters, levels, name="unet", encoding_block=default_block, decoding_block=default_block, downsample=default_downsample, upsample=default_upsample):
    encoding_levels = []

    # Encoder
    for level in range(levels):
        # Encode
        x = encoding_block(x,
                filters=filters * (2 ** level),
                name=join(name, "encode" + str(level + 1)))
        if level < levels - 1:
            encoding_levels.append(x)
            # Downsample
            x = downsample(x, pool_size=2, name=join(name, "downsample" + str(str(level + 1))))

    # Decoder
    for level in reversed(range(1, levels)):
        # Upsample
        x = upsample(x, pool_size=2, name=join(name, "upsample" + str(level + 1)))
        # Skip connection
        x = tf.keras.layers.Concatenate()([x, encoding_levels[level - 1]])
        # Decode
        x = decoding_block(x,
                filters=filters * (2 ** (level - 1)),
                name=join(name, "decode" + str(level + 1)))

    return x
