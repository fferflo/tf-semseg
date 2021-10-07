import tensorflow as tf
from .util import *
from . import config, shortcut

default_upsample = lambda x, factor, name=None, config=config.Config(): upsample(x, factor=factor, name=name, method="nearest", config=config)

def unet(x, filters, num_encode_units, num_decode_units, encode, decode, upsample=default_upsample, bottleneck=None, shortcut=shortcut.concat, name="unet", config=config.Config()):
    levels = len(num_encode_units)
    if not isinstance(encode, list):
        encode = [encode] * levels
    if not isinstance(decode, list):
        decode = [decode] * levels

    # Encoder
    encoding_levels = []
    for level in range(levels):
        for unit_index in range(num_encode_units[level]):
            x = encode[level](x,
                    filters=filters * (2 ** level),
                    stride=2 if (unit_index == 0 and level > 0) else 1,
                    name=join(name, f"encode{level + 1}", f"unit{unit_index + 1}"),
                    config=config)
        encoding_levels.append(x)

    if not bottleneck is None:
        x = bottleneck(x, name=join(name, "bottleneck"))

    # Decoder
    for level in reversed(range(1, levels)):
        x = upsample(x, factor=2, name=join(name, f"upsample{level + 1}"), config=config)
        x = shortcut(x, encoding_levels[level - 1], name=join(name, f"shortcut{level + 1}"))
        for unit_index in range(num_decode_units[level]):
            x = decode[level](x,
                    filters=filters * (2 ** (level - 1)),
                    name=join(name, f"decode{level + 1}", f"unit{unit_index + 1}"),
                    config=config)

    return x
