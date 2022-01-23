import tensorflow as tf
from . import config

def shortcut(dest, src, stride=1, activation=True, name=None, config=config.Config()):
    if stride > 1:
        src = conv_norm(src, kernel_size=1, stride=stride, bias=False, name=name, config=config)
        if activation:
            src = act(src, config=config)

    return tf.concat([dest, src], axis=-1)

def unet(x, filters, num_encode_units, num_decode_units, encode, decode, neck=None, name="unet", config=config.Config()):
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
                config=config,
            )
        encoding_levels.append(x)

    if not neck is None:
        x = neck(x, name=join(name, "neck"))

    # Decoder
    for level in reversed(range(1, levels)):
        x = upsample(x, stride=2, name=join(name, f"upsample{level + 1}"), config=config)
        x = shortcut(x, encoding_levels[level - 1], name=join(name, f"shortcut{level + 1}"), config=config)
        for unit_index in range(num_decode_units[level]):
            x = decode[level](
                x,
                filters=filters * (2 ** (level - 1)),
                name=join(name, f"decode{level + 1}", f"unit{unit_index + 1}"),
                config=config,
            )

    return x
