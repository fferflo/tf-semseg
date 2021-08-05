import tensorflow as tf
from .util import *
from . import resnet, erfnet, pspnet, config, senet, shortcut

def stem(rgb, depth, se_reduction=16, name=None, config=config.Config()):
    def resnet_stem_b_no_pool(x, name):
        x = conv_norm_act(x, filters=64, kernel_size=7, stride=2, name=name, config=config)
        return x
    rgb = resnet_stem_b_no_pool(rgb, name=join(name, "stem_rgb"))
    depth = resnet_stem_b_no_pool(depth, name=join(name, "stem_depth"))

    rgb_se = senet.squeeze_excite_channel(rgb, reduction=se_reduction, name=join(name, "se_rgb"), config=config)
    depth_se = senet.squeeze_excite_channel(depth, reduction=se_reduction, name=join(name, "se_depth"), config=config)

    rgb = rgb_se + depth_se

    rgb = pool(rgb, kernel_size=3, stride=2, mode="max", config=config)
    depth = pool(depth, kernel_size=3, stride=2, mode="max", config=config)

    return rgb, depth

def upsample(x, factor, name=None, config=config.Config()):
    filters = x.shape[-1]
    x = resize(x, factor * tf.shape(x)[1:-1], method="nearest", config=config)
    x = conv(x, filters, kernel_size=3, groups=filters, name=join(name, "conv"), bias=True, config=config)
    return x

def esanet(rgb, depth, classes, num_residual_units, filters, dilation_rates, strides, name=None, psp_bin_sizes=[1, 5], block=erfnet.non_bottleneck_block_1d, se_reduction=16, decoder_filters=[512, 256, 128], num_decoder_units=[3, 3, 3], config=config.Config()):
    rgb, depth = globals()["stem"](rgb, depth, se_reduction=se_reduction, name=join(name, "stem_b"), config=config)

    encoder_blocks = []

    # Encoder
    for block_index in range(len(num_residual_units)):
        for unit_index in range(num_residual_units[block_index]):
            def unit(x, name):
                return block(x,
                        filters=filters[block_index],
                        stride=strides[block_index] if unit_index == 0 else 1,
                        dilation_rate=(dilation_rates[block_index - 1] if block_index > 0 else 1) if unit_index == 0 else dilation_rates[block_index],
                        name=join(name, f"block{block_index + 1}", f"unit{unit_index + 1}"),
                        config=config)
            rgb = unit(rgb, name=join(name, "encode_rgb"))
            depth = unit(depth, name=join(name, "encode_depth"))

        rgb_se = senet.squeeze_excite_channel(rgb, reduction=se_reduction, name=join(name, f"block{block_index + 1}", "se_rgb"), config=config)
        depth_se = senet.squeeze_excite_channel(depth, reduction=se_reduction, name=join(name, f"block{block_index + 1}", "se_depth"), config=config)

        rgb = rgb_se + depth_se

        encoder_blocks.append(rgb)

    # Context module
    encoder_blocks[-1] = pspnet.psp(
        encoder_blocks[-1],
        resize_method="nearest",
        name=join(name, "psp"),
        bin_sizes=psp_bin_sizes,
        config=config
    )
    encoder_blocks[-1] = conv_norm_act(encoder_blocks[-1], filters=encoder_blocks[-1].shape[-1] // 2, kernel_size=1, stride=1, name=join(name, "psp", "final"), config=config)

    # Decoder
    x = encoder_blocks[-1]
    for block_index in range(len(num_decoder_units)):
        x = conv_norm_act(x, filters=decoder_filters[block_index], kernel_size=3, name=join(name, "decode", f"block{block_index + 1}", "initial"), config=config)
        for unit_index in range(num_decoder_units[block_index]):
            x = block(x, name=join(name, "decode", f"block{block_index + 1}", f"unit{unit_index + 1}"), config=config)
        x = upsample(x, factor=2, name=join(name, "decode", f"block{block_index + 1}", "upsample"), config=config)
        x = shortcut.add(x, encoder_blocks[-(block_index + 2)], name=join(name, "decode", f"block{block_index + 1}", "shortcut"), config=config)

    x = conv(x, classes, kernel_size=3, name=join(name, "decode", "final", "conv"), bias=True, config=config) # TODO: this should be initialized differently for training: https://github.com/TUI-NICR/ESANet/blob/56b7aff77e3fc05ce4ffe55142dc805b07956f22/src/models/model.py#L385
    x = upsample(x, factor=2, name=join(name, "decode", "final", "upsample1"), config=config)
    x = upsample(x, factor=2, name=join(name, "decode", "final", "upsample2"), config=config)

    return x
