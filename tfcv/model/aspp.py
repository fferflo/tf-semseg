from . import config, einops
from .util import *
import tensorflow as tf

# skip_global_norm is useful when using BN and with batch_size = 1, since that causes variance = 0
def aspp(x, filters=256, atrous_rates=[12, 24, 36], name="aspp", skip_global_norm=False, config=config.Config()):
    # 1x1 conv
    x0 = conv_norm_act(x, filters=filters, kernel_size=1, stride=1, name=join(name, f"1x1"), config=config)

    # Atrous convs
    xs = [conv_norm_act(x, filters=filters, kernel_size=3, stride=1, dilation_rate=d, name=join(name, f"atrous{i + 1}"), config=config) for i, d in enumerate(atrous_rates)]

    # Global pooling
    x1 = einops.apply("b s... c -> b c", x, reduction="mean")
    # TODO: explicit LayerNorm here, and other places where global pooling is used?
    x1 = (conv_act if skip_global_norm else conv_norm_act)(x1, filters=filters, kernel_size=1, stride=1, name=join(name, f"global"), config=config)
    x1 = einops.apply("b c -> b s... c", x1, output_shape=tf.shape(x0))

    x = tf.concat([x0] + xs + [x1], axis=-1)

    return x

def denseaspp(x, filters=128, bottleneck_factor=4, atrous_rates=[3, 6, 12, 18, 24], dropout=0.0, bias=False, name="dense-aspp", config=config.Config()):
    for i in range(len(atrous_rates)):
        x_orig = x
        x = (act_conv if i == 0 else norm_act_conv)(x, filters=filters * bottleneck_factor, kernel_size=1, stride=1, bias=bias,
                name=join(name, f"atrous{i + 1}", "1"), config=config)
        x = norm_act_conv(x, filters=filters, kernel_size=3, stride=1, dilation_rate=atrous_rates[i], bias=bias,
                name=join(name, f"atrous{i + 1}", "2"), config=config)
        x = tf.keras.layers.Dropout(dropout)(x)
        x = tf.concat([x, x_orig], axis=-1)
    return x
