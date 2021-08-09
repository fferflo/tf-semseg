from .util import join
from . import config, util
import tensorflow as tf

def aspp(x, filters=256, atrous_rates=[12, 24, 36], name="aspp", config=config.Config()):
    # 1x1 conv
    x0 = util.conv_norm_act(x, filters=filters, kernel_size=1, stride=1, name=join(name, f"1x1"), config=config)

    # Atrous convs
    xs = [util.conv_norm_act(x, filters=filters, kernel_size=3, stride=1, dilation_rate=d, name=join(name, f"atrous{i + 1}"), config=config) for i, d in enumerate(atrous_rates)]

    # Global pooling
    x1 = tf.reduce_mean(x, axis=list(range(len(x.shape)))[1:-1], keepdims=True)
    x1 = util.conv_norm_act(x1, filters=filters, kernel_size=1, stride=1, name=join(name, f"global"), config=config)
    x1 = tf.broadcast_to(x1, tf.shape(x0))

    x = tf.concat([x0] + xs + [x1], axis=-1)

    return x

def denseaspp(x, filters=128, bottleneck_factor=4, atrous_rates=[3, 6, 12, 18, 24], dropout=0, bias=False, name="dense-aspp", config=config.Config()):
    for i in range(len(atrous_rates)):
        x_orig = x
        x = (util.act_conv if i == 0 else util.norm_act_conv)(x, filters=filters * bottleneck_factor, kernel_size=1, stride=1, bias=bias,
                name=join(name, f"atrous{i + 1}", "1"), config=config)
        x = util.norm_act_conv(x, filters=filters, kernel_size=3, stride=1, dilation_rate=atrous_rates[i], bias=bias,
                name=join(name, f"atrous{i + 1}", "2"), config=config)
        x = util.dropout(x, rate=dropout, config=config)
        x = tf.concat([x, x_orig], axis=-1)
    return x
