import tensorflow as tf
from . import config, resnet, util

def stem(x, name=None, config=config.Config()):
    x = util.conv_norm_act(x, filters=64, kernel_size=3, stride=2, name=util.join(name, "1"), config=config)
    x = util.conv_norm_act(x, filters=64, kernel_size=3, stride=2, name=util.join(name, "2"), config=config)
    return x

def downsample(x, n, filters, skip_last_act, name=None, config=config.Config()):
    for i in range(n):
        x = util.conv(x, filters=filters if i == n - 1 else x.shape[-1], kernel_size=3, stride=2, bias=False, name=util.join(name, str(i), "conv"), config=config)
        x = util.norm(x, name=util.join(name, str(i), "norm"), config=config)
        if i < n - 1 or not skip_last_act:
            x = util.act(x, config=config)
    return x

def upsample(x, n, filters, name=None, config=config.Config()):
    x = util.conv(x, filters=filters, kernel_size=1, stride=1, bias=False, name=util.join(name, "conv"), config=config)
    x = util.norm(x, name=util.join(name, "norm"), config=config)
    x = util.upsample(x, (2 ** n), method="bilinear", config=config)
    return x

def fuse(x, n, filters, name=None, config=config.Config()):
    if n < 0:
        return downsample(x, -n, filters, skip_last_act=True, name=util.join(name, "downsample"), config=config)
    elif n == 0:
        return x
    else:
        return upsample(x, n, filters, name=util.join(name, "upsample"), config=config)

def module(xs, block, num_units, filters, name=None, config=config.Config()):
    # Apply blocks
    for branch_index in range(len(xs)):
        for unit_index in range(num_units[branch_index]):
            xs[branch_index] = block(xs[branch_index],
                    filters=filters[branch_index],
                    stride=1,
                    dilation_rate=1,
                    name=util.join(name, f"branch{branch_index + 1}", f"unit{unit_index + 1}"),
                    config=config)

    # Fuse
    new_xs = [None] * len(xs)
    for output_branch_index in range(len(xs)):
        dest_shape = tf.shape(xs[output_branch_index])
        slice_begin = dest_shape * 0
        inputs = []
        for input_branch_index in range(len(xs)):
            x = fuse(
                xs[input_branch_index],
                n=input_branch_index - output_branch_index,
                filters=filters[output_branch_index],
                name=util.join(name, f"fuse_branch{input_branch_index}to{output_branch_index}"),
                config=config
            )
            slice_end = tf.concat([dest_shape[:-1], [x.shape[-1]]], axis=0)
            x = tf.slice(x, slice_begin, slice_end)
            inputs.append(x)
        new_xs[output_branch_index] = tf.reduce_sum(tf.stack(inputs, axis=-1), axis=-1)
    xs = [util.act(x, config=config) for x in new_xs]

    return xs

def transition(xs, filters, name=None, config=config.Config()):
    orig_xs = [x for x in xs]

    # Old branches
    for branch_index in range(len(xs)):
        if xs[branch_index].shape[-1] != filters[branch_index]:
            xs[branch_index] = util.conv_norm_act(xs[branch_index], filters=filters[branch_index], kernel_size=3, stride=1, name=util.join(name, f"branch{branch_index + 1}"), config=config)

    # New branches
    last_x = orig_xs[-1]
    for branch_index in range(len(xs), len(filters)):
        # In the authors' code this establishes a new downscale pyramid from orig_xs[-1] for every new branch, even though reusing
        # the previously downsampled branches seems more reasonable. This doesn't ever come into effect since the network configs
        # only specify single new branches. We go with the reusing approach here.
        xs.append(downsample(last_x, 1, filters[branch_index], skip_last_act=False, name=util.join(name, f"branch{branch_index + 1}"), config=config))
        last_x = xs[-1]

    return xs

def hrnet(x, num_units, filters, blocks, num_modules, stem=True, name=None, config=config.Config()):
    if stem:
        x = globals()["stem"](x, name=util.join(name, "stem"), config=config)

    xs = [x]

    for block_index in range(len(num_units)):
        for module_index in range(num_modules[block_index]):
            xs = module(xs,
                        block=blocks[block_index],
                        num_units=num_units[block_index],
                        filters=filters[block_index],
                        name=util.join(name, f"block{block_index + 1}", f"module{module_index + 1}"),
                        config=config)

        if block_index < len(num_units) - 1:
            xs = transition(xs, filters[block_index + 1], name=util.join(name, f"block{block_index + 1}", "transition"), config=config)

    dest_shape = tf.shape(xs[0])
    slice_begin = dest_shape * 0
    for branch_index in range(1, len(xs)):
        x = xs[branch_index]
        x = util.upsample(x, 2 ** branch_index, method="bilinear", config=config)
        slice_end = tf.concat([dest_shape[:-1], [x.shape[-1]]], axis=0)
        x = tf.slice(x, slice_begin, slice_end)
        xs[branch_index] = x

    x = tf.concat(xs, axis=-1)
    return x

def hrnet_v2_w48(x, name=None, stem=True, config=config.Config()):
    return hrnet(x,
        num_units=[[4], [4, 4], [4, 4, 4], [4, 4, 4, 4]],
        filters=[[64], [48, 96], [48, 96, 192], [48, 96, 192, 384]],
        blocks=[resnet.bottleneck_block_v1, resnet.basic_block_v1, resnet.basic_block_v1, resnet.basic_block_v1],
        num_modules=[1, 1, 4, 3],
        stem=stem,
        name=name,
        config=config
    )
