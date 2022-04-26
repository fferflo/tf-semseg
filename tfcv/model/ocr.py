import tensorflow as tf
from .util import *
from . import config, decode, transformer, einops

def object_attention(query, keyvalue, filters_qkv, name=None, config=config.Config()):
    output_filters = query.shape[-1]

    query = conv_norm_act(query, filters=filters_qkv, kernel_size=1, stride=1, name=join(name, "query", "1"), config=config)
    query = conv_norm_act(query, filters=filters_qkv, kernel_size=1, stride=1, name=join(name, "query", "2"), config=config)

    key = keyvalue
    key = conv_norm_act(key, filters=filters_qkv, kernel_size=1, stride=1, name=join(name, "key", "1"), config=config)
    key = conv_norm_act(key, filters=filters_qkv, kernel_size=1, stride=1, name=join(name, "key", "2"), config=config)

    value = keyvalue
    value = conv_norm_act(value, filters=filters_qkv, kernel_size=1, stride=1, name=join(name, "down"), config=config)

    token_features = transformer.multihead_attention(query, key, value, heads=1, config=config)

    token_features = conv_norm_act(token_features, filters=output_filters, kernel_size=1, stride=1, name=join(name, "up"), config=config)
    return token_features

# TODO: this is the same as in psa
def spatial_softmax_pool(weight, value):
    weight = tf.keras.activations.softmax(weight, axis=list(range(len(weight.shape)))[1:-1])
    return einops.apply("b s... fq, b s... fv -> b fq fv", weight, value)

def ocr(x, regions, filters=512, filters_qkv=256, fix_bias_before_norm=True, name="ocr", config=config.Config()):
    region_weights = x
    region_weights = conv_norm_act(region_weights, kernel_size=1, stride=1, bias=not fix_bias_before_norm, name=join(name, "regions"), config=config)
    region_weights = decode.decode(region_weights, filters=regions, name=join(name, "regions", "decode"), config=config)

    x = conv_norm_act(x, filters=filters, kernel_size=3, stride=1, bias=not fix_bias_before_norm, name=join(name, "initial"), config=config)

    # Gather
    regions_features = spatial_softmax_pool(weight=region_weights, value=x) # [batch, regions, features]

    # Distribute
    context_features = object_attention(query=x, keyvalue=regions_features, filters_qkv=filters_qkv, name=join(name, "distribute"), config=config)

    x = tf.concat([context_features, x], axis=-1)

    return x
