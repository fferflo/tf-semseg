import tensorflow as tf
from .util import *
from . import config, decode, transformer

def object_attention(token_features, regions_features, filters_qkv, name=None, config=config.Config()):
    output_filters = token_features.shape[-1]

    query = token_features # [batch, token, features]
    query = conv_norm_act(query, filters=filters_qkv, kernel_size=1, stride=1, name=join(name, "query", "1"), config=config)
    query = conv_norm_act(query, filters=filters_qkv, kernel_size=1, stride=1, name=join(name, "query", "2"), config=config) # [batch, token, filters]

    key = regions_features # [batch, regions, features]
    key = conv_norm_act(key, filters=filters_qkv, kernel_size=1, stride=1, name=join(name, "key", "1"), config=config)
    key = conv_norm_act(key, filters=filters_qkv, kernel_size=1, stride=1, name=join(name, "key", "2"), config=config) # [batch, regions, filters]

    value = regions_features # [batch, regions, features]
    value = conv_norm_act(value, filters=filters_qkv, kernel_size=1, stride=1, name=join(name, "down"), config=config) # [batch, regions, filters]

    token_features = transformer.multihead_attention(query, key, value, heads=1, config=config) # [batch, token, features]

    token_features = conv_norm_act(token_features, filters=output_filters, kernel_size=1, stride=1, name=join(name, "up"), config=config)
    return token_features

def ocr(x, regions, filters=512, filters_qkv=256, fix_bias_before_norm=True, name="ocr", config=config.Config()):
    regions_probs = x
    regions_probs = conv_norm_act(regions_probs, kernel_size=1, stride=1, bias=not fix_bias_before_norm, name=join(name, "regions"), config=config)
    regions_probs = decode.decode(regions_probs, filters=regions, name=join(name, "regions", "decode"), config=config)
    regions_probs = transformer.tokenize(regions_probs)
    regions_probs = tf.nn.softmax(regions_probs, axis=1) # [batch, token, regions]

    x = conv_norm_act(x, filters=filters, kernel_size=3, stride=1, bias=not fix_bias_before_norm, name=join(name, "initial"), config=config)
    x_tokens = transformer.tokenize(x) # [batch, token, features]

    # Gather
    regions_features = tf.matmul(regions_probs, x_tokens, transpose_a=True) # [batch, regions, features]

    # Distribute
    context_features = object_attention(x_tokens, regions_features, filters_qkv=filters_qkv, name=join(name, "distribute"), config=config) # [batch, token, features]
    context_features = transformer.detokenize(context_features, tf.shape(x)[1:-1])

    x = tf.concat([context_features, x], axis=-1)

    return x
