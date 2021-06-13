import tensorflow as tf
from ... import esanet
from google_drive_downloader import GoogleDriveDownloader as gdd
import os, pyunpack
import numpy as np
from ...config import Config

color_mean = np.asarray([0.485, 0.456, 0.406])
color_std = np.asarray([0.229, 0.224, 0.225])
depth_mean = 2841.94941272766
depth_std = 1417.2594281672277

def preprocess(color, depth):
    depth_0 = depth == 0
    depth = (depth - depth_mean) / depth_std
    depth[depth_0] = 0

    color = color / 255.0
    color = (color - color_mean) / color_std

    return color, depth

def convert_name(layer):
    key = layer
    key = key.replace("/", ".")
    key = key.replace("stem_b.stem_", "encoder_")
    key = key.replace("encoder_rgb.conv", "encoder_rgb.conv1")
    key = key.replace("encoder_depth.conv", "encoder_depth.conv1")
    key = key.replace("encoder_rgb.norm", "encoder_rgb.bn1")
    key = key.replace("encoder_depth.norm", "encoder_depth.bn1")
    key = key.replace("encode_", "encoder_")
    has_unit = False
    if "unit" in key:
        has_unit = True
        start_index = key.index("unit")
        unit = key[start_index + 4:]
        end_index = start_index + 4 + unit.index(".")
        unit = unit[:unit.index(".")]
        unit = int(unit)
        key = key[:start_index] + str(unit - 1) + key[end_index:]
    if "decode" in key:
        key = key.replace("decode", "decoder")
        key = key.replace("block", "decoder_module_")
        key = key.replace("initial", "conv3x3")
        key = key.replace("norm", "bn")
    if "se_" in key:
        key = key.replace("stem_b.se_", "se_layer0.se_")
        key = key.replace("block", "se_layer")
        key = key.replace("se_rgb.conv1", "se_rgb.fc.0")
        key = key.replace("se_depth.conv1", "se_depth.fc.0")
        key = key.replace("se_rgb.conv2", "se_rgb.fc.2")
        key = key.replace("se_depth.conv2", "se_depth.fc.2")
    else:
        key = key.replace("block", "layer")
    if "encode" in key:
        key = key.replace("shortcut.conv", "downsample.0")
        key = key.replace("shortcut.norm", "downsample.1")
    if "psp" in key:
        key = key.replace("psp", "context_module.features")
        key = key.replace("pool1", "0")
        key = key.replace("pool5", "1")
        key = key.replace("conv", "1.conv")
        key = key.replace("norm", "1.bn")
        key = key.replace("features.final.1", "final_conv")
    key = key.replace("1.dim0.conv", "conv3x1_1")
    key = key.replace("1.dim1.conv", "conv1x3_1")
    key = key.replace("2.dim0.conv", "conv3x1_2")
    key = key.replace("2.dim1.conv", "conv1x3_2")
    if "decode" in key:
        if has_unit:
            offset = len("decoder.decoder_module_1.")
            key = key[:offset] + "decoder_blocks." + key[offset:]
            if key.endswith("bn"):
                key = key.replace("1.bn", "bn1")
            if key.endswith("bn"):
                key = key.replace("2.bn", "bn2")
    key = key.replace("1.norm", "bn1")
    key = key.replace("2.norm", "bn2")
    if "shortcut" in key:
        key = key.replace("decoder.decoder_module_3.shortcut", "skip_layer1.0")
        key = key.replace("decoder.decoder_module_2.shortcut", "skip_layer2.0")
        key = key.replace("decoder.decoder_module_1.shortcut", "skip_layer3.0")
        key = key.replace("norm", "bn")
    key = key.replace("decoder.final.conv", "decoder.conv_out")
    key = key.replace("decoder.final", "decoder")
    return key

def esanet_resnet_v1b_34_nbt1d_nyuv2(): # Expects depth as mm and rgb in [0.0, 255.0]
    config = Config(
        mode="pytorch",
        norm=lambda x, *args, **kwargs: tf.keras.layers.BatchNormalization(*args, momentum=0.9, epsilon=1e-5, **kwargs)(x)
    )

    # Create model
    input_rgb = tf.keras.layers.Input((None, None, 3))
    input_depth = tf.keras.layers.Input((None, None, 1))
    x = esanet.esanet(
        input_rgb, input_depth,
        classes=40,
        num_residual_units=[3, 4, 6, 3],
        filters=[64, 128, 256, 512],
        dilation_rates=[1, 1, 1, 1],
        strides=[1, 2, 2, 2],
        psp_bin_sizes=[1, 5],
        config=config
    )
    x = tf.keras.layers.Softmax()(x)
    model = tf.keras.Model(inputs=[input_rgb, input_depth], outputs=[x])

    # Fix batchnorm epsilons
    for layer in model.layers:
        if layer.name.endswith("/1/norm") or layer.name.endswith("/2/norm"):
            layer.epsilon = 1e-3
    model = tf.keras.Model(inputs=model.inputs, outputs=[model.output])

    # https://github.com/TUI-NICR/ESANet
    download_file = os.path.join(os.path.expanduser("~"), ".keras", "nyuv2_r34_NBt1D_scenenet.tar.gz")
    gdd.download_file_from_google_drive(file_id="1w_Qa8AWUC6uHzQamwu-PAqA7P00hgl8w", dest_path=download_file)
    weights_uncompressed = os.path.join(os.path.dirname(download_file), "nyuv2", "r34_NBt1D_scenenet.pth")
    if not os.path.isfile(weights_uncompressed):
        pyunpack.Archive(download_file).extractall(os.path.dirname(download_file))

    import torch

    all_weights = dict(torch.load(weights_uncompressed, map_location=torch.device("cpu"))["state_dict"])
    def get_weight(key):
        if not key in all_weights:
            print(f"Variable {key} not found in {os.path.basename(download_file)}")
            os._exit(-1)
        result = all_weights[key]
        del all_weights[key]
        return np.asarray(result)
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            key = convert_name(layer.name)

            if "conv" in layer.name:
                weights = get_weight(key + ".weight")
                weights = np.transpose(weights, (2, 3, 1, 0))
                if not layer.bias is None:
                    bias = get_weight(key + ".bias")
                    layer.set_weights([weights, bias])
                else:
                    assert (key + ".bias") not in all_weights
                    layer.set_weights([weights])
            elif "norm" in layer.name:
                weights = get_weight(key + ".weight")
                bias = get_weight(key + ".bias")
                running_mean = get_weight(key + ".running_mean")
                running_var = get_weight(key + ".running_var")
                layer.set_weights([weights, bias, running_mean, running_var])
            else:
                print(f"Invalid type of layer {layer.name}")
                os._exit(-1)
    for key in list(all_weights.keys()):
        if "num_batches_tracked" in key or "side_output" in key:
            del all_weights[key]
    keys = list(all_weights.keys())
    if len(keys) > 0:
        print(f"Failed to find layer for torch variables: {keys}")
        os._exit(-1)
    return model
