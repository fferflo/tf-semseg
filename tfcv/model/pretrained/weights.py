import tensorflow as tf
import numpy as np
import os, h5py

class LoadWeightsException(Exception):
    pass

def load_h5(file, model, convert_name, ignore=None):
    with h5py.File(file, "r") as f:
        keys = []
        f.visit(keys.append)
        all_weights = {key: np.asarray(f[key]) for key in keys if ":" in key}
    for var in model.variables:
        key = convert_name(var.name)
        if not key in all_weights:
            raise LoadWeightsException(f"Variable {key} not found in {os.path.basename(file)}")
        weights = all_weights[key]
        if weights.shape != var.shape:
            raise LoadWeightsException(f"Variable {key} expected shape {var.shape} but got shape {weights.shape}")
        var.assign(weights)
        del all_weights[key]
    for key in list(all_weights.keys()):
        if not ignore is None and ignore(key):
            del all_weights[key]
    keys = list(all_weights.keys())
    if len(keys) > 0:
        raise LoadWeightsException(f"Failed to find variable for weights: {keys}")

def load_ckpt(file, model, convert_name): # TODO: ignore layers?
    for v in model.variables:
        key = convert_name(v.name)
        new_var = tf.train.load_variable(file, key)
        v.assign(new_var)

def load_pth(file, model, convert_name, ignore=None, map={}):
    import torch

    all_weights = dict(torch.load(file, map_location=torch.device("cpu")))
    if "state_dict" in all_weights:
        all_weights = all_weights["state_dict"]
    if "model_state" in all_weights:
        all_weights = all_weights["model_state"]
    # for k in all_weights.keys():
    #     print(k)
    # for layer in model.layers:
    #     if len(layer.get_weights()) > 0:
    #         print(layer.name)
    # sys.exit(-1)

    def get_weight(keys, default_mapper=lambda x: x):
        if not isinstance(keys, list):
            keys = [keys]
        keys2 = [k for k in keys if k in all_weights]
        if len(keys2) == 0:
            raise LoadWeightsException(f"Variable {keys} not found in {os.path.basename(file)}")
        if len(keys2) > 1:
            raise LoadWeightsException(f"More than one input matches variable {keys} in {os.path.basename(file)}")
        key = keys2[0]

        result = all_weights[key]
        del all_weights[key]
        result = np.asarray(result)

        if key in map:
            result = map[key](result)
        else:
            result = default_mapper(result)

        return result
    def set_weights(layer, weights):
        dest_shapes = [w.shape for w in layer.get_weights()]
        src_shapes = [w.shape for w in weights]
        for dest_shape, src_shape in zip(dest_shapes, src_shapes):
            if not np.all(dest_shape == src_shape):
                raise LoadWeightsException(f"Layer {layer.name} with weight shapes {dest_shapes} got invalid weight shapes {src_shapes}")
        layer.set_weights(weights)

    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            key = convert_name(layer.name)

            if isinstance(layer, tf.keras.layers.Conv1D) or isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Conv3D):
                weights = get_weight([key + ".weight", key + "_weight"], lambda w: np.transpose(w, list(range(2, len(w.shape))) + [1, 0]))
                if len(weights.shape) != len(layer.get_weights()[0].shape):
                    min_len = min(len(weights.shape), len(layer.get_weights()[0].shape))
                    if np.any(weights.shape[-min_len:] != layer.get_weights()[0].shape[-min_len:]):
                        raise LoadWeightsException(f"Convolution layer {layer.name} with kernel shape {layer.get_weights()[0].shape} got invalid loaded kernel shape {weights.shape}")
                    else:
                        while len(weights.shape) > len(layer.get_weights()[0].shape):
                            weights = weights[0]
                        while len(weights.shape) < len(layer.get_weights()[0].shape):
                            weights = np.expand_dims(weights, axis=0)
                if not layer.bias is None:
                    bias = get_weight([key + ".bias", key + "_bias"])
                    set_weights(layer, [weights, bias])
                else:
                    if (key + ".bias") in all_weights:
                        raise LoadWeightsException(f"Convolution layer {layer.name} does not have bias, but found bias in weights file")
                    set_weights(layer, [weights])
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                weights = get_weight(key + ".weight")
                bias = get_weight(key + ".bias")
                running_mean = get_weight(key + ".running_mean")
                running_var = get_weight(key + ".running_var")
                set_weights(layer, [weights, bias, running_mean, running_var])
            elif isinstance(layer, tf.keras.layers.LayerNormalization):
                weights = get_weight(key + ".weight")
                bias = get_weight(key + ".bias")
                set_weights(layer, [weights, bias])
            elif isinstance(layer, tf.keras.layers.Embedding):
                weights = get_weight(key)[0]
                set_weights(layer, [weights])
            else:
                raise LoadWeightsException(f"Invalid type of layer {layer.name}")
    for key in list(all_weights.keys()):
        if "num_batches_tracked" in key or (not ignore is None and ignore(key)):
            del all_weights[key]
    keys = list(all_weights.keys())
    if len(keys) > 0:
        raise LoadWeightsException(f"Failed to find layer for torch variables: {keys}")
