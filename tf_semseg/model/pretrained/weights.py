import tensorflow as tf
import numpy as np
import os, sys, h5py

def load_h5(file, model, convert_name, ignore=None):
    with h5py.File(file, "r") as f:
        keys = []
        f.visit(keys.append)
        all_weights = {key: np.asarray(f[key]) for key in keys if ":" in key}
    for var in model.variables:
        key = convert_name(var.name)
        if not key in all_weights:
            print(f"Variable {key} not found in {os.path.basename(file)}")
            sys.exit(-1)
        weights = all_weights[key]
        if weights.shape != var.shape:
            print(f"Variable {key} expected shape {var.shape} but got shape {weights.shape}")
            sys.exit(-1)
        var.assign(weights)
        del all_weights[key]
    for key in list(all_weights.keys()):
        if not ignore is None and ignore(key):
            del all_weights[key]
    keys = list(all_weights.keys())
    if len(keys) > 0:
        print(f"Failed to find variable for weights: {keys}")
        sys.exit(-1)

def load_ckpt(file, model, convert_name): # TODO: ignore layers?
    for v in model.variables:
        key = convert_name(v.name)
        new_var = tf.train.load_variable(file, key)
        v.assign(new_var)

def load_pth(file, model, convert_name, ignore=None):
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
    def get_weight(key):
        if not key in all_weights:
            print(f"Variable {key} not found in {os.path.basename(file)}")
            sys.exit(-1)
        result = all_weights[key]
        del all_weights[key]
        return np.asarray(result)
    def set_weights(layer, weights):
        dest_shapes = [w.shape for w in layer.get_weights()]
        src_shapes = [w.shape for w in weights]
        for dest_shape, src_shape in zip(dest_shapes, src_shapes):
            if not np.all(dest_shape == src_shape):
                raise ValueError(f"Layer {layer.name} with weight shapes {dest_shapes} got invalid weight shapes {src_shapes}")
        layer.set_weights(weights)
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            key = convert_name(layer.name)

            if isinstance(layer, tf.keras.layers.Conv1D) or isinstance(layer, tf.keras.layers.Conv2D) or isinstance(layer, tf.keras.layers.Conv3D):
                weights = get_weight(key + ".weight")
                weights = np.transpose(weights, (2, 3, 1, 0))
                while len(weights.shape) > len(layer.get_weights()[0].shape):
                    weights = np.squeeze(weights, axis=0)
                if not layer.bias is None:
                    bias = get_weight(key + ".bias")
                    set_weights(layer, [weights, bias])
                else:
                    assert (key + ".bias") not in all_weights
                    set_weights(layer, [weights])
            elif isinstance(layer, tf.keras.layers.BatchNormalization):
                weights = get_weight(key + ".weight")
                bias = get_weight(key + ".bias")
                running_mean = get_weight(key + ".running_mean")
                running_var = get_weight(key + ".running_var")
                set_weights(layer, [weights, bias, running_mean, running_var])
            else:
                print(f"Invalid type of layer {layer.name}")
                sys.exit(-1)
    for key in list(all_weights.keys()):
        if "num_batches_tracked" in key or (not ignore is None and ignore(key)):
            del all_weights[key]
    keys = list(all_weights.keys())
    if len(keys) > 0:
        print(f"Failed to find layer for torch variables: {keys}")
        sys.exit(-1)
