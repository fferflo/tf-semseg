import tensorflow as tf
import numpy as np
import os, sys

def load_pth(file, model, convert_name, ignore=None):
    import torch

    all_weights = dict(torch.load(file, map_location=torch.device("cpu")))
    if "state_dict" in all_weights:
        all_weights = all_weights["state_dict"]
    # for k in all_weights.keys():
    #     print(k)
    # for layer in model.layers:
    #     if len(layer.get_weights()) > 0:
    #         print(layer.name)
    def get_weight(key):
        if not key in all_weights:
            print(f"Variable {key} not found in {os.path.basename(file)}")
            sys.exit(-1)
        result = all_weights[key]
        del all_weights[key]
        return np.asarray(result)
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            key = convert_name(layer.name)

            if "conv" in layer.name:
                weights = get_weight(key + ".weight")
                weights = np.transpose(weights, (2, 3, 1, 0))
                while len(weights.shape) > len(layer.get_weights()[0].shape):
                    weights = np.squeeze(weights, axis=0)
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
                sys.exit(-1)
    for key in list(all_weights.keys()):
        if "num_batches_tracked" in key or (not ignore is None and ignore(key)):
            del all_weights[key]
    keys = list(all_weights.keys())
    if len(keys) > 0:
        print(f"Failed to find layer for torch variables: {keys}")
        sys.exit(-1)
