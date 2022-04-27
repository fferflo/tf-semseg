import tensorflow as tf

def get_all(x, pred):
    if not hasattr(x, "_keras_history"):
        raise ValueError(f"Input must be symbolic tensor, got {type(x)}")

    done = set()
    result = list()
    def recurse(node):
        if not node in done:
            done.add(node)
            if pred(node.layer):
                if hasattr(node.output_tensors, "_keras_history"):
                    result.append(node.output_tensors)
                else:
                    result.extend(node.output_tensors)
            for n in node.parent_nodes:
                recurse(n)
    recurse(x.node)
    return result

def get_unique(x, pred):
    if not hasattr(x, "_keras_history"):
        raise ValueError(f"Input must be symbolic tensor, got {type(x)}")

    result = get_all(x, pred)
    if len(result) > 1:
        raise ValueError(f"Node has more than one predecessor matching the given predicate: {[t.node.layer.name for t in result]}")
    if len(result) == 0:
        raise ValueError("Node has no predecessor matching the given predicate")
    return result[0]

def update(x, pred, op, keep_weights=True):
    if not hasattr(x, "_keras_history"):
        raise ValueError(f"Input must be symbolic tensor, got {type(x)}")

    has_modified_output_ = {}
    def has_modified_output(node):
        if not id(node) in has_modified_output_:
            has_modified_output_[id(node)] = pred(node.layer) or any(has_modified_output(n) for n in node.parent_nodes)
        return has_modified_output_[id(node)]

    layers = {}
    def get_layer(layer):
        if not id(layer) in layers:
            layers[id(layer)] = type(layer).from_config(layer.get_config())
        return layers[id(layer)]

    node_to_output_tensors = {}
    def get_output_tensors(node):
        if not id(node) in node_to_output_tensors:
            if isinstance(node.layer, tf.keras.layers.InputLayer) or not has_modified_output(node):
                output_tensors = node.output_tensors
            elif pred(node.layer):
                input_tensors = [t for n in node.parent_nodes for t in get_output_tensors(n)]
                input_tensors = input_tensors[0] if len(input_tensors) == 1 else input_tensors
                new_layer = get_layer(node.layer)
                output_tensors = op(input_tensors, new_layer, node.layer.get_weights())
                if keep_weights and len(new_layer.get_weights()) > 0:
                    new_layer.set_weights(node.layer.get_weights())
            else:
                def recurse(x):
                    if isinstance(x, list):
                        return [recurse(y) for y in x]
                    elif isinstance(x, tuple):
                        return tuple([recurse(y) for y in x])
                    elif isinstance(x, dict):
                        return {recurse(k): recurse(v) for k, v in x.items()}
                    elif hasattr(x, "_keras_history"):
                        for parent_node in node.parent_nodes:
                            if parent_node is x.node:
                                result = get_output_tensors(parent_node)
                                assert len(result) == 1
                                result = result[0]
                                assert hasattr(result, "_keras_history")
                                return result
                        else:
                            assert False
                    else:
                        return x
                new_layer = get_layer(node.layer)
                call_args = recurse(node.call_args)
                call_kwargs = recurse(node.call_kwargs)
                output_tensors = new_layer(*call_args, **call_kwargs)
                if keep_weights:
                    new_layer.set_weights(node.layer.get_weights())
            output_tensors = [output_tensors] if hasattr(output_tensors, "_keras_history") else output_tensors
            assert not id(node) in node_to_output_tensors
            node_to_output_tensors[id(node)] = output_tensors

        return node_to_output_tensors[id(node)]

    result = get_output_tensors(x.node)
    assert len(result) == 1
    result = result[0]
    assert hasattr(result, "_keras_history")
    return result

def replace(x, pred, block, **kwargs):
    return update(x, pred, lambda x, layer, layer_weights: block(x, layer, layer_weights), **kwargs)

def remove(x, pred, **kwargs):
    return update(x, pred, lambda x, layer, layer_weights: x, **kwargs)

def insert(x, pred, block, position="after", **kwargs):
    if not position in ["after", "before"]:
        raise ValueError("Position must be one of 'after' or 'before'")
    return update(x, pred, lambda x, layer, layer_weights: block(layer(x), layer, layer_weights) if position == "after" else layer(block(x, layer, layer_weights)), **kwargs)
