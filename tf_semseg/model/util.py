import tensorflow as tf
from functools import partial

def join(*args):
    result = None
    for a in args:
        if result is None:
            result = a
        else:
            result = result + "/" + a
    return result

def default_norm(x, *args, **kwargs):
    return tf.keras.layers.BatchNormalization(*args, momentum=0.997, epsilon=1e-5, **kwargs)(x)

def conv_norm_relu(x, filters=None, stride=1, name=None, kernel_size=3, dilation_rate=1, norm=default_norm, cardinality=1):
    if filters is None:
        filters = x.shape[-1]
    x = Conv(filters, kernel_size=kernel_size, strides=stride, groups=cardinality, dilation_rate=dilation_rate, use_bias=False, padding="same", name=join(name, "conv"))(x)
    x = norm(x, name=join(name, "norm"))
    x = tf.keras.layers.ReLU()(x)
    return x

def repeat(x, n, block, name=None, **kwargs):
    for i in range(n):
        x = block(x, name=join(name, str(i + 1)), **kwargs)
    return x



########## Keras n-dimensional layers ##########

def Conv(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.Conv1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.Conv2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.Conv3D(*args, **kwargs)(x)
        else:
            assert False, "Unsupported number of dimensions"
    return constructor

def MaxPool(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.MaxPool1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.MaxPool2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.MaxPool3D(*args, **kwargs)(x)
        else:
            assert False, "Unsupported number of dimensions"
    return constructor

def AveragePool(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.AveragePooling1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.AveragePooling2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.AveragePooling3D(*args, **kwargs)(x)
        else:
            assert False, "Unsupported number of dimensions"
    return constructor

def UpSample(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.UpSampling1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.UpSampling2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.UpSampling3D(*args, **kwargs)(x)
        else:
            assert False, "Unsupported number of dimensions"
    return constructor
