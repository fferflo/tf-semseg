import tensorflow as tf

def ConvND(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.Conv1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.Conv2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.Conv3D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 2:
            return tf.keras.layers.Conv1D(*args, **kwargs)(x[:, tf.newaxis])[:, 0] # TODO: replace with dense layer?
        else:
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor

def ZeroPaddingND(padding, *args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            keras_padding = padding
            if isinstance(keras_padding, tuple):
                keras_padding = keras_padding[0]
            return tf.keras.layers.ZeroPadding1D(padding=keras_padding, *args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.ZeroPadding2D(padding=padding, *args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.ZeroPadding3D(padding=padding, *args, **kwargs)(x)
        else:
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor

def MaxPoolND(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.MaxPool1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.MaxPool2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.MaxPool3D(*args, **kwargs)(x)
        else:
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor

def AveragePoolingND(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.AveragePooling1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.AveragePooling2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.AveragePooling3D(*args, **kwargs)(x)
        else:
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor

def UpSamplingND(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.UpSampling1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.UpSampling2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.UpSampling3D(*args, **kwargs)(x)
        else:
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor

def SpatialDropoutND(*args, **kwargs):
    def constructor(x):
        if len(x.get_shape()) == 3:
            return tf.keras.layers.SpatialDropout1D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 4:
            return tf.keras.layers.SpatialDropout2D(*args, **kwargs)(x)
        elif len(x.get_shape()) == 5:
            return tf.keras.layers.SpatialDropout3D(*args, **kwargs)(x)
        else:
            raise ValueError(f"Unsupported number of dimensions {len(x.get_shape())}")
    return constructor
