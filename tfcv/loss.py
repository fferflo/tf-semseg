import tensorflow as tf

dont_care_threshold = 0.9

class BoundaryLabelRelaxed:
    def __init__(self, pool_size, loss):
        self.pool_size = pool_size
        self.inner_loss = loss
        self.name = "BoundaryLabelRelaxed" + str(self.pool_size) + "." + self.inner_loss.name

    def __call__(self, y_true, y_pred):
        y_true = tf.nn.max_pool(y_true, [1] + [self.pool_size] * (y_true.shape.ndims - 2) + [1], strides=1, padding="SAME")
        return self.inner_loss(y_true, y_pred)

class Weighted:
    def __init__(self, weight, loss):
        self.weight = tf.convert_to_tensor(weight)
        self.inner_loss = loss
        self.name = "Weighted." + self.inner_loss.name

    def __call__(self, y_true, y_pred):
        sample_weights = tf.gather(params=self.weight, indices=tf.argmax(y_true, axis=-1))
        return tf.cast(sample_weights, y_pred.dtype) * self.inner_loss(y_true, y_pred)

class CrossEntropy:
    def __init__(self):
        self.name = "CrossEntropy"

    def __call__(self, y_true, y_pred):
        reduced_dot = tf.reduce_sum(y_true * y_pred, axis=-1)
        reduced_dot = tf.where(tf.reduce_sum(y_true, axis=-1) > dont_care_threshold, reduced_dot, 1.0)

        epsilon = tf.convert_to_tensor(tf.keras.backend.epsilon(), dtype=reduced_dot.dtype)
        reduced_dot = tf.clip_by_value(reduced_dot, epsilon, 1.0)

        return -tf.math.log(reduced_dot)

class Dice:
    def __init__(self, classes=None, class_weights=None):
        self.name = "Dice"
        self.classes = classes
        self.class_weights = class_weights
        if not classes is None and not class_weights is None:
            assert len(classes) == len(class_weights)

    def __call__(self, y_true, y_pred):
        # TODO: dont_care = tf.reduce_sum(y_true, axis=-1) <= dont_care_threshold

        axes = tuple(range(0, len(y_pred.shape) - 1))

        smooth = 1e-3
        intersection = 2 * tf.reduce_sum(y_true * y_pred, axes) + smooth
        union = tf.reduce_sum(y_true + y_pred, axes) + smooth
        ratio = intersection / union

        if self.classes != None:
            ratio = tf.gather(self.classes, ratio)
        loss = 1.0 - ratio
        if not self.class_weights is None:
            loss = loss * self.class_weights
        return loss

class Focal:
    def __init__(self, alpha = 0.25, gamma = 2.0, loss = CrossEntropy()):
        self.inner_loss = loss
        self.alpha = alpha
        self.gamma = gamma
        self.name = "Focal." + self.inner_loss.name

    def __call__(self, y_true, y_pred):
        return self.alpha * tf.pow(1.0 - tf.reduce_sum(y_true * y_pred, axis=-1) / tf.reduce_sum(y_true, axis=-1), self.gamma) * self.inner_loss(y_true, y_pred)

class Bootstrapped: # TODO: OHEM
    def __init__(self, rate, loss):
        self.rate = rate
        self.inner_loss = loss
        self.name = "Bootstrapped." + self.inner_loss.name

    def __call__(self, *args):
        loss = self.inner_loss(*args)
        loss = tf.reshape(loss, shape=[-1])

        loss = tf.sort(loss, direction="DESCENDING")
        loss = loss[:tf.cast(self.rate * tf.cast(tf.shape(loss)[0], dtype=tf.float32), dtype=tf.int32)]

        return loss
