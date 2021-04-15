import tensorflow as tf

dont_care_threshold = 0.9

class aggregator:
    def __init__(self):
        self.probs = None

    def add(self, probs, weight = None):
        if not weight is None:
            probs = self.weight(probs, weight)
        self.probs = self.accumulate(self.probs, probs)

    def add_all(self, fn, inputs):
        initializer = self.accumulate(None, fn(inputs[0]))
        inputs = inputs[1:]
        @tf.function
        def accumulate(acc, input):
            return self.accumulate(acc, fn(input))
        self.add(tf.foldl(fn=accumulate, elems=inputs, initializer=initializer))

class sum(aggregator):
    def __init__(self):
        super().__init__()

    def accumulate(self, acc, new):
        if acc is None:
            return new
        else:
            return acc + new

    def weight(self, probs, weight):
        return probs * weight

    def get(self):
        return self.probs / tf.reduce_sum(self.probs, axis=-1, keepdims=True)

class product(aggregator):
    def __init__(self):
        super().__init__()

    def accumulate(self, acc, new):
        valid = tf.reduce_sum(new, axis=-1, keepdims=True) > dont_care_threshold
        if acc is None:
            return tf.where(valid, new, 1.0)
        else:
            return tf.where(valid, acc * new, acc)

    def weight(self, probs, weight):
        return tf.math.pow(probs, weight)

    def get(self):
        return self.probs / tf.reduce_sum(self.probs, axis=-1, keepdims=True)

class max(aggregator):
    def __init__(self):
        super().__init__()

    def accumulate(self, acc, new):
        if acc is None:
            return new
        else:
            return tf.maximum(acc, new)

    def weight(self, probs, weight):
        return tf.where(weight > 0, probs, 0.0)

    def get(self):
        return self.probs / tf.reduce_sum(self.probs, axis=-1, keepdims=True)
