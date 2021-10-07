import math
import tensorflow as tf
from . import aggregator

def flip(next_predictor, axis, aggregator_factory=aggregator.sum):
    def predict(batch):
        aggregator = aggregator_factory()

        batch1 = next_predictor(batch)
        aggregator.add(batch1)

        batch2 = tf.reverse(batch, axis=[axis])
        batch2 = next_predictor(batch2)
        batch2 = tf.reverse(batch2, axis=[axis])
        aggregator.add(batch2)

        return aggregator.get()
    return predict

def multi_scale(next_predictor, scales, aggregator_factory=aggregator.sum, interpolation=tf.image.ResizeMethod.BILINEAR):
    def predict(batch):
        aggregator = aggregator_factory()

        image_shape = batch.shape[1:-1]
        for scale in scales:
            if scale != 1.0:
                output_size = (int(image_shape[0] * scale), int(image_shape[1] * scale))
                resized = tf.image.resize(batch, output_size, method=interpolation)
            else:
                resized = batch
            resized = next_predictor(resized)
            resized = tf.image.resize(resized, image_shape, method=interpolation)

            aggregator.add(resized)

        return aggregator.get()
    return predict

def pad_to_min_size(next_predictor, min_size):
    tf_min_size = tf.convert_to_tensor(min_size)
    def predict(batch):
        image_shape = batch.shape[1:-1]
        all_greater = True
        for d in range(len(image_shape)):
            all_greater = all_greater and image_shape[d] >= min_size[d]
        if all_greater:
            return next_predictor(batch)
        else:
            start = tf.math.maximum(0, (tf_min_size - image_shape) // 2)
            end = start + image_shape

            paddings = [[0, 0]] + [[start[d], tf.math.maximum(image_shape[d], tf_min_size[d]) - end[d]] for d in range(len(image_shape))] + [[0, 0]]
            batch_padded = tf.pad(batch, paddings=paddings)

            batch_padded = next_predictor(batch_padded)

            start = tf.concat([[0], start, [0]], axis=0)
            size = tf.concat([[batch_padded.shape[0]], image_shape, [batch_padded.shape[-1]]], axis=0)
            batch = tf.slice(batch_padded, start, size)
            return batch
    return predict

def sliding(next_predictor, window_size, overlap, aggregator_factory=aggregator.sum):
    tf_window_size = tf.convert_to_tensor(window_size)
    def predict(batch):
        image_shape = batch.shape[1:-1]
        tf.debugging.assert_greater_equal(image_shape, tf_window_size)

        stride = [math.ceil(window_size[d] * (1 - overlap)) for d in range(len(image_shape))]
        tiles = [max(int(math.ceil((image_shape[d] - window_size[d]) / stride[d]) + 1), 1) for d in range(len(image_shape))]
        stride = tf.convert_to_tensor(stride)

        tile_positions = tf.meshgrid(*[tf.range(0, t) for t in tiles], indexing="ij")
        tile_positions = tf.stack(tile_positions, axis=-1)
        tile_positions = tf.reshape(tile_positions, [-1, len(tiles)])

        @tf.function
        def predict_tile(tile):
            start = tf.math.minimum([tile[d] * stride[d] for d in range(len(image_shape))], image_shape - tf_window_size)
            end = start + tf_window_size
            paddings = [[0, 0]] + [[start[d], image_shape[d] - end[d]] for d in range(len(image_shape))] + [[0, 0]]
            start = tf.concat([[0], start, [0]], axis=0)
            size = tf.concat([[batch.shape[0]], tf_window_size, [batch.shape[-1]]], axis=0)

            cropped_batch = tf.slice(batch, start, size)
            cropped_batch = next_predictor(cropped_batch)
            return tf.pad(cropped_batch, paddings=paddings)

        aggregator = aggregator_factory()
        aggregator.add_all(predict_tile, tile_positions)
        return aggregator.get()
    return pad_to_min_size(predict, window_size)
