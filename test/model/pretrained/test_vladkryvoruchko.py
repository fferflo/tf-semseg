# frankfurt_000000_000294_leftImg8bit

import unittest, tf_semseg, os, imageio
import numpy as np
import tensorflow as tf

datadir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestPretrained(unittest.TestCase):
    def test(self):
        model = tf_semseg.model.pretrained.vladkryvoruchko.pspnet_resnet_v1s_101_cityscapes()
        classes_num = model.layers[-1].output.shape[-1]
        predictor = tf_semseg.predict.sliding(model, (713, 713), 0.2)
        predictor = tf.function(predictor)

        color = imageio.imread(os.path.join(datadir, "data", "cityscapes", "color.png")).astype("float32")
        labels = imageio.imread(os.path.join(datadir, "data", "cityscapes", "labels.png"))

        labels = tf.one_hot(labels.astype("int32"), axis=-1, depth=classes_num)
        color_preprocessed = tf_semseg.model.pretrained.vladkryvoruchko.preprocess(color)

        prediction = predictor(np.expand_dims(color_preprocessed, axis=0))[0]

        metric = tf_semseg.metric.Accuracy(classes_num=classes_num)
        metric.update_state(labels, prediction)

        self.assertGreater(metric.result().numpy(), 0.8) # Assert at least 80% of pixel predictions are correct

if __name__ == '__main__':
    unittest.main()
