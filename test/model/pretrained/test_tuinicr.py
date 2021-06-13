import unittest
import tf_semseg, cv2, imageio, os
import numpy as np
import tensorflow as tf

datadir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

class TestPretrained(unittest.TestCase):
    def test(self):
        model = tf_semseg.model.pretrained.tuinicr.esanet_resnet_v1b_34_nbt1d_nyuv2()
        classes_num = model.layers[-1].output.shape[-1]

        color = imageio.imread(os.path.join(datadir, "data", "nyu-depth-v2", "color.png")).astype("float32")
        depth = imageio.imread(os.path.join(datadir, "data", "nyu-depth-v2", "depth.png")).astype("float32") * 0.1
        labels = imageio.imread(os.path.join(datadir, "data", "nyu-depth-v2", "labels.png"))

        color = cv2.resize(color, (640, 480), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
        labels = cv2.resize(labels, (640, 480), interpolation=cv2.INTER_NEAREST)

        labels = tf.one_hot(labels.astype("int32") - 1, axis=-1, depth=classes_num)
        color_preprocessed, depth_preprocessed = tf_semseg.model.pretrained.tuinicr.preprocess(color, depth)

        prediction = model([np.expand_dims(color_preprocessed, axis=0), np.expand_dims(np.expand_dims(depth_preprocessed, axis=0), axis=-1)], training=False)[0]

        metric = tf_semseg.metric.Accuracy(classes_num=classes_num)
        metric.update_state(labels, prediction)
        self.assertGreater(metric.result().numpy(), 0.5) # Assert at least 50% of pixel predictions are correct

if __name__ == '__main__':
    unittest.main()
