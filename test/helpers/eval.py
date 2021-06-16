import tf_semseg, cv2, imageio, os
import numpy as np
import tensorflow as tf

datadir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

def cityscapes(predictor, preprocess): # frankfurt_000000_000294_leftImg8bit
    color = imageio.imread(os.path.join(datadir, "cityscapes", "color.png")).astype("float32")
    labels = imageio.imread(os.path.join(datadir, "cityscapes", "labels.png"))

    color_preprocessed = preprocess(color)

    prediction = predictor(np.expand_dims(color_preprocessed, axis=0))[0]
    classes_num = prediction.shape[-1]

    labels = tf.one_hot(labels.astype("int32"), axis=-1, depth=classes_num)
    metric = tf_semseg.metric.Accuracy(classes_num=classes_num)
    metric.update_state(labels, prediction)

    return metric.result().numpy()

def nyu_depth_v2(predictor, preprocess):
    color = imageio.imread(os.path.join(datadir, "nyu-depth-v2", "color.png")).astype("float32")
    depth = imageio.imread(os.path.join(datadir, "nyu-depth-v2", "depth.png")).astype("float32") * 0.1 # Depth is stored as 10e-4 m
    labels = imageio.imread(os.path.join(datadir, "nyu-depth-v2", "labels.png"))

    color = cv2.resize(color, (640, 480), interpolation=cv2.INTER_LINEAR)
    depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
    labels = cv2.resize(labels, (640, 480), interpolation=cv2.INTER_NEAREST)

    color_preprocessed, depth_preprocessed = preprocess(color, depth)

    prediction = predictor([np.expand_dims(color_preprocessed, axis=0), np.expand_dims(np.expand_dims(depth_preprocessed, axis=0), axis=-1)])[0]
    classes_num = prediction.shape[-1]

    labels = tf.one_hot(labels.astype("int32") - 1, axis=-1, depth=classes_num)
    metric = tf_semseg.metric.Accuracy(classes_num=classes_num)
    metric.update_state(labels, prediction)

    return metric.result().numpy()
