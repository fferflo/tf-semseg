import tensorflow as tf
import numpy as np
from distinctipy import distinctipy

cityscapes_class_to_color = [ # TODO: move this to tfcv.data
    (128, 64,128),
    (244, 35,232),
    ( 70, 70, 70),
    (102,102,156),
    (190,153,153),
    (153,153,153),
    (250,170, 30),
    (220,220,  0),
    (107,142, 35),
    (152,251,152),
    ( 70,130,180),
    (220, 20, 60),
    (255,  0,  0),
    (  0,  0,142),
    (  0,  0, 70),
    (  0, 60,100),
    (  0, 80,100),
    (  0,  0,230),
    (119, 11, 32)
]

def colorize(segmentation, image=None, class_to_color=None, classes_num=None, dont_care_color=(0, 0, 0), dont_care_threshold=0.5):
    if len(segmentation.shape) == 3:
        dont_care = tf.reduce_sum(segmentation, axis=-1) < dont_care_threshold
        segmentation = tf.where(dont_care, -1, tf.argmax(segmentation, axis=-1))

    if class_to_color is None:
        if classes_num is None:
            classes_num = tf.math.reduce_max(segmentation).numpy() + 1
        class_to_color = np.asarray(distinctipy.get_colors(classes_num)) * 255.0
    else:
        if isinstance(class_to_color, str):
            if class_to_color == "cityscapes":
                class_to_color = cityscapes_class_to_color
        if not classes_num is None:
            assert classes_num == len(class_to_color)
        else:
            classes_num = len(class_to_color)

    # Add dont-care class at the end
    dont_care = tf.logical_or(segmentation >= len(class_to_color), segmentation < 0)
    class_to_color = tf.concat([class_to_color, [dont_care_color]], axis=0)
    segmentation_rgb = tf.where(dont_care, len(class_to_color) - 1, segmentation)

    # Gather segmentation colors
    segmentation_rgb = tf.reshape(segmentation_rgb, [-1])
    segmentation_rgb = tf.cast(segmentation_rgb, "int32")
    segmentation_rgb = tf.gather(class_to_color, segmentation_rgb)
    segmentation_rgb = tf.reshape(segmentation_rgb, tf.concat([tf.shape(segmentation), tf.shape(segmentation_rgb)[-1:]], axis=0))
    segmentation_rgb = tf.cast(segmentation_rgb, dtype=tf.uint8)

    # Overlay with color image
    if not image is None:
        image = 0.5 * tf.cast(image, "float32") + 0.5 * tf.cast(segmentation_rgb, "float32")
        image = tf.clip_by_value(image, 0.0, 255.0)
    else:
        image = segmentation_rgb

    return tf.cast(image, "uint8")
