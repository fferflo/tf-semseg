import tensorflow as tf

def colorize(segmentation, class_to_color, dont_care_color=(0, 0, 0)):
    colors = tf.where(segmentation >= len(class_to_color), len(class_to_color), segmentation)
    class_to_color = tf.convert_to_tensor(class_to_color + [dont_care_color])

    # Gather colors
    colors = tf.reshape(colors, [-1])
    colors = tf.cast(colors, "int32")
    colors = tf.gather(class_to_color, colors)
    colors = tf.reshape(colors, (segmentation.shape[0], segmentation.shape[1], 3))
    colors = tf.cast(colors, dtype=tf.uint8)

    return tf.cast(colors, "uint8")
