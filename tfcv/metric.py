import tensorflow as tf
import numpy as np

dont_care_threshold = 0.9

class BoundaryLabelRelaxed:
    def __init__(self, inner_metric, pool_size):
        self.pool_size = pool_size
        self.inner_metric = inner_metric
        self.name = "BoundaryLabelRelaxed" + str(self.pool_size) + "." + inner_metric.name

    def update_state(self, y_true, y_pred):
        y_true = tf.nn.max_pool(y_true, [1] + [self.pool_size] * (y_true.shape.ndims - 2) + [1], strides=1, padding="SAME")
        self.inner_metric.update_state(y_true, y_pred)

    def result(self):
        return self.inner_metric.result()

    def reset_state(self):
        self.inner_metric.reset_state()

class ConfusionMatrix(tf.keras.metrics.Metric):
    def __init__(self, classes_num, *args, allow_multiple_groundtruths=True, dontcare_prediction="forbidden", **kwargs):
        super().__init__(*args, **kwargs)
        self.classes_num = (classes_num + 1) if dontcare_prediction == "error" else classes_num
        self.allow_multiple_groundtruths = allow_multiple_groundtruths
        assert dontcare_prediction in ["forbidden", "ignore", "error"]
        self.dontcare_prediction = dontcare_prediction
        self.total_cm = self.add_weight(
            "total_confusion_matrix",
            shape=(self.classes_num, self.classes_num),
            initializer=tf.zeros_initializer,
            dtype=tf.float64)

    def update_state(self, y_true, y_pred):
        assert len(y_true.shape) == len(y_pred.shape)

        # Preprocess prediction
        if self.dontcare_prediction == "ignore":
            y_pred_valid = tf.reduce_sum(y_pred, axis=-1) > dont_care_threshold
            y_pred = tf.argmax(y_pred, axis=-1)
        elif self.dontcare_prediction == "error":
            y_pred_valid = tf.reduce_sum(y_pred, axis=-1) > dont_care_threshold
            y_pred = tf.where(y_pred_valid, tf.argmax(y_pred, axis=-1), self.classes_num - 1)
            y_pred_valid = tf.ones_like(y_pred, dtype="bool")
        else: # self.dontcare_prediction == "forbidden"
            tf.assert_greater(tf.reduce_sum(y_pred, axis=-1), dont_care_threshold)
            y_pred = tf.argmax(y_pred, axis=-1)
            y_pred_valid = tf.ones_like(y_pred, dtype="bool")

        # Preprocess groundtruth
        y_true_valid = tf.reduce_sum(y_true, axis=-1) > dont_care_threshold
        if self.allow_multiple_groundtruths:
            y_pred_one_hot = tf.one_hot(y_pred, depth=y_true.shape[-1])
            matches = tf.reduce_sum(y_true * y_pred_one_hot, axis=-1) > dont_care_threshold
            y_true = tf.where(matches, y_pred, tf.argmax(y_true - y_pred_one_hot, axis=-1))
        else:
            tf.debugging.assert_less_equal(tf.reduce_sum(y_true, axis=-1), 1.0 + 1e-6)
            y_true = tf.argmax(y_true, axis=-1)

        # Add to confusion matrix
        y_pred_valid = tf.reshape(y_pred_valid, [-1])
        y_true_valid = tf.reshape(y_true_valid, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.reshape(y_true, [-1])
        sample_weight = tf.where(tf.math.logical_and(y_true_valid, y_pred_valid), 1, 0)
        cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=self.classes_num, weights=sample_weight, dtype=self.total_cm.dtype)
        return self.total_cm.assign_add(cm)

    def result(self):
        return self.total_cm

    def reset_state(self):
        tf.keras.backend.set_value(self.total_cm, np.zeros((self.classes_num, self.classes_num)))

    def get_config(self):
        config = {"classes_num": self.classes_num}
        base_config = super(tf.keras.metrics.Metric, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def confusionmatrix_to_classious(confusion_matrix):
    if tf.is_tensor(confusion_matrix):
        sum_over_row = tf.reduce_sum(confusion_matrix, axis=0)
        sum_over_col = tf.reduce_sum(confusion_matrix, axis=1)
        true_positives =  tf.linalg.diag_part(confusion_matrix)

        denominator = sum_over_row + sum_over_col - true_positives

        return true_positives / denominator
    else:
        sum_over_row = np.sum(confusion_matrix, axis=0)
        sum_over_col = np.sum(confusion_matrix, axis=1)
        true_positives = np.diag(confusion_matrix)

        denominator = sum_over_row + sum_over_col - true_positives

        return true_positives / denominator

class ClassIoUs(ConfusionMatrix):
    def __init__(self, *args, name = "ClassIoUs", **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def result(self):
        cm = super().result()
        return tf.cast(confusionmatrix_to_classious(cm), dtype=self._dtype)

def classious_to_meaniou(ious):
    if tf.is_tensor(ious):
        num_valid_entries = tf.reduce_sum(tf.where(tf.math.is_nan(ious), 0.0, 1.0))
        ious = tf.where(tf.math.is_nan(ious), tf.zeros_like(ious), ious)

        return tf.cast(tf.math.divide_no_nan(tf.reduce_sum(ious, name="mean_iou"), num_valid_entries), dtype=ious.dtype)
    else:
        num_valid_entries = np.sum(np.where(np.isnan(ious), 0.0, 1.0))
        ious = np.where(np.isnan(ious), np.zeros_like(ious), ious)

        return np.nan_to_num(np.sum(ious, name="mean_iou") / num_valid_entries).astype(ious.dtype)

def confusionmatrix_to_meaniou(confusion_matrix):
    return classious_to_meaniou(confusionmatrix_to_classious(confusion_matrix))

class MeanIoU(ClassIoUs):
    def __init__(self, *args, name = "MeanIoU", **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def result(self):
        ious = super().result()
        return tf.cast(classious_to_meaniou(ious), dtype=self._dtype)

def confusionmatrix_to_accuracy(confusion_matrix):
    if tf.is_tensor(confusion_matrix):
        total = tf.reduce_sum(confusion_matrix)
        true_positives = tf.reduce_sum(tf.linalg.diag_part(confusion_matrix))

        return true_positives / total
    else:
        total = np.sum(confusion_matrix)
        true_positives = np.sum(np.diag(confusion_matrix))

        return true_positives / total

class Accuracy(ConfusionMatrix):
    def __init__(self, *args, name="Accuracy", **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def result(self):
        cm = super().result()
        return tf.cast(confusionmatrix_to_accuracy(cm), dtype=self._dtype)

class ClassAccuracies(ConfusionMatrix): # Recall
    def __init__(self, *args, name="ClassAccuracies", **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def result(self):
        cm = super().result()

        total = tf.cast(tf.reduce_sum(cm, axis=1), dtype=self._dtype)
        true_positives = tf.cast(tf.linalg.diag_part(cm), dtype=self._dtype)

        return true_positives / total

class ClassPrecision(ConfusionMatrix):
    def __init__(self, *args, name="ClassPrecision", **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def result(self):
        cm = super().result()

        total = tf.cast(tf.reduce_sum(cm, axis=0), dtype=self._dtype)
        true_positives = tf.cast(tf.linalg.diag_part(cm), dtype=self._dtype)

        return true_positives / total

class MeanClassAccuracy(ClassAccuracies):
    def __init__(self, *args, name="MeanClassAccuracy", **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def result(self):
        class_accs = super().result()

        valid = tf.where(tf.math.is_nan(class_accs), tf.zeros_like(class_accs), 1)
        class_accs = tf.where(tf.math.is_nan(class_accs), tf.zeros_like(class_accs), class_accs)

        return tf.reduce_sum(class_accs) / tf.reduce_sum(valid)

class ClassAccuracyStd(ClassAccuracies):
    def __init__(self, *args, name="ClassAccuracyStd", **kwargs):
        super().__init__(*args, name=name, **kwargs)

    def result(self):
        class_accs = super().result()

        num_valid = tf.reduce_sum(tf.where(tf.math.is_nan(class_accs), tf.zeros_like(class_accs), 1))

        mean = tf.reduce_sum(tf.where(tf.math.is_nan(class_accs), tf.zeros_like(class_accs), class_accs), keepdims=True) / num_valid
        diffs = tf.where(tf.math.is_nan(class_accs), tf.zeros_like(class_accs), class_accs - mean)

        std = tf.math.sqrt(tf.reduce_sum(diffs * diffs)) / num_valid

        return std
