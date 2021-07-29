import tf_semseg, eval
import tensorflow as tf

def test_deeplabv3plus_resnet101_v1b_cityscapes():
    model = tf_semseg.model.pretrained.vainf.deeplabv3plus_resnet101_v1b_cityscapes.create()
    predictor = lambda x: model(x, training=False)
    accuracy = eval.cityscapes(predictor, tf_semseg.model.pretrained.vainf.deeplabv3plus_resnet101_v1b_cityscapes.preprocess)
    print(f"Got accuracy {accuracy * 100.0}")
    assert accuracy > 0.979
