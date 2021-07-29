import tf_semseg, eval
import tensorflow as tf

def test_pspnet_resnet_v1s_101_cityscapes():
    model = tf_semseg.model.pretrained.vladkryvoruchko.pspnet_resnet_v1s_101_cityscapes.create()
    predictor = lambda x: model(x, training=False)
    predictor = tf_semseg.predict.sliding(predictor, (713, 713), 0.2)
    predictor = tf.function(predictor)
    accuracy = eval.cityscapes(predictor, tf_semseg.model.pretrained.vladkryvoruchko.pspnet_resnet_v1s_101_cityscapes.preprocess)
    print(f"Got accuracy {accuracy * 100.0}")
    assert accuracy > 0.981
