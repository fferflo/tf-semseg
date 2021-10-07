import tfcv, eval
import tensorflow as tf

def test_pspnet_resnet_v1s_101_cityscapes():
    model = tfcv.model.pretrained.vladkryvoruchko.pspnet_resnet_v1s_101_cityscapes.create()
    predictor = lambda x: model(x, training=False)
    predictor = tfcv.predict.sliding(predictor, (713, 713), 0.2)
    predictor = tf.function(predictor)
    accuracy = eval.cityscapes(predictor, tfcv.model.pretrained.vladkryvoruchko.pspnet_resnet_v1s_101_cityscapes.preprocess)
    print(f"Got accuracy {accuracy * 100.0}")
    assert accuracy > 0.981
