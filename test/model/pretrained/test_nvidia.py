import tf_semseg, eval
import tensorflow as tf

def test_hrnet_v2_w48_ocr_mscale_cityscapes():
    model = tf_semseg.model.pretrained.nvidia.hrnet_v2_w48_ocr_mscale_cityscapes.create()
    predictor = lambda x: model(x, training=False)

    predictor_singlescale = tf_semseg.model.mscale.predictor_singlescale(predictor, config=tf_semseg.model.pretrained.nvidia.hrnet_v2_w48_ocr_mscale_cityscapes.config)
    predictor_singlescale = lambda x, predictor=predictor_singlescale: tf.nn.softmax(predictor(x), axis=-1)
    accuracy = eval.cityscapes(predictor_singlescale, tf_semseg.model.pretrained.nvidia.hrnet_v2_w48_ocr_mscale_cityscapes.preprocess)
    print(f"Got single-scale accuracy {accuracy * 100.0}")
    assert accuracy > 0.87

    predictor_multiscale = tf_semseg.model.mscale.predictor_multiscale(predictor, [0.5, 1.0, 2.0], config=tf_semseg.model.pretrained.nvidia.hrnet_v2_w48_ocr_mscale_cityscapes.config)
    predictor_multiscale = lambda x, predictor=predictor_multiscale: tf.nn.softmax(predictor(x), axis=-1)
    accuracy = eval.cityscapes(predictor_multiscale, tf_semseg.model.pretrained.nvidia.hrnet_v2_w48_ocr_mscale_cityscapes.preprocess)
    print(f"Got multi-scale accuracy {accuracy * 100.0}")
    assert accuracy > 0.87
