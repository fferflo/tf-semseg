import tfcv, eval
import tensorflow as tf

def test_hrnet_v2_w48_ocr_mscale_cityscapes():
    model = tfcv.model.pretrained.nvidia.hrnet_v2_w48_ocr_mscale_cityscapes.create()
    predictor = lambda x: model(x, training=False)

    predictor_singlescale = tfcv.model.mscale.predictor_singlescale(predictor, config=tfcv.model.pretrained.nvidia.hrnet_v2_w48_ocr_mscale_cityscapes.config)
    predictor_singlescale = lambda x, predictor=predictor_singlescale: tf.nn.softmax(predictor(x), axis=-1)
    accuracy = eval.cityscapes(predictor_singlescale, tfcv.model.pretrained.nvidia.hrnet_v2_w48_ocr_mscale_cityscapes.preprocess)
    print(f"Got single-scale accuracy {accuracy * 100.0}")
    assert accuracy > 0.985

    predictor_multiscale = tfcv.model.mscale.predictor_multiscale(predictor, [0.5, 1.0, 2.0], config=tfcv.model.pretrained.nvidia.hrnet_v2_w48_ocr_mscale_cityscapes.config)
    predictor_multiscale = lambda x, predictor=predictor_multiscale: tf.nn.softmax(predictor(x), axis=-1)
    accuracy = eval.cityscapes(predictor_multiscale, tfcv.model.pretrained.nvidia.hrnet_v2_w48_ocr_mscale_cityscapes.preprocess)
    print(f"Got multi-scale accuracy {accuracy * 100.0}")
    assert accuracy > 0.985
