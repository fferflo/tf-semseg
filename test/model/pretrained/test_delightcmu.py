import tfcv, eval
import tensorflow as tf

def test_hrnet_v2_w48_ocr_psa_cityscapes():
    for sequential in [False, True]:
        model = tfcv.model.pretrained.delightcmu.hrnet_v2_w48_ocr_psa_cityscapes.create(sequential=sequential)
        predictor = lambda x: model(x, training=False)

        predictor_singlescale = tfcv.model.mscale.predictor_singlescale(predictor, config=tfcv.model.pretrained.delightcmu.hrnet_v2_w48_ocr_psa_cityscapes.config)
        predictor_singlescale = lambda x, predictor=predictor_singlescale: tf.nn.softmax(predictor(x), axis=-1)
        accuracy = eval.cityscapes(predictor_singlescale, tfcv.model.pretrained.delightcmu.hrnet_v2_w48_ocr_psa_cityscapes.preprocess)
        print(f"Got single-scale accuracy {accuracy * 100.0}")
        assert accuracy > 0.9845

        predictor_multiscale = tfcv.model.mscale.predictor_multiscale(predictor, [0.5, 1.0, 2.0], config=tfcv.model.pretrained.delightcmu.hrnet_v2_w48_ocr_psa_cityscapes.config)
        predictor_multiscale = lambda x, predictor=predictor_multiscale: tf.nn.softmax(predictor(x), axis=-1)
        accuracy = eval.cityscapes(predictor_multiscale, tfcv.model.pretrained.delightcmu.hrnet_v2_w48_ocr_psa_cityscapes.preprocess)
        print(f"Got multi-scale accuracy {accuracy * 100.0}")
        assert accuracy > 0.985
