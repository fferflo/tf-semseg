import tfcv, eval
import tensorflow as tf

def test_hrnet_v2_w48_cityscapes():
    model = tfcv.model.pretrained.hrnet.hrnet_v2_w48_cityscapes.create()
    predictor = lambda x: model(x, training=False)
    accuracy = eval.cityscapes(predictor, tfcv.model.pretrained.hrnet.hrnet_v2_w48_cityscapes.preprocess)
    print(f"Got accuracy {accuracy * 100.0}")
    assert accuracy > 0.980

def test_hrnet_v2_w48_ocr_cityscapes():
    model = tfcv.model.pretrained.hrnet.hrnet_v2_w48_ocr_cityscapes.create()
    predictor = lambda x: model(x, training=False)
    accuracy = eval.cityscapes(predictor, tfcv.model.pretrained.hrnet.hrnet_v2_w48_ocr_cityscapes.preprocess)
    print(f"Got accuracy {accuracy * 100.0}")
    assert accuracy > 0.983
