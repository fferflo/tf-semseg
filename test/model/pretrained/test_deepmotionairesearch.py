import tfcv, eval
import tensorflow as tf

def test_deepmotionairesearch_densenet161_denseapp_cityscapes():
    model = tfcv.model.pretrained.deepmotionairesearch.densenet161_denseaspp_cityscapes.create()
    predictor = lambda x: model(x, training=False)
    accuracy = eval.cityscapes(predictor, tfcv.model.pretrained.deepmotionairesearch.densenet161_denseaspp_cityscapes.preprocess)
    print(f"Got accuracy {accuracy * 100.0}")
    assert accuracy > 0.959
