import tf_semseg, eval
import tensorflow as tf

def test_deepmotionairesearch_densenet161_denseapp_cityscapes():
    model = tf_semseg.model.pretrained.deepmotionairesearch.densenet161_denseaspp_cityscapes.create()
    predictor = lambda x: model(x, training=False)
    accuracy = eval.cityscapes(predictor, tf_semseg.model.pretrained.deepmotionairesearch.densenet161_denseaspp_cityscapes.preprocess)
    print(f"Got accuracy {accuracy * 100.0}")
    assert accuracy > 0.959
