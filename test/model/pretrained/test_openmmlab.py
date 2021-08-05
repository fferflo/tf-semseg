import tf_semseg, eval
import tensorflow as tf

def test_upernet_vitb_ade20k():
    model = tf_semseg.model.pretrained.openmmlab.upernet_vitb_ade20k.create()
    predictor = lambda x: model(x, training=False)
    accuracy = eval.ade20k(predictor, tf_semseg.model.pretrained.openmmlab.upernet_vitb_ade20k.preprocess)
    print(f"Got accuracy {accuracy * 100.0}")
    assert accuracy > 0.909
