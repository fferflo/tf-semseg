import tfcv, eval
import tensorflow as tf

def test_esanet_resnet_v1b_34_nbt1d_nyuv2():
    model = tfcv.model.pretrained.tuinicr.esanet_resnet_v1b_34_nbt1d_nyuv2.create()
    predictor = lambda x: model(x, training=False)
    accuracy = eval.nyu_depth_v2(predictor, tfcv.model.pretrained.tuinicr.esanet_resnet_v1b_34_nbt1d_nyuv2.preprocess)
    print(f"Got accuracy {accuracy * 100.0}")
    assert accuracy > 0.71
