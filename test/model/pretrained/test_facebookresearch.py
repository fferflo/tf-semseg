import tfcv, eval, re

def test_convnext_imagenet():
    for name, builder in vars(tfcv.model.pretrained.facebookresearch).items():
        if re.match("convnext_[a-z]*_[a-z0-9]*_[0-9]*$", name):
            builder.create()

def test_convnext_upernet_ade20k():
    for name, builder in vars(tfcv.model.pretrained.facebookresearch).items():
        if re.match("convnext_[a-z]*_upernet_[a-z0-9]*_ade20k_[0-9]*$", name):
            model = builder.create()
            predictor = lambda x: model(x, training=False)
            accuracy = eval.ade20k(predictor, builder.preprocess)
            print(f"Model {name} got accuracy {accuracy * 100.0}")
            assert accuracy > 0.9
