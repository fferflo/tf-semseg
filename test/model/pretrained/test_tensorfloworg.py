import tfcv

def test_resnet_v1b_50_imagenet():
    tfcv.model.pretrained.tensorfloworg.resnet_v1b_50_imagenet.create(dilate=False)
    tfcv.model.pretrained.tensorfloworg.resnet_v1b_50_imagenet.create(dilate=True)

def test_resnet_v1b_101_imagenet():
    tfcv.model.pretrained.tensorfloworg.resnet_v1b_101_imagenet.create(dilate=False)
    tfcv.model.pretrained.tensorfloworg.resnet_v1b_101_imagenet.create(dilate=True)

def test_resnet_v1b_152_imagenet():
    tfcv.model.pretrained.tensorfloworg.resnet_v1b_152_imagenet.create(dilate=False)
    tfcv.model.pretrained.tensorfloworg.resnet_v1b_152_imagenet.create(dilate=True)
