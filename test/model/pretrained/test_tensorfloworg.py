import tf_semseg

def test_resnet_v1b_50_imagenet():
    tf_semseg.model.pretrained.tensorfloworg.resnet_v1b_50_imagenet.create(dilate=False)
    tf_semseg.model.pretrained.tensorfloworg.resnet_v1b_50_imagenet.create(dilate=True)

def test_resnet_v1b_101_imagenet():
    tf_semseg.model.pretrained.tensorfloworg.resnet_v1b_101_imagenet.create(dilate=False)
    tf_semseg.model.pretrained.tensorfloworg.resnet_v1b_101_imagenet.create(dilate=True)

def test_resnet_v1b_152_imagenet():
    tf_semseg.model.pretrained.tensorfloworg.resnet_v1b_152_imagenet.create(dilate=False)
    tf_semseg.model.pretrained.tensorfloworg.resnet_v1b_152_imagenet.create(dilate=True)
