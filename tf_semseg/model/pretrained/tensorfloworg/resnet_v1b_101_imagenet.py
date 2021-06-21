from ... import resnet
from .resnet_v1b_x_imagenet import convert_name, config, create_x, preprocess

def create(dilated=False):
    return create_x(dilated, resnet.resnet_v1_101, "http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz")
