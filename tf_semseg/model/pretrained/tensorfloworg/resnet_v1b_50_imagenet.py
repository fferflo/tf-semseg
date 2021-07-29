from ... import resnet
from .resnet_v1b_x_imagenet import convert_name, config, create_x, preprocess

def create(dilate=False):
    return create_x(dilate, resnet.resnet_v1_50, "http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz")
