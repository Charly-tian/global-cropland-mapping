from keras import layers
from keras import models
from ..backbones import build_encoder
from .utils import pyramid_pooling_module


def pspnet(input_shape=(128, 128, 3), n_class=1, one_hot=False,
           backbone_name='resnet50', backbone_weights=None,
           output_stride=16):
    img_input, features = build_encoder(input_shape, None, backbone_name, backbone_weights, output_stride)
    [f1, f2, f3, f4, f5] = features
    psp = pyramid_pooling_module(f4, (input_shape[0] // output_stride, input_shape[1] // output_stride))

    x = layers.Conv2D(512, (3, 3), strides=(1, 1), padding="same", use_bias=False)(psp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # x = layers.Dropout(0.1)(x)

    x = layers.Conv2D(n_class, (1, 1), use_bias=False)(x)
    x = layers.UpSampling2D((8, 8), interpolation='bilinear')(x)

    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)

    return models.Model(img_input, x)
