import tensorflow as tf
from keras import layers
from .utils import atrous_spatial_pyramid_pooling


def deeplabv3_decoder(features, n_class, atrous_rates, input_shape):
    [f1, f2, f3, f4, f5] = features
    # ASPP
    x = atrous_spatial_pyramid_pooling(f4, atrous_rates)
    x = layers.Lambda(lambda xx:
                      tf.cast(
                          tf.image.resize_bilinear(xx, input_shape[0:2], align_corners=True),
                          dtype=tf.float32))(x)
    x = layers.Conv2D(n_class, (1, 1), padding='same')(x)
    return x
