from keras import layers
from keras import models
from keras import regularizers
import tensorflow as tf

from ..backbones import build_encoder
from .. import WEIGHT_DECAY, KERNEL_INITIALIZER, DROPOUT


def sri_net(input_shape,
            n_class,
            backbone_name="resnet_v2_101",
            backbone_weights=None,
            output_stride=16,
            one_hot=False):
    img_input, features = build_encoder(input_shape, None, backbone_name, backbone_weights, output_stride)
    [f1, f2, f3, f4, f5] = features

    # 32->64
    net = spatial_residual_inception_v2(f4, 192)
    net = layers.Lambda(lambda xx: tf.image.resize_bilinear(xx, (input_shape[0] // 8, input_shape[1] // 8), align_corners=True))(net)

    # 64->128
    p3 = layers.Conv2D(int(net.shape[-1]//4), (1, 1), use_bias=False, activation=None,
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(f3)
    p3 = layers.BatchNormalization()(p3)
    p3 = layers.Activation("relu")(p3)
    net = layers.Concatenate()([net, p3])
    net = spatial_residual_inception_v2(net, 192)
    net = layers.Lambda(lambda xx: tf.image.resize_bilinear(xx, (input_shape[0] // 4, input_shape[1] // 4), align_corners=True))(net)

    # 128->512
    p2 = layers.Conv2D(int(net.shape[-1]//4), (1, 1), use_bias=False, activation=None,
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(f2)
    p2 = layers.BatchNormalization()(p2)
    p2 = layers.Activation("relu")(p2)
    net = layers.Concatenate()([net, p2])
    net = spatial_residual_inception_v2(net, 192)
    net = layers.Lambda(lambda xx: tf.image.resize_bilinear(xx, (input_shape[0], input_shape[1]), align_corners=True))(net)

    net = layers.Conv2D(256, (3, 3), use_bias=False, activation=None, padding="same",
                 kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(net)
    net = layers.BatchNormalization()(net)
    net = layers.Activation("relu")(net)

    x = layers.Conv2D(n_class, (1, 1), activation=None,
                    kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(net)
    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)

    return models.Model(img_input, x)


def spatial_residual_inception_v2(inputs, base_filters=192):
    x_short = inputs
    x_short = layers.Conv2D(base_filters + 64, (1, 1), use_bias=False, activation=None,
                     kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x_short)
    x_short = layers.BatchNormalization()(x_short)
    x_short = layers.Activation('relu')(x_short)

    # 1x1
    x_conv1x1 = layers.Conv2D(base_filters + 64, (1, 1), use_bias=False, activation=None,
                       kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x_short)
    x_conv1x1 = layers.BatchNormalization()(x_conv1x1)

    rates = [1, 2, 5, 7]
    xs_conv3x3 = []
    for rate in rates:
        x = layers.Conv2D(base_filters + 32, (1, 3), padding="same", use_bias=False, activation=None, dilation_rate=1,
                         kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x_short)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(base_filters + 64, (3, 1), padding="same", use_bias=False, activation=None, dilation_rate=rate,
                         kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x)
        x = layers.BatchNormalization()(x)
        xs_conv3x3.append(x)

    x_conv = layers.Concatenate()([x_conv1x1] + xs_conv3x3)
    x_conv = layers.Activation('relu')(x_conv)
    x_conv = layers.Conv2D(base_filters + 64, (1, 1), use_bias=False, activation=None,
                    kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x_conv)
    x_conv = layers.BatchNormalization()(x_conv)

    x = layers.Add()([x_short, x_conv])
    return layers.Activation("relu")(x)
