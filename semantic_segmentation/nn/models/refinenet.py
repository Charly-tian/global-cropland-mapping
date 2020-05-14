from keras import layers
from keras import models
from keras import regularizers
import tensorflow as tf

from ..backbones import build_encoder
from .. import WEIGHT_DECAY, KERNEL_INITIALIZER, DROPOUT


def residual_conv_unit(inputs, n_filters=256):
    x = layers.Activation("relu")(inputs)
    x = layers.Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
               kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
               kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x)
    # x = layers.BatchNormalization()(x)
    x = layers.Add()([x, inputs])

    return x


def multi_resolution_fusion(inputs, n_filters, target_shape):
    results = []
    for x in inputs:
        x = layers.Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
               kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x)
        x = layers.BatchNormalization()(x)
        # x = layers.Activation("relu")(x)
        x = layers.Lambda(lambda xx: tf.image.resize_bilinear(xx, target_shape, align_corners=True))(x)
        results.append(x)
    if len(results) > 1:
        return layers.Add()(results)
    return results[0]


def chained_residual_pooling(inputs,
                             pool_size=(5, 5),
                             n_filters=256):
    x_relu = layers.Activation("relu")(inputs)

    x = layers.MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding="same")(x_relu)
    x = layers.Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
               kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x)
    x = layers.BatchNormalization()(x)
    x_sum1 = layers.Add()([x_relu, x])

    x = layers.MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding="same")(x)
    x = layers.Conv2D(n_filters, (3, 3), padding="same", activation=None, use_bias=False,
               kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x)
    x = layers.BatchNormalization()(x)
    x_sum2 = layers.Add()([x_sum1, x])

    return x_sum2


def refine_block(inputs, rcu_filters, n_filters, target_shape):
    results = []
    for x, rcu_filter in zip(inputs, rcu_filters):
        x = residual_conv_unit(x, rcu_filter)
        x = residual_conv_unit(x, rcu_filter)
        results.append(x)
    x = multi_resolution_fusion(results, n_filters, target_shape)
    x = chained_residual_pooling(x, (5, 5), n_filters)
    x = residual_conv_unit(x, n_filters)
    return x


def bn_relu_deconv(inputs, n_filters, kernel_size, scale=2):
    x = layers.BatchNormalization()(inputs)
    x = layers.Activation("relu")(x)
    x = layers.Conv2DTranspose(n_filters, kernel_size, padding="same", activation=None, use_bias=False,
                        strides=(scale, scale), kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                        kernel_initializer=KERNEL_INITIALIZER)(x)
    return x


def refinenet_4cascaded(input_shape,
              n_class,
              backbone_name,
              backbone_weights=None,
              output_stride=16,
              one_hot=False,
              init_filters=256,
              upscaling_method="bilinear"):
    # 74.93 M  ==> (512->256) 62.55 M ==> (resnet50->mobilenetv2) 37.41 M
    img_input, features = build_encoder(input_shape, None, backbone_name, backbone_weights, output_stride)
    for i in range(len(features)):
        if i == 4:
            features[i] = layers.Conv2D(init_filters, 3, padding='same')(features[i])
        else:
            features[i] = layers.Conv2D(init_filters, 3, padding='same')(features[i])
        features[i] = layers.BatchNormalization()(features[i])
    [f1, f2, f3, f4, f5] = features
    f5 = refine_block([f5], [init_filters], init_filters, target_shape=(input_shape[0] // 32, input_shape[1] // 32))
    f4 = refine_block([f4, f5], [init_filters, init_filters], init_filters, target_shape=(input_shape[0] // 16, input_shape[1] // 16))
    f3 = refine_block([f3, f4], [init_filters, init_filters], init_filters, target_shape=(input_shape[0] // 8, input_shape[1] // 8))
    f2 = refine_block([f2, f3], [init_filters, init_filters], init_filters, target_shape=(input_shape[0] // 4, input_shape[1] // 4))

    if upscaling_method == "conv":
        x = bn_relu_deconv(f2, 256, 3, scale=2)
    else:
        x = layers.Lambda(lambda xx: tf.image.resize_bilinear(xx, (input_shape[0], input_shape[1]), align_corners=True))(f2)

    x = layers.Conv2D(n_class, (1, 1), activation=None,
                    kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x)
    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)

    return models.Model(img_input, x)


def refinenet_2cascaded(input_shape,
              n_class,
              backbone_name,
              backbone_weights=None,
              output_stride=16,
              one_hot=False,
              init_filters=256,
              upscaling_method="bilinear"):
    # 64.31 M ==>(512->256) 51.93 M ==> (resnet50->mobilenetv2) 26.79 M
    img_input, features = build_encoder(input_shape, None, backbone_name, backbone_weights, output_stride)
    for i in range(len(features)):
        if i == 4:
            features[i] = layers.Conv2D(init_filters, 3, padding='same')(features[i])
        else:
            features[i] = layers.Conv2D(init_filters, 3, padding='same')(features[i])
        features[i] = layers.BatchNormalization()(features[i])
    [f1, f2, f3, f4, f5] = features
    f4 = refine_block([f5, f4], [init_filters, init_filters], init_filters, target_shape=(input_shape[0] // 16, input_shape[1] // 16))
    f2 = refine_block([f2, f3, f4], [init_filters, init_filters, init_filters], init_filters, target_shape=(input_shape[0] // 4, input_shape[1] // 4))

    if upscaling_method == "conv":
        x = bn_relu_deconv(f2, 256, 3, scale=2)
    else:
        x = layers.Lambda(lambda xx: tf.image.resize_bilinear(xx, (input_shape[0], input_shape[1]), align_corners=True))(f2)

    x = layers.Conv2D(n_class, (1, 1), activation=None,
                    kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x)
    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)

    return models.Model(img_input, x)


def refinenet_1cascaded(input_shape,
              n_class,
              backbone_name,
              backbone_weights=None,
              output_stride=16,
              one_hot=False,
              init_filters=256,
              upscaling_method="bilinear"):
    # 59.00 M ==> (512->256) 46.61 M ==> (resnet50->mobilenetv2) 21.48 M
    img_input, features = build_encoder(input_shape, None, backbone_name, backbone_weights, output_stride)
    for i in range(len(features)):
        if i == 4:
            features[i] = layers.Conv2D(init_filters, 3, padding='same')(features[i])
        else:
            features[i] = layers.Conv2D(init_filters, 3, padding='same')(features[i])
        features[i] = layers.BatchNormalization()(features[i])
    [f1, f2, f3, f4, f5] = features
    f2 = refine_block([f2, f3, f4, f5], [init_filters, init_filters, init_filters, init_filters],
                      init_filters, target_shape=(input_shape[0] // 4, input_shape[1] // 4))

    if upscaling_method == "conv":
        x = bn_relu_deconv(f2, 256, 3, scale=2)
    else:
        x = layers.Lambda(lambda xx: tf.image.resize_bilinear(xx, (input_shape[0], input_shape[1]), align_corners=True))(f2)

    x = layers.Conv2D(n_class, (1, 1), activation=None,
                    kernel_regularizer=regularizers.l2(WEIGHT_DECAY), kernel_initializer=KERNEL_INITIALIZER)(x)
    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)

    return models.Model(img_input, x)