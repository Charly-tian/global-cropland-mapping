from keras import backend as K
from keras import layers
from keras.models import Model
import tensorflow as tf
from ..backbones.utils import _sep_conv_bn

# channel pool but keep dims: K.mean(inputs, axis=-1, keep_dims=True)
# global pool but keep dims: K.mean(inputs, axis=[1, 2], keepdims=True)
# resize image: tf.image.resize_bilinear(inputs, self.target_size, align_corners=True)


# class ResizeImageLayer(layers.Layer):
#     def __init__(self, target_size, **kwargs):
#         super(ResizeImageLayer, self).__init__(**kwargs)
#         self.target_size = target_size
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], self.target_size[0], self.target_size[1], input_shape[-1]
#
#     def _resize_function(self, inputs):
#         return tf.image.resize_bilinear(inputs, self.target_size, align_corners=True)
#
#     def call(self, inputs, **kwargs):
#         return self._resize_function(inputs=inputs)
#
#     def get_config(self):
#         config = {'target_size': self.target_size}
#         base_config = super(ResizeImageLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# class ChannelReduceLayer(layers.Layer):
#     def __init__(self, mode='mean'):
#         super(ChannelReduceLayer, self).__init__()
#         self.reduce_mode = mode
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[1], input_shape[2], 1
#
#     def _channel_reduce(self, inputs):
#         if self.reduce_mode == 'max':
#             return tf.reduce_max(inputs, axis=-1, keep_dims=True)
#         elif self.reduce_mode == 'avg':
#             return tf.reduce_mean(inputs, axis=-1, keep_dims=True)
#
#     def call(self, inputs, **kwargs):
#         return self._channel_reduce(inputs)
#
#     def get_config(self):
#         config = {'reduce_mode': self.reduce_mode}
#         base_config = super(ChannelReduceLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# class SpatialReduceLayer(layers.Layer):
#     def __init__(self, mode='mean'):
#         super(SpatialReduceLayer, self).__init__()
#         self.reduce_mode = mode
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], 1, 1, input_shape[-1]
#
#     def _channel_reduce(self, inputs):
#         if self.reduce_mode == 'max':
#             return tf.reduce_max(inputs, axis=[1, 2], keepdims=True)
#         elif self.reduce_mode == 'avg':
#             return tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
#
#     def call(self, inputs, **kwargs):
#         return self._channel_reduce(inputs)
#
#     def get_config(self):
#         config = {'reduce_mode': self.reduce_mode}
#         base_config = super(SpatialReduceLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#
# class ExtractSubChannelLayer(layers.Layer):
#     def __init__(self, channel_idxs):
#         super(ExtractSubChannelLayer, self).__init__()
#         self.channel_idxs = channel_idxs
#
#     def compute_output_shape(self, input_shape):
#         return input_shape[0], input_shape[1], input_shape[2], len(self.channel_idxs)
#
#     def _exact_subchannels(self, inputs):
#         return inputs[:, :, :, self.channel_idxs]
#
#     def call(self, inputs, **kwargs):
#         return self._exact_subchannels(inputs)
#
#     def get_config(self):
#         config = {'channel_idxs': self.channel_idxs}
#         base_config = super(ExtractSubChannelLayer, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))


def atrous_spatial_pyramid_pooling(inputs, atrous_rates=(6, 12, 24)):
    b4 = layers.GlobalAveragePooling2D()(inputs)
    b4 = layers.Lambda(lambda xx: K.expand_dims(xx, 1))(b4)
    b4 = layers.Lambda(lambda xx: K.expand_dims(xx, 1))(b4)

    b4 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(b4)
    b4 = layers.BatchNormalization(epsilon=1e-5)(b4)
    b4 = layers.Activation('relu')(b4)
    b4 = layers.UpSampling2D((int(inputs.shape[1]), int(inputs.shape[2])), interpolation='bilinear')(b4)
    # b4 = ResizeImageLayer(target_size=(int(inputs.shape[1]), int(inputs.shape[2])))(b4)

    b0 = layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(inputs)
    b0 = layers.BatchNormalization(epsilon=1e-5)(b0)
    b0 = layers.Activation('relu')(b0)

    bs = []
    for i in range(len(atrous_rates)):
        bi = _sep_conv_bn(inputs, 256, 'aspp{}'.format(i + 1), rate=atrous_rates[i],
                          depth_activation=True, epsilon=1e-5)
        bs.append(bi)

    # concatenate ASPP branches & project
    x = layers.Concatenate()([b4, b0] + bs)

    return x


def interp_block(inputs, level, feature_map_shape):
    ksize = (int(round(float(feature_map_shape[0]) / float(level))),
             int(round(float(feature_map_shape[1]) / float(level))))
    stride_size = ksize

    x = layers.MaxPooling2D(pool_size=ksize, strides=stride_size)(inputs)
    x = layers.Conv2D(64, (1, 1), activation=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Lambda(lambda xx: tf.image.resize_bilinear(xx, feature_map_shape, align_corners=True))(x)
    return x


def pyramid_pooling_module(inputs, feature_map_size):
    interp_block1 = interp_block(inputs, 1, feature_map_size)
    interp_block2 = interp_block(inputs, 2, feature_map_size)
    interp_block3 = interp_block(inputs, 3, feature_map_size)
    interp_block6 = interp_block(inputs, 6, feature_map_size)

    res = layers.concatenate([inputs, interp_block6, interp_block3, interp_block2, interp_block1])
    return res


def channel_attention(x, hidden_size=16):
    # 通道均值和最大值，加起来之后求sigmoid

    def shared_mlp(input_shape):
        x = layers.Input(shape=input_shape)
        in_planes = int(x.shape[-1])
        _x = layers.Conv2D(hidden_size, kernel_size=(1, 1), use_bias=False)(x)
        _x = layers.Activation('relu')(_x)
        _x = layers.Conv2D(in_planes, kernel_size=(1, 1), use_bias=False)(_x)
        return Model(x, _x)

    avg_out = layers.Lambda(lambda xx: K.mean(x, axis=[1, 2], keepdims=True))(x)
    max_out = layers.Lambda(lambda xx: K.max(x, axis=[1, 2], keepdims=True))(x)
    smlp = shared_mlp(input_shape=(1, 1, int(avg_out.shape[-1])))  # parameter sharing
    avg_out = smlp(avg_out)
    max_out = smlp(max_out)
    sum_out = layers.add([avg_out, max_out])
    o = layers.Activation('sigmoid')(sum_out)
    return o


def spatial_attention(x, conv_size=7):
    # 空间像素的均值+最大值得到的特征图，拼接之后，降维到1个波段，再接sigmoid，得到逐像素的权重
    max_out = layers.Lambda(lambda xx: K.max(x, axis=-1, keepdims=True))(x)
    avg_out = layers.Lambda(lambda xx: K.mean(x, axis=-1, keepdims=True))(x)
    cat_out = layers.concatenate([max_out, avg_out], axis=-1)
    if conv_size == 7:
        x = layers.Conv2D(1, kernel_size=(7, 7), padding='same', use_bias=False)(cat_out)
    else:
        x = layers.Conv2D(1, kernel_size=(3, 3), padding='same', use_bias=False)(cat_out)
    o = layers.Activation('sigmoid')(x)
    return o


def convolutional_block_attention_module(inputs, hidden_size=16, conv_size=7):
    # CBAM：通道注意力+像素注意力
    ca = channel_attention(inputs, hidden_size)
    x = layers.Multiply(name='channel_attention_x')([inputs, ca])
    sa = spatial_attention(x, conv_size)
    x = layers.Multiply(name='spatial_attention_x')([x, sa])
    return x


def my_feature_pyramid_attention(inputs):
    input_shape = K.int_shape(inputs)
    # global pool branch
    pool_branch = layers.AveragePooling2D(pool_size=input_shape[1])(inputs)
    pool_branch = layers.Conv2D(input_shape[-1], kernel_size=1, use_bias=False, kernel_initializer='he_normal')(pool_branch)
    pool_branch = ResizeImageLayer(target_size=(input_shape[1], input_shape[2]))(pool_branch)

    # direct branch
    direct_branch = layers.Conv2D(input_shape[-1], kernel_size=1, use_bias=False, kernel_initializer='he_normal')(inputs)
    direct_branch = layers.BatchNormalization()(direct_branch)

    # downsample branch
    conv_3_1 = layers.Conv2D(input_shape[-1]//4, kernel_size=3, padding='same', strides=2, use_bias=False, kernel_initializer='he_normal')(inputs)
    conv_3_1 = layers.BatchNormalization()(conv_3_1)
    conv_3_1 = layers.Activation('relu')(conv_3_1)

    conv_3_2 = layers.Conv2D(input_shape[-1], kernel_size=3, padding='same', strides=1, use_bias=False, kernel_initializer='he_normal')(conv_3_1)
    conv_3_2 = layers.BatchNormalization()(conv_3_2)
    conv_3_2 = layers.Activation('relu')(conv_3_2)

    upsampled_8 = layers.Conv2DTranspose(input_shape[-1], kernel_size=4, strides=(2, 2), padding='same', use_bias=False, kernel_initializer='he_normal')(conv_3_2)
    upsampled_8 = layers.BatchNormalization()(upsampled_8)

    x = layers.Multiply()([upsampled_8, direct_branch])
    x = layers.Add()([x, pool_branch])

    return x


def my_global_attention_upsample(x_high, x_low):
    low_shape = K.int_shape(x_low)

    x_low_mask = layers.Conv2D(low_shape[-1], kernel_size=3, padding='same', use_bias=False)(x_low)
    x_low_mask = layers.BatchNormalization()(x_low_mask)

    x_high_gp = SpatialReduceLayer(mode='avg')(x_high)
    x_high_gp = layers.Conv2D(low_shape[-1], kernel_size=1, use_bias=False)(x_high_gp)
    x_high_gp = layers.BatchNormalization()(x_high_gp)
    x_high_gp = layers.Activation('relu')(x_high_gp)
    x_high_gp = ResizeImageLayer(target_size=(low_shape[1], low_shape[2]))(x_high_gp)

    x_att = layers.Multiply()([x_high_gp, x_low_mask])

    x_h = layers.Conv2D(low_shape[-1], kernel_size=1, use_bias=False)(x_high)
    x_h = layers.BatchNormalization()(x_h)
    x = layers.Add()([x_h, x_att])
    x = layers.Activation('relu')(x)

    return x



