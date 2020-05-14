from keras import layers
from keras.models import Model
from keras import backend as K
from ..backbones import build_encoder
from ..backbones.utils import _sep_conv_bn
from .utils import atrous_spatial_pyramid_pooling


# def deeplabv3p_decoder(features, n_class, atrous_rates, input_shape):
#     [f1, f2, f3, f4, f5] = features
#
#     # ASPP
#     x = atrous_spatial_pyramid_pooling(f4, atrous_rates)
#
#     # multi scale feature fusion
#     x = layers.Conv2D(256, (1, 1), padding='same', use_bias=False)(x)
#     x = layers.BatchNormalization()(x)
#     x = layers.Activation('relu')(x)
#     x = layers.Dropout(0.1)(x)
#     # x = ResizeImageLayer(target_size=(int(f2.shape[1]), int(f2.shape[2])))(x)
#     x = layers.UpSampling2D((4, 4), interpolation='bilinear')(x)
#
#     dec_skip1 = layers.Conv2D(48, (1, 1), padding='same', use_bias=False)(f2)
#     dec_skip1 = layers.BatchNormalization()(dec_skip1)
#     dec_skip1 = layers.Activation('relu')(dec_skip1)
#
#     # multi level feature fusion
#     x = layers.Concatenate()([x, dec_skip1])
#     x = _sep_conv_bn(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
#     x = _sep_conv_bn(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)
#
#     x = layers.Conv2D(n_class, (1, 1), padding='same', name='logit')(x)
#     # x = ResizeImageLayer(target_size=(input_shape[0], input_shape[1]))(x)
#     x = layers.UpSampling2D((4, 4), interpolation='bilinear')(x)
#     return x


def deeplabv3p(input_shape=(128, 128, 4), n_class=1, one_hot=False,
               backbone_name='mobilenetv2', backbone_weights=None,
               output_stride=16, atrous_rates=(2, 3, 5)):
    def batch_relu(inputs):
        _x = layers.BatchNormalization()(inputs)
        _x = layers.Activation('relu')(_x)
        return _x

    img_input, features = build_encoder(input_shape, None, backbone_name, backbone_weights, output_stride)
    [f1, f2, f3, f4, f5] = features

    x_h = atrous_spatial_pyramid_pooling(f4, atrous_rates)
    x_h = layers.Conv2D(256, kernel_size=1, use_bias=False)(x_h)
    x_h = batch_relu(x_h)
    x_h = layers.Dropout(0.2)(x_h)
    x_h = layers.UpSampling2D((4, 4), interpolation='bilinear')(x_h)

    x_l = layers.Conv2D(48, kernel_size=1, use_bias=False)(f2)
    x_l = batch_relu(x_l)

    # feature_extractor = Model(img_input, [x_h, x_l])

    x = layers.Concatenate()([x_h, x_l])
    x = _sep_conv_bn(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
    x = _sep_conv_bn(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)

    x = layers.Conv2D(n_class, kernel_size=1, use_bias=False)(x)
    x = layers.UpSampling2D((4, 4), interpolation='bilinear')(x)

    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)

    return Model(img_input, x)


def deeplabv3p_lstm(input_shape=(128, 128, 16), n_class=1, one_hot=False,
               backbone_name='mobilenetv2', backbone_weights=None,
               output_stride=16, atrous_rates=(2, 3, 5), input_split=4):
    def batch_relu(inputs):
        _x = layers.BatchNormalization()(inputs)
        _x = layers.Activation('relu')(_x)
        return _x

    assert input_shape[-1] % input_split == 0
    img_input = layers.Input(shape=input_shape)
    img_inputs = []
    channels_per_split = input_shape[-1] // input_split
    for i in range(input_split):
        _x = layers.Lambda(lambda xx: xx[:, :, :, i*channels_per_split: (i+1)*channels_per_split])(img_input)
        img_inputs.append(_x)

    sub_img_input, features = build_encoder((input_shape[0], input_shape[1], input_shape[-1] // input_split),
                                            None, backbone_name, backbone_weights, output_stride)
    [f1, f2, f3, f4, f5] = features

    feature_extractor = Model(sub_img_input, [f2, f4])

    F2s = []
    F4s = []
    for i in range(input_split):
        F24 = feature_extractor(img_inputs[i])
        F2s.append(F24[0])
        F4s.append(F24[1])
    F2s = layers.Lambda(lambda xx: K.stack(xx, axis=1))(F2s)
    F4s = layers.Lambda(lambda xx: K.stack(xx, axis=1))(F4s)
    F2 = layers.ConvLSTM2D(filters=48, kernel_size=3, padding='same', activation='relu',
                           return_sequences=False)(F2s)
    F4 = layers.ConvLSTM2D(filters=48, kernel_size=3, padding='same', activation='relu',
                           return_sequences=False)(F4s)

    # 多层级特征融合
    x_h = atrous_spatial_pyramid_pooling(F4, atrous_rates)
    x_h = layers.Conv2D(256, kernel_size=1, use_bias=False)(x_h)
    x_h = batch_relu(x_h)
    x_h = layers.Dropout(0.2)(x_h)
    x_h = layers.UpSampling2D((4, 4), interpolation='bilinear')(x_h)

    x_l = layers.Conv2D(48, kernel_size=1, use_bias=False)(F2)
    x_l = batch_relu(x_l)

    x = layers.Concatenate()([x_h, x_l])
    x = _sep_conv_bn(x, 256, 'decoder_conv0', depth_activation=True, epsilon=1e-5)
    x = _sep_conv_bn(x, 256, 'decoder_conv1', depth_activation=True, epsilon=1e-5)

    x = layers.Conv2D(n_class, kernel_size=1, use_bias=False)(x)
    x = layers.UpSampling2D((4, 4), interpolation='bilinear')(x)

    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)

    return Model(img_input, x)





# def Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(512, 512, 3), classes=21, backbone='mobilenetv2',
#               OS=16, alpha=1., activation=None):
#     """ Instantiates the Deeplabv3+ architecture
#     Optionally loads weights pre-trained
#     on PASCAL VOC or Cityscapes. This model is available for TensorFlow only.
#     # Arguments
#         weights: one of 'pascal_voc' (pre-trained on pascal voc),
#             'cityscapes' (pre-trained on cityscape) or None (random initialization)
#         input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
#             to use as image input for the model.
#         input_shape: shape of input image. format HxWxC
#             PASCAL VOC model was trained on (512,512,3) images. None is allowed as shape/width
#         classes: number of desired classes. PASCAL VOC has 21 classes, Cityscapes has 19 classes.
#             If number of classes not aligned with the weights used, last layer is initialized randomly
#         backbone: backbone to use. one of {'xception','mobilenetv2'}
#         activation: optional activation to add to the top of the network.
#             One of 'softmax', 'sigmoid' or None
#         OS: determines input_shape/feature_extractor_output ratio. One of {8,16}.
#             Used only for xception backbone.
#         alpha: controls the width of the MobileNetV2 network. This is known as the
#             width multiplier in the MobileNetV2 paper.
#                 - If `alpha` < 1.0, proportionally decreases the number
#                     of filters in each layer.
#                 - If `alpha` > 1.0, proportionally increases the number
#                     of filters in each layer.
#                 - If `alpha` = 1, default number of filters from the paper
#                     are used at each layer.
#             Used only for mobilenetv2 backbone. Pretrained is only available for alpha=1.
#     # Returns
#         A Keras model instance.
#     # Raises
#         RuntimeError: If attempting to run this model with a
#             backend that does not support separable convolutions.
#         ValueError: in case of invalid argument for `weights` or `backbone`
#     """
#
#     if not (weights in {'pascal_voc', 'cityscapes', None}):
#         raise ValueError('The `weights` argument should be either '
#                          '`None` (random initialization), `pascal_voc`, or `cityscapes` '
#                          '(pre-trained on PASCAL VOC)')
#
#     if not (backbone in {'xception', 'mobilenetv2'}):
#         raise ValueError('The `backbone` argument should be either '
#                          '`xception`  or `mobilenetv2` ')
#
#     if input_tensor is None:
#         img_input = Input(shape=input_shape)
#     else:
#         img_input = input_tensor
#
#     if backbone == 'xception':
#         if OS == 8:
#             entry_block3_stride = 1
#             middle_block_rate = 2  # ! Not mentioned in paper, but required
#             exit_block_rates = (2, 4)
#             atrous_rates = (12, 24, 36)
#         else:
#             entry_block3_stride = 2
#             middle_block_rate = 1
#             exit_block_rates = (1, 2)
#             atrous_rates = (6, 12, 18)
#
#         x = Conv2D(32, (3, 3), strides=(2, 2),
#                    name='entry_flow_conv1_1', use_bias=False, padding='same')(img_input)
#         x = BatchNormalization(name='entry_flow_conv1_1_BN')(x)
#         x = Activation('relu')(x)
#
#         x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
#         x = BatchNormalization(name='entry_flow_conv1_2_BN')(x)
#         x = Activation('relu')(x)
#
#         x = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
#                             skip_connection_type='conv', stride=2,
#                             depth_activation=False)
#         x, skip1 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
#                                    skip_connection_type='conv', stride=2,
#                                    depth_activation=False, return_skip=True)
#
#         x = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
#                             skip_connection_type='conv', stride=entry_block3_stride,
#                             depth_activation=False)
#         for i in range(16):
#             x = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
#                                 skip_connection_type='sum', stride=1, rate=middle_block_rate,
#                                 depth_activation=False)
#
#         x = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
#                             skip_connection_type='conv', stride=1, rate=exit_block_rates[0],
#                             depth_activation=False)
#         x = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
#                             skip_connection_type='none', stride=1, rate=exit_block_rates[1],
#                             depth_activation=True)
#
#     else:
#         OS = 8
#         first_block_filters = _make_divisible(32 * alpha, 8)
#         x = Conv2D(first_block_filters,
#                    kernel_size=3,
#                    strides=(2, 2), padding='same',
#                    use_bias=False, name='Conv')(img_input)
#         x = BatchNormalization(
#             epsilon=1e-3, momentum=0.999, name='Conv_BN')(x)
#         x = Activation(relu6, name='Conv_Relu6')(x)
#
#         x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
#                                 expansion=1, block_id=0, skip_connection=False)
#
#         x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
#                                 expansion=6, block_id=1, skip_connection=False)
#         x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
#                                 expansion=6, block_id=2, skip_connection=True)
#
#         x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
#                                 expansion=6, block_id=3, skip_connection=False)
#         x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
#                                 expansion=6, block_id=4, skip_connection=True)
#         x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
#                                 expansion=6, block_id=5, skip_connection=True)
#
#         # stride in block 6 changed from 2 -> 1, so we need to use rate = 2
#         x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,  # 1!
#                                 expansion=6, block_id=6, skip_connection=False)
#         x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=7, skip_connection=True)
#         x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=8, skip_connection=True)
#         x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=9, skip_connection=True)
#
#         x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=10, skip_connection=False)
#         x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=11, skip_connection=True)
#         x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1, rate=2,
#                                 expansion=6, block_id=12, skip_connection=True)
#
#         x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=2,  # 1!
#                                 expansion=6, block_id=13, skip_connection=False)
#         x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
#                                 expansion=6, block_id=14, skip_connection=True)
#         x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1, rate=4,
#                                 expansion=6, block_id=15, skip_connection=True)
#
#         x = _inverted_res_block(x, filters=320, alpha=alpha, stride=1, rate=4,
#                                 expansion=6, block_id=16, skip_connection=False)
#
#     # end of feature extractor
#
#     # branching for Atrous Spatial Pyramid Pooling
#
#     # Image Feature branch
#     shape_before = tf.shape(x)
#     b4 = GlobalAveragePooling2D()(x)
#     # from (b_size, channels)->(b_size, 1, 1, channels)
#     b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
#     b4 = Lambda(lambda x: K.expand_dims(x, 1))(b4)
#     b4 = Conv2D(256, (1, 1), padding='same',
#                 use_bias=False, name='image_pooling')(b4)
#     b4 = BatchNormalization(name='image_pooling_BN', epsilon=1e-5)(b4)
#     b4 = Activation('relu')(b4)
#     # upsample. have to use compat because of the option align_corners
#     size_before = tf.keras.backend.int_shape(x)
#     b4 = Lambda(lambda x: tf.compat.v1.image.resize(x, size_before[1:3],
#                                                     method='bilinear', align_corners=True))(b4)
#     # simple 1x1
#     b0 = Conv2D(256, (1, 1), padding='same', use_bias=False, name='aspp0')(x)
#     b0 = BatchNormalization(name='aspp0_BN', epsilon=1e-5)(b0)
#     b0 = Activation('relu', name='aspp0_activation')(b0)
#
#     # there are only 2 branches in mobilenetV2. not sure why
#     if backbone == 'xception':
#         # rate = 6 (12)
#         b1 = _sep_conv_bn(x, 256, 'aspp1',
#                           rate=atrous_rates[0], depth_activation=True, epsilon=1e-5)
#         # rate = 12 (24)
#         b2 = _sep_conv_bn(x, 256, 'aspp2',
#                           rate=atrous_rates[1], depth_activation=True, epsilon=1e-5)
#         # rate = 18 (36)
#         b3 = _sep_conv_bn(x, 256, 'aspp3',
#                           rate=atrous_rates[2], depth_activation=True, epsilon=1e-5)
#
#         # concatenate ASPP branches & project
#         x = Concatenate()([b4, b0, b1, b2, b3])
#     else:
#         x = Concatenate()([b4, b0])
#
#     x = Conv2D(256, (1, 1), padding='same',
#                use_bias=False, name='concat_projection')(x)
#     x = BatchNormalization(name='concat_projection_BN', epsilon=1e-5)(x)
#     x = Activation('relu')(x)
#     x = Dropout(0.1)(x)
#     # DeepLab v.3+ decoder
#
#     if backbone == 'xception':
#         # Feature projection
#         # x4 (x2) block
#         size_before2 = tf.keras.backend.int_shape(x)
#         x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
#                                                         skip1.shape[1:3],
#                                                         method='bilinear', align_corners=True))(x)
#
#         dec_skip1 = Conv2D(48, (1, 1), padding='same',
#                            use_bias=False, name='feature_projection0')(skip1)
#         dec_skip1 = BatchNormalization(
#             name='feature_projection0_BN', epsilon=1e-5)(dec_skip1)
#         dec_skip1 = Activation('relu')(dec_skip1)
#         x = Concatenate()([x, dec_skip1])
#         x = _sep_conv_bn(x, 256, 'decoder_conv0',
#                          depth_activation=True, epsilon=1e-5)
#         x = _sep_conv_bn(x, 256, 'decoder_conv1',
#                          depth_activation=True, epsilon=1e-5)
#
#     # you can use it with arbitary number of classes
#     if (weights == 'pascal_voc' and classes == 21) or (weights == 'cityscapes' and classes == 19):
#         last_layer_name = 'logits_semantic'
#     else:
#         last_layer_name = 'custom_logits_semantic'
#
#     x = Conv2D(classes, (1, 1), padding='same', name=last_layer_name)(x)
#     size_before3 = tf.keras.backend.int_shape(img_input)
#     x = Lambda(lambda xx: tf.compat.v1.image.resize(xx,
#                                                     size_before3[1:3],
#                                                     method='bilinear', align_corners=True))(x)
#
#     # Ensure that the model takes into account
#     # any potential predecessors of `input_tensor`.
#     # if input_tensor is not None:
#     #     inputs = get_source_inputs(input_tensor)
#     # else:
#     inputs = img_input
#
#     if activation in {'softmax', 'sigmoid'}:
#         x = tf.keras.layers.Activation(activation)(x)
#
#     model = Model(inputs, x, name='deeplabv3plus')
#
#     # load weights
#     # if weights == 'pascal_voc':
#     #     if backbone == 'xception':
#     #         weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
#     #                                 WEIGHTS_PATH_X,
#     #                                 cache_subdir='models')
#     #     else:
#     #         weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels.h5',
#     #                                 WEIGHTS_PATH_MOBILE,
#     #                                 cache_subdir='models')
#     #     model.load_weights(weights_path, by_name=True)
#     # elif weights == 'cityscapes':
#     #     if backbone == 'xception':
#     #         weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels_cityscapes.h5',
#     #                                 WEIGHTS_PATH_X_CS,
#     #                                 cache_subdir='models')
#     #     else:
#     #         weights_path = get_file('deeplabv3_mobilenetv2_tf_dim_ordering_tf_kernels_cityscapes.h5',
#     #                                 WEIGHTS_PATH_MOBILE_CS,
#     #                                 cache_subdir='models')
#     #     model.load_weights(weights_path, by_name=True)
#     return model
