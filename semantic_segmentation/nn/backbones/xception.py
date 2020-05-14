import os
from keras import layers
from keras.models import Model
from .utils import _sep_conv_bn, _conv2d_same
from ...utils import log

pretrained_url = 'https://github.com/fchollet/deep-learning-models/' \
                 'releases/download/v0.4/' \
                 'xception_weights_tf_dim_ordering_tf_kernels_notop.h5'


def _xception_block(inputs, depth_list, prefix, skip_connection_type, stride,
                    rate=1, depth_activation=False):
    """ Basic building block of modified Xception network
        Args:
            inputs: input tensor
            depth_list: number of filters in each SepConv layer. len(depth_list) == 3
            prefix: prefix before name
            skip_connection_type: one of {'conv','sum','none'}
            stride: stride at last depthwise conv
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & pointwise convs
            """
    residual = inputs
    for i in range(3):
        residual = _sep_conv_bn(residual,
                                depth_list[i],
                                prefix + '_separable_conv{}'.format(i + 1),
                                stride=stride if i == 2 else 1,
                                rate=rate,
                                depth_activation=depth_activation)
        if i == 1:
            skip = residual
    if skip_connection_type == 'conv':
        shortcut = _conv2d_same(inputs, depth_list[-1], prefix + '_shortcut',
                                kernel_size=1,
                                stride=stride)
        shortcut = layers.BatchNormalization(name=prefix + '_shortcut_BN')(shortcut)
        outputs = layers.add([residual, shortcut])
    elif skip_connection_type == 'sum':
        outputs = layers.add([residual, inputs])
    elif skip_connection_type == 'none':
        outputs = residual
    else:
        raise ValueError('`skip_connection_type` expected to be in [`conv`, `sum`, `none`],'
                         ' but got {}'.format(skip_connection_type))
    return outputs, skip


def xceptionv0(input_shape=(512, 512, 3), input_tensor=None, pretrained_weights_path=None, output_stride=16):
    """ the original xception model """
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name='img_input')
    else:
        img_input = input_tensor
    x = layers.BatchNormalization()(img_input)
    channel_axis = -1
    output_stride = output_stride

    x = layers.Conv2D(32, (3, 3),
                      strides=(2, 2),
                      use_bias=False,
                      name='block1_conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
    x = layers.Activation('relu', name='block1_conv1_act')(x)
    x = layers.Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
    x = layers.Activation('relu', name='block1_conv2_act')(x)

    residual = layers.Conv2D(128, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block2_sepconv2_act')(x)
    x = layers.SeparableConv2D(128, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block2_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)
    f1 = x

    x = layers.MaxPooling2D((3, 3),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(256, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = layers.SeparableConv2D(256, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block3_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)
    f2 = x

    x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                            padding='same',
                            name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = layers.Conv2D(728, (1, 1),
                             strides=(2, 2),
                             padding='same',
                             use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block4_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)
    f3 = x

    # simply remove the max pooling layer
    if output_stride > 16:
        x = layers.MaxPooling2D((3, 3), strides=(2, 2),
                                padding='same',
                                name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv1')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv1_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv2')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv2_bn')(x)
        x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = layers.SeparableConv2D(728, (3, 3),
                                   padding='same',
                                   use_bias=False,
                                   name=prefix + '_sepconv3')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = layers.Conv2D(1024, (1, 1), strides=(2, 2),
                             padding='same', use_bias=False)(x)
    residual = layers.BatchNormalization(axis=channel_axis)(residual)

    x = layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = layers.SeparableConv2D(728, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = layers.SeparableConv2D(1024, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block13_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block13_sepconv2_bn')(x)
    f4 = x

    if output_stride > 8:
        x = layers.MaxPooling2D((3, 3),
                                strides=(2, 2),
                                padding='same',
                                name='block13_pool')(x)
    x = layers.add([x, residual])

    x = layers.SeparableConv2D(1536, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv1_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = layers.SeparableConv2D(2048, (3, 3),
                               padding='same',
                               use_bias=False,
                               name='block14_sepconv2')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='block14_sepconv2_bn')(x)
    x = layers.Activation('relu', name='block14_sepconv2_act')(x)
    f5 = x

    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        log('load pretrained encoder weights from `{}`'.format(pretrained_weights_path))
        Model(img_input, x, name='xceptionv0_encoder').load_weights(pretrained_weights_path)

    if output_stride == 8:
        features = [f1, f2, f5, f5, f5]
    elif output_stride == 16:
        features = [f1, f2, f3, f5, f5]
    else:
        features = [f1, f2, f3, f4, f5]

    return img_input, features


def xception(input_shape=(512, 512, 3), input_tensor=None, pretrained_weights_path=None, output_stride=16):
    img_input = input_tensor if input_tensor is not None else layers.Input(shape=input_shape, name='img_input')
    x = layers.BatchNormalization()(img_input)

    if output_stride == 8:
        entry_block3_stride = 1
        middle_block_rate = 2  # ! Not mentioned in paper, but required
        exit_block_rates = (2, 4)
        exit_block1_stride = 1
    elif output_stride == 16:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        exit_block1_stride = 1
    else:
        entry_block3_stride = 2
        middle_block_rate = 1
        exit_block_rates = (1, 2)
        exit_block1_stride = 2

    x = layers.Conv2D(32, (3, 3), strides=(2, 2), name='entry_flow_conv1_1',
                      use_bias=False, padding='same')(x)
    x = layers.BatchNormalization(name='entry_flow_conv1_1_BN')(x)
    x = layers.Activation('relu')(x)

    x = _conv2d_same(x, 64, 'entry_flow_conv1_2', kernel_size=3, stride=1)
    x = layers.BatchNormalization(name='entry_flow_conv1_2_BN')(x)
    x = layers.Activation('relu')(x)

    # entry flow
    x, f1 = _xception_block(x, [128, 128, 128], 'entry_flow_block1',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)

    x, f2 = _xception_block(x, [256, 256, 256], 'entry_flow_block2',
                            skip_connection_type='conv', stride=2,
                            depth_activation=False)

    x, f3 = _xception_block(x, [728, 728, 728], 'entry_flow_block3',
                            skip_connection_type='conv', stride=entry_block3_stride,
                            depth_activation=False)

    # middle flow
    for i in range(16):
        x, _ = _xception_block(x, [728, 728, 728], 'middle_flow_unit_{}'.format(i + 1),
                               skip_connection_type='sum', stride=1, rate=middle_block_rate,
                               depth_activation=False)

    # exit flow
    x, f4 = _xception_block(x, [728, 1024, 1024], 'exit_flow_block1',
                            skip_connection_type='conv', stride=exit_block1_stride, rate=exit_block_rates[0],
                            depth_activation=False)
    x, _ = _xception_block(x, [1536, 1536, 2048], 'exit_flow_block2',
                           skip_connection_type='none', stride=1, rate=exit_block_rates[1],
                           depth_activation=True)
    f5 = x

    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        log('load pretrained encoder weights from `{}`'.format(pretrained_weights_path))
        Model(img_input, x, name='xception_encoder').load_weights(pretrained_weights_path, by_name=True, skip_mismatch=True)

    if output_stride == 8:
        features = [f1, f2, f5, f5, f5]
    elif output_stride == 16:
        features = [f1, f2, f3, f5, f5]
    else:
        features = [f1, f2, f3, f4, f5]

    return img_input, features
