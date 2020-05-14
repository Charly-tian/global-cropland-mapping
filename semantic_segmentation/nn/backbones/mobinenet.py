import os
from keras import backend as K
from keras import layers
from keras.models import Model
from ...utils import log


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def correct_pad(backend, inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.

    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.

    # Returns
        A tuple.
    """
    img_dim = 2 if backend.image_data_format() == 'channels_first' else 1
    input_size = backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))


def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, rate=1, skip_connection=False):
    channel_axis = -1

    in_channels = K.int_shape(inputs)[channel_axis]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)
    feature = None

    if block_id:
        # Expand
        x = layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = layers.BatchNormalization(axis=channel_axis,
                                      epsilon=1e-3,
                                      momentum=0.999,
                                      name=prefix + 'expand_BN')(x)
        x = layers.ReLU(6., name=prefix + 'expand_relu')(x)
        feature = x
    else:
        prefix = 'expanded_conv_'

    # Depthwise
    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(K, x, 3),
                                 name=prefix + 'pad')(x)
    x = layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               dilation_rate=(rate, rate),
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'depthwise_BN')(x)

    x = layers.ReLU(6., name=prefix + 'depthwise_relu')(x)

    # Project
    x = layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name=prefix + 'project_BN')(x)
    if skip_connection:
        return layers.Add(name=prefix + 'add')([inputs, x]), feature
    # if in_channels == pointwise_filters and stride == 1:
    #     return layers.Add(name=prefix + 'add')([inputs, x])
    return x, feature

from keras.applications.mobilenetv2 import MobileNetV2
def mobilenetv2(input_shape=(512, 512, 3), input_tensor=None, pretrained_weights_path=None,
                alpha=1., output_stride=16):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        img_input = input_tensor

    channel_axis = -1
    output_stride = output_stride
    if output_stride == 8:
        res_block_strides = [1, 1]
        res_block_rates = [2, 4]
    elif output_stride == 16:
        res_block_strides = [2, 1]
        res_block_rates = [1, 2]
    else:
        res_block_strides = [2, 2]
        res_block_rates = [1, 1]

    first_block_filters = _make_divisible(32 * alpha, 8)
    x = layers.ZeroPadding2D(padding=correct_pad(K, img_input, 3),
                             name='Conv1_pad')(img_input)
    x = layers.Conv2D(first_block_filters,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name='Conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='bn_Conv1')(x)
    x = layers.ReLU(6., name='Conv1_relu')(x)

    x, _ = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                               expansion=1, block_id=0, rate=1, skip_connection=False)

    x, f1 = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1, rate=1, skip_connection=False)
    x, _ = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                               expansion=6, block_id=2, rate=1, skip_connection=True)

    x, f2 = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3, rate=1, skip_connection=False)
    x, _ = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                               expansion=6, block_id=4, rate=1, skip_connection=True)
    x, _ = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                               expansion=6, block_id=5, rate=1, skip_connection=True)

    x, f3 = _inverted_res_block(x, filters=64, alpha=alpha, stride=res_block_strides[0],
                                expansion=6, block_id=6, rate=1, skip_connection=False)
    x, _ = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                               expansion=6, block_id=7, rate=res_block_rates[0], skip_connection=True)
    x, _ = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                               expansion=6, block_id=8, rate=res_block_rates[0], skip_connection=True)
    x, _ = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                               expansion=6, block_id=9, rate=res_block_rates[0], skip_connection=True)

    x, _ = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                               expansion=6, block_id=10, rate=res_block_rates[0], skip_connection=False)
    x, _ = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                               expansion=6, block_id=11, rate=res_block_rates[0], skip_connection=True)
    x, _ = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                               expansion=6, block_id=12, rate=res_block_rates[0], skip_connection=True)

    x, f4 = _inverted_res_block(x, filters=160, alpha=alpha, stride=res_block_strides[1],
                                expansion=6, block_id=13, rate=res_block_rates[0], skip_connection=False)
    x, _ = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                               expansion=6, block_id=14, rate=res_block_rates[1], skip_connection=True)
    x, _ = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                               expansion=6, block_id=15, rate=res_block_rates[1], skip_connection=True)

    x, _ = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                               expansion=6, block_id=16, rate=res_block_rates[1], skip_connection=False)

    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280

    x = layers.Conv2D(last_block_filters,
                      kernel_size=1,
                      use_bias=False,
                      name='Conv_1')(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  epsilon=1e-3,
                                  momentum=0.999,
                                  name='Conv_1_bn')(x)
    x = layers.ReLU(6., name='out_relu')(x)
    f5 = x

    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        log('load pretrained encoder weights from `{}`'.format(pretrained_weights_path))
        Model(img_input, x).load_weights(pretrained_weights_path, by_name=True, skip_mismatch=True)

    if output_stride == 8:
        features = [f1, f2, f5, f5, f5]
    elif output_stride == 16:
        features = [f1, f2, f3, f5, f5]
    else:
        features = [f1, f2, f3, f4, f5]

    return img_input, features
