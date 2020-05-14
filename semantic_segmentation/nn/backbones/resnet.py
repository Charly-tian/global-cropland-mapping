import os
from keras import layers
from keras.models import Model
from ...utils import log

resnet50_pretrained_url = 'https://github.com/fchollet/deep-learning-models/' \
                          'releases/download/v0.2/'\
                          'resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block, rate=1):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, dilation_rate=rate,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               rate=1):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.

    # Returns
        Output tensor for the block.

    Note that from stage 3,
    the first conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      dilation_rate=rate,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet50(input_shape=(512, 512, 3), input_tensor=None, pretrained_weights_path=None,
             output_stride=16):
    from keras.applications.resnet50 import ResNet50
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name='img_input')
    else:
        img_input = input_tensor
    bn_axis = 3

    if output_stride == 8:
        strides = [1, 1]
        rates = [2, 4]
    elif output_stride == 16:
        strides = [2, 1]
        rates = [1, 2]
    else:
        strides = [2, 2]
        rates = [1, 1]

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    f1 = x

    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = x

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a', strides=2)
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=strides[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', rate=rates[0])
    f4 = x

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=strides[1])
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', rate=rates[1])
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', rate=rates[1])
    f5 = x

    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        log('load pretrained encoder weights from `{}`'.format(pretrained_weights_path))
        Model(img_input, x, name='resnet50_encoder').load_weights(pretrained_weights_path, by_name=True, skip_mismatch=True)

    if output_stride == 8:
        features = [f1, f2, f5, f5, f5]
    elif output_stride == 16:
        features = [f1, f2, f3, f5, f5]
    else:
        features = [f1, f2, f3, f4, f5]

    return img_input, features


def resnet101(input_shape=(512, 512, 3), input_tensor=None, pretrained_weights_path=None,
              output_stride=16):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name='img_input')
    else:
        img_input = input_tensor
    x = layers.BatchNormalization()(img_input)
    bn_axis = 3
    output_stride = output_stride
    if output_stride == 8:
        strides = [1, 1]
        rates = [2, 4]
    elif output_stride == 16:
        strides = [2, 1]
        rates = [1, 2]
    else:
        strides = [2, 2]
        rates = [1, 1]

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    f1 = x

    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # stage 2, 3 units
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')
    f2 = x

    # stage 3, 4 units
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')
    f3 = x

    # stage 4, 23 units
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a', strides=strides[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='g', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='h', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='i', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='j', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='k', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='l', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='m', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='n', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='o', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='p', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='q', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='r', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='s', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='t', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='u', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='v', rate=rates[0])
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='w', rate=rates[0])
    f4 = x

    # stage 5, 3 units
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a', strides=strides[1])
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b', rate=rates[1])
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c', rate=rates[1])
    f5 = x

    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        log('load pretrained encoder weights from `{}`'.format(pretrained_weights_path))
        Model(img_input, x, name='resnet101_encoder').load_weights(pretrained_weights_path, by_name=True, skip_mismatch=True)

    if output_stride == 8:
        features = [f1, f2, f5, f5, f5]
    elif output_stride == 16:
        features = [f1, f2, f3, f5, f5]
    else:
        features = [f1, f2, f3, f4, f5]

    return img_input, features
