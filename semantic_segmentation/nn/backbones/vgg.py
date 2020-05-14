import os
from keras import layers
from keras.models import Model
from ...utils import log

vgg16_pretrained_url = 'https://github.com/fchollet/deep-learning-models/' \
                       'releases/download/v0.1/' \
                       'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'
vgg19_pretrained_url = 'https://github.com/fchollet/deep-learning-models/' \
                       'releases/download/v0.1/' \
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'


def vgg16(input_shape=(512, 512, 3), input_tensor=None, pretrained_weights_path=None,
          output_stride=16):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name='img_input')
    else:
        img_input = input_tensor
    x = layers.BatchNormalization()(img_input)

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    if output_stride > 16:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    if output_stride > 8:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        log('load pretrained encoder weights from `{}`'.format(pretrained_weights_path))
        Model(img_input, x, name='vgg16_encoder').load_weights(pretrained_weights_path)

    if output_stride == 8:
        features = [f1, f2, f5, f5, f5]
    elif output_stride == 16:
        features = [f1, f2, f3, f5, f5]
    else:
        features = [f1, f2, f3, f4, f5]

    return img_input, features


def vgg19(input_shape=(512, 512, 3), input_tensor=None, pretrained_weights_path=None,
          output_stride=16):
    if input_tensor is None:
        img_input = layers.Input(shape=input_shape, name='img_input')
    else:
        img_input = input_tensor
    x = layers.BatchNormalization()(img_input)

    # Block 1
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    f1 = x

    # Block 2
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    f2 = x

    # Block 3
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    f3 = x

    # Block 4
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv4')(x)
    if output_stride > 16:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    f4 = x

    # Block 5
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv4')(x)
    if output_stride > 8:
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    f5 = x

    if pretrained_weights_path is not None and os.path.exists(pretrained_weights_path):
        log('load pretrained encoder weights from `{}`'.format(pretrained_weights_path))
        Model(img_input, x, name='vgg19_encoder').load_weights(pretrained_weights_path)

    if output_stride == 8:
        features = [f1, f2, f5, f5, f5]
    elif output_stride == 16:
        features = [f1, f2, f3, f5, f5]
    else:
        features = [f1, f2, f3, f4, f5]

    return img_input, features
