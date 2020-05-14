from keras import layers
from keras import models
from keras import backend as K
from keras import regularizers
# from .utils import my_global_attention_upsample, convolutional_block_attention_module
from .. import WEIGHT_DECAY, KERNEL_INITIALIZER, DROPOUT


def unet_mini(input_shape, n_class, n_filter=32, one_hot=False):
    img_input = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(n_filter, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(img_input)
    conv1 = layers.Dropout(DROPOUT)(conv1)
    conv1 = layers.Conv2D(n_filter, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(n_filter * 2, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(pool1)
    conv2 = layers.Dropout(DROPOUT)(conv2)
    conv2 = layers.Conv2D(n_filter * 2, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(n_filter * 4, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(pool2)
    conv3 = layers.Dropout(DROPOUT)(conv3)
    conv3 = layers.Conv2D(n_filter * 4, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv3)

    up1 = layers.concatenate([layers.UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = layers.Conv2D(n_filter * 2, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(up1)
    conv4 = layers.Dropout(DROPOUT)(conv4)
    conv4 = layers.Conv2D(n_filter * 2, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv4)

    up2 = layers.concatenate([layers.UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = layers.Conv2D(n_filter, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(up2)
    conv5 = layers.Dropout(DROPOUT)(conv5)
    conv5 = layers.Conv2D(n_filter, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv5)

    x = layers.Conv2D(n_class, (1, 1), padding='same',
                      kernel_initializer=KERNEL_INITIALIZER,
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                     )(conv5)
    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)
    return models.Model(img_input, x)


# unet with dropout
def unet(input_shape=(128, 128, 4), n_class=1, num_filters=64, one_hot=False):
    img_input = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(num_filters * 1, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(img_input)
    conv1 = layers.Dropout(DROPOUT)(conv1)
    conv1 = layers.Conv2D(num_filters * 1, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(pool1)
    conv2 = layers.Dropout(DROPOUT)(conv2)
    conv2 = layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = layers.Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(pool2)
    conv3 = layers.Dropout(DROPOUT)(conv3)
    conv3 = layers.Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv3)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)

    conv4 = layers.Conv2D(num_filters * 8, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(pool3)
    conv4 = layers.Dropout(DROPOUT)(conv4)
    conv4 = layers.Conv2D(num_filters * 8, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv4)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)

    conv5 = layers.Conv2D(n_class * 16, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(pool4)
    conv5 = layers.Dropout(DROPOUT)(conv5)
    conv5 = layers.Conv2D(num_filters * 16, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv5)

    up1 = layers.concatenate([layers.UpSampling2D((2, 2))(conv5), conv4], axis=-1)
    conv6 = layers.Conv2D(num_filters * 8, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(up1)
    conv6 = layers.Dropout(DROPOUT)(conv6)
    conv6 = layers.Conv2D(num_filters * 8, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv6)

    up2 = layers.concatenate([layers.UpSampling2D((2, 2))(conv6), conv3], axis=-1)
    conv7 = layers.Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(up2)
    conv7 = layers.Dropout(DROPOUT)(conv7)
    conv7 = layers.Conv2D(num_filters * 4, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv7)

    up3 = layers.concatenate([layers.UpSampling2D((2, 2))(conv7), conv2], axis=-1)
    conv8 = layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(up3)
    conv8 = layers.Dropout(DROPOUT)(conv8)
    conv8 = layers.Conv2D(num_filters * 2, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv8)

    up4 = layers.concatenate([layers.UpSampling2D((2, 2))(conv8), conv1], axis=-1)
    conv9 = layers.Conv2D(num_filters * 1, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(up4)
    conv9 = layers.Dropout(DROPOUT)(conv9)
    conv9 = layers.Conv2D(num_filters * 1, (3, 3), activation='relu', padding='same',
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(conv9)

    x = layers.Conv2D(n_class, (1, 1), padding='same', use_bias=False, activation=None,
                      kernel_initializer=KERNEL_INITIALIZER,
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                     )(conv9)
    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)
    return models.Model(img_input, x)


# unet with bn
def unet_bn(input_shape=(512, 512, 3), n_class=2, num_filters=32, one_hot=False):
    def conv_block(inputs, num_filters=32):
        x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    def encoder_block(inputs, num_filters=32):
        x = conv_block(inputs, num_filters)
        # replace the MaxPooling to 3x3 Conv with strides
        # pool = layers.Conv2D(num_filters, (3, 3), padding='same', strides=2,
        #                      kernel_initializer=KERNEL_INITIALIZER,
        #                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
        # pool = layers.BatchNormalization()(pool)
        # pool = layers.Activation('relu')(pool)
        pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(x)
        return x, pool

    def decoder_block(input_tensor, concat_tensor, num_filters=32):
        x = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same',
                                   kernel_initializer=KERNEL_INITIALIZER,
                                   kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.concatenate([x, concat_tensor], axis=-1)

        x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(num_filters, (3, 3), padding='same', kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x

    img_input = layers.Input(shape=input_shape)
    conv0, pool1 = encoder_block(img_input, num_filters * 1)
    conv1, pool2 = encoder_block(pool1, num_filters * 2)
    conv2, pool3 = encoder_block(pool2, num_filters * 4)
    conv3, pool4 = encoder_block(pool3, num_filters * 8)
    conv4 = conv_block(pool4, num_filters * 16)
    uconv3 = decoder_block(conv4, conv3, num_filters * 8)
    uconv2 = decoder_block(uconv3, conv2, num_filters * 4)
    uconv1 = decoder_block(uconv2, conv1, num_filters * 2)
    uconv0 = decoder_block(uconv1, conv0, num_filters * 1)

    x = layers.Conv2D(n_class, (1, 1), padding='same', use_bias=False,
                      kernel_initializer=KERNEL_INITIALIZER)(uconv0)

    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)
    return models.Model(img_input, x)


def batchActivate(x):
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding,
                      kernel_initializer=KERNEL_INITIALIZER,
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                     )(x)
    if activation:
        x = batchActivate(x)
    return x


def residual_block(blockInput, num_filters=16, batch_activate=False):
    x = batchActivate(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = layers.Add()([x, blockInput])
    if batch_activate:
        x = batchActivate(x)
    return x


# u net model with residual block and dropout to avoid over-fitting
def resunet(input_shape=(128, 128, 4), n_class=1, num_filters=64, one_hot=False):
    img_input = layers.Input(shape=input_shape)
    # 1/2
    conv1 = layers.Conv2D(num_filters * 1, (3, 3), activation=None, padding="same",
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(img_input)
    conv1 = residual_block(conv1, num_filters * 1)
    conv1 = residual_block(conv1, num_filters * 1, True)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    pool1 = layers.Dropout(DROPOUT)(pool1)

    # 1/4
    conv2 = layers.Conv2D(num_filters * 2, (3, 3), activation=None, padding="same",
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(pool1)
    conv2 = residual_block(conv2, num_filters * 2)
    conv2 = residual_block(conv2, num_filters * 2, True)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    pool2 = layers.Dropout(DROPOUT)(pool2)

    # 1/8
    conv3 = layers.Conv2D(num_filters * 4, (3, 3), activation=None, padding="same",
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(pool2)
    conv3 = residual_block(conv3, num_filters * 4)
    conv3 = residual_block(conv3, num_filters * 4, True)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    pool3 = layers.Dropout(DROPOUT)(pool3)

    # 1/16
    conv4 = layers.Conv2D(num_filters * 8, (3, 3), activation=None, padding="same",
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(pool3)
    conv4 = residual_block(conv4, num_filters * 8)
    conv4 = residual_block(conv4, num_filters * 8, True)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    pool4 = layers.Dropout(DROPOUT)(pool4)

    # Middle
    convm = layers.Conv2D(num_filters * 16, (3, 3), activation=None, padding="same",
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(pool4)
    convm = residual_block(convm, num_filters * 16)
    convm = residual_block(convm, num_filters * 16, True)
    # convm = convolutional_block_attention_module(convm, hidden_size=start_neurons, conv_size=7)

    # 1/8
    deconv4 = layers.UpSampling2D((2, 2), interpolation='bilinear')(convm)
    # uconv4 = my_global_attention_upsample(deconv4, conv4)
    uconv4 = layers.concatenate([deconv4, conv4])
    uconv4 = layers.Dropout(DROPOUT)(uconv4)

    uconv4 = layers.Conv2D(num_filters * 8, (3, 3), activation=None, padding="same",
                          kernel_initializer=KERNEL_INITIALIZER,
                          kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(uconv4)
    uconv4 = residual_block(uconv4, num_filters * 8)
    uconv4 = residual_block(uconv4, num_filters * 8, True)

    # 1/4
    deconv3 = layers.UpSampling2D((2, 2), interpolation='bilinear')(uconv4)
    # uconv3 = my_global_attention_upsample(deconv3, conv3)
    uconv3 = layers.concatenate([deconv3, conv3])
    uconv3 = layers.Dropout(DROPOUT)(uconv3)

    uconv3 = layers.Conv2D(num_filters * 4, (3, 3), activation=None, padding="same",
                           kernel_initializer=KERNEL_INITIALIZER,
                           kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                           )(uconv3)
    uconv3 = residual_block(uconv3, num_filters * 4)
    uconv3 = residual_block(uconv3, num_filters * 4, True)

    # 1/2
    deconv2 = layers.UpSampling2D((2, 2), interpolation='bilinear')(uconv3)
    # uconv2 = my_global_attention_upsample(deconv2, conv2)
    uconv2 = layers.concatenate([deconv2, conv2])
    uconv2 = layers.Dropout(DROPOUT)(uconv2)

    uconv2 = layers.Conv2D(num_filters * 2, (3, 3), activation=None, padding="same",
                           kernel_initializer=KERNEL_INITIALIZER,
                           kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                           )(uconv2)
    uconv2 = residual_block(uconv2, num_filters * 2)
    uconv2 = residual_block(uconv2, num_filters * 2, True)

    # 1/1
    deconv1 = layers.UpSampling2D((2, 2), interpolation='bilinear')(uconv2)
    uconv1 = layers.concatenate([deconv1, conv1])
    # uconv1 = my_global_attention_upsample(deconv1, conv1)
    uconv1 = layers.Dropout(DROPOUT)(uconv1)

    uconv1 = layers.Conv2D(num_filters * 1, (3, 3), activation=None, padding="same",
                           kernel_initializer=KERNEL_INITIALIZER,
                           kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                           )(uconv1)
    uconv1 = residual_block(uconv1, num_filters * 1)
    uconv1 = residual_block(uconv1, num_filters * 1, True)

    x = layers.Conv2D(n_class, (1, 1), padding="same", activation=None, use_bias=False,
                      kernel_initializer=KERNEL_INITIALIZER,
                      kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                     )(uconv1)
    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)
    return models.Model(img_input, x)


def resunet_lstm(input_shape=(128, 128, 16), n_class=1, n_filter=64, one_hot=False, input_split=4):
    def resunet_encoder():
        img_input = layers.Input(shape=(input_shape[0], input_shape[1], input_shape[-1] // input_split))
        # 256 -> 128
        conv1 = layers.Conv2D(n_filter * 1, (3, 3), activation=None, padding="same",
                           kernel_initializer=KERNEL_INITIALIZER,
                           kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                           )(img_input)
        conv1 = residual_block(conv1, n_filter * 1)
        conv1 = residual_block(conv1, n_filter * 1, True)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)
        pool1 = layers.Dropout(DROPOUT)(pool1)

        # 128 -> 64
        conv2 = layers.Conv2D(n_filter * 2, (3, 3), activation=None, padding="same",
                              kernel_initializer=KERNEL_INITIALIZER,
                              kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                              )(pool1)
        conv2 = residual_block(conv2, n_filter * 2)
        conv2 = residual_block(conv2, n_filter * 2, True)
        pool2 = layers.MaxPooling2D((2, 2))(conv2)
        pool2 = layers.Dropout(DROPOUT)(pool2)

        # 64 -> 32
        conv3 = layers.Conv2D(n_filter * 4, (3, 3), activation=None, padding="same",
                              kernel_initializer=KERNEL_INITIALIZER,
                              kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                              )(pool2)
        conv3 = residual_block(conv3, n_filter * 4)
        conv3 = residual_block(conv3, n_filter * 4, True)
        pool3 = layers.MaxPooling2D((2, 2))(conv3)
        pool3 = layers.Dropout(DROPOUT)(pool3)

        # 32 -> 16
        conv4 = layers.Conv2D(n_filter * 8, (3, 3), activation=None, padding="same",
                              kernel_initializer=KERNEL_INITIALIZER,
                              kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                              )(pool3)
        conv4 = residual_block(conv4, n_filter * 8)
        conv4 = residual_block(conv4, n_filter * 8, True)
        pool4 = layers.MaxPooling2D((2, 2))(conv4)
        pool4 = layers.Dropout(DROPOUT)(pool4)

        # Middle
        convm = layers.Conv2D(n_filter * 16, (3, 3), activation=None, padding="same",
                              kernel_initializer=KERNEL_INITIALIZER,
                              kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                              )(pool4)
        convm = residual_block(convm, n_filter * 16)
        convm = residual_block(convm, n_filter * 16, True)
        # convm = convolutional_block_attention_module(convm, hidden_size=n_filter, conv_size=7)

        return models.Model(img_input, [conv1, conv2, conv3, conv4, convm])

    assert input_shape[-1] % input_split == 0
    img_input = layers.Input(shape=input_shape)
    img_inputs = []
    channels_per_split = input_shape[-1] // input_split
    for i in range(input_split):
        _x = layers.Lambda(lambda xx: xx[:, :, :, i * channels_per_split: (i + 1) * channels_per_split])(img_input)
        img_inputs.append(_x)

    feature_extractor = resunet_encoder()
    F0s = []
    F1s = []
    F2s = []
    F3s = []
    F4s = []
    for i in range(input_split):
        F01234 = feature_extractor(img_inputs[i])
        F0s.append(F01234[0])
        F1s.append(F01234[1])
        F2s.append(F01234[2])
        F3s.append(F01234[3])
        F4s.append(F01234[4])

    F0s = layers.Lambda(lambda xx: K.stack(xx, axis=1))(F0s)
    F1s = layers.Lambda(lambda xx: K.stack(xx, axis=1))(F1s)
    F2s = layers.Lambda(lambda xx: K.stack(xx, axis=1))(F2s)
    F3s = layers.Lambda(lambda xx: K.stack(xx, axis=1))(F3s)
    F4s = layers.Lambda(lambda xx: K.stack(xx, axis=1))(F4s)

    F0 = layers.ConvLSTM2D(filters=n_filter, kernel_size=3, padding='same', activation='relu',
                           return_sequences=False)(F0s)
    F1 = layers.ConvLSTM2D(filters=n_filter, kernel_size=3, padding='same', activation='relu',
                           return_sequences=False)(F1s)
    F2 = layers.ConvLSTM2D(filters=n_filter, kernel_size=3, padding='same', activation='relu',
                           return_sequences=False)(F2s)
    F3 = layers.ConvLSTM2D(filters=n_filter, kernel_size=3, padding='same', activation='relu',
                           return_sequences=False)(F3s)
    F4 = layers.ConvLSTM2D(filters=n_filter, kernel_size=3, padding='same', activation='relu',
                           return_sequences=False)(F4s)

    # 16 -> 32
    deconv4 = layers.UpSampling2D((2, 2))(F4)
    uconv4 = layers.concatenate([deconv4, F3])
    uconv4 = layers.Dropout(DROPOUT)(uconv4)

    uconv4 = layers.Conv2D(n_filter * 8, (3, 3), activation=None, padding="same",
                           kernel_initializer=KERNEL_INITIALIZER,
                           kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(uconv4)
    uconv4 = residual_block(uconv4, n_filter * 8)
    uconv4 = residual_block(uconv4, n_filter * 8, True)

    # 32 -> 64
    deconv3 = layers.UpSampling2D((2, 2))(uconv4)
    uconv3 = layers.concatenate([deconv3, F2])
    uconv3 = layers.Dropout(DROPOUT)(uconv3)

    uconv3 = layers.Conv2D(n_filter * 4, (3, 3), activation=None, padding="same",
                           kernel_initializer=KERNEL_INITIALIZER,
                           kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                           )(uconv3)
    uconv3 = residual_block(uconv3, n_filter * 4)
    uconv3 = residual_block(uconv3, n_filter * 4, True)

    # 64 -> 128
    deconv2 = layers.UpSampling2D((2, 2))(uconv3)
    uconv2 = layers.concatenate([deconv2, F1])
    uconv2 = layers.Dropout(DROPOUT)(uconv2)

    uconv2 = layers.Conv2D(n_filter * 2, (3, 3), activation=None, padding="same",
                           kernel_initializer=KERNEL_INITIALIZER,
                           kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(uconv2)
    uconv2 = residual_block(uconv2, n_filter * 2)
    uconv2 = residual_block(uconv2, n_filter * 2, True)

    # 128 -> 256
    deconv1 = layers.UpSampling2D((2, 2))(uconv2)
    uconv1 = layers.concatenate([deconv1, F0])
    uconv1 = layers.Dropout(DROPOUT)(uconv1)

    uconv1 = layers.Conv2D(n_filter * 1, (3, 3), activation=None, padding="same",
                           kernel_initializer=KERNEL_INITIALIZER,
                           kernel_regularizer=regularizers.l2(WEIGHT_DECAY)
                          )(uconv1)
    uconv1 = residual_block(uconv1, n_filter * 1)
    uconv1 = residual_block(uconv1, n_filter * 1, True)

    x = layers.Conv2D(n_class, (1, 1), padding="same", activation=None)(uconv1)
    if n_class == 1 or (n_class == 2 and one_hot is False):
        x = layers.Activation('sigmoid')(x)
    else:
        x = layers.Activation('softmax')(x)
    return models.Model(img_input, x)
