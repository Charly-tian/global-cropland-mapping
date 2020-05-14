from keras import layers


def segnet_decoder(features, n_class):
    [f1, f2, f3, f4, f5] = features
    o = f5

    o = layers.UpSampling2D()(o)
    o = layers.Conv2D(512, 3, padding='same', use_bias=False, activation='relu')(o)
    o = layers.BatchNormalization()(o)

    o = layers.UpSampling2D()(o)
    o = layers.Conv2D(512, 3, padding='same', use_bias=False, activation='relu')(o)
    o = layers.BatchNormalization()(o)

    o = layers.UpSampling2D()(o)
    o = layers.Conv2D(256, 3, padding='same', use_bias=False, activation='relu')(o)
    o = layers.BatchNormalization()(o)

    o = layers.UpSampling2D()(o)
    o = layers.Conv2D(128, 3, padding='same', use_bias=False, activation='relu')(o)
    o = layers.BatchNormalization()(o)

    o = layers.UpSampling2D()(o)
    o = layers.Conv2D(64, 3, padding='same', use_bias=False, activation='relu')(o)
    o = layers.BatchNormalization()(o)

    o = layers.Conv2D(n_class, (1, 1), padding='same')(o)
    return o
