from keras import backend as K
from keras import layers
from keras.models import Model

from ..backbones import build_encoder
from .segnet import segnet_decoder
from .unet import unet, unet_mini, resunet, resunet_lstm
from .deeplab_v3 import deeplabv3_decoder
from .deeplab_v3p import deeplabv3p, deeplabv3p_lstm


def segmentation_model(model_name='deeplabv3p', input_shape=(512, 512, 3), input_tensor=None, n_class=21,
                       backbone_name='mobilenetv2', backbone_weights=None, output_stride=16, alpha=1.,
                       n_filter=64, one_hot=False):
    model_name = model_name.lower()
    assert model_name in {'unetmini', 'unet', 'resunet', 'deeplabv3p', 'deeplabv3plstm', 'resunetlstm'}, \
        "`model_name` expected to be in [], but got `{}`".format(model_name)

    if model_name == 'unet':
        model = unet(input_shape, n_class, n_filter, one_hot)
        return model
    elif model_name == 'unetmini':
        model = unet_mini(input_shape, n_class, n_filter, one_hot)
        return model
    elif model_name == 'resunet':
        model = resunet(input_shape, n_class, n_filter, one_hot)
        return model
    elif model_name == 'resunetlstm':
        model = resunet_lstm(input_shape, n_class, n_filter, one_hot, input_split=4)
        return model
    elif model_name == 'deeplabv3p':
        if output_stride == 8:
            atrous_rates = (12, 24, 36)
        else:
            # atrous_rates = (6, 12, 18)
            atrous_rates = (2, 3, 5)
        model = deeplabv3p(input_shape, n_class, one_hot, backbone_name, backbone_weights, output_stride, atrous_rates)
        return model
    elif model_name == 'deeplabv3plstm':
        if output_stride == 8:
            atrous_rates = (12, 24, 36)
        else:
            # atrous_rates = (6, 12, 18)
            atrous_rates = (2, 3, 5)
        model = deeplabv3p_lstm(input_shape, n_class, one_hot, backbone_name, backbone_weights, output_stride,
                                atrous_rates, input_split=4)
        return model

