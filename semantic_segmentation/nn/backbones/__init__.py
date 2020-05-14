from .vgg import vgg16, vgg19
from .resnet import resnet50, resnet101
from .mobinenet import mobilenetv2
from .xception import xceptionv0, xception
# a BN is added before the backbone


def build_encoder(input_shape=(512, 512, 3), input_tensor=None, backbone_name='xception',
                  backbone_weights=None, output_stride=16, alpha=1.):
    backbone_name = backbone_name.lower()
    assert output_stride in [8, 16, 32], '`output_stride` expected be in [8, 16, 32],' \
                                         ' but got {}'.format(output_stride)
    if backbone_name == 'vgg16':
        img_input, features = vgg16(input_shape=input_shape, input_tensor=input_tensor,
                                    pretrained_weights_path=backbone_weights,
                                    output_stride=output_stride)
    elif backbone_name == 'vgg19':
        img_input, features = vgg19(input_shape=input_shape, input_tensor=input_tensor,
                                    pretrained_weights_path=backbone_weights,
                                    output_stride=output_stride)
    elif backbone_name == 'resnet50':
        img_input, features = resnet50(input_shape=input_shape, input_tensor=input_tensor,
                                       pretrained_weights_path=backbone_weights,
                                       output_stride=output_stride)
    elif backbone_name == 'resnet101':
        img_input, features = resnet101(input_shape=input_shape, input_tensor=input_tensor,
                                        pretrained_weights_path=backbone_weights,
                                        output_stride=output_stride)
    elif backbone_name == 'mobilenetv2':
        img_input, features = mobilenetv2(input_shape=input_shape, input_tensor=input_tensor,
                                          pretrained_weights_path=backbone_weights,
                                          alpha=alpha, output_stride=output_stride)
    elif backbone_name == 'xceptionv0':
        img_input, features = xceptionv0(input_shape=input_shape, input_tensor=input_tensor,
                                         pretrained_weights_path=backbone_weights,
                                         output_stride=output_stride)
    elif backbone_name == 'xception':
        img_input, features = xception(input_shape=input_shape, input_tensor=input_tensor,
                                       pretrained_weights_path=backbone_weights,
                                       output_stride=output_stride)
    else:
        raise ValueError('`backbone` expected in [`vgg16`, `vgg19`, `resnet50`, `resnet101`, `mobilenetv2`,'
                         ' `xceptionv0`, `xception`], but got `{}`'.format(backbone_name))
    return img_input, features
