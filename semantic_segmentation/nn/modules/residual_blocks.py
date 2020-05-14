#
# from keras.layers import Conv2D
# from keras.layers import BatchNormalization
# from keras.layers import Activation
# from keras.layers import DepthwiseConv2D
# from keras.layers import Add
#
# from ..utils.util_functions import _make_divisible
#
# from keras.activations import relu
#
#
# def relu6(x):
#     return relu(x, max_value=6)
#
#
# def identity_block(input_tensor, kernel_size, filters, stage, block):
#     """The identity block is the block that has no conv layer at shortcut.
#     # Arguments
#         input_tensor: input tensor
#         kernel_size: defualt 3, the kernel size of middle conv layer at main path
#         filters: list of integers, the filterss of 3 conv layer at main path
#         stage: integer, current stage label, used for generating layer names
#         block: 'a','b'..., current block label, used for generating layer names
#     # Returns
#         Output tensor for the block.
#     """
#     filters1, filters2, filters3 = filters
#     bn_axis = 3
#
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#     x = Add()([x, input_tensor])
#     x = Activation('relu')(x)
#     return x
#
#
# def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
#     """conv_block is the block that has a conv layer at shortcut
#     # Arguments
#         input_tensor: input tensor
#         kernel_size: defualt 3, the kernel size of middle conv layer at main path
#         filters: list of integers, the filterss of 3 conv layer at main path
#         stage: integer, current stage label, used for generating layer names
#         block: 'a','b'..., current block label, used for generating layer names
#     # Returns
#         Output tensor for the block.
#     Note that from stage 3, the first conv layer at main path is with strides=(2,2)
#     And the shortcut should have strides=(2,2) as well
#     """
#     filters1, filters2, filters3 = filters
#     bn_axis = 3
#
#     conv_name_base = 'res' + str(stage) + block + '_branch'
#     bn_name_base = 'bn' + str(stage) + block + '_branch'
#
#     x = Conv2D(filters1, (1, 1), strides=strides,
#                name=conv_name_base + '2a')(input_tensor)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters2, kernel_size, padding='same', name=conv_name_base + '2b')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
#     x = Activation('relu')(x)
#
#     x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
#     x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)
#
#     shortcut = Conv2D(filters3, (1, 1), strides=strides,
#                       name=conv_name_base + '1')(input_tensor)
#     shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
#
#     x = Add()([x, shortcut])
#     x = Activation('relu')(x)
#     return x
#
#
# def _inverted_res_block(inputs, expansion, stride, alpha, filters, block_id, skip_connection, rate=1):
#     in_channels = inputs.shape[-1].value  # inputs._keras_shape[-1]
#     pointwise_conv_filters = int(filters * alpha)
#     pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
#     x = inputs
#     prefix = 'expanded_conv_{}_'.format(block_id)
#     if block_id:
#         # Expand
#
#         x = Conv2D(expansion * in_channels, kernel_size=1, padding='same',
#                    use_bias=False, activation=None,
#                    name=prefix + 'expand')(x)
#         x = BatchNormalization(epsilon=1e-3, momentum=0.999,
#                                name=prefix + 'expand_BN')(x)
#         x = Activation(relu6, name=prefix + 'expand_relu')(x)
#     else:
#         prefix = 'expanded_conv_'
#     # Depthwise
#     x = DepthwiseConv2D(kernel_size=3, strides=stride, activation=None,
#                         use_bias=False, padding='same', dilation_rate=(rate, rate),
#                         name=prefix + 'depthwise')(x)
#     x = BatchNormalization(epsilon=1e-3, momentum=0.999,
#                            name=prefix + 'depthwise_BN')(x)
#
#     x = Activation(relu6, name=prefix + 'depthwise_relu')(x)
#
#     # Project
#     x = Conv2D(pointwise_filters,
#                kernel_size=1, padding='same', use_bias=False, activation=None,
#                name=prefix + 'project')(x)
#     x = BatchNormalization(epsilon=1e-3, momentum=0.999,
#                            name=prefix + 'project_BN')(x)
#
#     if skip_connection:
#         return Add(name=prefix + 'add')([inputs, x])
#
#     # if in_channels == pointwise_filters and stride == 1:
#     #    return Add(name='res_connect_' + str(block_id))([inputs, x])
#
#     return x
