# from keras.layers import ZeroPadding2D
# from keras.layers import Lambda
#
#
# def one_side_pad(x):
#     x = ZeroPadding2D((1, 1))(x)
#     x = Lambda(lambda x: x[:, :-1, :-1, :])(x)
#     return x
