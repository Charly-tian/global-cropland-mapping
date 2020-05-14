import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
import numbers
import random
from collections.abc import Iterable

FILL_VALUE = 0  # fill value for data array
LABEL_FILL_VALUE = 0    # fill value for label array


class ToTensor:
    """ Convert numpy-array to tensorflow-tensor """

    def __call__(self, data_numpy, label_numpy):
        return tf.convert_to_tensor(data_numpy), tf.convert_to_tensor(label_numpy)


class ToNumpy:
    """ Convert tensorflow-tensor to numpy-array """

    def __call__(self, data_tensor, label_tensor):
        return data_tensor.eval(), label_tensor.eval()


class Normalize:
    """ Normalize the numpy-array through given mean & std for each channel """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data_numpy, label_numpy):
        return (data_numpy - self.mean) / self.std, label_numpy


class MinMaxScale:
    """ Scale the numpy-array through given min & max for each channel """

    def __init__(self, mins, maxs):
        self.mins = mins
        self.maxs = maxs

    def __call__(self, data_numpy, label_numpy):
        data_numpy = (data_numpy - self.mins) / (self.maxs - self.mins)
        return data_numpy.clip(0, 1), label_numpy


class Pad:
    """ Pad the image and label numpy-arrays """

    def __init__(self, target_height, target_width):
        self.target_height = target_height
        self.target_width = target_width

    def __call__(self, data_numpy, label_numpy):
        img_h, img_w, _ = data_numpy.shape
        pad_h = max(self.target_height - img_h, 0)
        pad_w = max(self.target_width - img_w, 0)
        data_numpy = np.lib.pad(data_numpy,
                                ((pad_h // 2, pad_h - pad_h // 2),
                                 (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                                'constant', constant_values=FILL_VALUE)  # change 255 to 0
        label_numpy = np.lib.pad(label_numpy,
                                 ((pad_h // 2, pad_h - pad_h // 2),
                                  (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
                                 'constant', constant_values=LABEL_FILL_VALUE)
        return data_numpy, label_numpy


class Resize:
    """ Resize the input numpy-array to given size, using PIL"""

    def __init__(self, size, interp_mode=Image.BILINEAR):
        assert isinstance(size, int) or (
            isinstance(size, Iterable) and len(size) == 2)
        self.size = size
        self.interp_mode = interp_mode

    def __call__(self, data_numpy):
        print("'Resize' is deprecated because of 'target_size' in dataio.py")


class CenterCrop:
    """ Crop the given numpy-array at the center """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, data_numpy, label_numpy):
        pad = Pad(target_height=self.size[0], target_width=self.size[1])
        data_numpy, label_numpy = pad(data_numpy, label_numpy)
        assert data_numpy.shape[1] >= self.size[1] and data_numpy.shape[0] >= self.size[0]
        x_left = data_numpy.shape[1] // 2 - self.size[1] // 2
        y_left = data_numpy.shape[0] // 2 - self.size[0] // 2

        return data_numpy[y_left: y_left + self.size[0], x_left: x_left + self.size[1]], \
            label_numpy[y_left: y_left + self.size[0],
                        x_left: x_left + self.size[1]]


class RandomCrop:
    """ Crop the given numpy-array at random position """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, data_numpy, label_numpy):
        pad = Pad(target_height=self.size[0], target_width=self.size[1])
        data_numpy, label_numpy = pad(data_numpy, label_numpy)
        assert data_numpy.shape[1] >= self.size[1] and data_numpy.shape[0] >= self.size[0]
        x = np.random.randint(0, data_numpy.shape[1] - self.size[1] + 1)
        y = np.random.randint(0, data_numpy.shape[1] - self.size[0] + 1)

        return data_numpy[y: y + self.size[0], x: x + self.size[1]], \
            label_numpy[y: y + self.size[0], x: x + self.size[1]]


def _flip(x, axis=0):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


class RandomHorizontalFlip:
    """ Randomly flip the numpy-arrays at the horizontal axis """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_numpy, label_numpy):
        if random.random() < self.p:
            return _flip(data_numpy, 1), _flip(label_numpy, 1)
        return data_numpy, label_numpy


class RandomVerticalFlip:
    """ Randomly flip the numpy-arrays at the vertical axis """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data_numpy, label_numpy):
        if random.random() < self.p:
            return _flip(data_numpy, 0), _flip(label_numpy, 0)
        return data_numpy, label_numpy


class RandomRotate:
    """ Randomly rotate the numpy-arrays within given angle """

    def __init__(self, angle_threshold, p=0.5):
        self.angle_threshold = angle_threshold
        self.p = p

    def __call__(self, data_numpy, label_numpy):
        if random.random() < self.p:
            angle = np.random.random_integers(-self.angle_threshold,
                                              self.angle_threshold)
            img_h, img_w, _ = data_numpy.shape
            M_rotate = cv2.getRotationMatrix2D(
                (img_w / 2, img_h / 2), angle, 1)
            data_numpy = cv2.warpAffine(data_numpy, M_rotate, (img_w, img_h))
            label_numpy = cv2.warpAffine(
                label_numpy, M_rotate, (img_w, img_h), flags=cv2.INTER_NEAREST)
            label_numpy = np.expand_dims(
                label_numpy, axis=-1) if label_numpy.ndim == 2 else label_numpy
        return data_numpy, label_numpy


class RandomScale:
    """ Randomly scale the numpy-arrays """

    def __init__(self, scale_range=(0.8, 1.2), p=0.5):
        self.scale_range = scale_range
        self.p = p

    def __call__(self, data_numpy, label_numpy):
        if random.random() < self.p:
            scale = np.random.random() * \
                (self.scale_range[1] - self.scale_range[0]) + \
                self.scale_range[0]
            img_h, img_w, _ = data_numpy.shape
            M_rotate = cv2.getRotationMatrix2D(
                (img_w / 2, img_h / 2), 0, scale)
            data_numpy = cv2.warpAffine(data_numpy, M_rotate, (img_w, img_h))
            label_numpy = cv2.warpAffine(
                label_numpy, M_rotate, (img_w, img_h), flags=cv2.INTER_NEAREST)
            label_numpy = np.expand_dims(
                label_numpy, axis=-1) if label_numpy.ndim == 2 else label_numpy
        return data_numpy, label_numpy


class RandomBlur:
    """ Randomly apply blur on the numpy-arrays """

    def __init__(self, kernel_size=(3, 3), p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, data_numpy, label_numpy):
        if random.random() < self.p:
            data_numpy = cv2.blur(data_numpy, self.kernel_size)
        return data_numpy, label_numpy


class RandomGammaTransform:
    """ Randomly apply gamma transform on the numpy-arrays """

    def __init__(self, gamma=0.5, p=0.5):
        self.gamma = gamma
        self.gamma_table = np.array(
            [int((x / 255.) ** self.gamma * 255) for x in range(256)])
        self.p = p

    def __call__(self, data_numpy, label_numpy):
        if random.random() < self.p:
            data_numpy = cv2.LUT(
                (data_numpy.clip(0, 1) * 255).astype(np.uint8), self.gamma_table) / 255.
        return data_numpy, label_numpy

# class ColorJiter:
#     """ apply color jilter to RGB images"""
#     def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
#         self.brightness = self._check_input(brightness, 'brightness')
#         self.contrast = self._check_input(contrast, 'contrast')
#         self.saturation = self._check_input(saturation, 'saturation')
#         self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
#                                      clip_first_on_zero=False)
#
#     def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
#         if isinstance(value, numbers.Number):
#             if value < 0:
#                 raise ValueError("If {} is a single number, it must be non negative.".format(name))
#             value = [center - value, center + value]
#             if clip_first_on_zero:
#                 value[0] = max(value[0], 0)
#         elif isinstance(value, (tuple, list)) and len(value) == 2:
#             if not bound[0] <= value[0] <= value[1] <= bound[1]:
#                 raise ValueError("{} values should be between {}".format(name, bound))
#         else:
#             raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
#
#         # if value is 0 or (1., 1.) for brightness/contrast/saturation
#         # or (0., 0.) for hue, do nothing
#         if value[0] == value[1] == center:
#             value = None
#         return value
#
#     @staticmethod
#     def get_params(brightness, contrast, saturation, hue):
#         transforms = []
#
#         if brightness is not None:
#             brightness_factor = random.uniform(brightness[0], brightness[1])
#             transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))
#
#         if contrast is not None:
#             contrast_factor = random.uniform(contrast[0], contrast[1])
#             transforms.append(Lambda(lambda img: Image(img, contrast_factor)))
#
#         if saturation is not None:
#             saturation_factor = random.uniform(saturation[0], saturation[1])
#             transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))
#
#         if hue is not None:
#             hue_factor = random.uniform(hue[0], hue[1])
#             transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))
#
#         random.shuffle(transforms)
#         transform = Compose(transforms)
#
#         return transform
#
#     def __call__(self, data_numpy, label_numpy):
#         transform = self.get_params(self.brightness, self.contrast,
#                                     self.saturation, self.hue)
#         return transform(data_numpy), label_numpy


# class RandomRotation:
#     def __init__(self, degree, p=0.5):
#         self.degree = degree
#         self.p = p

#     def __call__(self, data_numpy, label_numpy):
#         if random.random() < self.p:
#             rows, cols, _ = data_numpy.shape
#             mat = cv2.getRotationMatrix2D(
#                 ((cols-1)/2.0, (rows-1)/2.0), self.degree, 1)
#             data_numpy = cv2.warpAffine(data_numpy, mat, (cols, rows))
#             label_numpy = cv2.warpAffine(label_numpy, mat, (cols, rows))
#         return data_numpy, label_numpy


class Grayscale:
    """ Convert RGB img to grayscale """

    def __call__(self, data_numpy, label_numpy, rgb_rates=(0.299, 0.587, 0.114)):
        return data_numpy[:, :, 0] * rgb_rates[0] + data_numpy[:, :, 1] * rgb_rates[1]\
            + data_numpy[:, :, 2] * rgb_rates[2], label_numpy


class Lambda:
    """ Apply custom functions """

    def __init__(self, lamda):
        assert callable(lamda)
        self.lamda = lamda

    def __call__(self, data, label):
        return self.lamda(data, label)


class Compose:
    """ Compose multi augmentation methods"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data, label):
        for _t in self.transforms:
            data, label = _t(data, label)
        return data, label


class RandomApply:
    """ Randomly apply augmentation methods """

    def __init__(self, transforms, p=0.5):
        self.transforms = transforms
        self.p = p

    def __call__(self, data, label):
        for _t in self.transforms:
            if self.p < random.random():
                data, label = _t(data, label)
        return data, label
