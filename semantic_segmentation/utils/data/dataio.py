from PIL import Image
import numpy as np
from osgeo import gdal
import cv2


def load_gray_img(fn, dtype='int8', scale=1.0, target_size=None, interp_mode=Image.NEAREST):
    """ Load gray images using PIL.

    # Args:
        fn: str.
            File name of the input image.
        dtype: str, default `int8`.
            Data type of the input image, optional {'int8', 'int32', 'float32'}.
        scale: float, default 1.0.
            The Scale for image data values.
        target_size: tuple, default None.
            If set to None, the original image shape will remain. Otherwise, the image will
            be resized to the specific size.
        interp_mode: enumerate, default Image.NEAREST.
            Interpolation mode when applying resizing, optional {Image.LINEAR, Image.NEAREST,
            Image.BILINEAR, Image.ADAPTIVE, Image.AFFINE}.

    # Returns:
        np.ndarray with shape of (height, width, 1).
    """
    assert dtype in {'int8', 'int32', 'float32'},\
        "'dtype' must be one of {'int8', 'int32', 'float32'}, but got '%s'" % dtype
    assert scale != 0
    assert interp_mode in {Image.LINEAR, Image.NEAREST,
                           Image.BILINEAR, Image.ADAPTIVE, Image.AFFINE, None}
    mode_map = {'int8': 'L', 'int32': 'I', 'float32': 'F'}
    img_pil = Image.open(fn).convert(mode_map[dtype])
    if target_size is not None:
        img_pil = img_pil.resize(target_size, interp_mode)
    data_numpy = np.expand_dims(
        np.array(img_pil) * scale, axis=-1)  # (H, W, 1)
    return data_numpy


def load_rgb_img(fn, scale=1.0, target_size=None, interp_mode=Image.NEAREST):
    """ Load rgb images using PIL.

    # Args:
        fn: str.
            File name of the input image.
        dtype: str, default `int8`.
            Data type of the input image, optional {'int8', 'int32', 'float32'}.
        scale: float, default 1.0.
            The Scale for image data values.
        target_size: tuple, default None.
            If set to None, the original image shape will remain. Otherwise, the image will
            be resized to the specific size.
        interp_mode: enumerate, default Image.NEAREST.
            Interpolation mode when applying resizing, optional {Image.LINEAR, Image.NEAREST,
            Image.BILINEAR, Image.ADAPTIVE, Image.AFFINE}.

    # Returns:
        np.ndarray with shape of (height, width, 3).
    """
    assert scale != 0
    assert interp_mode in {Image.LINEAR, Image.NEAREST,
                           Image.BILINEAR, Image.ADAPTIVE, Image.AFFINE, None}
    img_pil = Image.open(fn).convert('RGB')
    if target_size is not None:
        img_pil = img_pil.resize(target_size, interp_mode)
    data_numpy = np.array(img_pil) * scale
    return data_numpy


def _gdal_load_tif_prop(fn):
    """ Load geotiff image properties.

    # Args:
        fn: str.
            File name of the input image.

    # Returns:
        dict of image properties.
    """
    ds = gdal.Open(fn, gdal.GA_ReadOnly)
    bands, rows, cols = ds.RasterCount, ds.RasterYSize, ds.RasterXSize
    geo_transform, projection = ds.GetGeoTransform(), ds.GetProjection()
    data_type, no_data = ds.GetRasterBand(
        1).DataType, ds.GetRasterBand(1).GetNoDataValue()
    props = {
        'bands': bands,
        'cols': cols,
        'rows': rows,
        'geo_transform': geo_transform,
        'projection': projection,
        'data_type': data_type,
        'no_data': no_data
    }
    del ds
    return props


def _gdal_load_tif_data(fn):
    """ Load geotiff image data matrix.

    # Args:
        fn: str.
            File name of the input image.

    # Returns:
        np.ndarray with shape of (height, width, channel).
    """
    ds = gdal.Open(fn, gdal.GA_ReadOnly)
    bands, rows, cols = ds.RasterCount, ds.RasterYSize, ds.RasterXSize
    data_type = ds.GetRasterBand(1).DataType
    dtype = np.uint8 if data_type == 1 else np.float
    data = np.zeros((rows, cols, bands), dtype=dtype)
    for i in range(bands):
        dt = ds.GetRasterBand(i + 1)
        data[:, :, i] = dt.ReadAsArray(0, 0, cols, rows).astype(dtype)
    del ds
    return data


def _gdal_load_tif_img(fn):
    """ Load geotiff image data matrix and properties.

    # Args:
        fn: str.
            File name of the input image.

    # Returns:
        np.ndarray with shape of (height, width, channel).
        dict of image properties.
    """
    ds = gdal.Open(fn, gdal.GA_ReadOnly)
    bands, rows, cols = ds.RasterCount, ds.RasterYSize, ds.RasterXSize
    geo_transform, projection = ds.GetGeoTransform(), ds.GetProjection()
    data_type, no_data = ds.GetRasterBand(
        1).DataType, ds.GetRasterBand(1).GetNoDataValue()
    props = {
        'bands': bands,
        'cols': cols,
        'rows': rows,
        'geo_transform': geo_transform,
        'projection': projection,
        'data_type': data_type,
        'no_data': no_data
    }
    dtype = np.uint8 if data_type == 1 else np.float
    data = np.zeros((rows, cols, bands), dtype=dtype)
    for i in range(bands):
        dt = ds.GetRasterBand(i + 1)
        data[:, :, i] = dt.ReadAsArray(0, 0, cols, rows).astype(dtype)
    del ds
    return data, props


def load_multi_channel_img(fn, scale=1.0, target_size=None, interp_mode=cv2.INTER_NEAREST):
    """ Load multi-spectral images using gdal+cv2.

    # Args:
        fn: str.
            File name of the input image.
        scale: float, default 1.0.
            The Scale for image data values.
        target_size: tuple, default None.
            If set to None, the original image shape will remain. Otherwise, the image will
            be resized to the specific size.
        interp_mode: enumerate, default cv2.INTER_NEAREST.
            Interpolation mode when applying resizing, optional {cv2.INTER_NEAREST, 
            cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}.

    # Returns:
        np.ndarray with shape of (height, width, channel).

    """
    assert scale != 0
    assert interp_mode in {cv2.INTER_NEAREST, cv2.INTER_LINEAR,
                           cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4}
    data_numpy = _gdal_load_tif_data(fn)
    if target_size is not None:
        res = np.zeros(shape=target_size +
                       (data_numpy.shape[2],), dtype=data_numpy.dtype)
        for i in range(data_numpy.shape[2]):
            res[:, :, i] = cv2.resize(data_numpy[:, :, i], dsize=(
                target_size[1], target_size[0]), interpolation=interp_mode)
        return res * scale
    return data_numpy * scale


def color_to_index(color_array, color_mapping, one_hot=True):
    """ Convert colorful label array to label index

    # Args:
        color_array: np.ndarray with shape of (height, width) or (height, width, 3).
            RGB or gray label array, where the pixel value is not the label index but the 
            rendering color.
        color_mapping: np.ndarray of shape (n_class,) or (n_class, 3), where n_class is the
            [total] class number.
            Note: the first element is the color of background (label 0).
            e.g., if colour_mapping=[0, 255], pixel equal to 255 are assigned with 1, otherwise 0.
            if colour_mapping=[[0, 0, 0], [255,255,255]], pixel equal to [255, 255, 255] are
            assigned with 1, otherwise 0.
        to_sparse: bool, default True.
            Whether to apply a argmax on the last axis to obtain sparse label array

    # Returns: 
        np.ndarray with shape of (height, width, n_class+1)
        Note: if a pixel value pair is not in the color_mapping, the value of that pixel in the 
        final one-hot array will be [0, 0, ..., 0]
    """
    assert color_mapping is not None
    if len(color_mapping) < 2:
        raise ValueError(
            "Invalid length of color map: {}. Expected >= 2!".format(len(color_mapping)))

    onehot_array = [
        np.zeros((color_array.shape[0], color_array.shape[1]), dtype=np.uint8)]
    for color in color_mapping[1:]:
        _equal = np.equal(color_array, color)
        onehot_array.append(np.all(_equal, axis=-1).astype(np.uint8))
    onehot_array = np.stack(onehot_array, axis=-1).astype(np.uint8)

    # if the color is not in the colour_mapping, assign 0 to represent background
    all_zeros = np.zeros(len(color_mapping), dtype=np.uint8)
    onehot_array[:, :, 0] = np.where(
        np.all(np.equal(onehot_array, all_zeros), axis=-1), 1, 0)

    if not one_hot:
        onehot_array = np.argmax(onehot_array, axis=-1).astype(np.uint8)
    return onehot_array


def index_to_color(label_array, color_mapping):
    """ Encode the 2-dim label array to colorful images.
    # Args:
        label_array: np.ndarray with shape of (height, width).
            The label index array.
        color_mapping: np.ndarray with shape of (n_class,) or (n_class, 3).
            Refer to the one in function 'color_to_label'

    # Returns: 
        np.ndarray of shape(height, width, 3) or (height, width), depending on the 
        dimension of color_mapping.
    """
    assert color_mapping is not None
    assert label_array.ndim == 2

    color_mapping = np.array(color_mapping)
    if color_mapping.ndim == 1 or (color_mapping.ndim == 2 and color_mapping.shape[1] == 3):
        return color_mapping[label_array.astype(np.uint8)]
    else:
        raise ValueError("Invalid color_mapping shape: {}. Expected to be (n,) or (n, 3)".format(
            color_mapping.shape))
