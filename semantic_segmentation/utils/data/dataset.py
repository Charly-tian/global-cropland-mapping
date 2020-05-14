import os
import h5py
import numpy as np
from ..data.dataio import load_gray_img, load_rgb_img, load_multi_channel_img, color_to_index

#
# class SegmentationDataset:
#     def __init__(self, img_root, label_root, img_suffix, label_suffix, fn_txt, n_class,
#                  data_color_mode='rgb', label_mode='color', label_one_hot=True,
#                  label_color_map=None, dtype='int8', value_scale=1.0, target_size=None, transforms=None):
#         """
#         :param img_root:
#         :param label_root:
#         :param img_suffix:
#         :param label_suffix:
#         :param fn_txt:
#         :param data_color_mode: 'gray', 'rgb', 'multi_channel'
#         :param label_mode: 'color' or 'index'
#         :param label_color_map: dict of color maps
#         :param transforms:
#         """
#         self.img_root = img_root
#         self.label_root = label_root
#         with open(fn_txt, 'r') as f:
#             fns = [line.strip() for line in f]
#         self.data_fns = [os.path.join(img_root, fn + img_suffix) for fn in fns]
#         self.label_fns = [os.path.join(label_root, fn + label_suffix) for fn in fns]
#
#         self.n_class = n_class
#         self.data_color_mode = data_color_mode
#         self.label_mode = label_mode
#         self.label_color_map = label_color_map
#         self.label_one_hot = label_one_hot
#         self.dtype = dtype
#         self.value_scale = value_scale
#         self.target_size = target_size
#         self.transforms = transforms
#
#     def __getitem__(self, index):
#         data_path = self.data_fns[index]
#         label_path = self.label_fns[index]
#
#         if self.data_color_mode == 'gray':
#             data = load_gray_img(data_path, self.dtype, self.value_scale, self.target_size)
#         elif self.data_color_mode == 'rgb':
#             data = load_rgb_img(data_path, self.value_scale, self.target_size)
#         elif self.data_color_mode == 'multi_channel':
#             data = load_multi_channel_img(data_path, self.value_scale, self.target_size)
#         else:
#             raise ValueError("data_color_mode must be in ['gray', 'rgb', 'multi_channel'], but got {}"
#                              .format(self.data_color_mode))
#
#         if self.label_mode == 'index':
#             label = load_gray_img(label_path, target_size=self.target_size)
#             if self.label_one_hot:
#                 label = color_to_index(label, range(self.n_class), one_hot=True)  # here, index is color.
#         elif self.label_mode == 'color':
#             assert self.label_color_map is not None and self.label_one_hot, "'label_color_map' should not be None when 'label_mode' is 'color'"
#             if isinstance(self.label_color_map[0], int):
#                 label = load_gray_img(label_path, target_size=self.target_size)
#             else:
#                 label = load_rgb_img(label_path, target_size=self.target_size)
#             label = color_to_index(label, self.label_color_map, one_hot=self.label_one_hot)  # one hot
#             # print(data_path, label_path)
#         else:
#             raise ValueError("label_mode must be in ['color', 'index'], but got {}"
#                              .format(self.label_mode))
#
#         if self.transforms is not None:
#             data, label = self.transforms(data, label)
#         return data, label
#
#     def __len__(self):
#         return len(self.data_fns)


class H5SegmentationDataset:
    """ Parse data array and label array from a H5 file for segmentation. Note that the H5 file
     contains keys of 'data' and 'label'.

    """

    def __init__(self, root, fn_txt, n_class, label_prob_to_cls=True,
                 label_one_hot=False, transforms=None):
        """
        # Args:
            root: str.
                Directory of *.h5 files.
            fn_txt: str.
                Text file that records the h5 file names (prefix, exclude the file format suffix, i.e., `.h5`)
            n_class: int.
                Number of label classes.
            label_prob_to_cls: bool.
                Whether to convert the probability of labels into categorical classes.
            label_one_hot: bool.
                Whether to encode the label with class index into one-hot array.
            transforms: list of transforms, or None.
                The transformations that performed on the data/label arrays.
        """
        with open(fn_txt, 'r') as f:
            fns = [line.strip() + '.h5' for line in f]
        root_fns = os.listdir(root)
        fns = [fn for fn in fns if fn in root_fns]
        self.fns = [os.path.join(root, fn) for fn in fns]
        self.n_class = n_class
        self.label_prob_to_cls = label_prob_to_cls
        self.label_one_hot = label_one_hot
        self.transforms = transforms

    def __getitem__(self, index):
        fn = self.fns[index]
        with h5py.File(fn, 'r') as f:
            data, label = f['data'][:], f['label'][:]
        # convert the probability of labels into categorical classes using a threshold of 0.5
        # if self.label_prob_to_cls:
        #     label = np.where(label >= 0.5, 1, 0)
        # encode the label index into one-hot array
        if self.label_one_hot:
            label = color_to_index(label, range(self.n_class), one_hot=True)
        # perform transformations
        if self.transforms is not None:
            data, label = self.transforms(data, label)
        return data, label

    def __len__(self):
        return len(self.fns)


# class MultiH5SegmentationDataset:
#     """ segementation dataset for multi inputs with *h5 files """

#     def __init__(self, root, aux_root, fn_txt, n_class, label_prob_to_cls=True,
#                  label_one_hot=False, transforms=None):
#         with open(fn_txt, 'r') as f:
#             fns = [line.strip() + '.h5' for line in f]
#         aux_fns = os.listdir(aux_root)
#         fns = [fn for fn in fns if fn in aux_fns]
#         self.main_fns = [os.path.join(root, fn) for fn in fns]
#         self.aux_fns = [os.path.join(aux_root, fn) for fn in fns]
#         self.n_class = n_class
#         self.label_prob_to_cls = label_prob_to_cls
#         self.label_one_hot = label_one_hot
#         self.transforms = transforms
#         assert len(self.main_fns) == len(self.aux_fns)

#     def __getitem__(self, idx):
#         main_fn = self.main_fns[idx]
#         aux_fn = self.aux_fns[idx]
#         with h5py.File(main_fn, 'r') as f:
#             data, label = f['data'][:], f["label"][:]
#         with h5py.File(aux_fn, 'r') as f:
#             aux_data = f["data"][:]
#         data = np.concatenate((data, aux_data), axis=-1)
#         # convert the probability of labels into categorical classes using a threshold of 0.5
#         # if self.label_prob_to_cls:
#         #     label = np.where(label >= 0.5, 1, 0)
#         # encode the label index into one-hot array
#         # if self.label_one_hot:
#         #     label = color_to_index(label, range(self.n_class), one_hot=True)
#         # perform transformations
#         if self.transforms is not None:
#             data, label = self.transforms(data, label)
#         return data, label

#     def __len__(self):
#         return len(self.main_fns)
