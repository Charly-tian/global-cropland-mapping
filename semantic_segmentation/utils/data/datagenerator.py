import numpy as np
import tensorflow as tf
from .dataio import color_to_index


def data_generator(dataset, batch_size=1, replacement=False, shuffle=False, drop_last=False):
    """ Continuously generate batch of data & labels for training / validation

    # Args:
        dataset: class.Dataset.
            Dataset object with attributes of `__getitem__` and `__len__`.
        batch_size : int, default 1.
            Size of examples in each mini-batch.
        replacement: bool, default False. 
            Whether to sample with replacement.
        shuffle: bool, default False.
            Whether to shuffle the sequence of samples. recommend False for
            validation set and True for training set.
        drop_last: bool, default False.
            Whether to drop the last mini-batch.

    # Returns: 
        tuple of (data, label), with size of (batch_size, height, width, channel)
    """
    n = len(dataset)
    if shuffle:
        if replacement:
            sampler_indices = np.random.randint(
                low=0, high=n, size=(n,), dtype=np.int64)
        else:
            sampler_indices = np.random.permutation(n)
    else:
        sampler_indices = range(n)

    while True:
        indices = []
        for idx in sampler_indices:
            indices.append(idx)
            if len(indices) == batch_size:
                batch = [dataset[i] for i in indices]
                data_batch = [_batch[0] for _batch in batch]
                label_batch = [_batch[1] for _batch in batch]
                yield (np.array(data_batch), np.array(label_batch))
                indices = []
        # if len(indices) > 0 and not drop_last:
        #     batch = [dataset[i] for i in indices]
        #     data_batch = [_batch[0] for _batch in batch]
        #     label_batch = [_batch[1] for _batch in batch]
        #     yield (np.array(data_batch), np.array(label_batch))
        #     indices = []


# def tfrecord_data_generator(dataset, with_sample_weights=False, min_t=0.3, max_t=0.7):
#     transforms = dataset.transforms
#     while True:
#         data = dataset.iterator.get_next()
#         data = tf.keras.backend.eval(data)
#         _data, _labels = data[:, :, :, :-1], data[:, :, :, -1:]
#         del data
#         if dataset.label_one_hot:
#             _labels = np.stack([color_to_index(_label, range(dataset.n_class), one_hot=dataset.label_one_hot)
#                                 for _label in _labels], axis=0)
#         _labels = np.where(_labels >= 0.5, 1, 0)
#         if transforms is not None:
#             _data, _labels = transforms(_data, _labels)
#         _sample_weights = None
#         if with_sample_weights:
#             _sample_weights = np.where(np.logical_or(
#                 _labels > max_t, _labels < min_t), 1, 0)
#         yield _data, _labels, _sample_weights
