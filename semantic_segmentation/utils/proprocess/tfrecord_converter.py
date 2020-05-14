import os
import h5py
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm, tqdm_notebook
from collections import OrderedDict
from .. import plot_image_label_per_channel


def get_TFRecord_feature_def(tfrecord_file, gzip=True):
    """ Get the feature definition of a tfrecord file.
    E.g.:
    ```
        path = 'D:/test.tfrecord.gz'
        get_TFRecord_feature_def(path)
    ```

    # Args:
        tfrecord_file: str.
            Absolute name of tfrecord file.
        gzip: bool, default True.
            Whether the tfrecord is compressed with gzip.
    """
    with tf.Session() as sess:
        if gzip:
            ds = tf.data.TFRecordDataset(
                [tfrecord_file], compression_type='GZIP')
        else:
            ds = tf.data.TFRecordDataset([tfrecord_file])
        ds = ds.batch(10)
        ds = ds.prefetch(buffer_size=100)
        iterator = ds.make_one_shot_iterator()

        batch_data = iterator.get_next()
        serialized_example = sess.run(batch_data)[0]
        example_proto = tf.train.Example.FromString(serialized_example)
        features = example_proto.features
        for key in features.feature:
            feature = features.feature[key]
            ftype = None
            if len(feature.bytes_list.value) > 0:
                ftype = 'bytes_list'
                fvalue = feature.bytes_list.value

            if len(feature.float_list.value) > 0:
                ftype = 'float_list'
                fvalue = feature.float_list.value

            if len(feature.int64_list.value) > 0:
                ftype = 'int64_list'
                fvalue = feature.int64_list.value

            result = '{0} : {1}'.format(key, ftype)
            print(result)


def _extract_data_label(features, in_size, out_size, feature_names, label_names):
    """ Extract data and label from the parsed features.

    # Args:
        features: dict.
            Features parsed from the TFRecord Dataset.
        in_size: tuple.
            The original size of data values in the features.
        out_size: tuple.
            The target size of data values.
        feature_names: list of strings.
            Keys that define the data features.
        label_names: list of strings.
            Keys that define the label features.

    # Returns:
        data: np.ndarray. 
            Extracted data array, or None.
        label: np.ndarray.
            Extracted label array, or None.
    """
    _ = (in_size[0] - out_size[0]) // 2
    if len(feature_names) == 0:
        data = None
    else:
        data = np.stack([features[k][_: _ + out_size[0], _: _ + out_size[1]]
                         for k in feature_names], axis=-1)
    if len(label_names) == 0:
        label = None
    else:
        label = np.stack([features[k][_: _ + out_size[0], _: _ + out_size[1]]
                          for k in label_names], axis=-1)
    return data, label


def TFRecord_to_H5(tfrecord_filenames, dst_h5_dir, in_size, out_size, feature_names, label_names):
    """ Convert TFRecord files to H5 files.
    E.g.:
    ```
        tfrecord_filenames = os.listdir('D:/test')
        dst_dir = 'D:/dst'
        filenames = [fn for fn in tfrecord_filenames if 'val' in fn]
        TFRecord_to_H5(filenames, dst_dir, in_size=(129, 129), out_size=(128, 128),
                       feature_names=['B', 'G', 'R', 'NIR', 'NDVI'], label_names=['Cropland'])
    ```
    # Args:
        tfrecord_filenames: list of strings.
            Absolute file names of tfrecord files.
        dst_h5_dir: str.
            Folder that contains the destination h5 files.
        in_size: tuple.
            The original size of data values in the features.
        out_size: tuple. 
            The destination size of data values.
        feature_names: list of strings.
            Keys that define the data features.
        label_names: list of strings. 
            Keys that define the label features.
    """
    if not os.path.exists(dst_h5_dir):
        os.makedirs(dst_h5_dir, exist_ok=True)

    features = dict()
    features['ID'] = tf.FixedLenFeature([], tf.float32)  # point id
    features['GridID'] = tf.FixedLenFeature([], tf.float32)  # grid id
    # features['CZID'] = tf.FixedLenFeature([], tf.float32)
    for data_feature_name in feature_names:
        features[data_feature_name] = tf.FixedLenFeature(in_size, tf.float32)
    for label_feature_name in label_names:
        features[label_feature_name] = tf.FixedLenFeature(in_size, tf.float32)
    assert out_size[0] <= in_size[0] and out_size[1] <= in_size[1]

    def _parse_feature(example_proto):
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features

    with tf.Session() as sess:
        dataset = tf.data.TFRecordDataset(
            tfrecord_filenames, compression_type='GZIP')
        dataset = dataset.map(_parse_feature, num_parallel_calls=4)
        iterator = dataset.make_one_shot_iterator()

        try:
            item = iterator.get_next()
            while True:
                parsed_features = sess.run(item)
                point_id, grid_id = int(parsed_features['ID']), int(
                    parsed_features['GridID'])
                dst_h5_file = os.path.join(
                    dst_h5_dir, '{}-{}.h5'.format(grid_id, point_id))
                # if os.path.exists(dst_h5_file):
                #     continue
                data, label = _extract_data_label(
                    parsed_features, in_size, out_size, feature_names, label_names)
                if True:
                    with h5py.File(dst_h5_file, mode='w') as f:
                        if data is not None:
                            f.create_dataset(
                                'data', data=data, compression='gzip')
                        if label is not None:
                            f.create_dataset(
                                'label', data=label, compression='gzip')
                del parsed_features, data, label
        except tf.errors.OutOfRangeError:
            pass
        except Exception as e:
            pass
            # print(e)
        del dataset, iterator
    sess.close()


def get_data_description(data_dir, feature_names, label_names, save_file):
    """ Statistic the mean & std of each channel and save to file.

    # Args:
        data_dir: str. 
            Folder that contains all the files to be computed.
        feature_names: list of strings.
            Keys that define the data features.
        label_names: list of strings. 
            Keys that define the label features.
        save_file: str.
            Destination file that save the statistics.
    """
    if save_file is None:
        save_file = data_dir + '.csv'
    data_records = []
    h5_filenames = os.listdir(data_dir)
    for h5_filename in tqdm(h5_filenames):
        record = OrderedDict()
        _ = h5_filename.split('-')
        grid_id, point_id = _[0], _[1][:-3]
        record['GridID'] = grid_id
        record['PointID'] = point_id
        try:
            with h5py.File(os.path.join(data_dir, h5_filename), 'r') as f:
                if len(feature_names) > 0:
                    data = f['data'][:]
                    data_mn = data.reshape(-1, len(feature_names)
                                           ).mean(axis=0).astype(np.str).tolist()
                    for k, v in zip(feature_names, data_mn):
                        record[k] = v
                if len(label_names) > 0:
                    label = f['label'][:]
                    label_mn = label.reshape(-1, len(label_names)
                                             ).mean(axis=0).astype(np.str).tolist()
                    for k, v in zip(label_names, label_mn):
                        record[k] = v
            data_records.append(record)
        except:
            print(h5_filename)
    pd.DataFrame.from_dict(data_records, orient='columns').to_csv(
        save_file, index=False)


def validate_H5(fn):
    """ test whether the converted h5 file is readable """
    with h5py.File(fn, 'r') as f:
        data = f['data'][:]
        label = f['label'][:]
        print(data.shape, label.shape)
        plot_image_label_per_channel(data, label[:, :, 0])


# get_data_description(data_dir='G:\crop\samples/20191231_128x128x4_uscan\h5_tmp', feature_names=['B', 'G', 'R', 'NIR'], label_names=['Cropland'], save_file=None)
