import os
import random
import time
from datetime import datetime
import tensorflow as tf
import keras.backend as K
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

plt.rcParams["image.cmap"] = 'viridis'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.serif'] = 'DejaVu Serif'
plt.rcParams['axes.unicode_minus'] = False
FONT_SIZE = 10


def save_model_to_serving(model, export_path='savedmodel', export_version=None):
    """ Save the keras model as tensorflow savedmodel format for online cloud service.

    # Args:
        model: keras model instance.
            The keras model to be transformed.
        export_path: str.
            The export directory.
        export_version: str.
            The model version.
    """
    dt = datetime.now().strftime('%Y%m%d%H%M')
    if export_version is None:
        export_version = 'v' + dt

    inputs = {}
    for input_node in model.inputs:
        inputs[input_node.name] = input_node
    outputs = {}
    for output_node in model.outputs:
        outputs[output_node.name] = output_node
    print('input def mapping: ', inputs)
    print('input def mapping: ', outputs)

    signature = tf.saved_model.signature_def_utils.predict_signature_def(
        inputs=inputs, outputs=outputs)
    export_path = os.path.join(
        tf.compat.as_bytes(export_path),
        tf.compat.as_bytes(str(export_version)))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(
        sess=K.get_session(),
        tags=[tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'serving_default': signature,
        },
        legacy_init_op=legacy_init_op
    )
    builder.save()


def log(msg, level='info'):
    """ Print logs in the terminal.

    # Args:
        level: str.
            optional {'info', 'warn', 'error'}.
    """
    t = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('{} >>>> {}'.format(t, msg))


def create_train_val_txt(image_dir, val_rate=0.25):
    """ Split the image set into training & validation proportionally.

    # Args:
        image_dir: str.
            The directory that contains all the samples.
        val_rate: float, default 0.25.
            The proportion of validation set.
    """
    fnames = os.listdir(image_dir)
    base_dir = os.path.dirname(image_dir)
    with open(base_dir + '/train.txt', 'w', encoding='utf-8') as f_train:
        with open(base_dir + '/val.txt', 'w', encoding='utf-8') as f_val:
            for fname in fnames:
                if random.random() < val_rate:
                    f_val.write('%s\n' % fname.split('.')[0])
                else:
                    f_train.write('%s\n' % fname.split('.')[0])


def create_train_val_txt_by_keyword(image_dir, train_keyword='train', prefix=''):
    """ Split the image set into training & validation by keywords. Sample file names
     with 'train_keyword' will be categorized as training samples. 

    # Args:
        image_dir: str.
            The directory that contains all the samples.
        train_keyword: str, default 'train'.
            The keyword of training samples.
        prefix: str, default ''.
            The prefix of saved text files.
    """

    fnames = os.listdir(image_dir)
    base_dir = os.path.dirname(image_dir)
    with open(base_dir + '/{}train.txt'.format(prefix), 'w', encoding='utf-8') as f_train:
        with open(base_dir + '/{}val.txt'.format(prefix), 'w', encoding='utf-8') as f_val:
            for fname in fnames:
                if train_keyword in fname:
                    f_train.write('%s\n' % fname.split('.')[0])
                else:
                    f_val.write('%s\n' % fname.split('.')[0])


def plot_image_label(rgb_img, label_img, vmin, vmax, names, overlay=True):
    """ Plot a rgb image and a label image.

    # Args:
        rgb_img: 3-D array or a PIL instance.
            The RGB formatted image to plot.
        label_img: 2-D array.
            The label array to plot.
        vmin: int. 
            The minimum value of label values.
        vmax: int. 
            The maximum value of label values.
        names: 1-D array.
            The names of labels.
        overlay: bool, default True.
            Whether to overlay the label on the RGB image.
    """
    grid_spec = gridspec.GridSpec(1, 3, width_ratios=[6, 6, 1])

    # plot rgb image
    plt.subplot(grid_spec[0])
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title('image', fontdict={"fontsize": 12})

    # plot label image
    plt.subplot(grid_spec[1])
    if overlay:
        plt.imshow(rgb_img)
        plt.imshow(label_img, vmin=vmin, vmax=vmax, alpha=0.7)
    else:
        plt.imshow(label_img, vmin=vmin, vmax=vmax)
    plt.axis('off')
    plt.title('label', fontdict={"fontsize": 12})

    unique_labels = np.unique(label_img)
    FULL_LABEL_MAP = np.arange(len(names)).reshape(len(names), 1)
    ax = plt.subplot(grid_spec[2])
    plt.imshow(
        FULL_LABEL_MAP[unique_labels].astype(np.uint8), interpolation='nearest', vmin=vmin, vmax=vmax)
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), np.array(names)[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')

    plt.show()


def plot_rgb_image(image, label):
    """ Plot an image and the corresponding label.

    # Args:
        img: np.ndarray.
            The image to plot.
        label_img: 2-D array.
            The label array to plot.
    """
    plt.subplot(1, 2, 1)
    plt.imshow((image[:, :, np.array([2, 1, 0])]), vmin=0, vmax=1)
    plt.subplot(1, 2, 2)
    plt.imshow(label)
    plt.show()


def plot_image_label_per_channel(img, label):
    """ Plot each channel of a multi-channel image and the label.

    # Args:
        img: np.ndarray.
            The image to plot.
        label_img: 2-D array.
            The label array to plot.
    """
    img_channels = img.shape[-1]
    n_subfig = img_channels + 1
    n_row = int(np.sqrt(n_subfig))
    n_col = np.ceil(n_subfig / n_row).astype(np.int)
    grid_spec = gridspec.GridSpec(n_row, n_col)
    for i_channel in range(img_channels):
        plt.subplot(grid_spec[i_channel])
        plt.imshow(img[:, :, i_channel])
        plt.colorbar()
        plt.axis('off')
        plt.title('Band {}'.format(i_channel + 1),
                  fontdict={'fontsize': FONT_SIZE})
    plt.subplot(grid_spec[img_channels])
    plt.imshow(label)
    plt.colorbar()
    plt.title('label')
    plt.axis('off')
    plt.show()


# def plot_batch_pred_label(preds, labels):
#     bs = preds.shape[0]
#     for i in range(bs):
#         plt.subplot(2, bs, i + 1)
#         plt.imshow(labels[i, :, :, 0])
#         plt.axis('off')
#         plt.subplot(2, bs, bs + i + 1)
#         plt.imshow(preds[i, :, :, 0])
#         plt.axis('off')
#     plt.show()

