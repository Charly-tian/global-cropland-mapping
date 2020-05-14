import tensorflow as tf
from keras import losses
from keras import backend as K

# USE_MASK = False  # mask spatial pixels according to their pixel values
# LABEL_SMOOTH = False  # use label smooth


# ========== loss / metrics for sparse labels ==========

def sbce(y_true, y_pred):
    """ sparse binary cross entropy for sparse labels

    # Args:
        y_true: (batch_size, height, width, 1).
        y_pred: (batch_size, height, width, 1).
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    return losses.binary_crossentropy(y_true_f, y_pred_f)


def scce(y_true, y_pred):
    """ sparse categorical cross entropy for sparse labels

    # Args:
        y_true: (batch_size, height, width, 1).
        y_pred: (batch_size, height, width, 1).
    """
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    return losses.sparse_categorical_crossentropy(y_true_f, y_pred_f)


# ========== loss / metrics for one-hot labels ==========


def cce(y_true, y_pred):
    """ categorical cross entropy for one-hot labels

    # Args:
        y_true: (batch_size, height, width, classes).
        y_pred: (batch_size, height, width, classes).
    """
    n_class = int(y_pred.shape[-1])
    y_true_f = tf.reshape(y_true, [-1, n_class])
    y_pred_f = tf.reshape(y_pred, [-1, n_class])

    return losses.categorical_crossentropy(y_true_f, y_pred_f)


def dc(y_true, y_pred):
    """ dice coefficient for one-hot labels """
    n_class = int(y_pred.shape[-1])
    y_true_f = tf.reshape(y_true, [-1, n_class])
    y_pred_f = tf.reshape(y_pred, [-1, n_class])
    coeff = (2 * tf.reduce_sum(y_true_f * y_pred_f, axis=0) + 1.) / \
        (tf.reduce_sum(y_true_f, axis=0) + tf.reduce_sum(y_pred_f, axis=0) + 1.)
    return tf.reduce_mean(coeff)


def dl(y_true, y_pred):
    """ dice loss for one-hot labels """
    return 1. - dc(y_true, y_pred)


def gdc(y_true, y_pred):
    """ generalised dice coefficient for one-hot labels """
    n_class = int(y_pred.shape[-1])
    y_true_f = tf.reshape(y_true, [-1, n_class])
    y_pred_f = tf.reshape(y_pred, [-1, n_class])
    w = tf.reduce_sum(y_true_f, axis=0)
    w = 1 / (w ** 2 + 1e-6)

    xn = tf.reduce_sum(w * tf.reduce_sum(y_true_f * y_pred_f, axis=0))
    xd = tf.reduce_sum(w * tf.reduce_sum(y_true_f + y_pred_f, axis=0))
    return 2 * xn / xd


def gdl(y_true, y_pred):
    """ generalised dice loss for one-hot labels """
    return 1 - gdc(y_true, y_pred)


def jc(y_true, y_pred):
    """ jaccard coefficient (IoU score) for one-hot labels """
    n_class = int(y_pred.shape[-1])
    y_true = tf.reshape(y_true, [-1, n_class])
    y_pred = tf.reshape(y_pred, [-1, n_class])
    intersection = tf.reduce_sum(y_true * y_pred, axis=0)
    iou = (intersection + 1) / (tf.reduce_sum(y_true + y_pred, axis=0) - intersection + 1)
    miou = tf.reduce_mean(iou)
    return miou


def jl(y_true, y_pred):
    """ jaccard loss (IoU loss) for one-hot labels """
    return 1 - jc(y_true, y_pred)


def cce_dl(y_true, y_pred):
    """ categorical cross entropy + dice loss for one-hot labels """
    return cce(y_true, y_pred) + dl(y_true, y_pred)


def cce_gdl(y_true, y_pred):
    """ categorical cross entropy + generalised dice loss for one-hot labels """
    return cce(y_true, y_pred) + gdl(y_true, y_pred)


def cce_jl(y_true, y_pred):
    """ categorical cross entropy + jaccard loss for one-hot labels """
    return cce(y_true, y_pred) + jl(y_true, y_pred)


def get_loss(loss_name):
    return globals().get(loss_name)
