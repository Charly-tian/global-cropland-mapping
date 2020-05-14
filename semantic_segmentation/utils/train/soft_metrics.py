"""
y_true and y_pred are both probabilities
`soft_{}` means the y_true is a probability distributed tensor

"""

import tensorflow as tf
from keras import losses
from keras import backend as K

USE_MASK = False
LABEL_SMOOTH = False


def _masked_tensor(y_true, y_pred, value_gt=0.25, value_lt=0.75):
    """ mask tensors that between `value_gt` and `value_lt` """
    mask = tf.where(tf.logical_or(y_true >= value_gt, y_true <= value_lt),
                    tf.ones_like(y_true), tf.zeros_like(y_true))
    y_true = tf.boolean_mask(y_true, mask)
    y_pred = tf.boolean_mask(y_pred, mask)
    return y_true, y_pred


def bce(y_true, y_pred):
    # if USE_MASK:
    #     y_true, y_pred = _masked_tensor(y_true, y_pred)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    # if LABEL_SMOOTH:
    #     y_true_f = tf.clip_by_value(y_true_f, 0.05, 0.95)
    # y_true = tf.where(y_true >= 0.5, tf.ones_like(y_true), tf.zeros_like(y_true))
    return losses.binary_crossentropy(y_true_f, y_pred_f)


def dice_coeff(y_true, y_pred):
    # if USE_MASK:
    #     y_true, y_pred = _masked_tensor(y_true, y_pred)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    # if LABEL_SMOOTH:
    #     y_true_f = tf.clip_by_value(y_true_f, 0.05, 0.95)
    # y_true = tf.where(y_true >= 0.5, tf.ones_like(y_true), tf.zeros_like(y_true))
    coeff = (2 * tf.reduce_sum(y_true_f * y_pred_f) + 1.) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + 1.)
    return coeff


def dice_loss(y_true, y_pred):
    return 1. - dice_coeff(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)


def generalized_dice_coeff(y_true, y_pred):
    # if LABEL_SMOOTH:
    #     y_true = tf.clip_by_value(y_true, 0.05, 0.95)
    y_true = tf.concat((1 - y_true, y_true), axis=-1)
    y_pred = tf.concat((1 - y_pred, y_pred), axis=-1)
    w = K.sum(y_true, axis=(0, 1, 2))
    w = 1 / (w ** 2 + 1e-6)
    xn = K.sum(w * K.sum(y_true * y_pred, (0, 1, 2)))
    xd = K.sum(w * K.sum(y_true + y_pred, (0, 1, 2)))
    gen_dice_coef = 2 * xn / xd
    return gen_dice_coef


def generalized_dice_loss(y_true, y_pred):
    return 1 - generalized_dice_coeff(y_true, y_pred)


def bce_gdice_loss(y_true, y_pred):
    return bce(y_true, y_pred) + generalized_dice_loss(y_true, y_pred)


def iou_score(y_true, y_pred):
    # if USE_MASK:
    #     y_true, y_pred = _masked_tensor(y_true, y_pred)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    # if LABEL_SMOOTH:
    #     y_true = tf.clip_by_value(y_true, 0.05, 0.95)
    # y_true = tf.where(y_true >= 0.5, tf.ones_like(y_true), tf.zeros_like(y_true))
    intersection = tf.reduce_sum(y_true * y_pred)
    iou = (intersection + 1) / (tf.reduce_sum(y_true + y_pred) - intersection + 1)

    return iou


def iou_loss(y_true, y_pred):
    return 1 - iou_score(y_true, y_pred)


def bce_iou_loss(y_true, y_pred):
    return bce(y_true, y_pred) + 2 * iou_loss(y_true, y_pred)


def iou_dice_loss(y_true, y_pred):
    return iou_loss(y_true, y_pred) + dice_loss(y_true, y_pred)


def bce_iou_dice_loss(y_true, y_pred):
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred) + iou_loss(y_true, y_pred)


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.):
    # if USE_MASK:
    #     y_true, y_pred = _masked_tensor(y_true, y_pred)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    y_true = tf.where(y_true >= 0.5, tf.ones_like(y_true), tf.zeros_like(y_true))

    y_pred = tf.clip_by_value(y_pred, 1e-12, 1. - (1e-12))
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    alpha = 0.85
    return -tf.reduce_sum(alpha * tf.pow(1. - pt_1, gamma) * tf.log(pt_1)) \
           - tf.reduce_sum((1 - alpha) * tf.pow(pt_0, gamma) * tf.log(1. - pt_0))


# # --------------------------- BINARY LOSSES ---------------------------
# def _lovasz_grad(gt_sorted):
#     """
#     Computes gradient of the Lovasz extension w.r.t sorted errors
#     See Alg. 1 in paper
#     """
#     gts = tf.reduce_sum(gt_sorted)
#     intersection = gts - tf.cumsum(gt_sorted)
#     union = gts + tf.cumsum(1. - gt_sorted)
#     jaccard = 1. - intersection / union
#     jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
#     return jaccard
#
#
# def lovasz_hinge(labels, y_pred, per_image=True, ignore=None):
#     """
#     Binary Lovasz hinge loss
#       logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
#       labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
#       per_image: compute the loss per image instead of per batch
#       ignore: void class id
#     """
#     def convert_to_logits(y_pred):
#         # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
#         y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
#         return tf.log(y_pred / (1 - y_pred))
#
#     logits = convert_to_logits(y_pred)
#     if per_image:
#         def treat_image(log_lab):
#             log, lab = log_lab
#             log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
#             log, lab = _flatten_binary_scores(log, lab, ignore)
#             return _lovasz_hinge_flat(log, lab)
#
#         losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
#
#         # Fixed python3
#         losses.set_shape((None,))
#
#         loss = tf.reduce_mean(losses)
#     else:
#         loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
#     return loss
#
#
# def _lovasz_hinge_flat(logits, labels):
#     """
#     Binary Lovasz hinge loss
#       logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
#       labels: [P] Tensor, binary ground truth labels (0 or 1)
#       ignore: label to ignore
#     """
#
#     def compute_loss():
#         labelsf = tf.cast(labels, logits.dtype)
#         signs = 2. * labelsf - 1.
#         errors = 1. - logits * tf.stop_gradient(signs)
#         errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
#         gt_sorted = tf.gather(labelsf, perm)
#         grad = _lovasz_grad(gt_sorted)
#         # loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
#         # ELU + 1
#         loss = tf.tensordot(tf.nn.elu(errors_sorted) + 1., tf.stop_gradient(grad), 1, name="loss_non_void")
#         return loss
#
#     # deal with the void prediction case (only void pixels)
#     loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
#                    lambda: tf.reduce_sum(logits) * 0.,
#                    compute_loss,
#                    strict=True,
#                    name="loss"
#                    )
#     return loss
#
#
# def _flatten_binary_scores(scores, labels, ignore=None):
#     """
#     Flattens predictions in the batch (binary case)
#     Remove labels equal to 'ignore'
#     """
#     scores = tf.reshape(scores, (-1,))
#     labels = tf.reshape(labels, (-1,))
#     if ignore is None:
#         return scores, labels
#     valid = tf.not_equal(labels, ignore)
#     vscores = tf.boolean_mask(scores, valid, name='valid_scores')
#     vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
#     return vscores, vlabels
#
#
# def mse(y_true, y_pred):
#     # http://ddrv.cn/a/331925
#     return K.mean(K.square(y_pred - y_true), axis=-1)


def get_soft_loss_metric(name):
    return globals().get(name)

