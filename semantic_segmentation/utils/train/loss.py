import tensorflow as tf
from keras import backend as K
from keras.losses import binary_crossentropy, categorical_crossentropy
from .score import jaccard_iou_score
from .score import dice_score
from .score import SMOOTH


def masked_categorical_crossentropy(y_true, y_pred):
    n_class = K.int_shape(y_pred)[-1]

    mask = K.reshape(y_true[:, :, :, -1], (-1,))
    y_true = K.reshape(y_true[:, :, :, :-1], (-1, n_class))
    y_pred_log = K.log(K.reshape(y_pred[:, :, :, :-1], (-1, n_class)))

    valid = tf.not_equal(mask, 1)
    y_true = tf.boolean_mask(y_true, valid)
    y_pred_log = tf.boolean_mask(y_pred_log, valid)

    cross_entropy = -K.sum(y_true * y_pred_log, axis=1)
    cross_entropy_mean = K.mean(cross_entropy)

    return cross_entropy_mean


def binary_weighted_cross_entropy(y_true, y_pred, beta=1.):
    """ 适用于二类分类，解决类别不平衡问题 """
    def convert_to_logits(y_pred):
        """ 将经过sigmoid变换后的概率值转回logit """
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.log(y_pred / (1 - y_pred))

    y_pred = convert_to_logits(y_pred)
    loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)
    return tf.reduce_mean(loss)


def binary_balanced_cross_entropy(y_true, y_pred, smooth=SMOOTH, beta=0.5):
    y_pred = K.clip(y_pred, smooth, 1. - smooth)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(beta * K.log(pt_1)) - K.sum((1 - beta) * K.log(1. - pt_0))


def binary_focal_loss(y_true, y_pred, alpha=0.25, gamma=2., smooth=SMOOTH):
    """ focal loss for imbalanced dataset """
    y_pred = K.clip(y_pred, smooth, 1. - smooth)
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))\
           - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


def dice_loss(y_true, y_pred, class_weights=1., smooth=SMOOTH, per_image=True):
    """ Dice loss function for imbalanced dataset, only for binary tasks """
    return 1 - dice_score(y_true, y_pred, class_weights, smooth, per_image)


def bce_dice_loss(y_true, y_pred, bce_weight=1., smooth=SMOOTH, per_image=True):
    """ alpha * binary cross entropy + jaccard """
    bce = K.mean(binary_crossentropy(y_true, y_pred))
    loss = bce * bce_weight + dice_loss(y_true, y_pred, smooth=smooth, per_image=per_image)
    return loss


def cce_dice_loss(y_true, y_pred, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True):
    """ alpha * categorical cross entropy + jaccard """
    cce = K.mean(categorical_crossentropy(y_true, y_pred) * class_weights)
    loss = cce * cce_weight + dice_loss(y_true, y_pred, class_weights, smooth, per_image)
    return loss


def jaccard_iou_loss(y_true, y_pred, class_weights=1., smooth=SMOOTH, per_image=True):
    """ jaccard loss """
    return 1 - jaccard_iou_score(y_true, y_pred, class_weights, smooth, per_image)


def bce_jaccard_loss(y_true, y_pred, bce_weight=1., smooth=SMOOTH, per_image=True):
    """ alpha * binary cross entropy + jaccard """
    bce = K.mean(binary_crossentropy(y_true, y_pred))
    loss = bce * bce_weight + jaccard_iou_loss(y_true, y_pred, smooth=smooth, per_image=per_image)
    return loss


def cce_jaccard_loss(y_true, y_pred, cce_weight=1., class_weights=1., smooth=SMOOTH, per_image=True):
    """ alpha * categorical cross entropy + jaccard """
    cce = K.mean(categorical_crossentropy(y_true, y_pred) * class_weights)
    loss = cce * cce_weight + jaccard_iou_loss(y_true, y_pred, class_weights, smooth, per_image)
    return loss


def tversky_loss(y_true, y_pred):
    pass


def exponential_logarithmic_loss(y_true, y_pred):
    raise NotImplementedError


def boundary_loss(y_true, y_pred):
    raise NotImplementedError


def conservative_loss(y_true, y_pred):
    raise NotImplementedError


def _lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        def treat_image(log_lab):
            log, lab = log_lab
            log, lab = tf.expand_dims(log, 0), tf.expand_dims(lab, 0)
            log, lab = _flatten_binary_scores(log, lab, ignore)
            return _lovasz_hinge_flat(log, lab)
        losses = tf.map_fn(treat_image, (logits, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = _lovasz_hinge_flat(*_flatten_binary_scores(logits, labels, ignore))
    return loss


def _lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """

    def compute_loss():
        labelsf = tf.cast(labels, logits.dtype)
        signs = 2. * labelsf - 1.
        errors = 1. - logits * tf.stop_gradient(signs)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort")
        gt_sorted = tf.gather(labelsf, perm)
        grad = _lovasz_grad(gt_sorted)
        loss = tf.tensordot(tf.nn.relu(errors_sorted), tf.stop_gradient(grad), 1, name="loss_non_void")
        return loss

    # deal with the void prediction case (only void pixels)
    loss = tf.cond(tf.equal(tf.shape(logits)[0], 0),
                   lambda: tf.reduce_sum(logits) * 0.,
                   compute_loss,
                   strict=True,
                   name="loss"
                   )
    return loss


def _flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = tf.reshape(scores, (-1,))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return scores, labels
    valid = tf.not_equal(labels, ignore)
    vscores = tf.boolean_mask(scores, valid, name='valid_scores')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vscores, vlabels


def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None, order='BHWC'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = _flatten_probas(prob, lab, ignore, order)
            return _lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = _lovasz_softmax_flat(*_flatten_probas(probas, labels, ignore, order), classes=classes)
    return loss


def _lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = tf.abs(fg - class_pred)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = _lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)
    return loss


def _flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas, order = tf.expand_dims(probas, 3), 'BHWC'
    if order == 'BCHW':
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels
