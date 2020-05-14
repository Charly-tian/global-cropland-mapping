
from keras import backend as K

SMOOTH = 1e-12


def acc_score(y_true, y_pred):
    raise NotImplementedError


def jaccard_iou_score(y_true, y_pred, class_weights=1., smooth=SMOOTH, per_image=True):
    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true + y_pred, axis=axes) - intersection
    iou = (intersection + smooth) / (union + smooth)

    # mean per image
    if per_image:
        iou = K.mean(iou, axis=0)

    # weighted mean per class
    iou = K.mean(iou * class_weights)

    return iou


def f_score(y_true, y_pred, class_weights=1., beta=1, smooth=SMOOTH, per_image=True):
    y_true = y_true[:, :, :, 1]
    y_pred = y_pred[:, :, :, 1]

    if per_image:
        axes = [1, 2]
    else:
        axes = [0, 1, 2]

    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    score = ((1 + beta ** 2) * tp + smooth) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)

    # mean per image
    if per_image:
        score = K.mean(score, axis=0)

    # weighted mean per class
    score = K.mean(score * class_weights)

    return score


def dice_score(y_true, y_pred, class_weights=1., smooth=SMOOTH, per_image=True):
    """ equal to f1_score """
    return f_score(y_true, y_pred, class_weights, 1, smooth, per_image)
