# TODO

import numpy as np

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export

from ..exceptions import ExceptionFactory

errors = {
    'ConfusionMatrixError': ExceptionFactory.register('ConfusionMatrixError')
}


@type_safe
@not_none
def _cmat1d(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    pass


@type_safe
@not_none
def _cmat2d(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    pass


@type_safe
@not_none
@export
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *,
                     normalize: bool = False) -> np.ndarray:
    '''
    Computes the confusion matrix given the labels and predictions

    Parameters:
        y_true: np.ndarray of shape (n_samples, n_classes) or (n_samples,)
            known labels or one hot encoded labels
        y_pred: np.ndarray of shape (n_samples, n_classes) or (n_samples,)
            predicted labels or one hot encoded labels
        normalize: bool, keyword only, default = False
            Scores are  the number of classifications
            under that label by default
            If set to True, scores will be a float between 0. and 1.

    Returns:
        np.ndarray: of shape (n_classes, n_classes), the confusion matrix
                    where axis=0 are the true labels
                    and axis=1 are the predicted labels
    '''
    if y_true.shape != y_pred.shape:
        raise errors['ConfusionMatrixError'](
            f'y_true and y_pred must have the same shapes, '
            f'{y_true.shape} != {y_pred.shape}'
        )

    if y_true.ndim == 1:
        cmat = _cmat1d(y_true, y_pred)
    elif y_true.ndim == 2:
        cmat = _cmat2d(y_true, y_pred)
    else:
        raise errors['ConfusionMatrixError'](
            f'y_true and y_pred must have dimensions <= 2, ({y_true.ndim} > 2)'
        )

    if normalize:
        cmat = cmat / np.sum(cmat)

    return cmat
