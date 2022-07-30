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
    targets = {y: x for x, y in enumerate(np.unique(y_true))}
    cmat = np.zeros((len(targets),) * 2, dtype=int)
    for true, pred in zip(y_true, y_pred):
        cmat[targets[true], targets[pred]] += 1
    return cmat


@type_safe
@not_none
def _cmat2d(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    targets = np.arange(y_true.shape[-1], dtype=int)
    cmat = np.zeros((len(targets),) * 2, dtype=int)
    for true, pred in zip(y_true, y_pred):
        cmat[targets[true.argmax()], targets[pred.argmax()]] += 1
    return cmat


@type_safe
@not_none(nullable=('normalize',))
@export
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *,
                     normalize: str = None) -> np.ndarray:
    '''
    Computes the confusion matrix given the labels and predictions
    Targets for 1d arrays are computed by np.unique, so the labels
    are sorted in increasing order

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

    if normalize is not None and normalize not in ['true', 'pred', 'all']:
        raise errors['ConfusionMatrixError'](
            f"normalize must be either 'true', 'pred' or all"
        )

    if y_true.ndim == 1:
        cmat = _cmat1d(y_true, y_pred)
    elif y_true.ndim == 2:
        cmat = _cmat2d(y_true, y_pred)
    else:
        raise errors['ConfusionMatrixError'](
            f'y_true and y_pred must have dimensions <= 2, ({y_true.ndim} > 2)'
        )

    if normalize is None:
        return cmat

    if normalize == 'all':
        divisor = np.array(np.sum(cmat))
    elif normalize == 'true':
        divisor = np.sum(cmat, axis=1)
    elif normalize == 'pred':
        divisor = np.sum(cmat, axis=0)

    divisor[divisor == 0] = 1
    return cmat / divisor
