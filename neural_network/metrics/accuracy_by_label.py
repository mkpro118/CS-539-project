import numpy as np

from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from .confusion_matrix import confusion_matrix


errors = {
    'AccuracyByLabelError': ExceptionFactory.register('AccuracyByLabelError'),
}


@type_safe
@not_none
@export
def accuracy_by_label(y_true: np.ndarray, y_pred: np.ndarray, *,
                      normalize: bool = True) -> np.ndarray:
    '''
    Calculates accuracy of the model by label

    Parameters:
        y_true: np.ndarray of shape (n_samples, n_classes)
            known one hot encoded labels
        y_pred: np.ndarray of shape (n_samples, n_classes)
            predicted one hot encoded labels
        normalize: bool, keyword only, default = True
            Scores are a float between 0. and 1. by default
            If set to False, scores will be the number of
            correct classifications under that label

    Returns:
        np.ndarray: each element of the array is the normalized accuracy of that label
                    if normalize is false, it's the number of
                    correct classifications under that label
    '''
    if y_true.shape != y_pred.shape:
        raise errors['AccuracyByLabelError'](
            f'y_true and y_pred must have the same shapes, '
            f'{y_true.shape} != {y_pred.shape}'
        )
    return (
        (
            correct := np.diagonal(
                (cmat := confusion_matrix(y_true, y_pred))
            )
        ) / np.sum(cmat, axis=0)
    ) if normalize else correct
