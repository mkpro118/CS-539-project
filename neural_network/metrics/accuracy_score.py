import numpy as np
from typing import Union

from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from .confusion_matrix import confusion_matrix


errors = {
    'AccuracyScoreError': ExceptionFactory.register('AccuracyScoreError'),
}


@type_safe
@not_none(nullable=('normalize',))
@export
def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                   normalize: bool = True) -> Union[float, int]:
    '''
    Calculates overall accuracy

    Parameters:
        y_true: np.ndarray of shape (n_samples, n_classes) or (n_samples,)
            known labels or one hot encoded labels
        y_pred: np.ndarray of shape (n_samples, n_classes) or (n_samples,)
            predicted labels or one hot encoded labels
        normalize: bool, keyword only, default = True
            By default score is a float between 0. and 1.
            If normalize is set to False, scores will be the
            number of correct classifications

    Returns:
        Union[float, int]: The accuracy score
    '''
    if y_true.shape != y_pred.shape:
        raise errors['AccuracyScoreError'](
            f'y_true and y_pred must have the same shapes, '
            f'{y_true.shape} != {y_pred.shape}'
        )

    cmat = confusion_matrix(y_true, y_pred)

    correct = int(np.trace(cmat))

    if not normalize:
        return correct

    return correct / len(y_true)
