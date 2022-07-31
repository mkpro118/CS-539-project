import numpy as np

from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from .confusion_matrix import confusion_matrix, multilabel_confusion_matrix

errors = {
    'RecallScoreError': ExceptionFactory.register('RecallScoreError'),
}


_recall = lambda tp, fn: tp / (tp + fn)


@type_safe
@not_none
def _score_binary_label(y_true: np.ndarray, y_pred: np.ndarray, *,
                        use_multiprocessing: bool = False) -> float:
    cmat = confusion_matrix(
        y_true,
        y_pred,
        use_multiprocessing=use_multiprocessing
    )

    if cmat.shape[-1] != 2:
        raise errors['RecallScoreError'](
            f'y_true has been detected to contain multiple labels, but average=\'binary\'. '
            f'Set the scoring setting from one of [\'micro\', \'macro\']'
        )

    tp, fn, *_ = cmat.ravel()
    return _recall(tp, fn)


'''TODO below'''


@type_safe
@not_none
def _score_multi_label_micro(y_true: np.ndarray, y_pred: np.ndarray, *,
                             use_multiprocessing: bool = False) -> float:
    mcm = multilabel_confusion_matrix(
        y_true,
        y_pred,
        use_multiprocessing=use_multiprocessing
    )

    tp = np.sum(mcm[:, 0, 0])
    fn = np.sum(mcm[:, 0, 1])

    return _recall(tp, fn)


@type_safe
@not_none
def _score_multi_label_macro(y_true: np.ndarray, y_pred: np.ndarray, *,
                             use_multiprocessing: bool = False) -> float:
    mcm = multilabel_confusion_matrix(
        y_true,
        y_pred,
        use_multiprocessing=use_multiprocessing
    )

    tp = mcm[:, 0, 0]
    fn = mcm[:, 0, 1]

    return np.mean(_recall(tp, fn))


'''TODO above'''

_scoring_functions = {
    'binary': _score_binary_label,
    'micro': _score_multi_label_micro,
    'macro': _score_multi_label_macro,
}


@type_safe
@not_none
@export
def recall_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                 average: str = 'binary', use_multiprocessing: bool = False) -> float:
    '''
    Calculates the recall score of the model

    For binary classifications, it is defined as
    Recall = True Positives / (True Positives + False Negatives)

    For multi-label classifications, it is computed in two ways
        micro-averaged:
            It is computed globally over all labels at once,
            giving equal weight to each decision. Tends to be dominated
            by the classifier's performance on the more common labels

            It is computes using the formula
                                 Sum of True Positives
            Recall = ----------------------------------------------
                     Sum of True Positives + Sum of False Negatives
            where all sums are by label

        macro-averaged:
            It is computed categorically (by label), giving equal weight
            to each label. Tends to be dominated by classifier's performance
            on rarer labels

            It is computes using the formula
                     Sum of recalls of each label
            Recall = ----------------------------
                           Number of labels


    Parameters:
        y_true: np.ndarray of shape (n_samples, n_classes)
            known one hot encoded labels
        y_pred: np.ndarray of shape (n_samples, n_classes)
            predicted one hot encoded labels
        use_multiprocessing: bool, default = False
            Set to true to use multiprocessing while computing the confusion
            matrix. Useful for large amounts of data

    Returns:
        float: The recall score
    '''
    if y_true.shape != y_pred.shape:
        raise errors['RecallScoreError'](
            f'y_true and y_pred must have the same shapes, '
            f'{y_true.shape} != {y_pred.shape}'
        )

    if average not in ['binary', 'micro', 'macro']:
        raise errors['RecallScoreError'](
            'average must be one of [\'binary\', \'micro\', \'macro\']'
        )

    return float(_scoring_functions[average](y_true, y_pred))
