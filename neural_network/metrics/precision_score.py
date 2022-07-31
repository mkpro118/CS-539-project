import numpy as np

from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from .confusion_matrix import confusion_matrix, multilabel_confusion_matrix

errors = {
    'PrecisionScoreError': ExceptionFactory.register('PrecisionScoreError'),
}


_precision = lambda tp, fp: tp / (tp + fp)


@type_safe
@not_none
def _score_binary_label(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    cmat = confusion_matrix(y_true, y_pred)

    if cmat.shape[-1] != 2:
        raise errors['PrecisionScoreError'](
            f'y_true has been detected to contain multiple labels, but average=\'binary\'. '
            f'Set the scoring setting from one of [\'micro\', \'macro\']'
        )

    tp, _, fp, _ = cmat.ravel()
    return _precision(tp, fp)


'''TODO below'''


@type_safe
@not_none
def _score_multi_label_micro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    tp = np.sum(mcm[:, 0, 0])
    fp = np.sum(mcm[:, 1, 0])

    return _precision(tp, fp)


@type_safe
@not_none
def _score_multi_label_macro(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    tp = mcm[:, 0, 0]
    fp = mcm[:, 1, 0]

    return np.mean(_precision(tp, fp))


'''TODO above'''

_scoring_functions = {
    'binary': _score_binary_label,
    'micro': _score_multi_label_micro,
    'macro': _score_multi_label_macro,
}


@type_safe
@not_none
@export
def precision_score(y_true: np.ndarray, y_pred: np.ndarray, *,
                    average: str = 'binary', use_multiprocessing: bool = False) -> float:
    '''
    Calculates the precision score of the model

    For binary classifications, it is defined as
    Precision = True Positives / (True Positives + False Positives)

    For multi-label classifications, it is computed in two ways
        micro-averaged:
            It is computed globally over all labels at once,
            giving equal weight to each decision. Tends to be dominated
            by the classifier's performance on the more common labels

            It is computes using the formula
                                    Sum of True Positives
            Precision = ----------------------------------------------
                        Sum of True Positives + Sum of False Positives
            where all sums are by label

        macro-averaged:
            It is computed categorically (by label), giving equal weight
            to each label. Tends to be dominated by classifier's performance
            on rarer labels

            It is computes using the formula
                        Sum of precisions of each label
            Precision = ------------------------------
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
        float: The precision score
    '''
    if y_true.shape != y_pred.shape:
        raise errors['PrecisionScoreError'](
            f'y_true and y_pred must have the same shapes, '
            f'{y_true.shape} != {y_pred.shape}'
        )

    if average not in ['binary', 'micro', 'macro']:
        raise errors['PrecisionScoreError'](
            'average must be one of [\'binary\', \'micro\', \'macro\']'
        )

    return float(_scoring_functions[average](y_true, y_pred))
