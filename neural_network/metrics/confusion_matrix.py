import numpy as np
from concurrent.futures import ProcessPoolExecutor

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export

from ..exceptions import ExceptionFactory

errors = {
    'ConfusionMatrixError': ExceptionFactory.register('ConfusionMatrixError')
}


class _cmat1d_helper:
    def __init__(self, targets):
        self.targets = targets
        self.cmat = np.zeros((len(targets),) * 2, dtype=int)

    def __call__(self, args):
        for true, pred in zip(*args):
            self.cmat[self.targets[true], self.targets[pred]] += 1


@type_safe
@not_none
def _cmat1d(y_true: np.ndarray, y_pred: np.ndarray, *,
            use_multiprocessing: bool = False) -> np.ndarray:
    targets = {y: x for x, y in enumerate(np.unique(y_true))}
    helper = _cmat1d_helper(targets)
    if use_multiprocessing:
        n = len(y_true) // 4
        y = (
            (y_true[:n], y_pred[:n]),
            (y_true[n: 2 * n], y_pred[n: 2 * n]),
            (y_true[2 * n:3 * n], y_pred[2 * n:3 * n]),
            (y_true[3 * n:], y_pred[3 * n:]),
        )
        with ProcessPoolExecutor() as executor:
            for _ in executor.map(helper, y):
                pass
    else:
        for _ in ((y_true, y_pred),):
            helper(_)
    return helper.cmat


class _cmat2d_helper:
    def __init__(self, targets):
        self.cmat = np.zeros((len(targets),) * 2, dtype=int)
        self.targets = targets

    def __call__(self, args):
        for true, pred in zip(*args):
            self.cmat[self.targets[true.argmax()], self.targets[pred.argmax()]] += 1


@type_safe
@not_none
def _cmat2d(y_true: np.ndarray, y_pred: np.ndarray, *,
            use_multiprocessing: bool = False) -> np.ndarray:
    targets = np.arange(y_true.shape[-1], dtype=int)
    helper = _cmat2d_helper(targets)
    if use_multiprocessing:
        n = len(y_true) // 4
        y = (
            (y_true[:n], y_pred[:n]),
            (y_true[n: 2 * n], y_pred[n: 2 * n]),
            (y_true[2 * n:3 * n], y_pred[2 * n:3 * n]),
            (y_true[3 * n:], y_pred[3 * n:]),
        )
        with ProcessPoolExecutor() as executor:
            for _ in executor.map(helper, y):
                pass
    else:
        for _ in ((y_true, y_pred),):
            helper(_)
    return helper.cmat


@type_safe
@not_none(nullable=('normalize',))
@export
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *,
                     normalize: str = None, use_multiprocessing: bool = False) -> np.ndarray:
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
        use_multiprocessing: bool, default = False
            Set to true to use multiprocessing while computing the confusion
            matrix. Useful for large amounts of data

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
        if len(y_true) == 1:
            raise errors['ConfusionMatrixError'](
                f'There must be at least 2 labels to construct a confusion matrix'
            )
        cmat = _cmat1d(y_true, y_pred, use_multiprocessing=use_multiprocessing)
    elif y_true.ndim == 2:
        if y_true.shape[-1] == 1:
            return confusion_matrix(y_true.flatten(), y_pred.flatten(), normalize=normalize)
        cmat = _cmat2d(y_true, y_pred, use_multiprocessing=use_multiprocessing)
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


class _mcm_helper:
    def __init__(self, cmat):
        self.cmat = cmat
        self.num_all = self.cmat.sum()
        self.mcm = np.empty((cmat.shape[-1], 2, 2), dtype=int)

    def __call__(self, label):
        tp = self.cmat[label, label]
        fp = self.cmat[:, label].sum() - tp
        fn = self.cmat[label].sum() - tp
        tn = self.num_all - (tp + fp + fn)
        self.mcm[label] = np.array([tp, fn, fp, tn], dtype=int).reshape((2, 2))


@type_safe
@not_none
@export
def multilabel_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, *,
                                use_multiprocessing: bool = False) -> np.ndarray:
    '''
    Computes the multilabel confusion matrix given the labels and predictions
    Targets for 1d arrays are computed by np.unique, so the labels
    are sorted in increasing order.

    Multilabel classifcations are done by considering a target class as positive,
    and all the other classes as negative. This means that there would be as many
    (2,2,) confusion matrices as there are unique labels

    Parameters:
        y_true: np.ndarray of shape (n_samples, n_classes) or (n_samples,)
            known labels or one hot encoded labels
        y_pred: np.ndarray of shape (n_samples, n_classes) or (n_samples,)
            predicted labels or one hot encoded labels
        use_multiprocessing: bool, default = False
            Set to True to use multiprocessing while computing the confusion
            matrix. Useful for large amounts of data

    Returns:
        np.ndarray: of shape (n_classes, 2, 2), array of confusion matrices
                    where in each matrix, axis=0 are the true labels
                    and axis=1 are the predicted labels
    '''
    cmat = confusion_matrix(y_true, y_pred, use_multiprocessing=use_multiprocessing)
    helper = _mcm_helper(cmat)

    if use_multiprocessing:
        with ProcessPoolExecutor() as executor:
            for _ in executor.map(helper, range(cmat.shape[-1])):
                pass
    else:
        for label in range(cmat.shape[-1]):
            helper(label)

    return helper.mcm
