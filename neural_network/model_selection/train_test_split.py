import numpy as np
from typing import Union, Iterable

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@type_safe
@not_none
@export
def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                     random_state: int = None, stratify: Union[bool, Iterable] = True) -> tuple:
    '''
    Splits the given dataset into training and testing data

    Parameters:
        X: numpy.ndarray
            The feature array to split
        y: numpy.ndarray
            The label array to split
        test_size: float, default = 0.2
            The ratio of samples in the testing set
        random_state: int, default = None
            Set a random seed for reproducible splits
        stratify: Union[bool, Iterable], default = True
            Return the same proportion of labels in training and testing sets
            Set to False to not stratify the data
            By default, argument y is used to stratitfy the data,
            the stratification strategy can be change by passing in

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The first element is the training feature set
            The second element is the testing feature set
            The third element is the training label set
            The fourth element is the testing label set
    '''
    assert len(X) == len(y), f'Inconsistent number of samples, (X) {len(X)} != {len(y)} (y)'

    test_length = int(X.shape[0] * test_size)

    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    if not stratify:
        # Number of elements in the test set

        # get a random permutation of the indices.
        idx = rng.permutation(X.shape[0])

        return (
            X[idx[test_length:]],
            X[idx[:test_length]],
            y[idx[test_length:]],
            y[idx[:test_length]],
        )

    train_size = 1 - test_size
    if isinstance(stratify, bool):
        stratify = y
    else:
        stratify = np.asarray(stratify)

    assert len(X) == len(stratify), f'Inconsistent number of samples, (X) {len(X)} != {len(stratify)} (stratify)'

    labels, counts = np.unique(stratify, return_counts=True)

    train_counts = sum([np.floor(train_size * count).astype(int) for count in counts])
    test_counts = sum([np.ceil(test_size * count).astype(int) for count in counts])

    train_mask = np.empty((train_counts,), dtype=int)
    test_mask = np.empty((test_counts,), dtype=int)

    train_allocated, test_allocated = 0, 0

    for label, count in zip(labels, counts):
        mask = np.argwhere(stratify == label).flatten()
        train_count = np.floor(count * train_size).astype(int)
        test_count = np.ceil(count * test_size).astype(int)
        train_mask[train_allocated: train_allocated + train_count] = mask[:train_count]
        train_allocated += train_count
        test_mask[test_allocated: test_allocated + test_count] = mask[::-1][test_count]
        test_allocated += test_count

    rng.shuffle(train_mask)
    rng.shuffle(test_mask)
    return (
        X[train_mask],
        X[test_mask],
        y[train_mask],
        y[test_mask],
    )
