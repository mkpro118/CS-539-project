import numpy as np

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@type_safe
@not_none
@export
def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2,
                     random_state: int = None) -> tuple:
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

    Returns:
        tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
            The first element is the training feature set
            The second element is the testing feature set
            The third element is the training label set
            The fourth element is the testing label set
    '''
    # Number of elements in the test set
    length = int(X.shape[0] * test_size)

    if random_state is not None:
        rng = np.random.default_rng(random_state)
    else:
        rng = np.random.default_rng()

    # get a random permutation of the indices.
    idx = rng.permutation(X.shape[0])

    return (
        X[idx[length:]],
        X[idx[:length]],
        y[idx[length:]],
        y[idx[:length]],
    )
