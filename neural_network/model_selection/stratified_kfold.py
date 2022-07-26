import numpy as np

from ..base import MetadataMixin, SaveMixin

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class StratifiedKFold(MetadataMixin, SaveMixin):
    '''
    Used to split data into training and validation data
    in a stratified manner
    '''

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: int = None):
        '''
        Initiliase the Stratified K-Fold Spliterator

        Parameters:
            n_splits: int, default = 5
                The number of splits to perform (the K)
            shuffle: bool, default = False
                Set to true to shuffle the indices before splitting
            random_state: int, default = None
                Set a Random State to have reproducible results

        '''
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        if shuffle:
            if self.random_state is not None:
                self._rng = np.random.default_rng(self.random_state)
            else:
                self._rng = np.random.default_rng()

    @type_safe(skip=('return',))
    @not_none
    def split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        '''
        Iterator that performs Stratified K-Fold split over the given array

        Parameters:
            X: numpy.ndarray
                The array to perform splits over
            y: numpy.ndarray
                The labels to use for stratification

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: The first array is the indices
            for the training set, the second array is the indices for the validating set
        '''

        _, y_idx, y_inv = np.unique(y, return_index=True, return_inverse=True)
        _, y_idx_inv = np.unique(y_idx, return_inverse=True)
        by_label = y_idx_inv[y_inv]

        n_classes = len(y_idx)
        counts = np.bincount(by_label)
        min_groups = np.min(counts)

        if np.all(self.n_splits > counts):
            raise ValueError(
                f'{self.n_splits=} cannot be greater than the'
                f' number of members in each class.'
            )
        if self.n_splits > min_groups:
            raise ValueError(
                f'The least populated class in y has only {min_groups}'
                f' members, which is less than {self.n_splits=}'
            )

        order = np.sort(by_label)
        allocation = np.array([np.bincount(order[i::self.n_splits], minlength=n_classes) for i in range(self.n_splits)])

        validation_folds = np.empty(len(y), dtype="i")
        for k in range(n_classes):
            folds_by_class = np.arange(self.n_splits).repeat(allocation[:, k])
            if self.shuffle:
                self._rng.shuffle(folds_by_class)
            validation_folds[by_label == k] = folds_by_class

        indices = np.arange(len(X))
        for i in range(self.n_splits):
            yield (
                indices[validation_folds != i],  # Training set
                indices[validation_folds == i],  # Validation set
            )
