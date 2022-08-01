import numpy as np
from typing import Union

from ..base.metadata_mixin import MetadataMixin
from ..base.save_mixin import SaveMixin

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class KFold(MetadataMixin, SaveMixin):
    '''
    Used to split data into training and validation data
    '''

    def __init__(self, n_splits: int = 5, shuffle: bool = False,
                 random_state: int = None):
        '''
        Initiliase the K-Fold Spliterator

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
    def split(self, X: Union[np.ndarray, list, tuple]) -> tuple:
        '''
        Iterator that performs a K-Fold split over the given array

        Parameters:
            X: Union[numpy.ndarray, list, tuple]
                The array to perform splits over

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: The first array is the indices
            for the training set, the second array is the indices for the validating set
        '''
        n_samples = len(X)
        X = np.asarray(X)

        indices = np.arange(n_samples)
        if self.shuffle:
            self._rng.shuffle(indices)

        current = 0
        for _ in range(self.n_splits):
            start, stop = current, current + (n_samples // self.n_splits)
            yield (
                np.concatenate((indices[0:start], indices[stop:])),  # Training
                indices[start:stop]  # Validating
            )
            current = stop
