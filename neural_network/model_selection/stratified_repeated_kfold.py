import numpy as np

from ..base import MetadataMixin, SaveMixin

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export

from .stratified_kfold import StratifiedKFold


@export
class StratifiedRepeatedKFold(MetadataMixin, SaveMixin):
    '''
    Used to split data into training and validation data
    in a stratified manner, repeated a given number of times
    '''

    def __init__(self, n_splits: int = 5, n_repeats: int = 10,
                 shuffle: bool = False, random_state: int = None):
        '''
        Initiliase the Stratified Repeated K-Fold Spliterator

        Parameters:
            n_splits: int, default = 5
                The number of splits to perform (the K)
            n_repeats: int, default = 10
                The number of time to repeat K-Fold
            shuffle: bool, default = False
                Set to true to shuffle the indices before splitting
            random_state: int, default = None
                Set a Random State to have reproducible results

        '''
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.shuffle = shuffle
        self.random_state = random_state
        if shuffle:
            if self.random_state is not None:
                self._rng = np.random.default_rng(self.random_state)
            else:
                self._rng = np.random.default_rng()
        else:
            self._rng = None

    @type_safe(skip=('return',))
    @not_none
    def split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        '''
        Iterator that performs a Stratified Repeated K-Fold split over the given array

        Parameters:
            X: numpy.ndarray
                The array to perform splits over

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: The first array is the indices
                for the training set, the second array is the indices for the validating set
        '''
        for i in range(self.n_repeats):
            splitter = StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=self.shuffle,
                random_state=self.random_state
            )
            for train, validate in splitter.split(X, y):
                yield train, validate
