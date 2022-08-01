import numpy as np
from typing import Union

from ..base.metadata_mixin import MetadataMixin
from ..base.save_mixin import SaveMixin

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export

from .stratified_kfold import StratifiedKFold


@export
class StratifiedRepeatedKFold(StratifiedKFold, MetadataMixin, SaveMixin):
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
        super().__init__(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state,
        )

        self.n_repeats = n_repeats

    @type_safe(skip=('return',))
    @not_none
    def split(self, X: Union[np.ndarray, list, tuple], y: Union[np.ndarray, list, tuple]) -> tuple:
        '''
        Iterator that performs a Stratified Repeated K-Fold split over the given array

        Parameters:
            X: Union[np.ndarray, list, tuple]
                The array to perform splits over
            y: Union[numpy.ndarray, list, tuple]
                The labels to use for stratification

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: The first array is the indices
                for the training set, the second array is the indices for the validating set
        '''
        for i in range(self.n_repeats):
            for train, validate in super().split(X, y):
                yield train, validate
