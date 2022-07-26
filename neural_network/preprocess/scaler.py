import numpy as np

from ..base import MetadataMixin, SaveMixin, TransformMixin
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class Scaler(MetadataMixin, SaveMixin, TransformMixin):
    '''
    Used to scale the features to a given range
    '''

    def __init__(self, start: float = -1., end: float = 1.):
        '''
        Parameters:
            start: float, default = -1.
                The lower bound for the scaled features
            end: float, default = 1.
                THe upper bound for the scaled features
        '''
        self.start = start
        self.end = end

    @type_safe(skip=('y', 'return'))
    @not_none(nullable=('y', ))
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> 'Scaler':
        '''
        Fits the instance with the given feature matrix to scale

        Parameters:
            X: numpy.ndarray of shape (n_samples, n_features)
                The feature matrix to fit the scaling parameters with

        Returns:
            self: The fitted Scaler instance
        '''

        # Compute the feature mininum and maximum values
        self.feature_min = np.min(X, axis=0)
        self.feature_max = np.max(X, axis=0)

        # Compute the feature range
        self.feature_range = self.feature_max - self.feature_min

        # Compute the required range
        self.range = self.end - self.start

        self.kwargs = kwargs
        self._attrs = ('feature_min', 'feature_max', 'feature_range', 'range')
        return self

    @type_safe(skip=('y',))
    @not_none(nullable=('y', ))
    def transform(self, X: np.ndarray, y: np.ndarray = None, *,
                  inplace: bool = False) -> np.ndarray:
        '''
        Transforms the given matrix according to the fitted scaling parameters

        Parameters:
            X: numpy.ndarray of shape (k, n_features)
                The features to scale

        Returns:
            numpy.ndarray: The scaled features
        '''

        # Copy the array if transformation is not inplace
        if not inplace:
            X = np.copy(X)

        # Scale all values
        X[:] = ((self.range) * (X - self.feature_min)) / (self.feature_range) + self.start
        return X
