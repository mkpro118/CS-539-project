import numpy as np

from ..base import MetadataMixin, SaveMixin, TransformMixin
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class Standardizer(MetadataMixin, SaveMixin, TransformMixin):
    '''
    Used to standardize the features to have zero mean and unity variance
    '''
    @type_safe(skip=('y', 'return'))
    @not_none(nullable=('y', ))
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> 'Standardizer':
        '''
        Fits the instance with the given feature matrix to standardize

        Parameters:
            X: numpy.ndarray of shape (n_samples, n_features)
                The feature matrix to fit the standardizing parameters with

        Returns:
            self: The fitted Standardizer instance
        '''

        # Compute the Mean and Standard deviation
        self.feature_means = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0)

        self.kwargs = kwargs
        self._attrs = ('feature_means', 'feature_std',)
        return self

    @type_safe(skip=('y',))
    @not_none(nullable=('y', ))
    def transform(self, X: np.ndarray, y: np.ndarray = None, *,
                  inplace: bool = False) -> np.ndarray:
        '''
        Transforms the given matrix according to the fitted standardizing parameters

        Parameters:
            X: numpy.ndarray of shape (k, n_features)
                The features to standardize

        Returns:
            numpy.ndarray: The standardized features
        '''

        # Copy the array if transformation is not inplace
        if not inplace:
            X = np.copy(X)

        # Standardize all values
        X[:] = (X - self.feature_means) / self.feature_std
        return X
