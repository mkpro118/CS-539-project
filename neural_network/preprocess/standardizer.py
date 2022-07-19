import numpy as np

from ..base import MetadataMixin, SaveMixin, TransformMixin
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class Standardizer(MetadataMixin, SaveMixin, TransformMixin):
    @type_safe(skip=('y', 'return'))
    @not_none(nullable=('y', ))
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> 'Standardizer':
        # Compute the Mean and Standard deviation
        self.feature_means = np.mean(X, axis=0)
        self.feature_std = np.std(X, axis=0)

        self.kwargs = kwargs
        return self

    def _check_is_fitted(self):
        _attrs = ('feature_means', 'feature_std',)

        if any(((getattr(self, attr, None) is None) for attr in _attrs)):
            raise ValueError('PrincipalComponentAnalysis object is not yet fitted!')

    @type_safe(skip=('y',))
    @not_none(nullable=('y', ))
    def transform(self, X: np.ndarray, y: np.ndarray = None, *,
                  inplace: bool = False) -> np.ndarray:

        # Copy the array if transformation is not inplace
        if not inplace:
            X = np.copy(X)

        # Standardize all values
        X[:] = (X - self.feature_means) / self.feature_std
        return X
