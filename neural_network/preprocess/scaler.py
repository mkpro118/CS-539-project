import numpy as np

from ..base import MetadataMixin, SaveMixin, TransformMixin
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class Scaler(MetadataMixin, SaveMixin, TransformMixin):
    def __init__(self, start: float = -1., end: float = 1.):
        self.start = start
        self.end = end

    @type_safe(skip=('y', 'return'))
    @not_none(nullable=('y', ))
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> 'Scaler':
        # Compute the feature mininum and maximum values
        self.feature_min = np.min(X, axis=0)
        self.feature_max = np.max(X, axis=0)
        self.feature_range = self.feature_max - self.feature_min
        self.range = self.end - self.start

        self.kwargs = kwargs
        return self

    def _check_is_fitted(self):
        _attrs = ('feature_min', 'feature_max', 'feature_range', 'range')

        if any(((getattr(self, attr, None) is None) for attr in _attrs)):
            raise ValueError('PrincipalComponentAnalysis object is not yet fitted!')

    @type_safe(skip=('y',))
    @not_none(nullable=('y', ))
    def transform(self, X: np.ndarray, y: np.ndarray = None, *,
                  inplace: bool = False) -> np.ndarray:

        # Copy the array if transformation is not inplace
        if not inplace:
            X = np.copy(X)

        # Scale all values
        X[:] = ((self.range) * (X - self.feature_min)) / (self.feature_range) + self.start
        return X
