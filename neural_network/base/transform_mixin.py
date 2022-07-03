import numpy as np

from .mixin import mixin
from ..utils.typesafety import type_safe


@mixin
class TransformMixin:

    @type_safe
    def fit_transform(self, X: np.ndarray, y: np.ndarray = None):
        self.fit(X, y)
        return self.transform(X, y)
