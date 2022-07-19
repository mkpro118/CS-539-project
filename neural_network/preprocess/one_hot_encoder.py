import numpy as np

from ..base import MetadataMixin, SaveMixin, TransformMixin
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class OneHotEncoder(MetadataMixin, SaveMixin, TransformMixin):

    @type_safe(skip=('y',))
    @not_none(nullable=('y',))
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> 'OneHotEncoder':
        self.label_vector = X
        self.n_samples = len(self.label_vector)
        self.classes = np.unique(self.label_vector)
        self.n_classes = len(self.classes)
        self.kwargs = kwargs
        return self

    @type_safe(skip=('y',))
    @not_none(nullable=('y',))
    def transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs):
        one_hot = np.zeros((self.n_samples, self.n_classes), dtype=int)
        for index, class_ in enumerate(self.classes):
            one_hot[self.label_vector == class_, index] = 1
        return one_hot
