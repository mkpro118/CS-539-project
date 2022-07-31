import numpy as np
from typing import Union

from ..base.metadata_mixin import MetadataMixin
from ..base.save_mixin import SaveMixin
from ..base.transform_mixin import TransformMixin
from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export

errors = {
    'OneHotEncoderError': ExceptionFactory.register('OneHotEncoderError'),
    'UniqueLabelError': ExceptionFactory.register('UniqueLabelError'),
    'UnkownLabelError': ExceptionFactory.register('UnkownLabelError'),
}


@export
class OneHotEncoder(MetadataMixin, SaveMixin, TransformMixin):
    '''
    Used to one hot encode discrete labels

    This method uses numpy.unique to determine different class labels
    and since numpy.unique returns a sorted array of unique values,
    the order of appearance of labels will not be preserved.

    The labels will be encoded in the way they appear in the array returned
    by numpy.unique, so the smallest label will correspond to the first index
    and the largest label will correspond to the last index of the
    one hot encoded vector
    '''

    @type_safe(skip=('y',))
    @not_none(nullable=('y',))
    def fit(self, X: Union[np.ndarray, list, tuple], y: np.ndarray = None, **kwargs) -> 'OneHotEncoder':
        '''
        Fits the encoder with the unique label values

        Paramters:
            X: numpy.ndarray of shape (n,)
                A 1 dimensional label vector to fit the encoder

        Returns:
            self: the fitted encoder instance
        '''
        self.classes = self._validate_params(X)
        self.n_classes = len(self.classes)

        self.kwargs = kwargs
        self._attrs = ('n_classes', 'classes')
        return self

    @type_safe(skip=('y',))
    @not_none(nullable=('y',))
    def transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> np.ndarray:
        '''
        Transform the given label vector into a one hot encoded vectors

        Parameters:
            X: numpy.ndarray of shape (n,)
                The label vector to transform into one hot encoded vectors

        Returns:
            numpy.ndarray: The matrix whose rows are the one hot encoded vectors
        '''
        self._check_is_fitted()
        one_hot = np.zeros((len(X), self.n_classes), dtype=int)
        for index, class_ in enumerate(self.classes):
            one_hot[X == class_, index] = 1
        return one_hot

    @type_safe
    @not_none
    def _validate_params(self, labels: Union[np.ndarray, list, tuple]) -> np.ndarray:
        labels = np.asarray(labels)

        if labels.ndim != 1:
            raise errors['OneHotEncoderError'](
                f'labels array must be 1 dimensional'
            )

        return np.unique(labels)
