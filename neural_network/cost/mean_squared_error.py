import numpy as np
from typing import Union
from numbers import Number

from ..base.cost_mixin import CostMixin
from ..base.metadata_mixin import MetadataMixin
from ..base.save_mixin import SaveMixin

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class MeanSquaredError(CostMixin, MetadataMixin, SaveMixin):
    name = 'mse'

    '''
    Provides static methods to compute the mean squared error loss
    and it's derivative.
    '''
    @staticmethod
    @type_safe
    @not_none
    def apply(y_true: np.ndarray, y_pred: np.ndarray) -> Union[Number, np.number]:
        '''
        Applies the mean squared error loss on the given labels and predictions

        Parameters:
            y_true: numpy.ndarray of shape (n_samples, n_labels)
                The known true labels (n_samples, n_labels)
            y_pred: numpy.ndarray of shape (n_samples, n_labels)
                The predicted values
        Returns:
            float: average loss over all samples
        '''
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    @type_safe
    @not_none
    def derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        '''
        Applies the derivative of mean squared error loss
        on the given labels and predictions

        Parameters:
            y_true: numpy.ndarray of shape (n_samples, n_labels)
                The known true labels (n_samples, n_labels)
            y_pred: numpy.ndarray of shape (n_samples, n_labels)
                The predicted values

        Returns:
            numpy.ndarray: The gradient with respect to each sample's known and predicted labels
        '''
        return y_pred - y_true
