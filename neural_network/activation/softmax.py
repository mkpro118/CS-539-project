import numpy as np

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from ..base.activation_mixin import ActivationMixin


@export
class Softmax(ActivationMixin):
    '''
    Provides methods for the Softmax activation function

    It is defined as
    f(X) = e ** X) / sum(e ** X)

    It's derivative is defined as (in conjunction with Cross Entropy)
    f'(X) = 1
    '''

    @staticmethod
    @type_safe
    @not_none
    def apply(X: np.ndarray) -> np.ndarray:
        '''
        Apply the Softmax activation function on X

        Parameters:
            X: np.ndarray
                The array to apply Softmax on

        Returns:
            np.ndarray: The activated array
        '''
        return (_ := np.exp(X)) / np.sum(_, axis=1, keepdims=True)

    @staticmethod
    @type_safe
    @not_none
    def derivative(X: np.ndarray) -> np.ndarray:
        '''
        Compute the derivative of Softmax for each value in X

        Parameters:
            X: np.ndarray
                The array to compute the derivative of Softmax on

        Returns:
            np.ndarray: The derivative with respect to the Softmax activation function
        '''
        return np.ones(X.shape, dtype=float)
