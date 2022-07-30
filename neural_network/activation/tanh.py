import numpy as np

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from ..base.activation_mixin import ActivationMixin


@export
class Tanh(ActivationMixin):
    '''
    Provides methods for the Tanh activation function

    It is defined as
    f(X) = (1 + e ** -X) ** -1

    It's derivative is defined as
    f'(X) = 1 - f(X) ** 2
    '''

    @staticmethod
    @type_safe
    @not_none
    def apply(X: np.ndarray) -> np.ndarray:
        '''
        Apply the Tanh activation function on X

        Parameters:
            X: np.ndarray
                The array to apply Tanh on

        Returns:
            np.ndarray: The activated array
        '''
        return np.tanh(X)

    @staticmethod
    @type_safe
    @not_none
    def derivative(X: np.ndarray) -> np.ndarray:
        '''
        Compute the derivative of Tanh for each value in X

        Parameters:
            X: np.ndarray
                The array to compute the derivative of Tanh on

        Returns:
            np.ndarray: The derivative with respect to the Tanh activation function
        '''
        return 1. - Tanh.apply(X) ** 2.
