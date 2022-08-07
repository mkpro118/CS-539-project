import numpy as np

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from ..base.activation_mixin import ActivationMixin


@export
class Sigmoid(ActivationMixin):
    name = 'sigmoid'

    '''
    Provides methods for the Sigmoid activation function

    It is defined as
    f(X) = (1 + e ** -X) ** -1

    It's derivative is defined as
    f'(X) = f(X) * (1 - f(X))
    '''

    @staticmethod
    @type_safe
    @not_none
    def apply(X: np.ndarray) -> np.ndarray:
        '''
        Apply the Sigmoid activation function on X

        Parameters:
            X: np.ndarray
                The array to apply Sigmoid on

        Returns:
            np.ndarray: The activated array
        '''
        return np.power((1 + np.exp(-X)), -1)

    @staticmethod
    @type_safe
    @not_none
    def derivative(X: np.ndarray) -> np.ndarray:
        '''
        Compute the derivative of Sigmoid for each value in X

        Parameters:
            X: np.ndarray
                The array to compute the derivative of Sigmoid on

        Returns:
            np.ndarray: The derivative with respect to the Sigmoid activation function
        '''
        fx = Sigmoid.apply(X)
        return fx * (1 - fx)
