import numpy as np

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from ..base.activation_mixin import ActivationMixin
from .leaky_relu import LeakyReLU


@export
class ReLU(LeakyReLU, ActivationMixin):
    '''
    Provides methods for the ReLU activation function

    It is defined as
    X = X {X >= 0}
    X = 0 {X < 0}

    It's derivative is defined as
    X = 1 {X > 0}
    X = 0 {X <= 0}
    '''

    @staticmethod
    @type_safe
    @not_none
    def apply(X: np.ndarray) -> np.ndarray:
        '''
        Apply the ReLU activation function on X

        Parameters:
            X: np.ndarray
                The array to apply ReLU on

        Returns:
            np.ndarray: The activated array
        '''
        return super(ReLU, ReLU).apply(X, coef=0.) + 0.

    @staticmethod
    @type_safe
    @not_none
    def derivative(X: np.ndarray) -> np.ndarray:
        '''
        Compute the derivative of ReLU for each value in X

        Parameters:
            X: np.ndarray
                The array to compute the derivative of ReLU on

        Returns:
            np.ndarray: The derivative with respect to the ReLU activation function
        '''
        return super(ReLU, ReLU).derivative(X, coef=0.) + 0.
