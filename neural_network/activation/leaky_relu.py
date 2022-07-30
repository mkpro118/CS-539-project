import numpy as np

from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export
from ..base.activation_mixin import ActivationMixin


@export
class LeakyReLU(ActivationMixin):
    '''
    Leaky ReLU is a variant of the ReLU function

    It is defined as
    X = X {X >= 0}
    X = coef * X {X < 0}

    It's derivative is defined as
    X = 1 {X > 0}
    X = - {X = 0}
    X = coef {X <= 0}

    By default, coef = 0.01
    coef can be changed by specifying a keyword argument `coef`
    on the apply and derivative methods
    '''

    @staticmethod
    @type_safe
    @not_none
    def apply(X: np.ndarray, *, coef: float = None) -> np.ndarray:
        '''
        Apply the Leaky ReLU activation function on X

        Parameters:
            X: np.ndarray
                The array to apply Leaky ReLU on
            coef: float, keyword only, default = 0.01
                The scaling coefficient for negative values

        Returns:
            np.ndarray: The activated array
        '''
        if coef is None:
            coef = 0.01
        X = X.astype(float)
        X[X < 0] = coef * X[X < 0]
        return X

    @staticmethod
    @type_safe
    @not_none
    def derivative(X: np.ndarray, *, coef: float = None) -> np.ndarray:
        '''
        Compute the derivative of Leaky ReLU for each value in X

        Parameters:
            X: np.ndarray
                The array to compute the derivative of Leaky ReLU on
            coef: float, keyword only, default = 0.01
                The scaling coefficient for negative values

        Returns:
            np.ndarray: The derivative with respect to the Leaky ReLU activation function
        '''
        if coef is None:
            coef = 0.01
        X = X.astype(float)
        X[X > 0] = 1
        X[X == 0] = 0
        X[X < 0] = coef
        return X
