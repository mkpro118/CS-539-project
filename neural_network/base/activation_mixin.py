import numpy as np
from functools import wraps

from .mixin import mixin
from .layer_mixin import LayerMixin
from ..utils.typesafety import type_safe, not_none


@mixin  # Prevents instantiation
class ActivationMixin:
    '''
    Mixin Class for all Activation Layers

    Provides methods to performs forward and backward propagation through this
    layer

    Requires all derived class to define at least 2 methods,
    `apply` and `derivative`, which contain the logic and
    mathematical operations for the activation function

    Both `apply` and `derivative` must take in exactly 1
    postional argument. It is guaranteed that the arguments passed into both
    `apply` and `derivative` is of type numpy.ndarray
    '''

    def __init__(self):
        '''
        Ensure that `apply` and `derivative` exist
        '''
        # To display all errors at once
        errors = []

        # Ensure that the subclass defines method `apply`
        activation_fn = getattr(self, 'apply', None)
        if not callable(activation_fn):
            errors.append(
                f'{self.__class__} must explicitly define the '
                f'`apply(input_: np.ndarray) -> np.ndarray` '
                f'function to specify the activation function'
            )

        # Ensure that the subclass defines method `derivative`
        derivative_fn = getattr(self, 'derivative', None)
        if not callable(derivative_fn):
            errors.append(
                f'{self.__class__} must explicitly define the '
                f'`derivative(input_: np.ndarray) -> np.ndarray` '
                f'function to specify the derivative of the activation function'
            )

        if (n := len(errors)) > 0:
            errors = '\n'.join(errors)
            raise TypeError(f'{n} errors in {self.__class__}\n{errors}')

    def __str__(self):
        '''
        Mixin method for to return a string representation of the activation
        layer

        Format: '<ClassName> | Activation Layer'
        '''
        return f'{self.__class__} | Activation Layer'

    def __repr__(self):
        '''
        Mixin method for to return a string representation of the activation
        layer

        Format: '<ClassName> | Activation Layer'
        '''
        return str(self)
