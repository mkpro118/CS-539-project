import numpy as np

from .mixin import mixin
from .layer_mixin import LayerMixin
from ..utils.typesafety import type_safe


@mixin
class ActivationMixin(LayerMixin):
    def __init__(self):
        errors = []

        activation_fn = getattr(self, 'activation', None)
        if not callable(activation_fn):
            errors.append(
                f'{self.__class__} must explicitly define the '
                f'`activation_derivative(self, input_)` function to specify '
                f'the derivative of the activation function'
            )

        derivative_fn = getattr(self, 'activation_derivative', None)
        if not callable(derivative_fn):
            errors.append(
                f'{self.__class__} must explicitly define the '
                f'`activation(self, input_)` function to specify the activation function'
            )

        if (n := len(errors)) > 0:
            errors = '\n'.join(errors)
            raise TypeError(f'{n} errors in {self.__class__}\n{errors}')

    def __str__(self):
        return f'{self.__class__} | Activation Layer'

    def __repr__(self):
        return str(self)

    @type_safe
    def backward(self, input_):
        return input_ * self.activation_derivative(input_)

    @type_safe
    def forward(self, input_):
        return self.activation(input_)
