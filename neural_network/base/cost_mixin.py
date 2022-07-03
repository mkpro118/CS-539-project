import numpy as np

from .mixin import mixin
from .layer_mixin import LayerMixin
from ..utils.typesafety import type_safe


@mixin
class CostMixin(LayerMixin):
    def __init__(self):
        errors = []

        cost_fn = getattr(self, 'cost', None)
        if not callable(cost_fn):
            errors.append(
                f'{self.__class__} must explicitly define the '
                f'`cost_derivative(self, input_)` function to specify '
                f'the derivative of the cost function'
            )

        derivative_fn = getattr(self, 'cost_derivative', None)
        if not callable(derivative_fn):
            errors.append(
                f'{self.__class__} must explicitly define the '
                f'`cost(self, input_)` function to specify the cost function'
            )

        if (n := len(errors)) > 0:
            errors = '\n'.join(errors)
            raise TypeError(f'{n} errors in {self.__class__}\n{errors}')

    def __str__(self):
        return f'{self.__class__} | Cost Layer'

    def __repr__(self):
        return str(self)

    @type_safe(skip=('self',))
    def backward(self, input_: np.ndarray) -> np.ndarray:
        return input_ * self.cost_derivative(input_)

    @type_safe(skip=('self',))
    def forward(self, input_: np.ndarray) -> np.ndarray:
        return self.cost(input_)
