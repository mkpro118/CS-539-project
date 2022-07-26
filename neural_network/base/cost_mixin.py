from .mixin import mixin
from .layer_mixin import LayerMixin


@mixin  # Prevents instantiation
class CostMixin(LayerMixin):
    '''
    Mixin Class for all Cost Layers

    Provides methods to performs forward and backward propagation through this
    layer

    Requires all derived class to define at least 2 methods,
    `cost` and `cost_derivative`, which contain the logic and
    mathematical operations for the cost function

    Both `cost` and `cost_derivative` must take in exactly 1 positional
    argument. It is guaranteed that the arguments passed into both `cost`
    and `cost_derivative` is of type numpy.ndarray

    Inherited from LayerMixin
        `get_metadata`
    '''

    def __init__(self):
        errors = []

        # Ensure that the subclass defines method `apply`
        cost_fn = getattr(self, 'apply', None)
        if not callable(cost_fn):
            errors.append(
                f'{self.__class__} must explicitly define the '
                f'`apply(input_: np.ndarray) -> np.ndarray` '
                f'function to specify the cost function'
            )

        # Ensure that the subclass defines method `derivative`
        derivative_fn = getattr(self, 'derivative', None)
        if not callable(derivative_fn):
            errors.append(
                f'{self.__class__} must explicitly define the '
                f'`derivative(input_: np.ndarray) -> np.ndarray` '
                f'function to specify the derivative of the cost function'
            )

        if (n := len(errors)) > 0:
            errors = '\n'.join(errors)
            raise TypeError(
                f'{n} error{"s" if n > 1 else ""} in {self.__class__}\n{errors}'
            )

    def __str__(self):
        return f'{self.__class__} | Cost Layer'

    def __repr__(self):
        return str(self)
