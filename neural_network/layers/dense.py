from typing import Union, Callable
from numbers import Integral, Real
import numpy as np

from ..base.layer import Layer
from ..base.activation_mixin import ActivationMixin
from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


errors = {
    'DenseLayerError': ExceptionFactory.register('DenseLayerError'),
}


@export
class Dense(Layer):
    @type_safe
    @not_none(nullable=('input_shape', 'activation', 'weights_constraints', 'bias_constraints',))
    def __init__(self, nodes: Union[int, Integral, np.integer], *,
                 input_shape: Union[int, np.ndarray, list, tuple] = None,
                 activation: Union[str, ActivationMixin] = None,
                 trainable: bool = True,
                 use_bias: bool = True,
                 weights_constraints: Union[np.ndarray, list, tuple] = None,
                 bias_constraints: Union[np.ndarray, list, tuple] = None,
                 learning_rate: Union[np.floating, np.integer, float, Real, int, Integral] = 1e-2):
        super().__init__(
            activation=activation,
            trainable=trainable,
            use_bias=use_bias,
            weights_constraints=weights_constraints,
            bias_constraints=bias_constraints
        )
        self.nodes = int(nodes)
        self.learning_rate = float(learning_rate)

        if input_shape:
            self.input_shape = np.asarray(input_shape)

    @type_safe
    def build(self, input_shape: Union[np.ndarray, list, tuple, int, Integral, np.integer] = None):
        if isinstance(input_shape, (int, Integral, np.integer)):
            input_shape = (input_shape,)
        input_shape = np.asarray(input_shape, dtype=int)

        if hasattr(self, 'input_shape') and not np.array_equal(self.input_shape, input_shape):
            raise errors['DenseLayerError'](
                f'The given input dimension input_shape={self.input_shape} does not match'
                f' the previous layer\'s output dimension {input_shape}'
            )

        self.input_shape = input_shape

        self.weights = self.generate_weights((*self.input_shape, self.nodes))
        self.bias = self.generate_weights((1, self.nodes))
        if not self.use_bias:
            self.bias[:] = 0

        return self.nodes

    @type_safe
    @not_none
    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.shape[-1] != self.input_shape[-1]:
            raise errors['DenseLayerError'](
                f'given input\'s feature shape is not equal to the expected input feature shape, '
                f'{X.shape[-1]} != {self.input_shape[-1]}'
            )
        self._X = X
        return self.apply_activation(self._X @ self.weights + self.bias)

    @type_safe
    @not_none
    def backward(self, gradient: np.ndarray, optimizer: Callable):
        self._gradient = self.weights.T @ gradient
        self.optimize(gradient)
        return self._gradient

    @type_safe
    def optimize(self, gradient: np.ndarray):
        self.weights -= (self.learning_rate / len(self._x)) * (self._X.T @ gradient)
        self.bias -= (self.learning_rate / len(self._x)) * gradient
