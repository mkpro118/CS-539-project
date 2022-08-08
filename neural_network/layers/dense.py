from typing import Union
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

    pos_params = (
        'nodes',
    )

    kw_params = (
        'input_shape',
        'activation',
        'trainable',
        'use_bias',
        'weights_constraints',
        'bias_constraints',
        'learning_rate',
    )

    _attrs = {
        'activation',
        'bias',
        'bias_constraints',
        'built',
        'input_shape',
        'learning_rate',
        'name',
        'nodes',
        'output_shape',
        'trainable',
        'use_bias',
        'weights',
        'weights_constraints',
    }

    @type_safe
    @not_none(nullable=('input_shape', 'activation', 'weights_constraints', 'bias_constraints',))
    def __init__(self, nodes: Union[int, Integral, np.integer], *,
                 input_shape: Union[int, np.ndarray, list, tuple] = None,
                 activation: Union[str, ActivationMixin] = None,
                 trainable: bool = True,
                 use_bias: bool = True,
                 weights_constraints: Union[np.ndarray, list, tuple] = None,
                 bias_constraints: Union[np.ndarray, list, tuple] = None,
                 learning_rate: Union[np.floating, np.integer, float, Real, int, Integral] = 1e-2,
                 name: str = None):
        super().__init__(
            activation=activation,
            trainable=trainable,
            use_bias=use_bias,
            weights_constraints=weights_constraints,
            bias_constraints=bias_constraints,
            name=name
        )
        self.nodes = int(nodes)
        self.learning_rate = float(learning_rate)

        if input_shape:
            if isinstance(input_shape, (int, Integral, np.integer)):
                self.input_shape = np.asarray((input_shape,))
            else:
                self.input_shape = np.asarray(input_shape)

    @type_safe
    @not_none(nullable=('input_shape',))
    def build(self,
              _id: int,
              input_shape: Union[np.ndarray, list, tuple, int, Integral, np.integer] = None):

        self._id = _id
        self.name = self.name or f'Dense Layer'
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

        self.built = True
        self.output_shape = (self.nodes,)
        return self.output_shape

    @type_safe
    @not_none
    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if X.shape[-1] != self.input_shape[-1]:
            raise errors['DenseLayerError'](
                f'given input\'s feature shape is not equal to the expected input feature shape, '
                f'{X.shape[-1]} != {self.input_shape[-1]}'
            )
        self._X = X
        result = super().apply_activation(self._X @ self.weights + self.bias)
        return result

    @type_safe
    @not_none
    def backward(self, gradient: np.ndarray, **kwargs):
        _gradient = gradient @ self.weights.T
        self.optimize(gradient)
        return _gradient

    @type_safe
    @not_none
    def optimize(self, gradient: np.ndarray):
        self.weights -= (self.learning_rate / len(self._X)) * (self._X.T @ gradient)
        if self.use_bias:
            self.bias -= (self.learning_rate / len(self._X))

    def __str__(self):
        if not self.built:
            return f'Dense Layer with {self.nodes} nodes (uninitialized)'
        return f'{self.name} with {self.nodes} nodes'
