from typing import Union
from numbers import Integral, Real
import numpy as np
from scipy.signal import convolve2d, correlate2d

from ..base.layer import Layer
from ..base.activation_mixin import ActivationMixin
from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


errors = {
    'ConvolutionalLayerError': ExceptionFactory.register('ConvolutionalLayerError'),
}


@export
class Convolutional(Layer):

    pos_params = (
        'filters',
        'kernel_size',
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
        'channels',
        'filters',
        'height',
        'input_shape',
        'kernel_size',
        'learning_rate',
        'name',
        'output_shape',
        'trainable',
        'use_bias',
        'weights',
        'weights_constraints',
        'width',
    }

    @type_safe
    @not_none(nullable=('input_shape', 'activation', 'weights_constraints', 'bias_constraints',))
    def __init__(self, filters: Union[int, Integral, np.integer],
                 kernel_size: Union[int, Integral, np.integer], *,
                 input_shape: Union[np.ndarray, list, tuple] = None,
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
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.learning_rate = float(learning_rate)

        if input_shape is not None:
            self.input_shape = np.asarray(input_shape)
            if self.input_shape.ndim != 1 or len(self.input_shape) != 3:
                raise errors['ConvolutionalLayerError'](
                    f'Input shape must be a tuple, list or ndarray containing exactly 3 values, '
                    f'(number of channels, image height, image width), found {input_shape}'
                )

    @type_safe
    @not_none(nullable=('input_shape',))
    def build(self,
              _id: int,
              input_shape: Union[np.ndarray, list, tuple, int, Integral, np.integer] = None):
        self._id = _id
        self.name = self.name or f'Convolutional Layer'

        input_shape = np.asarray(input_shape, dtype=int)
        if input_shape.ndim != 1 or len(input_shape) != 3:
            raise errors['ConvolutionalLayerError'](
                f'Input shape must be a tuple, list or ndarray containing exactly 3 values, '
                f'(number of channels, image height, image width), found {input_shape}'
            )

        if hasattr(self, 'input_shape') and not np.array_equal(self.input_shape, input_shape):
            raise errors['ConvolutionalLayerError'](
                f'The given input dimension input_shape={self.input_shape} does not match'
                f' the previous layer\'s output dimension {input_shape}'
            )

        self.input_shape = input_shape
        self.channels, self.height, self.width = self.input_shape

        self.output_shape = (
            self.filters,
            self.height - self.kernel_size + 1,
            self.width - self.kernel_size + 1,
        )
        self.weights = self.generate_weights((
            self.filters,
            self.channels,
            self.kernel_size,
            self.kernel_size,
        ))
        self.bias = self.generate_weights(self.output_shape)
        if not self.use_bias:
            self.bias = np.zeros_like(self.bias)

        self.built = True
        return self.output_shape

    @type_safe
    @not_none
    def forward(self, X: np.ndarray) -> np.ndarray:
        if (X.shape[1:] != self.input_shape).any():
            raise errors['ConvolutionalLayerError'](
                f'given input\'s feature shape is not equal to the expected input feature shape, '
                f'{X.shape[1:]} != {self.input_shape}'
            )
        self._X = X
        output = np.empty(shape=(len(X), *self.output_shape))
        for x in range(len(X)):
            for f in range(self.filters):
                for c in range(self.channels):
                    output[x, f] = correlate2d(self._X[x, c], self.weights[f, c], mode='valid')
        output = output + self.bias
        return self.apply_activation(output)

    @type_safe
    @not_none
    def backward(self, gradient: np.ndarray):
        kernels_gradient = np.empty(self.weights.shape)
        _gradient = np.empty((len(gradient), *self.input_shape))
        for x in range(len(self._X)):
            for f in range(self.filters):
                for c in range(self.channels):
                    kernels_gradient[f, c] = correlate2d(self._X[x, c], gradient[x, f], mode='valid')
                    _gradient[x, c] += convolve2d(gradient[x, f], self.weights[f, c], mode='full')

        self.optimize(kernels_gradient, gradient)
        return _gradient

    @type_safe
    @not_none
    def optimize(self, kernels_gradient: np.ndarray, gradient: np.ndarray):
        self.weights -= (self.learning_rate / len(self._X)) * kernels_gradient
        if self.use_bias:
            self.bias = self.bias - (self.learning_rate / len(self._X))

    def __str__(self):
        if not self.built:
            return (
                f'Convolutional Layer with {self.filters} filter{"s" if self.filters > 1 else ""}, '
                f'{self.kernel_size}x{self.kernel_size} kernel '
                f'(uninitialized)'
            )
        return (
            f'{self.name} with {self.filters} filter{"s" if self.filters > 1 else ""}, '
            f'{self.channels} channel{"s" if self.channels > 1 else ""}, '
            f'{self.kernel_size}x{self.kernel_size} kernel'
        )
