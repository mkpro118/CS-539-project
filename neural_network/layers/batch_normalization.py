from numbers import Integral, Real
from typing import Union
import numpy as np

from ..base.layer import Layer
from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export

errors = {
    'BatchNormLayerError': ExceptionFactory.register('BatchNormLayerError'),
}


@export
class BatchNormalization(Layer):
    pos_params = tuple()

    kw_params = {
        'name',
    }

    _attrs = (
        'bias',
        'bias_constraints',
        'built',
        'input_shape',
        'learning_rate',
        'momentum',
        'moving_mean',
        'moving_var',
        'name',
        'output_shape',
        'trainable',
        'use_bias',
        'weights',
        'weights_constraints',
    )

    @type_safe
    def __init__(self, *, input_shape: Union[int, np.ndarray, list, tuple] = None,
                 trainable: bool = True,
                 use_bias: bool = True,
                 weights_constraints: Union[np.ndarray, list, tuple] = None,
                 bias_constraints: Union[np.ndarray, list, tuple] = None,
                 learning_rate: Union[np.floating, np.integer, float, Real, int, Integral] = 1e-2,
                 name: str = None):
        super().__init__(
            activation=None,
            trainable=trainable,
            use_bias=use_bias,
            weights_constraints=weights_constraints,
            bias_constraints=bias_constraints,
            name=name
        )

        self.learning_rate = float(learning_rate)

        if input_shape:
            if isinstance(input_shape, (int, Integral, np.integer)):
                self.input_shape = np.asarray((input_shape,))
            else:
                self.input_shape = np.asarray(input_shape)

    @type_safe
    @not_none
    def build(self, _id: int,
              input_shape: Union[list, tuple, np.ndarray]) -> tuple:
        self._id = _id
        self.name = self.name or f'Batch Normalization Layer'
        self.input_shape = np.asarray(input_shape)
        self.output_shape = tuple(input_shape)

        self.momentum = self.generate_weights((1,), mean=0.5, std=0.5)
        self.momentum = np.clip(self.momentum, 0.1, 0.9)

        self.eps = 1e-10

        if len(self.input_shape) == 1:
            weight_shape = (1, self.output_shape[0])
        elif len(self.input_shape) == 3:
            weight_shape = (1, self.output_shape[0], 1, 1)
        else:
            raise errors['BatchNormLayerError'](
                f'Input shape = {input_shape} is not supported, '
                f'Only outputs from dense (len(shape)=1) or convolutional (len(shape)=3) layers are supported'
            )

        self.weights = self.generate_weights(weight_shape)
        self.bias = self.generate_weights(weight_shape)

        if not self.use_bias:
            self.bias[:] = 0

        self.moving_mean = np.zeros(weight_shape)
        self.moving_var = np.ones(weight_shape)

        self.built = True
        return tuple((_ for _ in self.output_shape))

    @type_safe
    @not_none
    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        is_training = kwargs.get('is_training', False)

        if not is_training:
            self._X_norm = (X - self.moving_mean) / np.sqrt(self.moving_var + self.eps)
        else:
            self._X = X
            assert len(self._X.shape) in (2, 4)
            if len(self._X.shape) == 2:
                self._mean = self._X.mean(axis=0)
                self._var = ((self._X - self._mean) ** 2).mean(axis=0)
            else:
                self._mean = self._X.mean(axis=(0, 2, 3), keepdims=True)
                self._var = ((self._X - self._mean) ** 2).mean(axis=(0, 2, 3), keepdims=True)
            self._X_norm = (self._X - self._mean) / np.sqrt(self._var + self.eps)

            self.moving_mean = self.momentum * self.moving_mean + (1.0 - self.momentum) * self._mean
            self.moving_var = self.momentum * self.moving_var + (1.0 - self.momentum) * self._var
        return self.weights * self._X_norm + self.bias

    @type_safe
    @not_none
    def backward(self, gradient: np.ndarray, **kwargs) -> np.ndarray:
        dgamma = np.sum(gradient * self._X_norm, axis=0)
        dbeta = np.sum(gradient, axis=0)

        m = self._X.shape[0]
        t = 1. / np.sqrt(self._var + self.eps)

        diff = self._X - self._mean

        dx = (self.weights * t / m) * (m * gradient - np.sum(gradient, axis=0) - t**2 * (diff) * np.sum(gradient * (diff), axis=0))

        self.optimize(dgamma, dbeta)

        return dx

    @type_safe
    @not_none
    def optimize(self, gradient_gamma: np.ndarray, gradient_beta: np.ndarray):
        self.weights = self.weights - self.learning_rate * gradient_gamma
        if self.use_bias:
            self.bias = self.bias - self.learning_rate * gradient_beta

    @property
    def trainable(self):
        return self._trainable

    @trainable.setter
    def trainable(self, value):
        pass

    def __str__(self):
        return f'{self.name}'
