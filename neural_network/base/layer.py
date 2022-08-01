from typing import Union, Callable
from numbers import Integral, Real
import numpy as np

from .mixin import mixin
from .activation_mixin import ActivationMixin
from .metadata_mixin import MetadataMixin
from .save_mixin import SaveMixin

from ..activation import __name_to_symbol_map__ as symbol_map
from ..exceptions import ExceptionFactory
from ..preprocess import Scaler
from ..utils.typesafety import type_safe, not_none
from ..utils.functools import MethodInvalidator

# List of activation functions mapped to their names
activation_symbol_map = {name.lower(): symbol for name, symbol in symbol_map.items()}

errors = {
    'WeightInitializationError': ExceptionFactory.register('WeightInitializationError'),
    'OptimizationError': ExceptionFactory.register('OptimizationError'),
    'InvalidInvocationError': ExceptionFactory.register('InvalidInvocationError'),
    'InvalidConstraintsError': ExceptionFactory.register('InvalidConstraintsError'),
    'InvalidActivationError': ExceptionFactory.register('InvalidActivationError'),
    'NotImplementedError': ExceptionFactory.register('NotImplementedError'),
}


@mixin  # Prevents instantiation
class Layer(MetadataMixin, SaveMixin):
    '''
    Base class for all Layers.

    Inherited from MetadataMixin
        method `get_metadata` to computer layer's metadata
    '''

    @type_safe
    @not_none
    def __init__(self, *, activation: Union[str, ActivationMixin] = None,
                 use_bias: bool = True,
                 trainable: bool = True,
                 weights_constraints: Union[np.ndarray, list, tuple] = None,
                 bias_constraints: Union[np.ndarray, list, tuple] = None,
                 name: str = None):
        '''
        Parameters: all params are keyword only
            activation: Union[str, ActivationMixin], default = None
                The activation function to apply to the result of this layer
            use_bias: bool, default = True
                Add a bias to the result of this layer
            trainable: bool, default = True
                Sets the layers variables to be trainable or not
            weights_constrains: Union[np.ndarray, list, tuple] of shape (2,), default = None
                Sets a bound on this layer's weights
            bias_constrains: Union[np.ndarray, list, tuple] of shape (2,), default = None
                Sets a bound on this layer's bias
            name: str, default = None
                The name of this layer, must be unique in a model
        '''
        self.activation = activation
        self.use_bias = use_bias
        self._trainable = trainable
        self.weights_constraints = weights_constraints
        self.bias_constraints = bias_constraints
        self.name = name

        self._check_activation()
        self._check_constraints()

        self._rng = np.random.default_rng()
        self.built = False

    @type_safe
    @not_none
    def _check_activation(self):
        if self.activation is None:
            return MethodInvalidator.register(self.apply_activation)

        if isinstance(self.activation, ActivationMixin):
            return

        try:
            self.activation = activation_symbol_map[self.activation.replace('_', '').lower()]
        except KeyError:
            raise errors['InvalidActivationError'](
                f'activation={self.activation} is not a recognized activation function'
            )

    @MethodInvalidator.check_validity(invalid_logic=lambda self, X: X)
    @type_safe
    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        return self.activation.apply(X)

    @type_safe
    def _check_constraints(self):
        if not self.use_bias:
            MethodInvalidator.register(self._check_bias_constraints)
            MethodInvalidator.register(self.ensure_bias_constraints)

        self._check_weights_constraints()
        self._check_bias_constraints()

    @type_safe
    def _check_weights_constraints(self):
        if self.weights_constraints is None:
            return MethodInvalidator.register(self.ensure_weight_constraints)

        constraint = np.asarray(self.weights_constraints)

        if constraint.ndim != 1 or constraint.shape[-1] != 2:
            raise errors['InvalidConstraintsError'](
                f'weights_constraints parameter must be 1 dimensional with the first index being the '
                f'lower bound of the weights, and the second index being the upper '
                f'bound of the weights, found {constraint=}'
            )

    @MethodInvalidator.check_validity
    def _check_bias_constraints(self):
        if self.bias_constraints is None:
            return MethodInvalidator.register(self.ensure_bias_constraints)

        constraint = np.asarray(self.bias_constraints)

        if constraint.ndim != 1 or constraint.shape[-1] != 2:
            raise errors['InvalidConstraintsError'](
                f'bias_constraints parameter must be 1 dimensional with the first index being the '
                f'lower bound of the bias, and the second index being the upper '
                f'bound of the bias, found {constraint=}'
            )

    @MethodInvalidator.check_validity
    def ensure_weight_constraints(self):
        np.clip(
            self.weights,
            self.weights_constraints[0],
            self.weights_constraints[1],
            out=self.weights
        )

    @MethodInvalidator.check_validity
    def ensure_bias_constraints(self):
        np.clip(
            self.bias,
            self.bias_constraints[0],
            self.bias_constraints[1],
            out=self.bias
        )

    def build(self, *args, **kwargs):
        raise errors['NotImplementedError'](
            f'build is not implemented'
        )

    def optimize(self, *args, **kwargs):
        if self.trainable:
            raise errors['NotImplementedError'](
                f'optimization is not implemented'
            )

    def forward(self, *args, **kwargs):
        raise errors['NotImplementedError'](
            f'forward propagation is not implemented'
        )

    def backward(self, *args, **kwargs):
        raise errors['NotImplementedError'](
            f'backward propagation is not implemented'
        )

    def __call__(self, *args, backward, **kwargs):
        if not self.built:
            return self.build(*args, **kwargs)

        if backward:
            return self.backward(*args, **kwargs)
        else:
            return self.forward(*args, **kwargs)

    @type_safe
    @not_none(nullable=('scale',))
    def generate_weights(self, shape: Union[np.ndarray, list, tuple], *,
                         mean: Union[np.floating, np.integer, float, Real, int, Integral] = None,
                         std: Union[np.floating, np.integer, float, Real, int, Integral] = None,
                         scale: Union[np.ndarray, list, tuple, Scaler] = None,
                         from_rng: np.random.Generator = None) -> np.ndarray:
        _shape = tuple(shape)

        _mean = mean if mean else 0.0
        _std = std if std else 1.0

        if from_rng:
            _weights = from_rng.normal(loc=_mean, scale=_std, size=_shape)
        else:
            _weights = self._rng.normal(loc=_mean, scale=_std, size=_shape)

        if not scale:
            return _weights

        if (mean or std) and scale:
            raise errors['WeightInitializationError'](
                f'The weights cannot be scaled if mean or standard deviation is specified'
            )

        if isinstance(scale, Scaler):
            return scale.fit_transform(_weights)

        try:
            return Scaler(scale[0], scale[1]).fit_transform(_weights)
        except IndexError:
            raise errors['WeightInitializationError'](
                f'scale parameter must be 1 dimensional with the first index being the '
                f'lower bound of the scaled values, and the second index being the upper '
                f'bound of the scaled values, found {scale=}'
            )

    @property
    @type_safe
    def trainable(self) -> bool:
        return self._trainable

    @trainable.setter
    @type_safe
    @not_none
    def trainable(self, value: bool):
        self._trainable = value

        optimizer = self.__dict__.get('optimizer', None)

        if optimizer is None:
            return

        if self._trainable:
            MethodInvalidator.validate(optimizer)
        else:
            MethodInvalidator.register(optimizer)

    @type_safe
    def __str__(self):
        return f'{self.__class__.__name__} Layer'

    @type_safe
    def __repr__(self):
        return str(self)
