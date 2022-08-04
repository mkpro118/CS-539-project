from typing import Union
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
from ..utils.exports import export

# List of activation functions mapped to their names
activation_symbol_map = {name.lower(): symbol for name, symbol in symbol_map.items()}
activation_symbol_map_inv = {v: k for k, v in activation_symbol_map.items()}
activation_symbol_map_inv[None] = None


errors = {
    'WeightInitializationError': ExceptionFactory.register('WeightInitializationError'),
    'OptimizationError': ExceptionFactory.register('OptimizationError'),
    'InvalidInvocationError': ExceptionFactory.register('InvalidInvocationError'),
    'InvalidConstraintsError': ExceptionFactory.register('InvalidConstraintsError'),
    'InvalidActivationError': ExceptionFactory.register('InvalidActivationError'),
    'NotImplementedError': ExceptionFactory.register('NotImplementedError'),
}


@export
@mixin  # Prevents instantiation
class Layer(MetadataMixin, SaveMixin):
    '''
    Base class for all Layers.

    Descendant classes must implement the following methods
    build(input_dim): To initialize layer's variables
    optimize(args): For trainable layers, this optimization logic for the variables
    forward(input): Apply this layer on the given inputs in forward propagation
    backward(input, gradient): Apply this layer on the given inputs in backward propagation

    It is also recommended to implement __init__, to define custom attributes and variables

    Inherited from MetadataMixin
        method `get_metadata` to compute layer's metadata
    Inherited from SaveMixin
        method `save` to save layer's metadata
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
                If of type string, must be one of the following
                    'leaky_relu'
                    'relu'
                    'sigmoid'
                    'softmax'
                    'tanh'
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

        if not self._trainable:
            MethodInvalidator.register(self.optimize)

        self._rng = np.random.default_rng()
        self.built = False

    @type_safe
    @not_none
    def _check_activation(self):
        '''
        Assigns the activation function to be applied to the result of this layer

        If activation=None, invalidates the apply_activation method
        If activation inherits from ActivationMixin, the given activation is assigned
        If activation is of type string,
        '''
        if self.activation is None:
            return MethodInvalidator.register(self.apply_activation)

        if isinstance(self.activation, ActivationMixin):
            self._activation = activation_symbol_map_inv.get(self.activation, self.activation.__class__.__name__)
            return

        try:
            self._activation = self.activation
            self.activation = activation_symbol_map[self.activation.replace('_', '').lower()]
        except KeyError:
            raise errors['InvalidActivationError'](
                f'activation={self.activation} is not a recognized activation function'
            )

    @MethodInvalidator.check_validity(invalid_logic=lambda self, X: X)
    @type_safe
    def apply_activation(self, X: np.ndarray) -> np.ndarray:
        '''
        Applies the activation function to the given inputs

        If activation=None, returns the inputs given

        Parameters:
            X: np.ndarray
                The inputs to apply the activation function on

        Returns:
            np.ndarray: The activated inputs
        '''
        return self.activation.apply(X)

    @type_safe
    def _check_constraints(self):
        '''
        Shortcut to check the constraints for the weights and biases

        if use_bias=False, invalidates the ensure_bias_constraints() method
        if bias_constraints=None, invalidates the ensure_bias_constraints() method
        if weights_constraints=None, invalidates the ensure_weights_constraints() method
        '''
        if not self.use_bias:
            MethodInvalidator.register(self._check_bias_constraints)
            MethodInvalidator.register(self.ensure_bias_constraints)

        self._check_weights_constraints()
        self._check_bias_constraints()

    @type_safe
    def _check_weights_constraints(self):
        '''
        Checks the constrains for the weights

        if weights_constraints=None, invalidates the ensure_weight_constraints() method
        '''
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
        '''
        Checks the constrains for the weights

        if weights_constraints=None, invalidates the ensure_weight_constraints() method
        '''
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
        '''
        Clips the weights inplace to conform to the constraints

        if weights_constraints=None, this method is not executed
        '''
        np.clip(
            self.weights,
            self.weights_constraints[0],
            self.weights_constraints[1],
            out=self.weights
        )

    @MethodInvalidator.check_validity
    def ensure_bias_constraints(self):
        '''
        Clips the bias inplace to conform to the constraints

        if bias_constraints=None, this method is not executed
        '''
        np.clip(
            self.bias,
            self.bias_constraints[0],
            self.bias_constraints[1],
            out=self.bias
        )

    def build(self, *args, **kwargs):
        '''
        Defines the logic for generating the layer's variables
        NOTE: This method must set the `built` attribute of the layer to True

        Must be implemented by descendent classes

        Parameters:
            Must be defined by subclasses

        Returns:
            Must be defined by subclasses
        '''
        raise errors['NotImplementedError'](
            f'build is not implemented'
        )

    def optimize(self, *args, **kwargs):
        '''
        Defines the logic for optimization during back propagation for trainable layers

        Must be implemented by descendent classes

        Parameters:
            Must be defined by subclasses

        Returns:
            Must be defined by subclasses
        '''
        if self.trainable:
            raise errors['NotImplementedError'](
                f'optimization is not implemented'
            )

    def forward(self, *args, **kwargs):
        '''
        Defines the layer's logic for forward propagation

        Must be implemented by descendent classes

        Parameters:
            Must be defined by subclasses

        Returns:
            Must be defined by subclasses
        '''
        raise errors['NotImplementedError'](
            f'forward propagation is not implemented'
        )

    def backward(self, *args, **kwargs):
        '''
        Defines the layer's logic for back propagation

        Must be implemented by descendent classes

        Parameters:
            Must be defined by subclasses

        Returns:
            Must be defined by subclasses
        '''
        raise errors['NotImplementedError'](
            f'backward propagation is not implemented'
        )

    @type_safe
    @not_none(nullable=('scale',))
    def generate_weights(self, shape: Union[np.ndarray, list, tuple], *,
                         mean: Union[np.floating, np.integer, float, Real, int, Integral] = None,
                         std: Union[np.floating, np.integer, float, Real, int, Integral] = None,
                         scale: Union[np.ndarray, list, tuple, Scaler] = None,
                         from_rng: np.random.Generator = None) -> np.ndarray:
        _shape = tuple(shape)

        _mean = mean if mean else 0.0
        _std = std if std else 0.1

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

        optimizer = getattr(self, 'optimize', None)

        if optimizer is None or not callable(optimizer):
            return

        if self._trainable:
            MethodInvalidator.validate(optimizer)
        else:
            MethodInvalidator.register(optimizer)

    @type_safe
    @not_none
    def get_metadata(self):
        data = super().get_metadata()
        data.update({
            'activation': self._activation,
        })
        return data

    @type_safe
    def __str__(self):
        return f'{self.__class__.__name__} Layer'

    @type_safe
    def __repr__(self):
        return str(self)
