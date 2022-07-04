import numpy as np

from .mixin import mixin
from .layer_mixin import LayerMixin
from ..utils.typesafety import type_safe


@mixin  # Prevents instantiation
class ActivationMixin(LayerMixin):
    '''
    Mixin Class for all Activation Layers

    Provides methods to performs forward and backward propagation through this
    layer

    Requires all derived class to define at least 2 methods,
    `activation` and `activation_derivative`, which contain the logic and
    mathematical operations for the activation function

    Both `activation` and `activation_derivative` must take in exactly 1
    postional argument. It is guaranteed that the arguments passed into both
    `activation` and `activation_derivative` is of type numpy.ndarray

    Methods:
        `backward(input_: numpy.ndarray) -> numpy.ndarray`:
            Performs backward propagation, using `activation_derivative`

        `forward(input_: numpy.ndarray) -> numpy.ndarray`:
            Performs forward propagation, using `activation`


    Inherited from LayerMixin
        `get_metadata`
    '''

    def __init__(self):
        '''
        Ensure that `activation` and `activation_derivative` exist
        '''
        # To display all errors at once
        errors = []

        # Ensure that the subclass defines method `activation`
        activation_fn = getattr(self, 'activation', None)
        if not callable(activation_fn):
            errors.append(
                f'{self.__class__} must explicitly define the '
                f'`activation(input_: np.ndarray) -> np.ndarray` '
                f'function to specify the activation function'
            )

        # Ensure that the subclass defines method `activation_derivative`
        derivative_fn = getattr(self, 'activation_derivative', None)
        if not callable(derivative_fn):
            errors.append(
                f'{self.__class__} must explicitly define the '
                f'`activation_derivative(input_: np.ndarray) -> np.ndarray` '
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

    @type_safe
    def backward(self, input_: np.ndarray) -> np.ndarray:
        '''
        To perform backward propagation, using `activation_derivative`

        Parameters:
            input_: numpy.ndarray of shape (n_samples, n_features)
                The input to this layer to update paramters and perform gradient
                descent

        Returns:
            numpy.ndarray: Product of the input and this layer's activation
            function derivative
        '''
        return input_ * self.activation_derivative(input_)

    @type_safe
    def forward(self, input_: np.ndarray) -> np.ndarray:
        '''
        To perform forward propagation, using `activation`

        Parameters:
            input_: numpy.ndarray of shape (n_samples, n_features)
                The input to this layer to be activated

        Returns:
            numpy.ndarray: Activated input
        '''
        return self.activation(input_)
