'''Allow classes to be accesed from package level imports'''

from .leaky_relu import LeakyReLU, __name_to_symbol_map__ as leaky_relu_symbol_map
from .relu import ReLU, __name_to_symbol_map__ as relu_symbol_map
from .sigmoid import Sigmoid, __name_to_symbol_map__ as sigmoid_symbol_map
from .softmax import Softmax, __name_to_symbol_map__ as softmax_symbol_map
from .tanh import Tanh, __name_to_symbol_map__ as tanh_symbol_map

from ..base.activation_mixin import ActivationMixin
from ..utils.exports import export


@export
class NoActivation(ActivationMixin):
    @staticmethod
    def apply(X):
        return X

    @staticmethod
    def derivative(X):
        return X


__name_to_symbol_map__ = {
    **leaky_relu_symbol_map,
    **relu_symbol_map,
    **sigmoid_symbol_map,
    **softmax_symbol_map,
    **tanh_symbol_map,
    'NoActivation': NoActivation,
}
