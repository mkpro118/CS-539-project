# To be implemented

from typing import Union
from numbers import Integral
import numpy as np

from ..base.layer_mixin import LayerMixin
from ..base.activation_mixin import ActivationMixin
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class Dense(LayerMixin):
    @type_safe
    @not_none(nullable=('input_dim', 'activation', 'weights_constraints', 'bias_constraints',))
    def __init__(self, nodes: Union[int, Integral, np.integer], *,
                 input_dim: Union[np.ndarray, list, tuple] = None,
                 activation: Union[str, ActivationMixin] = None,
                 trainable: bool = True,
                 use_bias: bool = True,
                 weights_constraints: Union[np.ndarray, list, tuple] = None,
                 bias_constraints: Union[np.ndarray, list, tuple] = None):
        super().__init__(
            activation=activation,
            trainable=trainable,
            use_bias=use_bias,
            weights_constraints=weights_constraints,
            bias_constraints=bias_constraints
        )
        self.nodes = int(nodes)
        if input_dim:
            self.input_dim = input_dim

    @type_safe
    def build(self):
        pass
