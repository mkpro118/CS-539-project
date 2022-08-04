from typing import Union
import numpy as np

from ..base.layer import Layer
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class Flatten(Layer):
    @type_safe
    def __init__(self, *, name: str = None):
        super().__init__(
            trainable=False,
            use_bias=False,
            name=name
        )

    @type_safe
    @not_none
    def build(self, _id: int,
              input_shape: Union[list, tuple, np.ndarray]) -> tuple:
        self._id = _id
        self.input_shape = np.asarray(input_shape)
        self.output_shape = np.prod(self.input_shape, dtype=int)
        self.built = True
        return (self.output_shape, )

    @type_safe
    @not_none
    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 2:
            return X
        result = X.reshape(len(X), self.output_shape)
        return result

    @type_safe
    @not_none
    def backward(self, X) -> np.ndarray:
        result = X.reshape(len(X), *self.input_shape)
        return result