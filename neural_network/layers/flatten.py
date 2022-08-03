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
        self.built = True
        self.output_shape = 1
        for shape in input_shape:
            self.output_shape *= shape
        return self.output_shape

    @type_safe
    @not_none
    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.reshape(X, (len(X), self.output_shape))

    @type_safe
    @not_none
    def backward(self, X) -> np.ndarray:
        return np.reshape(X, (len(X), self.input_shape))
