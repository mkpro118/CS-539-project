from typing import Union

from ..base.cost_mixin import CostMixin
from ..base.classifier_mixin import ClassifierMixin
from ..base.layers import Layer
from ..base.model import Model
from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


errors = {
    'SequentialModelError': ExceptionFactory.register('SequentialModelError'),
}


@export
class Sequential(Model, ClassifierMixin):
    @type_safe
    def __init__(self, *, layers: Layer = None, from_model: 'Sequential' = None):
        self.layers = []

        if from_model is not None:
            if not isinstance(from_model, Sequential):
                raise errors['SequentialModelError'](
                    f'from_model argument must be a sequential model, found {from_model} '
                    f'of type {type(from_model)}'
                )
            self.layers.extend(from_model.layers)

        if layers is not None:
            self.layers.extend(layers)

    @type_safe
    @not_none
    def add(self, *layers: tuple):
        if not layers:
            raise errors['SequentialModelError'](
                f'Found no layers to add'
            )
        for layer in layers:
            if not isinstance(layer, Layer):
                raise errors['SequentialModelError'](
                    f'{layer} of type {type(layer)} is not a descendant of neural_network.base.Layer. '
                    f'If {layer} is a custom layer, esnure that it is a subclass of neural_network.base.Layer.'
                )
            self.layers.append(layer)

    @type_safe
    @not_none
    def compile(self, cost: Union[str, CostMixin],
                metrics: Union[list, tuple]) -> 'Sequential':
        if not self.layers:
            raise errors['SequentialModelError'](
                'There must be at least one layer before the model can compile'
            )
