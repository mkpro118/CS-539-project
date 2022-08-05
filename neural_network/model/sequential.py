from typing import Union
from numbers import Integral
import numpy as np

from ..base.cost_mixin import CostMixin
from ..base.classifier_mixin import ClassifierMixin
from ..base.layer import Layer
from ..base.model import Model
from ..exceptions import ExceptionFactory
from ..layers import Convolutional, Dense, Flatten
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


errors = {
    'SequentialModelError': ExceptionFactory.register('SequentialModelError'),
}

known_layers = {
    'Convolutional': Convolutional,
    'Dense': Dense,
    'Flatten': Flatten,
}


@export
class Sequential(Model, ClassifierMixin):
    @type_safe
    def __init__(self, *, layers: Layer = None, from_model: 'Sequential' = None):
        super().__init__()
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
        super().compile(cost=cost, metrics=metrics)

        for layer in self.layers:
            if hasattr(layer, 'input_shape'):
                next_dim = layer.input_shape
                break
        else:
            raise errors['SequentialModelError'](
                f'The first layer must have a defined input shape. Use the '
                f'keyword argument input_shape on the first layer to define an input shape'
            )

        for _id, layer in enumerate(self.layers):
            next_dim = layer.build(_id, next_dim)

        for layer in reversed(self.layers):
            if layer.activation is None:
                continue

            self.final_activation = layer.activation
            break

        if (
            self.final_activation.name == 'softmax' and self.cost.name != 'crossentropy'
        ) or (
            self.final_activation.name != 'softmax' and self.cost.name == 'crossentropy'
        ):
            raise errors['SequentialModelError'](
                f'Final layer activation and cost function are not compatible. '
                f'If the final layer uses \'softmax\', use the \'crossentropy\' cost. '
                f'For any other final layer activation use \'mse\''
            )

        self.built = True

    @type_safe
    @not_none
    def fit(self,
            X: np.ndarray = None,
            y: np.ndarray = None, *,
            epochs: Union[int, Integral, np.integer] = None,
            batch_size: Union[int, Integral, np.integer] = None,
            steps_per_epoch: Union[int, Integral, np.integer] = None,
            shuffle: bool = True,
            validation_data: Union[np.ndarray, list, tuple] = None,
            verbose: bool = True):

        super().fit(
            verbose=verbose,
        )

        if y.ndim not in (1, 2,):
            raise errors['SequentialModelError'](
                f'Training labels must be a 1 or 2 dimensional array'
            )

        self._train(X, y, validation_data, epochs, batch_size, steps_per_epoch, shuffle)

    def _train(self, X, y, validation_data, epochs, batch_size, steps_per_epoch, shuffle):
        if not epochs:
            epochs = 10
        for epoch in range(epochs):
            if self.verbose:
                print(f'\nEpoch {epoch + 1: >{len(str(epochs))}}/{epochs}')
            self._run_epoch(X, y, batch_size, steps_per_epoch, shuffle)

            if validation_data:
                targets = validation_data[1]
                predictions = self.predict(validation_data[0])

                error = self.cost.apply(targets, predictions)
                self.history['validation']['loss'].append(error)
                error = np.around(error, 4)

                if y.ndim == 2:
                    predictions = (predictions == predictions.max(axis=1)[:, None]).astype(int)

                metrics = {}
                for metric in self.metrics:
                    acc = metric(targets, predictions)
                    self.history['validation'][f'{metric.__name__}'].append(acc)
                    metrics[f'{metric.__name__}'] = np.around(acc, 4)
                if self.verbose:
                    print(f'\n  Validation loss: {error}', end=' ')
                    print(' | '.join(map(lambda x: f'{x[0]}: {x[1]}', metrics.items())))

            targets = y
            predictions = self.predict(X)

            error = self.cost.apply(targets, predictions)

            self.history['overall']['loss'].append(error)
            error = np.around(error, 4)

            if y.ndim == 2:
                predictions = (predictions == predictions.max(axis=1)[:, None]).astype(int)

            metrics = {}
            for metric in self.metrics:
                acc = metric(targets, predictions)
                self.history['overall'][f'{metric.__name__}'].append(acc)
                metrics[f'{metric.__name__}'] = np.around(acc, 4)

            if self.verbose:
                print(f'  Overall loss: {error}', end=' ')
                print(' | '.join(map(lambda x: f'{x[0]}: {x[1]}', metrics.items())))

    def _run_epoch(self, X, y, batch_size, steps_per_epoch, shuffle):
        if batch_size and steps_per_epoch:
            batch_size = len(X) // steps_per_epoch + 1
        elif batch_size and not steps_per_epoch:
            steps_per_epoch = len(X) // batch_size
        elif steps_per_epoch and not batch_size:
            batch_size = len(X) // steps_per_epoch + 1
        else:
            steps_per_epoch = 5
            batch_size = len(X) // steps_per_epoch + 1

        batches = self._get_batches(X, y, batch_size, shuffle)
        for step, (X, y) in zip(range(steps_per_epoch), batches):
            if self.verbose:
                print(f'  Step {step + 1: >{len(str(steps_per_epoch))}}/{steps_per_epoch}', end=' ... ')
            self._run_batch(X, y)
            if self.verbose:
                print(f'done')

    def _run_batch(self, X, y):
        predictions = self.predict(X)
        error_gradient = self.cost.derivative(y, predictions) * self.final_activation.derivative(predictions)
        for layer in reversed(self.layers):
            error_gradient = layer.backward(error_gradient)

    def predict(self, X: np.ndarray, *, classify: bool = False):
        for layer in self.layers:
            X = layer.forward(X)
        if classify:
            return (X == X.max(axis=1)[:, None]).astype(int)
        return X

    def get_metadata(self):
        allowed_keys = {
            'built',
            'metrics',
            'trainable',
            'verbose',
        }
        data = super().get_metadata()
        data.update({'metrics': data['metrics_names']})
        data = {k: v for k, v in data.items() if k in allowed_keys}
        data['layers'] = {}
        for layer in self.layers:
            data['layers'].update({f'layer{layer._id}': layer.get_metadata()})
            data['layers'][f'layer{layer._id}'].update({'type': layer.__class__.__name__})
        return data

    @classmethod
    def build_from_config(cls, config):
        pass
