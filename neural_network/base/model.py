from typing import Union
from numbers import Real, Integral
import numpy as np

from .activation_mixin import ActivationMixin
from .mixin import mixin
from .metadata_mixin import MetadataMixin
from .save_mixin import SaveMixin

from ..exceptions import ExceptionFactory
from ..utils.typesafety import type_safe, not_none
from ..model_selection import KFold

from ..metrics import (
    accuracy_score,
    accuracy_by_label,
    precision_score,
    recall_score,
)

from ..cost import CrossEntropy, MeanSquaredError

errors = {
    'UncompiledModelError': ExceptionFactory.register('UncompiledModelError'),
    'UnknownCostError': ExceptionFactory.register('UnknownCostError'),
    'UnknownMetricError': ExceptionFactory.register('UnknownMetricError'),
    'NotImplementedError': ExceptionFactory.register('NotImplementedError'),
}

known_metrics = {
    'accuracy_score': accuracy_score,
    'accuracy_by_label': accuracy_by_label,
    'precision_score': precision_score,
    'recall_score': recall_score,
}

known_costs = {
    'crossentropy': CrossEntropy,
    'mse': MeanSquaredError,
}


@mixin  # Prevents instantiation
class Model(MetadataMixin, SaveMixin):
    '''
    Mixin for easier definition of Model classes

    Inherited from MetadataMixin
        `get_metadata`

    Inherited from SaveMixin
        `save`
    '''

    def __init__(self):
        self.history = []
        self.checkpoints = []

    @type_safe
    @not_none
    def compile(self, cost: Union[str, ActivationMixin], metrics: Union[list, tuple]):
        if isinstance(cost, str):
            cost = known_costs.get(cost, None)
            if cost is None:
                raise errors['UnknownCostError'](
                    f'{cost=} is not a recognized cost function. Known cost functions are '
                    f'{", ".join(known_costs.keys())}. Alternatively custom cost functions can '
                    f'be defined by subclassing neural_network.base.cost_mixin.CostMixin'
                )

        self.cost = cost

        self.metrics = set()
        for metric in metrics:
            if isinstance(metric, str):
                metric = known_metrics.get(metric, None)
                if metric is None:
                    raise errors['UnknownmetricError'](
                        f'{metric=} is not a recognized metric function. Known metrics are '
                        f'{", ".join(known_metrics.keys())}'
                    )
            self.metrics.add(metric)

        self._attrs = ('cost', 'metrics')

    @type_safe
    @not_none
    def _check_compiled(self):
        if not all((hasattr(self, attr) for attr in self._attrs)):
            raise errors['UncompiledModelError'](
                f'Cannot fit a model before compiling it. Use model.compile(cost=cost, metrics=metrics) '
                f'to compile the model'
            )

    @type_safe
    @not_none
    def fit(self,
            X: np.ndarray = None,
            y: np.ndarray = None, *,
            epochs: Union[int, Integral, np.integer] = None,
            batch_size: Union[int, Integral, np.integer] = None,
            steps_per_epoch: Union[int, Integral, np.integer] = None,
            shuffle: Union[int, Integral, np.integer] = None,
            validation_data: Union[np.ndarray, list, tuple] = None,
            validation_split: Union[float, np.floating, Real] = None,
            validation_batch_size: Union[int, Integral, np.integer] = None,
            validator: KFold = None,
            verbose: bool = True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.shuffle = shuffle
        self.validation_data = validation_data
        self.validation_split = validation_split
        self.validation_batch_size = validation_batch_size
        self.validator = validator
        self.verbose = verbose

    @type_safe
    @not_none
    def _get_batches(self, X: np.ndarray, y: np.ndarray,
                     batch_size: int, shuffle: bool):
        n_samples = len(X)
        indices = np.arange(n_samples)
        if shuffle:
            np.random.default_rng().shuffle(indices)
        for i in range(0, n_samples, batch_size):
            idxs = indices[i: min(i + batch_size, n_samples)]
            yield X[idxs], y[idxs]

    def checkpoint(self):
        self.checkpoints.append(self.get_metadata())

    def predict(self, *args, **kwargs):
        raise errors['NotImplementedError'](
            f'Descendant classes must define their implementation of the predict method'
        )

    def evaluate(self, *args, **kwargs):
        raise errors['NotImplementedError'](
            f'Descendant classes must define their implementation of the evalute method'
        )
