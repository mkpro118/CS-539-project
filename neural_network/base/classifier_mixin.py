from typing import Callable, Union
import numpy as np

from .mixin import mixin

from ..metrics import (
    accuracy_score,
    accuracy_by_label,
    confusion_matrix,
    precision_score,
    recall_score,
)
from ..utils.typesafety import type_safe, not_none

# default metric to use
DEFAULT_METRIC = 'accuracy_score'

# Facilitate lookup of functions
NAME_TO_SYMBOL_MAP = {
    'accuracy_score': accuracy_score,
    'accuracy_by_label': accuracy_by_label,
    'confusion_matrix': confusion_matrix,
    'precision_score': precision_score,
    'recall_score': recall_score,
}


@mixin  # Prevents instantiation
class ClassifierMixin:
    '''
    Mixin Class for Classifiers

    Provides a `score` method to compute model accuracy

    Methods:
        `score(X: numpy.ndarray, y: numpy.ndarray,
              metric: Union[str, Callable]) -> numpy.ndarray`:
            Computes the accuracy of the model with the given metric
    '''

    @type_safe
    @not_none
    def score(self, X: np.ndarray, y: np.ndarray,
              metric: Union[str, Callable] = DEFAULT_METRIC) -> Union[np.ndarray, float]:
        '''
        Computes the accuracy of the model with the given metric

        Parameters:
            X: numpy.ndarray of shape (n_samples, n_features)
                The feature matrix to predict and score
            y: numpy.ndarray of shape (n_samples, )
                The label vector corresponding to the feature matrix
            metric: str or Callable, defaults to 'accuracy_score'
                The metric to use for scoring the predictions

                If `metric` is of type str, valid values are:
                    'accuracy_score',
                    'accuracy_by_label',
                    'average_precision_score',
                    'average_recall_score',
                    'confusion_matrix',
                    'correct_classification_rate',
                    'precision_score',
                    'recall_score'

                If `metric` is a callable, it should take two positional
                parameters, `y_true` and `y_pred` (in that order!). `y_true` is
                the vector containing the correct labels, `y_pred` is the vector
                containing the model's predictions. It is guranteed that
                `y_true` and `y_pred` are of type numpy.ndarray

        Returns:
            float or numpy.ndarray: The score computed by the given metric

            metrics that return float:
                'accuracy_score'
                'average_precision_score'
                'average_recall_score'
                'correct_classification_rate'

            metrics that return numpy.ndarray:
                'accuracy_by_label'
                'confusion_matrix'
                'precision_score'
                'recall_score'

        Raises:
            ValueError: If an unrecognized metric is passed to `metric`
        '''
        # If a custom metric is specified
        if callable(metric):
            return metric(y, self.predict(X))

        # Look for default metrics
        score_fn = NAME_TO_SYMBOL_MAP.get(metric, None)

        if score_fn is None:
            raise ValueError(
                f'{metric} is an unrecognized metric. '
                f'Supported metrics are {", ".join(NAME_TO_SYMBOL_MAP.keys())}'
                f'\nFor custom metrics, pass a scoring function '
                f'with the keyword argument metric'
            )

        return score_fn(y, self.predict(X))
