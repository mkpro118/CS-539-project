from typing import Callable, Union
import numpy as np

from mixin import mixin

from ..metrics import (
    accuracy_score,
    accuracy_by_label,
    average_precision_score,
    average_recall_score,
    confusion_matrix,
    correct_classification_rate,
    precision_score,
    recall_score,
)
from ..utils.typesafety import type_safe


DEFAULT_METRIC = 'accuracy_score'

NAME_TO_SYMBOL_MAP = {
    'accuracy_score': accuracy_score,
    'accuracy_by_label': accuracy_by_label,
    'average_precision_score': average_precision_score,
    'average_recall_score': average_recall_score,
    'confusion_matrix': confusion_matrix,
    'correct_classification_rate': correct_classification_rate,
    'precision_score': precision_score,
    'recall_score': recall_score,
}


@mixin
class ClassifierMixin:

    @type_safe
    def score(self, X: np.ndarray, y: np.ndarray,
              metric: Union[str, Callable] = DEFAULT_METRIC) -> np.ndarray:
        # If a custom metric is specified
        if callable(metric):
            return metric(y, self.predict(X))

        # Look for default metrics
        score_fn = NAME_TO_SYMBOL_MAP.get(metric, None)

        if score_fn is None:
            raise ValueError(
                f'{metric} is an unrecognized metric. For custom metrics, '
                f'pass a scoring function with the keyword argument metric'
            )

        return score_fn(y, self.predict(X))
