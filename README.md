# Update Logs

## Change Log (July 5, 2022)
+ Module [`neural_network.model`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/model) has been extended to include two more models
    + Added `decision_tree.py`, `k_nearest_neighbors.py`, both of which are currently not implemented.


## Change Log (July 4, 2022)
+ Module [`neural_network.base`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/base) has been added and fully implemented. Authors are welcome to add any other mixins as they please.
    + Removed `decompostion_mixin.py`, `exception_mixin.py`, `fit_mixin.py`, `solver_mixin.py`
    + Renamed `save_model_mixin.py` to [`save_mixin.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/save_mixin.py)
    + Added [`mixin.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/mixin.py), [`metadata_mixin.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/metadata_mixin.py)
    + Added docstrings to all files under `neural_network.base`

## Change Log (July 2, 2022)
+ The neural_network api is now more modular
+ More details have been added
+ Module `neural_network.base` has been added, which now contains mixins for other classes
    + Some mixins have been added, authors are welcome to add more as they please.
+ Module [`neural_network.utils`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/utils) has been added, which currently contains some utility decorators. Authors are welcome to add more utility functions as they please
    + [`exceptions_handling.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/exception_handling.py)
        + [`safeguard`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/exception_handling.py#L14) decorator has been implemented
        + [`warn`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/exception_handling.py#L83) decorator has been implemented
    + [`exports.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/exports.py)
        + [`export`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/exports.py#L10) decorator has been implemented
    + [`typesafety`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/typesafety.py)
        + [`type_safe`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/typesafety.py#L10) decorator has been implemented
        + [`not_none`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/typesafety.py#L161) decorator has been implemented
+ Module `neural_network.aux_math` has been added, which contains auxillary functions to properly perform mathematical operations on tensors.

## New Project Structure

```bash
.
├── data/
│    ├── images/
│    │    ├── bishop/*.jpg
│    │    ├── knight/*.jpg
│    │    ├── pawn/*.jpg
│    │    ├── queen/*.jpg
│    │    ├── rook/*.jpg
│    ├── dataset.py
├── neural_network/
│    ├── activation/
│    │    ├── __init__.py
│    │    ├── leaky_relu.py
│    │    ├── relu.py
│    │    ├── sigmoid.py
│    │    ├── softmax.py
│    │    ├── tanh.py
│    ├── aux_math/
│    │    ├── __init__.py
│    │    ├── mat_dot.py
│    │    ├── mat_mul.py
│    │    ├── correlate.py
│    │    ├── convolve.py
│    ├── base/
│    │    ├── __init__.py
│    │    ├── activation_mixin.py
│    │    ├── cost_mixin.py
│    │    ├── classifier_mixin.py
│    │    ├── layer_mixin.py
│    │    ├── metadata_mixin.py
│    │    ├── mixin.py
│    │    ├── model_mixin.py
│    │    ├── save_mixin.py
│    │    ├── transform_mixin.py
│    ├── cost/
│    │    ├── __init__.py
│    │    ├── categorical_cross_entropy.py
│    │    ├── ln_norm_error.py
│    │    ├── mean_squared_error.py
│    ├── decomposition/
│    │    ├── __init__.py
│    │    ├── linear_discriminant_analysis.py
│    │    ├── principal_component_analysis.py
│    ├── exceptions/
│    │    ├── __init__.py
│    │    ├── exception_factory.py
│    ├── layers/
│    │    ├── __init__.py
│    │    ├── convolutional.py
│    │    ├── dense.py
│    │    ├── reshape.py
│    ├── metrics/
│    │    ├── __init__.py
│    │    ├── accuracy.py
│    │    ├── accuracy_by_label.py
│    │    ├── average_precision_score.py
│    │    ├── average_recall_score.py
│    │    ├── confusion_matrix.py
│    │    ├── correct_classification_rate.py
│    │    ├── precision_score.py
│    │    ├── recall_score.py
│    ├── model/
│    │    ├── __init__.py
│    │    ├── decision_tree.py
│    │    ├── k_nearest_neighbors.py
│    │    ├── sequential.py
│    ├── model_selection/
│    │    ├── __init__.py
│    │    ├── kfold.py
│    │    ├── repeated_kfold.py
│    │    ├── stratified_kfold.py
│    │    ├── stratified_repeated_kfold.py
│    │    ├── train_test_split.py
│    ├── preprocess/
│    │    ├── __init__.py
│    │    ├── imputer.py
│    │    ├── one_hot_encoder.py
│    │    ├── scaler.py
│    │    ├── standardizer.py
│    ├── utils/
│    │    ├── __init__.py
│    │    ├── exceptions_handling.py
│    │    ├── exports.py
│    │    ├── typesafety.py
│    ├── __init__.py
├── __init__.py
├── main.py
├── .gitignore
├── README.md
```
