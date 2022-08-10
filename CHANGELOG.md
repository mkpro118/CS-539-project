
# Update Logs

# Change Log (August 9, 2022)
+ Top Level
    + Added [`Custom Keras Model Experiment`](https://github.com/mkpro118/CS-539-project/blob/main/Custom%20Keras%20Model%20Experiment.ipynb), experiment had a **26.78%** Accuracy
    + Added [`Custom Model Experiment`](https://github.com/mkpro118/CS-539-project/blob/main/Custom%20Model%20Experiment.ipynb), experiment had a **83.77%** Accuracy
    + Added [`Experiment With ResNet50`](https://github.com/mkpro118/CS-539-project/blob/main/Experiment%20WIth%20ResNet50.ipynb), experiment had a **100%** Accuracy
    + Added [`Experiment with 10 layers`](https://github.com/mkpro118/CS-539-project/blob/main/Experiment%20with%2010%20layers.ipynb), experiment _failed_
+ Moved change log to [`CHANGELOG`](https://github.com/mkpro118/CS-539-project/blob/main/CHANGELOG.md)

## Change Log (August 7, 2022)
+ ### Refactored code to be compatible with `python3 version >=3.6`
    + See commit history to view the files changed
+ Module [`neural_network.layers`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/layers)
    + Added [`BatchNormalization`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/layers/batch_normalization.py) Layer
+ Module [`neural_network.utils`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/utils)
    + Added [`timeit.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/timeit.py), to track training time

## Change Log (August 6, 2022)
+ Module [`data`](https://github.com/mkpro118/CS-539-project/tree/main/data)
    + [`__init__.py`](https://github.com/mkpro118/CS-539-project/blob/main/data/__init__.py), added function `load_data` to load the images and labels
    + Improved the implementation of [`resize.py`](https://github.com/mkpro118/CS-539-project/blob/main/data/resize.py) and [`grayscale.py`](https://github.com/mkpro118/CS-539-project/blob/main/data/grayscale.py)
+ Top Level
    + Added file [`environment.yaml`] specifying python environment requirements
    + Added file [`requirements.txt`] specifying required packages

## Change Log (August 5-6, 2022)
+ Module [`data`](https://github.com/mkpro118/CS-539-project/tree/main/data)
    + Larger set of raw images have been added and renamed. (Total 651 files)
+ Module [`neural_network.base`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/base)
    + [`layer.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/layer.py): Added support for history, checkpoints, summary and properties `trainable_params`, `non_trainable_params`
    + [`model.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/model.py): Added support for history, checkpoints
+ Module [`neural_network.layers`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/layers)
    + [`convolutional.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/layers/convolutional.py), [`dense.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/layers/dense.py), [`flatten.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/layers/flatten.py) added support for building from config dictionaries, `__str__` implemented
+ Module [`neural_network.model`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/model)
    + [`sequential.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model/sequential.py)
        + Minor change: Added summary, support for history, checkpoints, building models from config dictionaries.
        + **Major change**: `fit` now takes in a _bool_ **keyword** argument `get_trainer`, which returns a _generator object_ than will run one epoch of training each time `next` is called. When executed in this manner, the call to `next` will return a dictionary of the loss and accuracy after running that epoch of training. If validation data is provided, the loss and accuracy over the validation data is also returned. The dictionary returned is of exactly the following format.
            + If validation data is **NOT** specified
            ```py
            {
                'overall': {
                    'loss': loss,
                    'accuracy': accuracy,
                },
            }
            ```
            + If validation data is specified
            ```py
            {
                'overall': {
                    'loss': loss,
                    'accuracy': accuracy,
                },
                'validation': {
                    'loss': loss,
                    'accuracy': accuracy,
                },
            }
            ```
        Using the generator from `get_trainer` gives users better control over the number of steps in training, as well as flexibilty to use the step by step data for inference


## Change Log (August 4, 2022)
+ Module `neural_network.aux_math` has been removed, all required functionality is available in the scipy library
+ Module [`neural_network.base`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/base)
    + [`layer.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/layer.py): Invalidated optimizer functions for non trainable layers
+ Module [`neural_network.layers`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/layers)
    + Implemented [`convolutional.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/layers/convolutional.py)
    + Implemented [`flatten.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/layers/flatten.py)
    + Fixed errors in [`dense.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/layers/dense.py)
+ Module [`neural_network.model`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/model)
    + [`sequential.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model/sequential.py): Fixed a bug in computing the batch sizes, prettified verbose outputs
+ Module [`neural_network.utils`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/utils)
    + Fixed a major bug in [`functools.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/functools.py), inherited methods had the same `__qualname__`, which caused multiple layers to have activation methods rendered invalid. Using `id` in conjunction with `__qualname__` now
+ Top Level [`Convolutional Example.py`](https://github.com/mkpro118/CS-539-project/blob/main/Convolutional%20Example.ipynb)
    + Added an example program demonstrating the usage of the [`Convolutional Example.ipynb`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/layers/convolutional.py) layer

## Change Log (August 3, 2022)
+ Modules [`neural_network.activation`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/activation), [`neural_network.cost`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/cost)
    + Added a `name` property to all classes for easier inverse lookups
    + Fixed divide by zero error in `log2` in [`cross_entropy.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/cost/cross_entropy.py)
+ Module [`neural_network.base`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/base)
    + [`model.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/model.py) has been implemented, TODO: Add docs
+ Module [`neural_network.layers`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/layers)
    + [`dense.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/layers/dense.py) has been implemented, TODO: Add docs
+ Module [`neural_network.metrics`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/metrics)
    + Fixed zero division error in [`accuracy_by_label.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/accuracy_by_label.py)
+ Module [`neural_network.model`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/model)
    + [`sequential.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model/sequential.py) has been implemented, TODO: Add docs
+ Top Level [`main.py`](https://github.com/mkpro118/CS-539-project/blob/main/main.py)
    + Added an example program demonstrating the usage of the [`neural_network`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network) library

## Change Log (August 1, 2022)
+ Module [`neural_network.base`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/base)
    + Renamed `layer_mixin.py` to [`layer.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/layer.py#L29), since it's a base class, not a mixin
    + Renamed `model_mixin.py` to [`model.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/model.py#L29), since it's a base class, not a mixin
    + Refactored [`layer.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/layer.py#L29) to use [`utils.functools.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/functools.py), Added partial docs
    + Refactored ['metadata_mixin.py'](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/metadata_mixin.py) to filter callables since they are not saveable
+ Module [`neural_network.layers`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/layers)
    + Started implementing [`dense.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/layers/dense.py)
+ Module [`neural_network.model`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/model)
    + Started implementing [`sequential.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model/sequential.py)
+ Module [`neural_network.model_selection`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/model_selection)
    + Refactored [`kfold.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model_selection/kfold.py) to add support for `list`s and `tuple`s
    + Refactored [`repeated_kfold.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model_selection/repeated_kfold.py) to add support for `list`s and `tuple`s, and to use `KFold` as the base class
    + Refactored [`stratified_kfold.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model_selection/stratified_kfold.py) to add support for `list`s and `tuple`s, and to use `KFold` as the base class
    + Refactored [`stratified_repeated_kfold.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model_selection/stratified_repeated_kfold.py) to add support for `list`s and `tuple`s, and to use `KFold` as the base class
+ Module [`neural_network.utils`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/utils)
    + Added file [`functools.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/functools.py)
    + Refactored [`typesafety.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/typesafety.py) to use [`functools.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/functools.py), Removed unnecessary checks

## Change Log (July 31, 2022)
+ Module [`neural_network.activation`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/activation)
    + Added `__name_to_symbol_map__` in [`__init__.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/activation/__init__.py)
+ Module [`neural_network.base`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/base)
    + Removed unused import in [`activation_mixin.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/activation_mixin.py)
    + Removed unused import in [`classifier_mixin.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/classifier_mixin.py)
    + Added functionality in [`layer.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/layer.py#L29)
    + TODO: Add documentation in [`layer.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/layer.py)
+ Module [`neural_network.layers`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/layers)
    + Started the implementation of [`dense.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/layers/dense.py)
    + TODO: Complete implementation and add documentation
+ Module [`neural_network.metrics`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/metrics)
    + Added support for multiprocessing on all metrics
    + Added multilabel support for [`precision_score`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/precision_score.py#L75-L132) and [`recall_score`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/recall_score.py#L75-L132)
    + Added [`multilabel_confusion_matrix`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/confusion_matrix.py#L162-L201)
+ Module [`neural_network.preprocess`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/preprocess)
    + Refactored [`one_hot_encoder.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/preprocess/one_hot_encoder.py), added parameter validation
    + Refactored [`scaler.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/preprocess/scaler.py), added better type checks for `__init__`
+ Module [`neural_network.utils`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/utils)
    + Added support for subclass checks in [`typesafety.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/typesafety.py#L123)


## Change Log (July 30, 2022)
+ Module [`neural_network.metrics`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/metrics)
    + Refactored [`accuracy_score.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/accuracy_score.py)
    + Refactored [`confusion_matrix.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/confusion_matrix.py)
    + Removed file [`average_precision_score.py`] since it's redundant given [`precision_score.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/precision_score.py)
    + Removed file [`average_recall_score.py`] since it's redundant given [`recall_score.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/recall_score.py)
+ Refactored all [`neural_network.base`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/base) dependent modules to avoid circular imports


## Change Log (July 29, 2022)
+ Module [`neural_network.metrics`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/metrics)
    + Refactored [`accuracy_by_label.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/accuracy_by_label.py)

+ Module [`neural_network.aux_math`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/aux_math)
    + Refactored [`convolve.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/aux_math/convolve.py)

## Change Log (July 28, 2022)
+ Module [`neural_network.activation`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/activation) has been fully implemented (with docs)
    + Added [`leaky_relu.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/activation/leaky_relu.py)
    + Added [`relu.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/activation/relu.py)
    + Added [`sigmoid.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/activation/sigmoid.py)
    + Added [`softmax.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/activation/softmax.py)
    + Added [`tanh.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/activation/tanh.py)
+ Module [`neural_network.aux_math`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/aux_math)
    + Added support for 3D and 4D convolutions in [`convolve.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/aux_math/convolve.py)
+ Module [`neural_network.base`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/base)
    + Refactored [`activation_mixin.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/activation_mixin.py); Removed LayerMixin, renamed methods to be simiar to  [`CostMixin`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/cost_mixin.py#L5)
    + Removed imports for `correct_classification_rate`, since that is exactly what [`accuracy_score`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/accuracy_score.py) does
    + Refactored [`cost_mixin.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/cost_mixin.py); Removed LayerMixin
+ Module [`neural_network.exceptions`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/exceptions) has been fully implemented (with docs)
    + Added [`exception_factory.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/exceptions/exception_factory.py)
+ Module [`neural_network.metrics`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/metrics) has been implemented, needs refactoring and documentation
    + Added [`accuracy_by_label.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/accuracy_by_label.py)
    + Added [`accuracy_score.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/accuracy_score.py)
    + Added [`average_precision_score.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/average_precision_score.py)
    + Added [`average_recall_score.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/average_recall_score.py)
    + Added [`confusion_matrix.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/confusion_matrix.py)
    + Added [`precision_score.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/precision_score.py)
    + Added [`recall_score.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/metrics/recall_score.py)
+ Module [`neural_network.utils`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/utils)
    + Removed try-except during imports in [`exception_handling.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/utils/exception_handling.py)
+ All Modules
    + Refactored `__init__.py` to have module docstrings

## Change Log (July 27, 2022)
+ Module [`neural_network.aux_math`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/aux_math)
    + Added [`convolve.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/aux_math/convolve.py) with docs, containing two methods
        + [`convolve`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/aux_math/convolve.py#L86)
        + [`convolve_transpose`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/aux_math/convolve.py#L175)
    + Might add support for channels in convolution later

## Change Log (July 26, 2022)
+ Module [`neural_network.base`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/base)
    + Renamed required methods in class CostMixin ([`cost_mixin.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/cost_mixin.py)) from `cost` to `apply`, and `cost_derivative` to `derivative`
+ Module [`neural_network.cost`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/cost) has been fully implemented (with docs)
    + Added [`cross_entropy.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/cost/cross_entropy.py)
    + Added [`mean_squared_error.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/cost/mean_squared_error.py)
+ Module [`neural_network.model_selection`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/model_selection) has been fully implemented (with docs)
    + Added [`kfold.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model_selection/kfold.py)
    + Added [`repeated_kfold.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model_selection/repeated_kfold.py)
    + Added [`stratified_kfold.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model_selection/stratified_kfold.py)
    + Added [`stratified_repeated_kfold.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model_selection/stratified_repeated_kfold.py)
    + Added [`train_test_split.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/model_selection/train_test_split.py)
+ Module [`neural_network.decomposition`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/decomposition)
    + Added docs to [`linear_discriminant_analysis.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/decomposition/linear_discriminant_analysis.py)
    + Added docs to [`principal_component_analysis.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/decomposition/principal_component_analysis.py)
+ Module [`neural_network.preprocess`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/preprocess)
    + Removed unnecessary variables in [`one_hot_encoder.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/preprocess/one_hot_encoder.py), [`scaler.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/preprocess/scaler.py) and [`standardizer.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/preprocess/standardizer.py)
+ Fixed a return type bug in [`neural_network.preprocess.one_hot_encoder.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/preprocess/one_hot_encoder.py)
+ Fixed a bug in the `_check_fitted` method of [`neural_network.base.transform_mixin.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/base/transform_mixin.py)


## Change Log (July 20, 2022)
+ Renamed `neural_network.cost.categorical_cross_entropy.py` to [`neural_network.cost.cross_entropy.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/cost/cross_entropy.py)
+ Removed `neural_network.cost.ln_norm_distance.py` as it is unused

## Change Log (July 19, 2022)
+ Module [`decomposition`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/decomposition) has been fully implemented
    + Added [`linear_discriminant_analysis.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/decomposition/linear_discriminant_analysis.py)
    + Added [`principal_component_analysis.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/decomposition/principal_component_analysis.py)
+ Module [`preprocess`](https://github.com/mkpro118/CS-539-project/tree/main/neural_network/preprocess) has been fully implemented
    + Added [`one_hot_encoder.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/preprocess/one_hot_encoder.py)
    + Added [`scaler.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/preprocess/scaler.py)
    + Added [`standardizer.py`](https://github.com/mkpro118/CS-539-project/blob/main/neural_network/preprocess/standardizer.py)
+ Minor bug fixes in `neural_network.utils.typesafety.py` and `neural_network.base.classifier_mixin.py`
+ Removed `neural_network.preprocess.imputer.py` as it is unused

## Change Log (July 18, 2022)
+ Created partial dataset
    + Added, grayscaled and resized **bishop** images
    + Added, grayscaled and resized **knight** images
    + Added, grayscaled and resized **pawn** images
    + Added script [`grayscale.py`](https://github.com/mkpro118/CS-539-project/blob/main/data/grayscale.py)
    + Added script [`resize.py`](https://github.com/mkpro118/CS-539-project/blob/main/data/resize.py)
+ Dependencies now include Pillow (PIL)

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

## Project Structure

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
│    │    ├── correlate.py
│    │    ├── convolve.py
│    ├── base/
│    │    ├── __init__.py
│    │    ├── activation_mixin.py
│    │    ├── cost_mixin.py
│    │    ├── classifier_mixin.py
│    │    ├── layer.py
│    │    ├── metadata_mixin.py
│    │    ├── mixin.py
│    │    ├── model_mixin.py
│    │    ├── save_mixin.py
│    │    ├── transform_mixin.py
│    ├── cost/
│    │    ├── __init__.py
│    │    ├── cross_entropy.py
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
│    │    ├── flatten.py
│    ├── metrics/
│    │    ├── __init__.py
│    │    ├── accuracy.py
│    │    ├── accuracy_by_label.py
│    │    ├── confusion_matrix.py
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
├── .gitattributes
├── .gitignore
├── Convolutional Example.ipynb
├── environment.yaml
├── Example.ipynb
├── main.py
├── README.md
├── requirements.txt
```
