Project Structure:
```bash
.
├── neural_network
│    ├── activation.py
│    ├── cost.py
│    ├── layer.py
│    ├── model.py
│    ├── model_selection.py
│    ├── decomposition.py
│    ├── metrics.py
│    ├── preprocess.py
│    ├── utils.py
│    ├── exceptions.py
│    ├── __init__.py
├── data
├── main.py
├── __init__.py
├── .gitignore
├── README.md
```

+ `activation.py`
    + `Activation` base/abstract class.
    + Derived `Activation` classes, which are layers in the Neural Network.

+ `costfunction.py`
    + `CostFunction` base/abstract class.
    + Derived `CostFunction` classes, which is the final layer in the Neural Network.

+ `layer.py`
    + `Layer` base/abstract class.
    + Derived `Layer` classes, primarily Dense, which are layers in the Neural Network.

+ `model.py`
    + `Model` base/abstract class.
    + Derived `Model` classes, primarily Sequential.

+ `model_selection.py`
    + `KFold` class used to validate models
    + `RepeatedKFold` class, derived from `KFold` to use every fold as validation set
    + `train_test_split` function to divide the data into training and testing sets.

+ `decomposition.py`
    + `PCA` class, to perform Principal Component Analysis
    + `LDA` class, to perform Linear Discriminant Analysis

+ `metrics.py`
    + TBD
    + Possibilities include
        + accuracy
        + precision
        + confusion matrix
        + etc...

+ `preprocess.py`
    + `normalize` function to scale the data within the range $-n \le x le n$
    + `standardize` function to transform data to have $\text{Mean} = 0$ and $\text{Standard Deviation} = 1$
    + `impute` function to impute missing data, if any
    + `one_hot_encode` function to one hot encode the labels

+ `utils.py`
    + `export_function` decorator to export functions
    + `export_class` decorator to export classes (only for semantics, logically equivalent to `export_function`)
    + `safeguard` decorator to handle exceptions

+ `exceptions.py`
    + TBD

+ `__init__.py`
    + Initialize folder as a module

+ data
    + The data to build the model (most probably `csv` files)

+ `main.py`
    + Use the Neural Network to build a model.
