import numpy as np
from ..base.metadata_mixin import MetadataMixin
from ..base.save_mixin import SaveMixin
from ..base.transform_mixin import TransformMixin
from ..utils.typesafety import type_safe, not_none
from ..utils.exports import export


@export
class PrincipalComponentAnalysis(MetadataMixin, SaveMixin, TransformMixin):
    '''
    Used to reduce feature dimension by performing Principal Component Analysis
    '''

    @type_safe
    def __init__(self, n_components: int = None, *, solver: str = 'svd'):
        '''
        Initalize the PCA instance

        Parameters:
            n_components: int, default = None
                Number of components in the reduced dimension matrix
            solver: str, default = 'svd'
                Solver method to use, can be 'svd' or 'eigen'
        '''
        self.n_components = n_components
        self.solver = solver
        self._solver_fn = self._eigen_solver if self.solver == 'eigen' else self._svd_solver

    @type_safe(skip=('return',))
    @not_none(nullable=('y',))
    def fit(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> 'PrincipalComponentAnalysis':
        '''
        Fits the PCA instance with the given features and labels

        Parameters:
            X: np.ndarray of shape (n_samples, n_features)
                The feature matrix to use for PCA

        Returns:
            self: a fitted PCA instance
        '''
        self.X = X

        if X.ndim != 2:
            raise ValueError('Parameter X must be 2 dimensional')

        self.n_samples, self.n_features = X.shape

        # If number of components is not specified, set it to 1
        if self.n_components is None:
            self.n_components = 1

        # Cannot project into a bigger space
        # So n_components must be smaller than the number of features
        if self.n_components >= self.n_features:
            raise ValueError(
                f'Number of components is {self.n_components}, '
                f'but feature dimension is only {self.n_features}'
            )

        # Solve for projection matrix
        self._solver_fn()

        # Ignore complex parts (they show up at +0.00j in some calculations)
        self.right_singular_vectors = self.right_singular_vectors.real

        # Compute the projection matrix, only used in the `get_approx_matrix` method
        self.projection_matrix = self.right_singular_vectors @ self.right_singular_vectors.T
        self.projection_matrix = self.projection_matrix.real

        self._attrs = (
            'X', 'n_samples', 'n_features',
            'right_singular_vectors', 'projection_matrix',
        )
        if self.solver == 'svd':
            self._attrs += ('U', 'S', 'Vh')
        elif self.solver == 'eigen':
            self._attrs += ('eig_vals', 'eig_vecs')
        return self

    def _svd_solver(self) -> None:
        # Perform singular value decomposition
        self.U, self.S, self.Vh = np.linalg.svd(self.X, full_matrices=False)
        # Right Singular Vectors contain Vh[0 to n_components - 1].T
        self.right_singular_vectors = self.Vh[:self.n_components].T

    def _eigen_solver(self) -> None:
        eig_vals, eig_vecs = np.linalg.eig(self.X.T @ self.X)

        # Sorted indices eigen values, in decreasing order
        sorted_indices = np.argsort(np.abs(eig_vals))[::-1]

        # modify eig_vals and eig_vecs to be sorted in decreasing order
        self.eig_vals = eig_vals[sorted_indices]
        self.eig_vecs = eig_vecs[:, sorted_indices]

        # Right Singular Vectors contain eig_vecs[first `n_components` columns]
        self.right_singular_vectors = self.eig_vecs[:, : self.n_components]

    @type_safe(skip=('y',))
    @not_none(nullable=('y',))
    def transform(self, X: np.ndarray, y: np.ndarray = None, **kwargs) -> np.ndarray:
        '''
        Reduces the feature dimensions of a given matrix to the
        number of components specified in the PCsA instance

        Parameters:
            X: np.ndarray of shape (m, n)
                The matrix to transform
        Returns:
            numpy.ndarray: The transformed matrix
        '''
        # Ensure the instance is fitted before transforming
        self._check_is_fitted()
        return X @ self.right_singular_vectors

    @type_safe
    @not_none
    def get_approx_matrix(self, X: np.ndarray) -> np.ndarray:
        # Ensure the instance is fitted before projection
        self.check_is_fitted()
        return X @ self.projection_matrix
