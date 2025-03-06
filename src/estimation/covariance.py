############################################################################
### QPMwP - COVARIANCE
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     18.01.2025
# First version:    18.01.2025
# --------------------------------------------------------------------------



# # Standard library imports
from typing import Union, Optional
from sklearn.covariance import LedoitWolf, MinCovDet  # For shrinkage & robust covariance estimation
from sklearn.covariance import LedoitWolf, MinCovDet  # For shrinkage & robust covariance estimation
from arch import arch_model  # For GARCH model
# Third party imports
import numpy as np
import pandas as pd



# TODO:

# [ ] Add covariance functions:
#    [ ] cov_linear_shrinkage
#    [ ] cov_nonlinear_shrinkage
#    [ ] cov_factor_model
#    [ ] cov_robust
#    [ ] cov_ewma (expoential weighted moving average)
#    [ ] cov_garch
#    [ ] cov_dcc (dynamic conditional correlation)
#    [ ] cov_pc_garch (principal components garch)
#    [ ] cov_ic_garch (independent components analysis)
#    [ ] cov_constant_correlation


# [ ] Add helper methods:
#    [ ] is_pos_def
#    [ ] is_pos_semidef
#    [ ] is_symmetric
#    [ ] is_correlation_matrix
#    [ ] is_diagonal
#    [ ] make_symmetric
#    [ ] make_pos_def
#    [ ] make_correlation_matrix (from covariance matrix)
#    [ ] make_covariance_matrix (from correlation matrix)


# helper methods:
def is_pos_def(matrix: np.ndarray) -> bool:
    """Check if a matrix is positive definite."""
    return np.all(np.linalg.eigvals(matrix) > 0)

def is_pos_semidef(matrix: np.ndarray) -> bool:
    """Check if a matrix is positive semi-definite."""
    return np.all(np.linalg.eigvals(matrix) >= 0)

def is_symmetric(matrix: np.ndarray) -> bool:
    """Check if a matrix is symmetric."""
    return np.allclose(matrix, matrix.T)

def make_symmetric(matrix: np.ndarray) -> np.ndarray:
    """Force a matrix to be symmetric."""
    return (matrix + matrix.T) / 2

def make_pos_def(matrix: np.ndarray) -> np.ndarray:
    """Convert a matrix into a positive definite matrix."""
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-6, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T

def make_correlation_matrix(cov_matrix: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to a correlation matrix."""
    std_dev = np.sqrt(np.diag(cov_matrix))
    return cov_matrix / np.outer(std_dev, std_dev)

def make_covariance_matrix(corr_matrix: np.ndarray, std_dev: np.ndarray) -> np.ndarray:
    """Convert a correlation matrix to a covariance matrix."""
    return corr_matrix * np.outer(std_dev, std_dev)


# --------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------

def cov_pearson(X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    return np.cov(X, rowvar=False) if isinstance(X, np.ndarray) else X.cov()

def cov_linear_shrinkage(X: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance estimation."""
    return LedoitWolf().fit(X).covariance_

def cov_nonlinear_shrinkage(X: np.ndarray) -> np.ndarray:
    """Placeholder for a nonlinear shrinkage estimation (requires more advanced methods)."""
    return LedoitWolf().fit(X).covariance_  # Approximate for now

def cov_factor_model(X: np.ndarray, num_factors: int = 5) -> np.ndarray:
    """Estimate covariance using a factor model (PCA-based)."""
    U, S, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
    factor_cov = np.dot(U[:, :num_factors] * S[:num_factors], Vt[:num_factors])
    return np.cov(factor_cov, rowvar=False)

def cov_robust(X: np.ndarray) -> np.ndarray:
    """Robust covariance estimation using Minimum Covariance Determinant."""
    return MinCovDet().fit(X).covariance_

def cov_ewma(X: np.ndarray, lambda_: float = 0.94) -> np.ndarray:
    """Exponentially weighted moving average covariance matrix."""
    weights = np.array([(1 - lambda_) * lambda_ ** i for i in range(len(X))][::-1])
    weighted_cov = np.cov(X.T, aweights=weights)
    return weighted_cov

def cov_garch(X: np.ndarray) -> np.ndarray:
    """Estimate covariance using univariate GARCH models for each asset."""
    n_assets = X.shape[1]
    cov_matrix = np.zeros((n_assets, n_assets))
    for i in range(n_assets):
        model = arch_model(X[:, i], vol='Garch', p=1, q=1)
        res = model.fit(disp='off')
        cov_matrix[i, i] = res.conditional_volatility[-1]**2
    return cov_matrix

def cov_constant_correlation(X: np.ndarray) -> np.ndarray:
    """Assume all asset correlations are the same and estimate covariance accordingly."""
    avg_corr = np.mean(np.corrcoef(X, rowvar=False))
    std_dev = np.std(X, axis=0)
    return avg_corr * np.outer(std_dev, std_dev)

class CovarianceSpecification(dict):

    def __init__(self,
                 method='pearson',
                #  check_positive_definite=False,
                 **kwargs):
        super().__init__(
            method=method,
            # check_positive_definite=check_positive_definite,
        )
        self.update(kwargs)


class Covariance:

    def __init__(self,
                 spec: Optional[CovarianceSpecification] = None,
                 **kwargs):
        self.spec = CovarianceSpecification() if spec is None else spec
        self.spec.update(kwargs)
        self._matrix: Union[pd.DataFrame, np.ndarray, None] = None

    @property
    def spec(self):
        return self._spec

    @spec.setter
    def spec(self, value):
        if isinstance(value, CovarianceSpecification):
            self._spec = value
        else:
            raise ValueError(
                'Input value must be of type CovarianceSpecification.'
            )
        return None

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        if isinstance(value, (pd.DataFrame, np.ndarray)):
            self._matrix = value
        else:
            raise ValueError(
                'Input value must be a pandas DataFrame or a numpy array.'
            )
        return None

    def estimate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        inplace: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray, None]:

        estimation_method = self.spec['method']

        if estimation_method == 'pearson':
            cov_matrix = cov_pearson(X=X)
        else:
            raise ValueError(
                'Estimation method not recognized.'
            )

        # if self.spec.get('check_positive_definite'):
        #     if not isPD(covmat):
        #         covmat = nearestPD(covmat)

        if inplace:
            self.matrix = cov_matrix
            return None
        else:
            return cov_matrix

