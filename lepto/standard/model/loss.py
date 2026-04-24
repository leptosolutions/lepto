import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, Union
from scipy.sparse import csr_matrix


def poisson_loss(x: ArrayLike,
                 X: Union[np.ndarray, csr_matrix],
                 y: ArrayLike,
                 weights: ArrayLike,
                 offset_X: float = 0.0
                 ) -> Tuple[float, np.ndarray]:
    
    """
    Compute the Poisson log-likelihood and its gradient.

    
    The Poisson GLM assumes:
        η = X @ x + offset_X        (linear predictor)
        μ = exp(η)                  (mean parameter via log link)

    Loss:
        L = Σ w_i * [ μ_i - y_i * log(μ_i) ]
    Gradient:
        ∂L/∂x = Xᵀ @ [ w ⊙ ( μ - y ) ]

    Parameters
    ----------
    x : 1D array-like
        Coefficient vector for Poisson regression.

    X : ndarray or scipy.sparse.csr_matrix
        Design matrix (features).

    y : 1D array-like
        Target vector (counts).

    weights : 1D array-like
        Observation weights applied to each row of X.

    offset_X : float, optional, default=0.0
        Offset term added to the linear predictor.

    Returns
    -------
    loss : float
        Negative log-likelihood value.

    grad : ndarray
        Gradient of the loss with respect to coefficients `x`.
    """
    # Convert inputs to arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Linear predictor
    linear_pred = X @ x + offset_X

    # Mean parameter via log link
    mu = np.exp(linear_pred)

    # Negative log-likelihood
    loss = weights @ (mu - y * np.log(mu))

    # Gradient
    grad = X.T @ (weights * (mu - y))


    return loss, grad

def gaussian_loss(x: ArrayLike,
                 X: Union[np.ndarray, csr_matrix],
                 y: ArrayLike,
                 weights: ArrayLike,
                 offset_X: float = 0.0
                 ) -> Tuple[float, np.ndarray]:
    
    """
    Compute the Gaussian log-likelihood and its gradient.

    
    For a Gaussian GLM with identity link:
        η = X @ x + offset_X
        μ = η

    Loss:
        L = 0.5 * Σ w_i * (y_i - μ_i)²
    Gradient:
        ∂L/∂x = Xᵀ @ [ w ⊙ (μ - y) ]

    Parameters
    ----------
    x : 1D array-like
        Coefficient vector for Gaussian regression.

    X : ndarray or scipy.sparse.csr_matrix
        Design matrix (features).

    y : 1D array-like
        Target vector (counts).

    weights : 1D array-like
        Observation weights applied to each row of X.

    offset_X : float, optional, default=0.0
        Offset term added to the linear predictor.

    Returns
    -------
    loss : float
        Negative log-likelihood value.

    grad : ndarray
        Gradient of the loss with respect to coefficients `x`.
    """
    
    # Convert inputs to arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Linear predictor and mean
    mu = X @ x + offset_X

    # Negative log-likelihood
    loss = 0.5 * weights @ ((mu - y) ** 2)

    # Gradient wrt coefficients
    grad = X.T @ (weights * (mu - y))


    return loss, grad

def binomial_loss(
    x: ArrayLike,
    X: Union[np.ndarray, csr_matrix],
    y: ArrayLike,
    weights: ArrayLike,
    offset_X: float = 0.0
) -> Tuple[float, np.ndarray]:
    """
    Compute the Binomial negative log-likelihood and its gradient.

    For a Binomial GLM with logit link:
        η = X @ x + offset_X
        μ = sigmoid(η) = 1 / (1 + exp(-η))

    Loss:
        L = Σ w_i * [ -y_i log(μ_i) - (1 - y_i) log(1 - μ_i) ]
    Gradient:
        ∂L/∂x = Xᵀ @ [ w ⊙ (μ - y) ]

    Parameters
    ----------
    x : 1D array-like
        Coefficient vector for Binomial regression (shape: n_features,).

    X : ndarray or scipy.sparse.csr_matrix
        Design matrix of features (shape: n_samples x n_features).

    y : 1D array-like
        Binary target values (0 or 1) (shape: n_samples,).

    weights : 1D array-like
        Observation weights applied per sample (shape: n_samples,).

    offset_X : float, optional (default=0.0)
        Constant offset added to the linear predictor.

    Returns
    -------
    loss : float
        Binomial negative log-likelihood (unnormalized).

    grad : ndarray
        Gradient of the loss with respect to coefficients `x` (shape: n_features,).

    """
    # Convert inputs to arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Linear predictor
    score = X @ x + offset_X

    # Sigmoid function for μ
    mu = 1.0 / (1.0 + np.exp(-score))

    # Negative log-likelihood
    loss = weights @ (-y * np.log(mu) - (1.0 - y) * np.log(1.0 - mu))

    # Gradient wrt coefficients
    grad = X.T @ (weights * (mu - y))

    return loss, grad

def gamma_loss(
    x: ArrayLike,
    X: Union[np.ndarray, csr_matrix],
    y: ArrayLike,
    weights: ArrayLike,
    offset_X: float = 0.0
) -> Tuple[float, np.ndarray]:
    """
    Compute the Gamma negative log-likelihood and its gradient.

    For a Gamma GLM with log link:
        η = X @ x + offset_X
        μ = exp(η)

    Loss:
        L = Σ w_i * [ log(μ_i) + y_i / μ_i ]
    Gradient:
        ∂L/∂x = Xᵀ @ [ w ⊙ ( (dμ/dη)/μ - y * (dμ/dη)/μ² ) ]
        where dμ/dη = μ (since log link)

    Parameters
    ----------
    x : 1D array-like
        Coefficient vector for Gamma regression (shape: n_features,).

    X : ndarray or scipy.sparse.csr_matrix
        Design matrix of features (shape: n_samples x n_features).

    y : 1D array-like
        Positive target values (shape: n_samples,).

    weights : 1D array-like
        Observation weights applied per sample (shape: n_samples,).

    offset_X : float, optional (default=0.0)
        Constant offset added to the linear predictor.

    Returns
    -------
    loss : float
        Gamma negative log-likelihood (unnormalized).

    grad : ndarray
        Gradient of the loss with respect to coefficients `x` (shape: n_features,).

    Notes
    -----
    - Ensure y > 0 (Gamma domain).
    - If you need an average loss, divide externally by sum(weights).
    """
    # Convert inputs to arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Linear predictor
    score = X @ x + offset_X

    # Mean and derivative
    mu = np.exp(score)          # μ = exp(η)
    mu_deriv = mu               # dμ/dη = μ (log link)

    # Negative log-likelihood
    loss = weights @ (np.log(mu) + y / mu)

    # Gradient
    grad = X.T @ (weights * ((mu_deriv / mu) - y * (mu_deriv / (mu ** 2))))

    return loss, grad

def tweedie_loss(
    x: ArrayLike,
    X: Union[np.ndarray, csr_matrix],
    y: ArrayLike,
    weights: ArrayLike,
    offset_X: float = 0.0,
    power: float = 1.5
) -> Tuple[float, np.ndarray]:
    """
    Compute the Tweedie negative log-likelihood and its gradient.

    For a Tweedie GLM with log link:
        η = X @ x + offset_X
        μ = exp(η)

    Loss:
        L = Σ w_i * [ -y_i * μ_i^(1 - p)/(1 - p) + μ_i^(2 - p)/(2 - p) ]
    Gradient:
        ∂L/∂x = Xᵀ @ [ w ⊙ ( -y_i * μ_i^(-p) * dμ/dη + μ_i^(1 - p) * dμ/dη ) ]
        where dμ/dη = μ (log link)

    Parameters
    ----------
    x : 1D array-like
        Coefficient vector for Tweedie regression (shape: n_features,).

    X : ndarray or scipy.sparse.csr_matrix
        Design matrix of features (shape: n_samples x n_features).

    y : 1D array-like
        Positive target values (shape: n_samples,).

    weights : 1D array-like
        Observation weights applied per sample (shape: n_samples,).

    power : float, optional (default=1.5)
        Tweedie power parameter (must satisfy 1 < power < 2).
        - power = 1 → Poisson
        - power = 2 → Gamma
        - 1 < power < 2 → Compound Poisson-Gamma

    offset_X : float, optional (default=0.0)
        Constant offset added to the linear predictor.

    Returns
    -------
    loss : float
        Tweedie negative log-likelihood (unnormalized).

    grad : ndarray
        Gradient of the loss with respect to coefficients `x` (shape: n_features,).

    Raises
    ------
    ValueError
        If `power` is not in the interval (1, 2).

    Notes
    -----
    - Ensure y ≥ 0 and μ > 0.
    - If you need an average loss, divide externally by sum(weights).
    """
    if not (1 < power < 2):
        raise ValueError("Tweedie power must be between 1 and 2 (exclusive).")

    # Convert inputs to arrays
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    weights = np.asarray(weights, dtype=float)

    # Linear predictor
    score = X @ x + offset_X

    # Mean and derivative
    mu = np.exp(score)          # μ = exp(η)
    mu_deriv = mu               # dμ/dη = μ (log link)

    # Negative log-likelihood
    loss = weights @ (
        (-y * (mu ** (1 - power)) / (1 - power)) +
        ((mu ** (2 - power)) / (2 - power))
    )

    # Gradient
    grad = X.T @ (
        weights * (
            (-y * (mu ** (-power)) * mu_deriv) +
            ((mu ** (1 - power)) * mu_deriv)
        )
    )

    return loss, grad

