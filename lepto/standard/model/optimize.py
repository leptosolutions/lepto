import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple, Optional, Union
from scipy.sparse import csr_matrix
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

from lepto.standard.model.loss import gaussian_loss, poisson_loss, gamma_loss, tweedie_loss

def ridge_penalty(
    x: ArrayLike,
    lam: float,
    matrix_penalty: Union[np.ndarray, csr_matrix],
    offset_D: float = 0.0
) -> Tuple[float, np.ndarray]:
    """
    Compute the ridge penalty and its gradient.

    The penalty is defined as:
        P(x) = λ * || Mx + offset_D ||²
    where:
        - λ is the regularization weight
        - M is the penalty matrix
        - x is the coefficient vector

    Parameters
    ----------
    x : array-like (shape: n_features,)
        Coefficient vector on which to apply the penalty.

    lam : float
        Regularization weight (λ).

    matrix_penalty : ndarray or scipy.sparse.csr_matrix (shape: m x n_features)
        Penalty matrix M.

    offset_D : float, optional (default=0.0)
        Constant offset added to the product Mx.

    Returns
    -------
    penalty : float
        Ridge penalty value at x.

    grad : ndarray (shape: n_features,)
        Gradient of the penalty with respect to x.
    """
    # Convert x to array
    x = np.asarray(x, dtype=float)

    # Compute product
    prod = matrix_penalty @ x + offset_D

    # Penalty value
    penalty = lam * np.sum(prod ** 2)

    # Gradient
    grad = 2.0 * lam * (matrix_penalty.T @ prod)

    return penalty, grad


class GLMFit(BaseEstimator, RegressorMixin):
    """
        Generalized Linear Model (GLM) optimizer with optional quadratic difference penalty,
        designed to be compatible with scikit-learn's estimator API.

        This class fits a GLM using L-BFGS optimization, optionally applying a structured
        penalty term to encourage smoothness or other constraints on coefficients.

        **Objective Function**
        ----------------------
        The optimizer minimizes:
            J(β) = L_family(β; X, y, w) + λ · || D β + c ||²

        where:
        - L_family : Negative log-likelihood for the chosen GLM family.
        - λ (`lam`) : Penalty strength for the quadratic term.
        - D : Penalty matrix (e.g., difference operator for smoothness).
        - c : Offset term derived from `offset_betas` (fixed coefficients).

        **Supported Families**
        -----------------------
        - "gaussian"  : Identity link, normal distribution.
        - "poisson"   : Log link, Poisson distribution.
        - "gamma"     : Log link, Gamma distribution.
        - "tweedie"   : Log link, Tweedie distribution (power ∈ (1, 2)).
        - "binomial"  : Logit link, Bernoulli distribution.

        **Assumptions**
        ---------------
        - The design matrix `X` must include an intercept column if required
        (e.g., last column of ones). This class does **not** automatically add an intercept.

        Parameters
        ----------
        lam : float, default=1e-2
            Penalty strength λ for the quadratic penalty term.
        family : {"gaussian", "poisson", "gamma", "tweedie", "binomial"}, default="poisson"
            GLM family specifying the likelihood and link function.
        tweedie_power : float, default=1.5
            Power parameter for Tweedie family; must satisfy 1 < power < 2.
        max_iter : int, default=200
            Maximum number of iterations for the L-BFGS optimizer.
        tol : float, default=1e-6
            Convergence tolerance for optimization.
        verbose : int, default=0
            Verbosity level (0 = silent, >0 = print optimization status).

        Attributes
        ----------
        betas : ndarray of shape (n_features,)
            Estimated coefficients after fitting.
        m : scipy.optimize.OptimizeResult
            Result object returned by `scipy.optimize.minimize`.
        is_fitted_ : bool
            Indicator set to True after a successful fit.
        offset_betas : ndarray or None
            Optional vector of fixed coefficients; NaNs indicate parameters to optimize.
        x0 : ndarray
            Initial guess for coefficients used by the optimizer.

        Notes
        -----
        - The penalty term allows structured regularization (e.g., smoothness via difference matrices).
        - Intercept rebasing is applied after optimization to align predicted mean with observed mean.
        - Compatible with scikit-learn pipelines and cross-validation.

        Examples
        --------
        >>> glm = GLMFit(lam=0.01, family="poisson")
        >>> X = np.random.rand(100, 3)
        >>> D = continuous_matrix(size=3)  # Example penalty matrix
        >>> y = np.random.poisson(lam=2.0, size=100)
        >>> glm.fit((X, D), y, sample_weight=np.ones_like(y))
        >>> preds = glm.predict((X, D))
        >>> preds[:5]
        array([2.01, 1.98, 2.05, 2.10, 1.95])
        """


    def __init__(self, lam=1e-2, family="poisson",
        tweedie_power: float = 1.5,
        max_iter: int = 200,
        tol: float = 1e-6,
        verbose: int = 0,
):
        
        self.lam = lam
        self.family = family
        self.tweedie_power = tweedie_power
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose

        # Internal state
        self.x0: Optional[np.ndarray] = None
        self.offset_betas: Optional[np.ndarray] = None
        self.betas: Optional[np.ndarray] = None
        self.m = None
        self.is_fitted_ = False

        # Internal mapping to the corresponding loss function
        self._loss_fn_map = {
            "gaussian": gaussian_loss,
            "poisson": poisson_loss,
            "gamma": gamma_loss,
            "tweedie": tweedie_loss, 
        }
        if self.family == "tweedie":
            if not (1.0 < self.tweedie_power < 2.0):
                raise ValueError("`tweedie_power` must be in (1, 2) for Tweedie.")
            self._loss_kwargs = {"power": self.tweedie_power}
        else:
            self._loss_kwargs = {}

    def fit(self, XD, y, sample_weight, offset_betas=None):
        """
        Fit the Generalized Linear Model (GLM) with optional structured penalty.

        This method optimizes the GLM coefficients using L-BFGS, minimizing:
            J(β) = L_family(β; X, y, w) + λ · || D β + c ||²

        Parameters
        ----------
        XD : tuple (X, D)
            X : ndarray or csr_matrix of shape (n_samples, n_features)
                Design matrix including intercept column if required.
            D : ndarray or csr_matrix of shape (m, n_features)
                Penalty matrix (e.g., difference operator). Can be None for no penalty.
        y : array-like of shape (n_samples,)
            Target variable.
        sample_weight : array-like of shape (n_samples,), optional
            Observation weights for likelihood computation.
        offset_betas : array-like of shape (n_features,), optional
            Coefficient offsets:
            - NaNs indicate parameters to optimize.
            - Non-NaNs are fixed and moved to offset contributions.

        Returns
        -------
        self : GLMFit
            Fitted estimator instance.

        Raises
        ------
        ValueError
            If `offset_betas` length does not match number of features in X.

        Notes
        -----
        - After optimization, intercept is rebased to align predicted mean with observed mean.
        - Supports structured penalties for smoothness or graph constraints.

        Examples
        --------
        >>> glm = GLMFit(lam=0.01, family="poisson")
        >>> X = np.random.rand(100, 3)
        >>> D = continuous_matrix(size=3)
        >>> y = np.random.poisson(lam=2.0, size=100)
        >>> glm.fit((X, D), y, sample_weight=np.ones_like(y))
        >>> glm.is_fitted_
        True
        """

        # Unpack 
        X, D = XD

        # Initialize betas
        x0 = self._generate_x0(X, y, sample_weight)
        # Apply offset logic (NaNs → optimize; fixed → offsets)
        offset_X, offset_D, X_red, D_red, x0_red = self._offset_betas(offset_betas, X, D, x0)
        
        # Optimizes the loss function
        self.m = minimize(
            fun=self._total_loss,
            x0=x0_red,
            args=(X_red, y, sample_weight, self.lam, D_red, offset_X, offset_D),
            jac=True,
            method="L-BFGS-B",
            tol=self.tol,
            options={"maxiter": self.max_iter},
        )
        if not self.m.success and self.verbose:
            print(f"[GLMFit] Optimize status: {self.m.status} — {self.m.message}")

        # Round optimization
        self.m.x = np.round(self.m.x, 5)
        # Map back to full coefficient vector
        self._remap_betas(offset_betas)
        self.is_fitted_ = True
        # Rebase to match mean in train
        self._rebase_intercept(XD, y, sample_weight)
        return self


    def predict(self, XD) -> np.ndarray:
        
        """
        Predict expected response E[y|X] under the fitted GLM.

        For binomial family, returns probabilities P(y=1|X).

        Parameters
        ----------
        XD : tuple (X, D)
            X : ndarray or csr_matrix of shape (n_samples, n_features)
                Design matrix used for prediction.
            D : ndarray or csr_matrix
                Penalty matrix (ignored during prediction, included for API consistency).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted mean response under the chosen GLM family.

        Raises
        ------
        RuntimeError
            If called before `fit`.
        ValueError
            If `family` is unsupported.

        Examples
        --------
        >>> glm = GLMFit(family="binomial")
        >>> glm.fit((X, D), y, sample_weight=np.ones_like(y))
        >>> probs = glm.predict((X, D))
        >>> probs[:5]
        array([0.52, 0.48, 0.55, 0.50, 0.49])
        """

        if not self.is_fitted_:
            raise RuntimeError("Call `fit` before `predict`.")

        X, D = XD
        linear = X @ self.betas
        if self.family == "gaussian":
            return np.asarray(linear, dtype=float)
        if self.family in ("poisson", "gamma", "tweedie"):
            return np.exp(linear)
        if self.family == "binomial":
            return 1.0 / (1.0 + np.exp(-linear))
        raise ValueError(f"Unsupported family for predict: {self.family}")



    def _generate_x0(self, X, y, w) -> np.ndarray:
        
        """
        Generate initial coefficient vector for optimization.

        Heuristic:
        - All coefficients start at zero.
        - Intercept (last column) initialized based on family:
            gaussian  : weighted mean(y)
            poisson/gamma/tweedie : log(mean(y))
            binomial  : logit(mean(y)) with clipping to avoid extremes.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Design matrix.
        y : array-like of shape (n_samples,)
            Target variable.
        w : array-like of shape (n_samples,)
            Sample weights.

        Returns
        -------
        x0 : ndarray of shape (n_features,)
            Initial guess for coefficients.

        Raises
        ------
        ValueError
            If `family` is unsupported.

        Notes
        -----
        - Uses small epsilon to avoid log(0) or division by zero.
        """

        n_features = X.shape[1]
        x0 = np.zeros(n_features, dtype=float)
        ybar = float(np.sum(w * y) / np.sum(w))
        eps = np.finfo(float).eps
        if self.family == "gaussian":
            x0[-1] = ybar
        elif self.family in ("poisson", "gamma", "tweedie"):
            x0[-1] = np.log(max(ybar, eps))
        elif self.family == "binomial":
            ybar = np.clip(ybar, eps, 1.0 - eps)
            x0[-1] = np.log(ybar / (1.0 - ybar)) 
        else:
            raise ValueError(f"Unsupported family: {self.family}")
        return x0

    def _offset_betas(
        self,
        offset_betas: Optional[ArrayLike],
        X: Union[np.ndarray, csr_matrix],
        D: Optional[Union[np.ndarray, csr_matrix]],
        x0: np.ndarray,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Union[np.ndarray, csr_matrix], Optional[Union[np.ndarray, csr_matrix]], np.ndarray]:
        
        """
        Apply offset coefficients and reduce design matrices for optimization.

        Logic:
        - NaNs in `offset_betas` → optimize these coefficients.
        - Non-NaNs → fixed; their contributions moved to offsets:
            offset_X = X @ fixed_part
            offset_D = D @ fixed_part
        - Reduce X, D, and x0 to optimizing columns.

        Parameters
        ----------
        offset_betas : array-like or None
            Vector of offsets; NaNs mark coefficients to optimize.
        X : ndarray or csr_matrix
            Full design matrix.
        D : ndarray or csr_matrix or None
            Penalty matrix.
        x0 : ndarray
            Initial coefficients.

        Returns
        -------
        offset_X : ndarray of shape (n_samples,)
            Contribution of fixed coefficients to linear predictor.
        offset_D : ndarray or scalar
            Contribution of fixed coefficients to penalty term.
        X_red : ndarray or csr_matrix
            Reduced design matrix for optimizing coefficients.
        D_red : ndarray or csr_matrix
            Reduced penalty matrix.
        x0_red : ndarray
            Reduced initial coefficients.

        Raises
        ------
        ValueError
            If `offset_betas` length does not match number of columns in X.
        """

        if offset_betas is None:
            self.offset_betas = None
            return np.zeros(X.shape[0], dtype=float), 0, X, D, x0

        self.offset_betas = np.asarray(offset_betas, dtype=float)
        if self.offset_betas.shape[0] != X.shape[1]:
            raise ValueError("`offset_betas` length must equal number of columns in X.")

        mask_opt = np.isnan(self.offset_betas)     # True → optimize
        fixed = np.where(mask_opt, 0.0, self.offset_betas)

        offset_X = X @ fixed
        offset_D = 0 if D is None else D @ fixed

        X_red = X[:, mask_opt]
        D_red = 0 if D is None else D[:, mask_opt]
        x0_red = x0[mask_opt]
        return offset_X, offset_D, X_red, D_red, x0_red


    def _total_loss(self, x, X, y, weights, lam, D, offset_X, offset_D):

        """
        Compute total objective and gradient for optimization.

        Objective:
            J(β) = L_family(β; X, y, w) + λ · || D β + c ||²

        Parameters
        ----------
        x : ndarray
            Current coefficient vector (optimized subset).
        X : ndarray or csr_matrix
            Reduced design matrix.
        y : ndarray
            Target variable.
        weights : ndarray
            Sample weights.
        lam : float
            Penalty strength.
        D : ndarray or csr_matrix
            Reduced penalty matrix.
        offset_X : ndarray
            Contribution of fixed coefficients to linear predictor.
        offset_D : ndarray or scalar
            Contribution of fixed coefficients to penalty term.

        Returns
        -------
        loss : float
            Objective value.
        grad : ndarray
            Gradient vector.
        """

        loss_fn = self._loss_fn_map[self.family]
        _loss, _loss_grad = loss_fn(
            x=x,
            X=X,
            y=y,
            weights=weights,
            offset_X=offset_X,
            **self._loss_kwargs)
        _penalty, _penalty_grad = ridge_penalty(x, lam, D, offset_D)
        return _loss + _penalty, _loss_grad + _penalty_grad

    def _remap_betas(self, offset_betas: Optional[ArrayLike]) -> None:
        
        """
        Reconstruct full coefficient vector after optimization.

        If `offset_betas` was provided:
        - Fill optimized positions with solution from optimizer.
        - Keep fixed positions unchanged.

        Parameters
        ----------
        offset_betas : array-like or None
            Original offset vector with NaNs marking optimized positions.

        Returns
        -------
        None
            Updates `self.betas` in place.
        """

        if offset_betas is None:
            self.betas = np.asarray(self.m.x, dtype=float)
            return
        full = np.asarray(offset_betas, dtype=float).copy()
        mask_opt = np.isnan(full)
        full[mask_opt] = self.m.x
        self.betas = np.asarray(full, dtype=float)

    def _rebase_intercept(self, XD, y, w):   
        """
        Adjust the model intercept so that the mean of predictions matches
        the observed mean in the training set.

        This method is useful after regularization (Ridge, Lasso, graph penalties)
        which can introduce a global bias in predictions. The adjustment modifies
        only the last coefficient (intercept) without changing other parameters.

        Parameters
        ----------
        XD : (X, D)
            X : ndarray or csr_matrix, shape (n_samples, n_features)
            D : ndarray or csr_matrix, shape (m, n_features)

        y : array-like
            Observed target values corresponding to XD.

        w : array-like
            Sample weights corresponding to XD.

        Behavior by family
        ------------------
        - "gaussian": adjustment on the raw scale (difference of means).
        - "poisson", "gamma", "tweedie": adjustment on the log scale (log of means).
        - "binomial": adjustment on the logit scale (log(p/(1-p))).

        Notes
        -----
        - A small epsilon is used to avoid numerical issues such as log(0) or division by zero.
        - This correction changes the model structure (intercept) rather than applying
        a post-hoc prediction shift.
        - For the binomial family, proportions are clipped to (eps, 1 - eps).

        Returns
        -------
        None
            Updates `self.betas[-1]` (intercept) in place.
        """
        eps = 1e-8
        y_pred_mean = np.sum(self.predict(XD) * w) / np.sum(w)
        y_mean = np.sum(y * w) / np.sum(w)
        if self.family == "gaussian":
            self.betas[-1] = self.betas[-1] + (y_mean - y_pred_mean)
        elif self.family in ("poisson", "gamma", "tweedie"):
            y_mean = max(y_mean, eps)
            y_pred_mean = max(y_pred_mean, eps)
            self.betas[-1] = self.betas[-1] + (np.log(y_mean) - np.log(y_pred_mean))
        elif self.family == "binomial":     
            y_mean = np.clip(y_mean, eps, 1 - eps)
            y_pred_mean = np.clip(y_pred_mean, eps, 1 - eps)
            self.betas[-1] = self.betas[-1] + (np.log(y_mean / (1 - y_mean)) - np.log(y_pred_mean / (1 - y_pred_mean)))
