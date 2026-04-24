
import warnings
from typing import Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin

from lepto.standard.model.optimize import ridge_penalty

def _sigmoid(z: np.ndarray) -> np.ndarray:
    
    """
    Numerically stable sigmoid function.

    Parameters
    ----------
    z : ndarray of shape (n_samples,)
        Input linear predictor.

    Returns
    -------
    ndarray
        Element‑wise sigmoid transformation of `z`.
    """

    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


def _softplus(u: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1+exp(u))."""
    # softplus(u) = max(u,0) + log1p(exp(-abs(u)))
    return np.maximum(u, 0.0) + np.log1p(np.exp(-np.abs(u)))



class MonotonePriceLogit(BaseEstimator, ClassifierMixin):
    
    """
    Logistic regression model with a price-dependent monotonicity constraint.

    Model form
    ----------
    The predicted probability is:

        p = sigmoid( X1 @ beta1 + price * (X2 @ beta2) )

    where:
    - `X1` and `X2` are two feature blocks,
    - `beta1` and `beta2` are their respective coefficient vectors.

    Global monotonicity with respect to `price`
    -------------------------------------------
    A monotonic constraint is imposed on the linear term involving `price`:

    * direction='increasing'  →  X2 @ beta2 >= margin
    * direction='decreasing'  →  X2 @ beta2 <= -margin
                                (equivalently, -(X2 @ beta2) >= margin)

    The optional `margin` parameter enforces strict monotonicity and helps avoid
    degenerate solutions such as beta2 ≈ 0.

    Parameters
    ----------
    lam : float, default=1e-2
        L2 regularization strength applied through the penalty matrices `D1` and `D2`
        (ridge-type penalty).
    lam_behaviour : float
        Strength of the monotonicity penalty. 
        0.0 means "no monotonicity encouragement".
    direction : {'increasing', 'decreasing'}, default='decreasing'
        Desired monotonic relationship between the predicted probability and `price`.
    margin : float, default=0.0
        Minimal monotonicity margin. A value of 0 enforces non-strict monotonicity.
    max_iter : int, default=1000
        Maximum number of SLSQP optimizer iterations.
    tol : float, default=1e-6
        Convergence tolerance for SLSQP.
    verbose : bool, default=False
        If True, prints optimizer progress information.
    mono_k : float
        Smoothness factor for the soft hinge. Higher ≈ closer to hard hinge.


    Notes
    -----
    - The model supports *offset coefficients* allowing some elements of `beta1` or
    `beta2` to be fixed (non-optimized) while others remain free.
    - Inequality constraints are implemented through SLSQP to enforce monotonicity.
    - Regularization is applied separately on the `beta1` and `beta2` blocks.

    Attributes
    ----------
    coef_glm1_ : ndarray of shape (n_features_glm1,)
        Estimated coefficients for X1.
    coef_glm2_ : ndarray of shape (n_features_glm2,)
        Estimated coefficients for X2.
    classes_ : ndarray of shape (2,)
        Class labels (always [0, 1]).
    n_iter_ : int
        Number of iterations performed by the optimizer.
    opt_status_ : str
        Message returned by the optimizer regarding convergence state.
    n_glm1_ : int
        Number of coefficients in the first block.
    n_glm2_ : int
        Number of coefficients in the second block.

    """


    def __init__(self,
                 lam = 1e-2,
                 lam_behaviour = 0,
                 direction: str = "decreasing",
                 margin: float = 0.0,
                 max_iter: int = 50000,
                 mono_k: float = 300.0,  
                 tol: float = 1e-6,
                 verbose: bool = False):
        self.lam = lam
        self.lam_behaviour = lam_behaviour
        self.direction = direction
        self.margin = float(margin)
        self.max_iter = int(max_iter)
        self.mono_k = float(mono_k)
        self.tol = tol
        self.verbose = verbose
        self.rows_non_monotone = None


    def _total_loss(self, x, Xs, price, y, weights, lam, Ds, offset_Xs, offset_Ds):
        
        """
        Compute total objective value and gradient: 
        negative log-likelihood + ridge penalties.

        Parameters
        ----------
        x : ndarray
            Current flattened parameter vector `[beta1, beta2]`.
        Xs : tuple (X1, X2)
            Feature blocks for the two GLM components.
        price : ndarray
            Price values modifying the second linear term.
        y : ndarray
            Binary target values in {0, 1}.
        weights : ndarray or None
            Optional sample weights.
        lam : float
            Regularization strength.
        Ds : tuple (D1, D2)
            Penalty matrices for `beta1` and `beta2`.
        offset_Xs : tuple (offset_X1, offset_X2)
            Linear predictor offsets from fixed coefficients.
        offset_Ds : tuple (offset_D1, offset_D2)
            Penalty offsets from fixed coefficients.

        Returns
        -------
        loss : float
            Total loss value.
        grad : ndarray
            Gradient of the total loss with respect to `x`.
        """

        X1, X2 = Xs
        D1, D2 = Ds
        offset_X1, offset_X2 = offset_Xs
        offset_D1, offset_D2 = offset_Ds
        n1, n2 = X1.shape[1], X2.shape[1]
        x1 = x[:n1]
        x2 = x[n1:n1+n2]
        _loss, _loss_grad = self._loss_and_grad(
            theta=x,
            X1=X1,
            X2=X2,
            price=price,
            y=y,
            sample_weight=weights,
            offset_X1=offset_X1,
            offset_X2=offset_X2)
        _penalty1, _penalty_grad1 = ridge_penalty(x1, lam, D1, offset_D1)
        _penalty2, _penalty_grad2 = ridge_penalty(x2, lam, D2, offset_D2)
        
        _mono_pen, _mono_grad2 = self._mono_penalty_and_grad(
                    beta2=x2,
                    X2=X2,
                    offset_X2=offset_X2,
                )
        
        
        total = _loss + _penalty1 + _penalty2 + _mono_pen
        grad = _loss_grad + np.concatenate([_penalty_grad1, _penalty_grad2])
        grad[n1:n1+n2] += _mono_grad2

        return total, grad

    def _loss_and_grad(self, theta, X1, X2,
                       price, y,
                       sample_weight, offset_X1, offset_X2):
        
        """
        Compute the negative log-likelihood and its gradient.

        Parameters
        ----------
        theta : ndarray
            Concatenated parameter vector `[beta1, beta2]`.
        X1, X2 : ndarray
            Feature matrices for the first and second linear components.
        price : ndarray
            Price vector scaling the X2 @ beta2 term.
        y : ndarray
            Binary target vector.
        sample_weight : ndarray or None
            Optional per-sample weight.
        offset_X1, offset_X2 : ndarray
            Linear predictor offsets from fixed coefficients.

        Returns
        -------
        loss : float
            Negative log-likelihood.
        grad : ndarray
            Gradient with respect to all model coefficients.
        """

        n1, n2 = X1.shape[1], X2.shape[1]
        beta1 = theta[:n1]
        beta2 = theta[n1:n1+n2]

        z = X1 @ beta1 + price * (X2 @ beta2) + offset_X1 + offset_X2
        p = _sigmoid(z)

        if sample_weight is None:
            sw = 1.0
        else:
            sw = np.asarray(sample_weight, dtype=float).reshape(-1)
            if sw.shape[0] != y.shape[0]:
                raise ValueError("sample_weight should have the same length as y.")

        eps = 1e-12
        loss = -np.sum((sw) * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))

        resid = (p - y) * (sw)
        # gradients
        grad_beta1 = X1.T @ resid
        # Support both dense and sparse X2 for grad_beta2
        if hasattr(X2, 'multiply'):
            # X2 is sparse (e.g., csr_matrix)
            X2_weighted = X2.multiply(price[:, None])
        else:
            # X2 is dense
            X2_weighted = price[:, None] * X2
        grad_beta2 = X2_weighted.T @ resid
        grad = np.concatenate([grad_beta1, grad_beta2])
        return loss, grad
    
    
    def _mono_penalty_and_grad(
        self,
        beta2: np.ndarray,
        X2: np.ndarray,
        offset_X2: np.ndarray,
    ) -> Tuple[float, np.ndarray]:
        """
        Soft monotonicity penalty + gradient wrt beta2.

        penalty = lam_behaviour * sum (h_i^2),
        h_i = softplus(-k*g_i)/k ≈ max(0, -g_i)
        """
        if self.lam_behaviour <= 0.0:
            return 0.0, np.zeros_like(beta2)

        v = X2 @ beta2 + offset_X2  # (n_samples,)

        if self.direction == "increasing":
            g = v - self.margin          # want g >= 0
            dg_dv = 1.0
        else:
            g = -v - self.margin         # want g >= 0  <=> v <= -margin
            dg_dv = -1.0

        k = self.mono_k

        # u = -k*g; softplus(u)/k approximates max(0, -g)
        u = -k * g
        sp = _softplus(u)
        h = sp / k

        penalty = self.lam_behaviour * np.sum(h * h)

        # d softplus(u)/du = sigmoid(u)
        sig_u = _sigmoid(u)

        # h = softplus(u)/k with u = -k*g
        # dh/dg = (1/k) * dsoftplus/du * du/dg = (1/k) * sig_u * (-k) = -sig_u
        # d(h^2)/dg = 2*h*dh/dg = -2*h*sig_u
        dpen_dg = self.lam_behaviour * (-2.0 * h * sig_u)     # (n_samples,)

        # dg/dv is +/- 1
        dpen_dv = dpen_dg * dg_dv                        # (n_samples,)

        grad_beta2 = X2.T @ dpen_dv                      # (n_features_beta2,)
        return penalty, grad_beta2


    # --- fit/predict ---
    def fit(self, XsDs, y, price, sample_weight=None, offsets_betas=None):
        
        """
        Fit the monotone logistic regression model with constrained optimization.

        Parameters
        ----------
        XsDs : tuple (X1, X2, D1, D2)
            - X1, X2 : feature matrices for each coefficient block
            - D1, D2 : corresponding penalty matrices
        y : ndarray of shape (n_samples,)
            Binary target values {0, 1}.
        price : ndarray of shape (n_samples,)
            Price vector used to modulate the second linear term.
        sample_weight : ndarray or None, optional
            Optional per-sample weights.
        offsets_betas : tuple or None
            Offsets for (beta1, beta2).  
            NaNs mark coefficients to be optimized; non‑NaNs are fixed.

        Returns
        -------
        self : MonotonePriceLogit
            Fitted estimator.

        Notes
        -----
        Uses SLSQP to solve:
            minimize  negative log-likelihood + ridge penalties
            subject to monotonicity constraints on X2 @ beta2.
        """

        # Unpack and validate
        X1, X2, D1, D2 = XsDs
        if offsets_betas is None:
            offset_betas1 = offset_betas2 = None
        else:
            offset_betas1, offset_betas2 = offsets_betas

        y = np.asarray(y, dtype=float).reshape(-1)
        price = np.asarray(price, dtype=float).reshape(-1)

        if not (X1.shape[0] == X2.shape[0] == y.shape[0] == price.shape[0]):
            raise ValueError("X, y and price must have the same length.")

        n1, n2 = X1.shape[1], X2.shape[1]

        # initialisation
        theta0 = np.zeros(n1 + n2, dtype=float)
        n1, n2 = X1.shape[1], X2.shape[1]
        x01 = theta0[:n1]
        x02 = theta0[n1:n1+n2]
        
        # Apply offset logic (NaNs → optimize; fixed → offsets)
        offset_X1, offset_D1, X1_red, D1_red, x01_red = self._offset_betas(offset_betas1, X1, D1, x01)
        offset_X2, offset_D2, X2_red, D2_red, x02_red = self._offset_betas(offset_betas2, X2, D2, x02)
        n1_red = X1_red.shape[1]
        n2_red = X2_red.shape[1]
        theta0_red = np.concatenate([x01_red, x02_red])

        
        def f_obj(th):
            return self._total_loss(
                th,
                (X1_red, X2_red),
                price,
                y,
                sample_weight,
                self.lam,
                (D1_red, D2_red),
                (offset_X1, offset_X2),
                (offset_D1, offset_D2),
            )
        
        res = minimize(
            fun=lambda th: f_obj(th)[0],
            x0=theta0_red,
            jac=lambda th: f_obj(th)[1],
            method="L-BFGS-B",
            tol=self.tol,
            options={"maxiter": self.max_iter},
        )
        if not res.success:
            print(f"[MonotonePriceLogit] Optimizer: {res.message}")

        th = res.x
        betas1 = th[:n1_red]
        betas2 = th[n1_red:n1_red+n2_red]

        # Map back to full coefficient vector
        self.coef_glm1_ = self._remap_betas(betas1, offset_betas1)
        self.coef_glm2_ = self._remap_betas(betas2, offset_betas2)

        self.classes_ = np.array([0, 1])
        self.n_iter_ = res.nit
        self.opt_status_ = res.message
        self.n_glm1_ = n1
        self.n_glm2_ = n2
        
        # Check monotonicity 
        if self.lam_behaviour>0:
            self.rows_non_monotone = self.monotonicity_violations_rows(X2, offset_X2)
            if len(self.rows_non_monotone):
                warnings.warn("Monotonicity constraint is not respected for each row, increase lam_behaviour and check violated rows in object rows_non_monotone")
        return self

    def _offset_betas(
        self,
        offset_betas,
        X,
        D,
        x0,
    ):
        """
        Apply fixed coefficients (offsets) and reduce matrices for optimization.

        Parameters
        ----------
        offset_betas : array-like or None
            Coefficient offsets.  
            - NaN → optimize this coefficient  
            - non-NaN → coefficient held fixed
        X : ndarray or sparse matrix
            Full design matrix.
        D : ndarray or sparse matrix or None
            Penalty matrix corresponding to X.
        x0 : ndarray
            Initial parameter vector.

        Returns
        -------
        offset_X : ndarray
            Contribution of fixed coefficients to the linear predictor.
        offset_D : ndarray or scalar
            Contribution of fixed coefficients in the penalty term.
        X_red : ndarray
            Design matrix restricted to coefficients being optimized.
        D_red : ndarray or scalar
            Reduced penalty matrix.
        x0_red : ndarray
            Initial parameter vector restricted to optimized coefficients.

        Raises
        ------
        ValueError
            If offset_betas has shape inconsistent with X.
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
    
    def decision_function(self, XsDs, price):
        """
        Compute the linear predictor (logits).

        Parameters
        ----------
        XsDs : tuple (X1, X2, D1, D2)
            Feature matrices (penalty matrices ignored here).
        price : ndarray of shape (n_samples,)
            Price vector scaling the X2 @ beta2 term.

        Returns
        -------
        ndarray
            Linear predictor z = X1 @ beta1 + price * (X2 @ beta2).
        """

        X1, X2, D1, D2 = XsDs
        return X1 @ self.coef_glm1_ + price * (X2 @ self.coef_glm2_)

    def predict(self, X, price):
        
        """
        Predict class probabilities.

        Parameters
        ----------
        X : tuple (X1, X2, D1, D2)
            Feature matrices (penalty matrices unused here).
        price : ndarray of shape (n_samples,)
            Price vector used in the model.

        Returns
        -------
        ndarray
            Predicted probabilities in (0, 1) via the sigmoid function.
        """

        z = self.decision_function(X, price)
        return _sigmoid(z)
    
    def _remap_betas(self, betas, offset_betas) -> None:
        
        """
        Reconstruct full coefficient vector after optimization.

        If `offset_betas` was provided:
        - Fill optimized positions with solution from optimizer.
        - Keep fixed positions unchanged.

        Parameters
        ----------
        betas : array-like
            betas.

        offset_betas : array-like or None
            Original offset vector with NaNs marking optimized positions.

        Returns
        -------
        None
            Updates `self.betas` in place.
        """

        if offset_betas is None:
            return np.asarray(betas, dtype=float)
        full = np.asarray(offset_betas, dtype=float).copy()
        mask_opt = np.isnan(full)
        full[mask_opt] = betas
        return np.asarray(full, dtype=float)
    
    
    def monotonicity_violations_rows(
        self,
        X2: np.ndarray,
        offset: Optional[np.ndarray] = None
    ) :
        """
        Test monotonicity row-wise and return indices where it's violated.

        By default uses fitted coef_glm2_ and instance direction/margin.

        Parameters
        ----------
        X2 : ndarray (n_samples, n_features_glm2)
        offset : ndarray or None
            Optional additive offset in v = X2@beta2 + offset.

        Returns
        -------
        List[int]
            Row indices i where monotonicity is NOT respected.
        """
        v = X2 @ np.asarray(self.coef_glm2_, dtype=float)
        if offset is not None:
            v = v + np.asarray(offset, dtype=float).reshape(-1)

        if self.direction == "increasing":
            ok = v >= self.margin
        else:
            ok = v <= -self.margin

        return np.where(~ok)[0].tolist()
