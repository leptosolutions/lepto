
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
r2_score, log_loss,
mean_poisson_deviance,
mean_gamma_deviance,
mean tweedie_deviance)

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom modules
import lepto
from lepto.standard.model.transformers import GLMData
from lepto.standard.model.optimize import GLMFit


class GLMDiff(BaseEstimator, RegressorMixin):
    """ High-level interface for fitting Generalized Linear Models (GLMs) with structured penalties (continuous, categorical, or graph-based) using a scikit-learn compatible API.
    This class combines:
    Data preprocessing and penalty matrix construction (GLMData").
    GLM optimization with optional difference penalty (GLMFit`).
    A unified pipeline for end-to-end modeling.
    Parameters
    family:{"gaussian", "poisson", "gamma", "tweedie' "binomial"}, default="poisson" GLM family specifying likelihood and link function.
    tweedie_power float, default=1.5
    Power parameter for Tweedie family; must satisfy 1 < power < 2.
    fit_intercept: bool, default=True
    Whether to include an intercept term in the model.
    nbins
    int, default=20
    Number of bins for discretizing numeric variables.
    lam: float, default=1e-2
    Regularization strength for the quadratic penalty term.
    penalty_choice: dict, optional