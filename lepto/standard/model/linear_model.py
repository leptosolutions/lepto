# Standard libraries
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    r2_score,
    log_loss,
    mean_poisson_deviance,
    mean_gamma_deviance,
    mean_tweedie_deviance
)

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Custom modules
import lepto
from lepto.standard.model.transformers import GLMData
from lepto.standard.model.optimize import GLMFit


class GLMDiff(BaseEstimator, RegressorMixin):
    
    """
    High-level interface for fitting Generalized Linear Models (GLMs) with structured penalties
    (continuous, categorical, or graph-based) using a scikit-learn compatible API.

    This class combines:
    - Data preprocessing and penalty matrix construction (`GLMData`).
    - GLM optimization with optional difference penalty (`GLMFit`).
    - A unified pipeline for end-to-end modeling.

    Parameters
    ----------
    family : {"gaussian", "poisson", "gamma", "tweedie", "binomial"}, default="poisson"
        GLM family specifying likelihood and link function.
    tweedie_power : float, default=1.5
        Power parameter for Tweedie family; must satisfy 1 < power < 2.
    fit_intercept : bool, default=True
        Whether to include an intercept term in the model.
    nbins : int, default=20
        Number of bins for discretizing numeric variables.
    lam : float, default=1e-2
        Regularization strength for the quadratic penalty term.
    penalty_choice : dict, optional
        Mapping of variable names to penalty specifications:
        Example:
        {
            'age': {'penalty': 'continuous'},
            'region': {'penalty': 'graph', 'graph': adjacency_matrix}
        }
    offset_betas : array-like of shape (n_features,), optional
        Offset coefficients:
        - NaNs indicate parameters to optimize.
        - Non-NaNs are fixed during optimization.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Combined pipeline of data preprocessing and GLM fitting.
    coef : ndarray
        Estimated coefficients excluding intercept.
    intercept : float
        Estimated intercept term.
    summary : dict
        JSON-like summary of fitted model including coefficients and link function.
    is_fitted_ : bool
        Flag indicating whether the model has been fitted.

    Notes
    -----
    - Supports structured penalties for smoothness and categorical differences.
    - Compatible with scikit-learn tools (cross-validation, pipelines).
    - Automatically computes a summary of coefficients after fitting.

    Examples
    --------
    >>> model = GLMDiff(family="poisson", lam=0.01)
    >>> X = pd.DataFrame({'age': [25, 40, 60], 'region': ['A', 'B', 'A']})
    >>> y = np.random.poisson(lam=2.0, size=3)
    >>> model.fit(X, y)
    >>> preds = model.predict(X)
    >>> model.summary
    {'intercept': ..., 'coefficients': {...}, 'link': 'log'}
    """

    def __init__(self,
                 family="poisson",
                 tweedie_power = 1.5,
                 fit_intercept=True,
                 nbins=20,
                 lam=1e-2,
                 penalty_choice=None,
                 offset_betas=None,
                 categories='auto',
                 sparse_output=False):

        self.family = family
        self.tweedie_power = tweedie_power
        self.fit_intercept = fit_intercept
        self.categories = categories
        self.nbins = nbins
        self.lam = lam
        self.penalty_choice = penalty_choice
        self.offset_betas = offset_betas
        self.sparse_output = sparse_output

        self.data = GLMData(fit_intercept=self.fit_intercept,
                            nbins=self.nbins,
                            categories=self.categories,
                            sparse_output=self.sparse_output)
        self.model = GLMFit(lam=self.lam, 
                            family=self.family,
                            tweedie_power=self.tweedie_power)

        self.pipeline = Pipeline([('data', self.data),
                                  ('model', self.model)])

    def fit(self,
            X,
            y,
            sample_weight=None):
        
        """
        Fit the GLM model with structured penalties.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            Target variable.
        sample_weight : array-like of shape (n_samples,), optional
            Observation weights. Defaults to uniform weights if None.

        Returns
        -------
        self : GLMDiff
            Fitted estimator instance.

        Notes
        -----
        - Internally calls `GLMData.fit` for preprocessing and penalty matrix construction.
        - Calls `GLMFit.fit` for optimization.
        - Computes and stores coefficient summary after fitting.

        Examples
        --------
        >>> model.fit(X, y)
        >>> model.coef, model.intercept
        """

        # Default weights
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        else:
            sample_weight = np.array(sample_weight)

        # Convert y
        y = np.array(y)

        # Fit model
        self.pipeline.fit(X,
                          y,
                          data__penalty_choice=self.penalty_choice,
                          model__sample_weight=sample_weight,
                          model__offset_betas=self.offset_betas)

        self.is_fitted_ = True

        # Save coef
        self.coef = self.pipeline.named_steps['model'].betas[:-1]
        self.intercept = self.pipeline.named_steps['model'].betas[-1]

        # Compute summary
        self.compute_summary()


    def predict(self, X):
        
        """
        Predict expected response E[y|X] under the fitted GLM.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted mean response under the chosen GLM family.

        Raises
        ------
        RuntimeError
            If called before `fit`.

        Examples
        --------
        >>> preds = model.predict(X)
        >>> preds[:5]
        array([2.01, 1.98, 2.05, 2.10, 1.95])
        """

        return self.pipeline.predict(X)
    
    def compute_summary(self):
        
        """
        Compute a structured summary of the fitted model.

        Summary includes:
        - Intercept term.
        - Coefficients for each variable and category/bin.
        - Link function based on GLM family.

        Returns
        -------
        None
            Updates `self.summary` attribute with a JSON-like dictionary.

        Examples
        --------
        >>> model.compute_summary()
        >>> model.summary
        {'intercept': ..., 'coefficients': {'age': {...}, 'region': {...}}, 'link': 'log'}
        """

        categories_var = self.pipeline.named_steps['data']._get_categories_var()
        var_num = self.pipeline.named_steps['data'].var_num
        var_cat = self.pipeline.named_steps['data'].var_cat

        # Init
        self.summary = {}

        # get coefficients
        self.summary['intercept'] = self.intercept
        self.summary['coefficients'] = {}
        i = 0
        for var in (var_num + var_cat):
            self.summary['coefficients'][var] = {}
            for mod in categories_var[var][1:] :
                self.summary['coefficients'][var][mod] = self.coef[i]
                i = i + 1

        # get link
        fam = getattr(self, "family", "gaussian").lower()
        if fam in ['poisson', 'gamma', 'tweedie']:
            self.summary["link"] = 'log'
        elif fam in ['gaussian']:
            self.summary["link"] = 'identity'
        elif fam in ['binomial']:
            self.summary["link"] = 'logit'

        # version
        self.summary["version"] = lepto.__version__

    def compute_summary_df(self):
        return transform_json_to_df(self.summary)

    def plot(self, X, y, sample_weights, var, pred=None):
        
        """
        Analyze and visualize the effect of a single variable on predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            Observed target values.
        sample_weights : array-like of shape (n_samples,)
            Observation weights.
        var : str
            Variable name to analyze.
        pred : array-like of shape (n_samples,)
            Prediction over X.
        

        Returns
        -------
        plot : object
            Interactive plot (depends on `analyse_var` implementation).

        Raises
        ------
        ValueError
            If `var` is not found among numeric or categorical variables.

        Notes
        -----
        - For numeric variables, bins are mapped to interval labels.
        - For categorical variables, original categories are used.

        Examples
        --------
        >>> model.plot(X, y, sample_weights=np.ones(len(y)), var='age')
        """

        # Variables
        var_num = self.pipeline.named_steps['data'].var_num
        var_cat = self.pipeline.named_steps['data'].var_cat
        categories_var = self.pipeline.named_steps['data']._get_categories_var()
        # Process beta
        betas_select = pd.DataFrame(list(self.summary['coefficients'][var].items()), columns=['modality', 'coefficient'])
        betas_select = pd.concat([betas_select, 
                                  pd.DataFrame({"modality": list(set(categories_var[var]) - set(betas_select['modality'].values))[0],  "coefficient": 0}, index=[betas_select.shape[0]])]).sort_values('modality')
        # Data frame without OneHot
        if var in var_cat:
            X_new = pd.DataFrame(self.pipeline.named_steps['data'].transformer.named_steps['col_transformer']['trcat'].transform(X[var_cat]), columns=var_cat)
        elif var in var_num:
            X_new = pd.DataFrame(self.pipeline.named_steps['data'].transformer.named_steps['col_transformer']['trnum'].transform(X[var_num]), columns=var_num)
            X_new[var] = X_new[var].map({i: categories_var[var][i] for i in range(len(categories_var[var]))})
        else: 
            raise ValueError("Unkown variable {}.".format(var))
        # Plot
        return analyse_var(variable=X_new[var],
                                   y=y,
                                   sample_weights=sample_weights,
                                   preds=self.predict(X) if pred is None else pred,
                                   coef=betas_select,
                                   var_name=var)
    
    
    
    def score(self, X, y, sample_weight=None):
        """
        Default scoring used by GridSearchCV when `scoring` is None.

        Behavior by family:
        - 'gaussian': returns r2_score
        - 'poisson': returns pseudo r2_score
        - 'gamma'  : returns pseudo r2_score
        - 'tweedie': returns pseudo r2_score (with self.tweedie_power)
        - 'binomial': returns pseudo r2_score if predict() returns probabilities in (0,1);
                      otherwise falls back to accuracy with threshold 0.5.

        Notes
        -----
        * GridSearchCV maximizes the score; deviance is minimized in GLMs. Therefore
          we return the negative mean deviance for Poisson/Gamma/Tweedie.
        * If `sample_weight` was used in fit, you can pass it here to compute a weighted score.
        * For binomial, if probabilities are available (recommended), log_loss is robust. If
          only labels are available, accuracy is used as a fallback.

        Parameters
        ----------
        X : array-like
            Input features
        y : array-like
            Target values
        sample_weight : array-like or None
            Optional sample weights

        Returns
        -------
        float
            The score (higher is better).
        """
        y = np.asarray(y)
        mu = np.asarray(self.predict(X))  # Expect response-scale predictions

        # Normalize sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        else:
            sample_weight = None

        # Numerical stability guard
        eps = 1e-12

        fam = getattr(self, "family", "gaussian").lower()

        if fam == "gaussian":
            # R^2 (higher is better)
            return r2_score(y, mu, sample_weight=sample_weight)

        elif fam == "poisson":
            # Ensure positive means
            mu = np.clip(mu, eps, None)
            # Negative mean deviance (higher is better)
            return  1- mean_poisson_deviance(y, mu, sample_weight=sample_weight) / mean_poisson_deviance(y, np.ones(len(y)) *np.average(y, weights=sample_weight), sample_weight=sample_weight)

        elif fam == "gamma":
            # Gamma requires positive y and mu; clip mu and guard y
            mu = np.clip(mu, eps, None)
            y_safe = np.clip(y, eps, None)
            return 1 - mean_gamma_deviance(y_safe, mu, sample_weight=sample_weight) / mean_gamma_deviance(y_safe, np.ones(len(y_safe)) * np.average(y_safe, weights=sample_weight), sample_weight=sample_weight)

        elif fam == "tweedie":
            # Tweedie with power in (1,2) typically (compound Poisson-Gamma)
            p = getattr(self, "tweedie_power", 1.5)
            mu = np.clip(mu, eps, None)
            y_safe = np.clip(y, eps, None) if p > 1 else y  # y must be >=0 for many p
            return 1 - mean_tweedie_deviance(y_safe, mu, power=p, sample_weight=sample_weight) / mean_tweedie_deviance(y_safe, np.ones(len(y_safe)) *np.average(y_safe, weights=sample_weight), power=p, sample_weight=sample_weight)

        elif fam == "binomial":
            mu = np.clip(mu, eps, 1.0 - eps)
            return 1 - log_loss(y, mu, sample_weight=sample_weight, labels=[0, 1]) / log_loss(y, np.full_like(y, np.average(y, weights=sample_weight)), sample_weight=sample_weight, labels=[0, 1])


        else:
            raise ValueError(f"Unsupported family for scoring: {self.family!r}")
        
    def variable_importance(self):
        """
        Compute variable importance based on the mean of absolute coefficients for each variable.

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Figure with variable importance bars.
        """
        if not hasattr(self, "summary"):
            self.compute_summary()
        importance = {}
        for var, coef_dict in self.summary["coefficients"].items():
            # Mean absolute values for all levels/bins
            importance[var] = np.mean(np.abs(list(coef_dict.values())))
        
        # Plot importance 
        color_importance = "#1f77b4"
        fig = go.Figure(go.Bar(
            x=list(importance.values()),
            y=list(importance.keys()),
            orientation='h',
            marker=dict(color=color_importance, opacity=0.7)
        ))
        fig.update_layout(
            title="Variables importance",
            xaxis_title="Importance",
            yaxis_title="Variable",
            yaxis={'categoryorder':'total ascending'},
            template="plotly_white",
            height=600,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
        )
        return fig



def analyse_var(variable, y, sample_weights, preds, coef, var_name=""):
    
    """
    Plot Observed vs Predicted by category of `variable`, with Exposure on a secondary y-axis,
    using Plotly Express traces.

    Parameters
    ----------
    variable : array-like
    y : array-like
        Target on the response scale (e.g., frequency, severity, probability).
    sample_weights : array-like or None, optional
        Weights (exposure). If None, all weights are set to 1.
    preds : array-like or None, optional
        Predictions on the same response scale as `y`. If None, zeros are used.
    coef : array-like or None, optional
    var_name : str, optional
        Label for the variable, used in the title.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        Figure with Observed/Predicted lines and Exposure bars (secondary y).
    """

    
    # --- Inputs normalization ---
    variable = np.array(variable)
    y = np.array(y)

    if sample_weights is None:
        sample_weights = np.ones_like(y, dtype=float)
    else:
        sample_weights = np.array(sample_weights, dtype=float)

    if preds is None:
        preds_new = np.zeros_like(y, dtype=float)
    else:
        preds_new = np.array(preds, dtype=float)


    # --- Build working frame ---
    df = pd.DataFrame(
        {
            "var": variable,
            "y": y,
            "w": sample_weights,
            "preds": preds_new,
        }
    )
    df["y_w"] = df["y"] * df["w"]
    df["preds_w"] = df["preds"] * df["w"]

    # --- Aggregations by category ---
    # exposure = sum of weights
    gb_expo = df.groupby("var", observed=False)["w"].sum().rename("exposure")
    # observed = sum(y * w) / sum(w)
    gb_obs = (df.groupby("var", observed=False)["y_w"].sum() / gb_expo).rename("observed")
    # predicted = sum(preds * w) / sum(w)
    gb_pred = (df.groupby("var", observed=False)["preds_w"].sum() / gb_expo).rename("predicted")
    df_gb = pd.concat([gb_obs, gb_pred, gb_expo], axis=1).reset_index()
    

    # Colors
    color_observed = "#800080"  # purple
    color_predicted = "#FFD700" # yellow
    color_exposure = "#1f77b4"  # blue
    color_coef = "#2ca02c"      # green

    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Observed line
    fig.add_trace(go.Scatter(
        x=df_gb["var"], y=df_gb["observed"],
        mode="lines+markers",
        name="Observed",
        line=dict(color=color_observed),
        marker=dict(color=color_observed)
    ), secondary_y=False)
    categories = df_gb["var"].values

    # Predicted line
    if preds is not None:
        fig.add_trace(go.Scatter(
            x=df_gb["var"], y=df_gb["predicted"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color=color_predicted),
            marker=dict(color=color_predicted)
        ), secondary_y=False)

    # Exposure bars
    fig.add_trace(go.Bar(
        x=df_gb["var"], y=df_gb["exposure"],
        name="Exposure",
        marker=dict(color=color_exposure),
        opacity=0.5
    ), secondary_y=True)

    # Coefficients line (if provided)
    if coef is not None and isinstance(coef, pd.DataFrame):
        fig.add_trace(go.Scatter(
            x=coef["modality"], y=coef["coefficient"],
            mode="lines+markers",
            name="Coefficients",
            line=dict(color=color_coef),
            marker=dict(color=color_coef)
        ), secondary_y=False)
        categories = coef["modality"].values

    # Layout
    fig.update_layout(
        title=f"Distribution: {var_name}" if var_name else "Distribution",
        xaxis=dict(title=var_name or "Variable",
                   type="category",
                   categoryorder="array",
                   categoryarray=categories),
        yaxis=dict(title="Value"),
        yaxis2=dict(title="Exposure"),
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )

    return fig

def transform_json_to_df(model_dict) -> pd.DataFrame:
    """
    Build a wide dataframe like:

    Base    <intercept>

    age (level,value) | segment (level,value) | region (level,value)
    ...
    """
    # Retreive data
    intercept = model_dict.get("intercept", np.nan)
    coeffs = model_dict.get("coefficients", {})

    # Build per-feature 2-col tables: (level, value)
    blocks = {}
    for feat, mapping in coeffs.items():
        # mapping can have keys like intervals (str) or numeric categories (e.g., 1.0)
        rows = [(str(k), v) for k, v in mapping.items()]
        df_block = pd.DataFrame(rows, columns=[feat, "value"])
        blocks[feat] = df_block

    # Make all blocks same height by padding with empty rows
    max_len = max((len(b) for b in blocks.values()), default=0)
    padded = []
    for feat, df_block in blocks.items():
        pad_n = max_len - len(df_block)
        if pad_n > 0:
            pad_df = pd.DataFrame({feat: [""] * pad_n, "value": [""] * pad_n})
            df_block = pd.concat([df_block, pad_df], ignore_index=True)

        df_block = df_block.rename(columns={"value": ""})
        df_block = pd.DataFrame([df_block.columns.tolist()] + df_block.values.tolist())
        df_block[2] = ''
        padded.append(df_block)
    wide = pd.concat(padded, axis=1)

    # Create data
    len_data = wide.shape[1]
    top = pd.DataFrame([["Base", "", intercept] + [""] * (wide.shape[1] - 3)],
                        columns=[f"col{i}" for i in range(0, len_data)])

    sep = pd.DataFrame([[""] * len_data], columns=top.columns)
    sep = sep.loc[np.repeat(sep.index, 2)].reset_index(drop=True)

    col_name = pd.DataFrame([[x for feat in blocks.keys() for x in (feat, "", "")]], columns=top.columns)

    sep2 = pd.DataFrame([[""] * len_data], columns=top.columns)
    wide2 = wide.copy()
    wide2.columns = top.columns

    # Concat all
    summary_frame = pd.concat([top, sep, col_name, sep2, wide2], ignore_index=True)

    return summary_frame
