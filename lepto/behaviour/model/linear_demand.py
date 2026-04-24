import numpy as np
import pandas as pd

# Sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss

# Plotly
import plotly.graph_objects as go

# Internal
import lepto
from lepto.behaviour.model.transformers import GLMDemandData
from lepto.behaviour.model.optimize import MonotonePriceLogit
from lepto.standard.model.linear_model import analyse_var, transform_json_to_df

class GLMDemand(BaseEstimator, RegressorMixin):
    
    """
    Generalized Linear Model for demand with monotone price elasticity.

    This class provides a high‑level user API combining:
    - preprocessing of static and elasticity features,
    - constrained logistic regression via `MonotonePriceLogit`,
    - optional penalty structures and fixed‑coefficient offsets.

    The model uses a two‑block representation:
    * static component  → demand level
    * elasticity component → price sensitivity
    Both contribute to a final logistic‑regression–style probability model.

    Parameters
    ----------
    var_glm_static : list of str
        Variables used in the static component of the GLM.

    var_glm_elasticity : list of str
        Variables used in the price‑elasticity component.

    var_behaviour : str
        Behaviour variable

    direction : {'increasing', 'decreasing'}, default='decreasing'
        Direction of the monotonicity constraint with respect to price.

    fit_intercept : bool, default=True
        Whether to include intercept terms in both static and elasticity blocks.

    nbins : int, default=20
        Number of bins used to discretize numerical covariates.

    lam : float, default=1e-2
        Ridge penalty strength propagated to the underlying logistic model.

    static_penalty_choice : dict or None, optional
        Penalty structure for static variables.
        Example:
            {'age': {'penalty': 'continuous'},
            'region': {'penalty': 'graph', 'graph': matrix}}
        where `matrix.shape == (k, k)` encodes graph penalties.

    elasticity_penalty_choice : dict or None, optional
        Same as `static_penalty_choice` but for elasticity variables.

    static_offset_betas : 1d array-like or None, optional
        Offset coefficients for the static block.  
        Use `np.nan` to denote coefficients to be optimized.
        Example:
            >>> offset_betas = np.array([1.1, 2.1, np.nan, np.nan])

    elasticity_offset_betas : 1d array-like or None, optional
        Offset coefficients for the elasticity block, same convention as above.

    categories_static : 'auto' or dict, default='auto'
        Category structure used by `GLMData` for interpreting variable encodings.

    categories_dynamic : 'auto' or dict, default='auto'
        Category structure used by `GLMData` for interpreting variable encodings.

    sparse_output : bool, default=False
        Whether to return sparse matrices for the transformed data.

    Notes
    -----
    - The class delegates preprocessing to `GLMDemandData` and estimation to
    `MonotonePriceLogit`.
    - A scikit‑learn `Pipeline` manages data transforms and model fitting.
    """

    def __init__(self,
                 var_glm_static,
                 var_glm_elasticity,
                 var_behaviour,
                 direction='decreasing',
                 fit_intercept=True,
                 nbins=20,
                 lam=1e-2,
                 lam_behaviour = 0,
                 static_penalty_choice=None,
                 elasticity_penalty_choice=None,
                 static_offset_betas=None,
                 elasticity_offset_betas=None,
                 categories_static="auto",
                 categories_elasticity="auto",
                 sparse_output=False):

        self.var_glm_static = var_glm_static
        self.var_glm_elasticity = var_glm_elasticity
        self.var_behaviour = var_behaviour
        self.direction = direction
        self.fit_intercept = fit_intercept
        self.nbins = nbins
        self.lam = lam
        self.lam_behaviour = lam_behaviour
        self.static_penalty_choice = static_penalty_choice
        self.elasticity_penalty_choice = elasticity_penalty_choice
        self.static_offset_betas = static_offset_betas
        self.elasticity_offset_betas = elasticity_offset_betas
        self.offsets_betas = (self.static_offset_betas, self.elasticity_offset_betas)
        self.categories_static = categories_static
        self.categories_elasticity = categories_elasticity
        self.sparse_output = sparse_output

        self.data = GLMDemandData(
            var_glm_static=self.var_glm_static,
            var_glm_elasticity=self.var_glm_elasticity,
            fit_intercept=self.fit_intercept,
            nbins=self.nbins,
            categories_elasticity=self.categories_elasticity,
            categories_static=self.categories_static,
            sparse_output=self.sparse_output)
        
        self.model = MonotonePriceLogit(lam=self.lam,
                                        lam_behaviour=self.lam_behaviour,
                                        direction=self.direction)

        self.pipeline = Pipeline([('data', self.data),
                                  ('model', self.model)])

    def fit(self,
            X,
            y,
            sample_weight=None):
        """

        Parameters
        ----------
        X : pandas frame, shape [n_samples, n_features]

        y : 1d array-like
            Ground truth (correct) labels

        sample_weight : 1d array-like
            Weights to be applied over X

        Return
        --------
        self
        """
        # Default weights
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        else:
            sample_weight = np.array(sample_weight)

        # Convert y
        y = np.array(y)

        # Convert behaviour
        behaviour = np.array(X[self.var_behaviour].values)

        # X
        X = X.loc[:, X.columns != self.var_behaviour]

        # Fit model
        self.pipeline.fit(X,
                          y,
                          data__static_penalty_choice=self.static_penalty_choice,
                          data__dynamic_penalty_choice=self.elasticity_penalty_choice,
                          model__price=behaviour,
                          model__sample_weight=sample_weight,
                          model__offsets_betas=self.offsets_betas)

        self.is_fitted_ = True

        # Save coef
        self.coef_static = self.pipeline.named_steps['model'].coef_glm1_[:-1]
        self.coef_elasticity = self.pipeline.named_steps['model'].coef_glm2_[:-1]
        self.intercept_static = self.pipeline.named_steps['model'].coef_glm1_[-1]
        self.intercept_elasticity = self.pipeline.named_steps['model'].coef_glm2_[-1]

        # Compute summary
        self.compute_summary()


    def predict(self, X):
        """
        Predict over X


        Parameters
        ----------
        X : pandas frame, shape [n_samples, n_features]

        Return
        --------
        pred : 1d array
            Prediction
        """
        # Convert behaviour
        behaviour = np.array(X[self.var_behaviour].values)

        # X
        X = X.loc[:, X.columns != self.var_behaviour]

        return self.pipeline.predict(X, price=behaviour)
    
    def compute_summary(self):
        
        """
        Compute a structured summary of the fitted model.

        Summary includes:
        - Intercept term.
        - Coefficients for each variable and category/bin.
        - Link function

        Returns
        -------
        None
            Updates `self.summary` attribute with a JSON-like dictionary.

        Examples
        --------
        >>> model.compute_summary()
        """
        # Init
        self.summary = {}
        self.summary['static'] = {}
        self.summary['elasticity'] = {}

        # get coefficients
        self.summary['static']['intercept'] = self.intercept_static
        self.summary['elasticity']['intercept'] = self.intercept_elasticity
        self.summary['static']['coefficients'] = {}
        self.summary['elasticity']['coefficients'] = {}

        for model in ['static', 'elasticity']:
            if model == 'static':
                pipeline_model =self.pipeline.named_steps['data'].static_transformer
            else :
                pipeline_model =self.pipeline.named_steps['data'].dynamic_transformer
            categories_var = pipeline_model._get_categories_var()
            var_num = pipeline_model.var_num
            var_cat = pipeline_model.var_cat
            i = 0
            for var in (var_num + var_cat):
                self.summary[model]['coefficients'][var] = {}
                for mod in categories_var[var][1:] :
                    if model == 'static':
                        self.summary[model]['coefficients'][var][mod] = self.coef_static[i]
                    else :
                        self.summary[model]['coefficients'][var][mod] = self.coef_elasticity[i]
                    i = i + 1

        # get link
        self.summary["link"] = 'logit'
        # Version
        self.summary["version"] = lepto.__version__

    def compute_summary_df(self):
        all_df = {}
        for model in ['static', 'elasticity']:
            all_df[model] = transform_json_to_df(self.summary[model])
        return all_df


    def plot(self, X, y, sample_weights, var, model, pred=None):
        
        """
        Analyze and visualize the effect of a single variable on predictions.

        Parameters
        ----------
        X : pandas frame of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,)
            Observed target values.
        sample_weights : array-like of shape (n_samples,)
            Observation weights.
        var : str
            Variable name to analyze.
        model : str ['static', 'elasticity']
            Model to visulize var on
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
        # Choose model
        if model=="static":
            model_pipeline = self.pipeline.named_steps['data'].static_transformer
        elif model=="elasticity":
            model_pipeline = self.pipeline.named_steps['data'].dynamic_transformer
    
        # Variables
        var_num = model_pipeline.var_num
        var_cat = model_pipeline.var_cat
        categories_var = model_pipeline._get_categories_var()
        # Process beta
        betas_select = pd.DataFrame(list(self.summary[model]['coefficients'][var].items()), columns=['modality', 'coefficient'])
        betas_select = pd.concat([betas_select, 
                                  pd.DataFrame({"modality": list(set(categories_var[var]) - set(betas_select['modality'].values))[0],  "coefficient": 0}, index=[betas_select.shape[0]])]).sort_values(['modality'])
        # Data frame without OneHot
        if var in var_cat:
            X_new = pd.DataFrame(model_pipeline.transformer.named_steps['col_transformer']['trcat'].transform(X[var_cat]), columns=var_cat)
        elif var in var_num:
            X_new = pd.DataFrame(model_pipeline.transformer.named_steps['col_transformer']['trnum'].transform(X[var_num]), columns=var_num)
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

        Parameters
        ----------
        X : pandas frame
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
        # Convert y
        y = np.asarray(y)

        # Mu
        mu = np.asarray(self.predict(X)) 

        # Normalize sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
        else:
            sample_weight = None

        # Numerical stability guard
        eps = 1e-12

        mu = np.clip(mu, eps, 1.0 - eps)
        return 1 - log_loss(y, mu, sample_weight=sample_weight, labels=[0, 1]) / log_loss(y, np.full_like(y, np.average(y, weights=sample_weight)), sample_weight=sample_weight, labels=[0, 1])

    def variable_importance(self, model):
        """
        Compute variable importance for both static and elasticity models.

        Parameters
        ----------
        model : str ['static', 'elasticity']
            Model to visualize variable importance for

        Returns
        -------
        fig : plotly.graph_objects.Figure
            Figure with variable importance bars.
        """
        if not hasattr(self, "summary"):
            self.compute_summary()
        importance = {'static': {}, 'elasticity': {}}
        for models in ['static', 'elasticity']:
            for var, coef_dict in self.summary[models]["coefficients"].items():
                importance[models][var] = np.mean(np.abs(list(coef_dict.values())))

        # Plot importance
        importance_df = pd.DataFrame({'Variable': list(importance[model].keys()), 'Importance': list(importance[model].values())})
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        color_importance = "#1f77b4"
        fig = go.Figure(go.Bar(
            x=list(importance[model].values()),
            y=list(importance[model].keys()),
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
