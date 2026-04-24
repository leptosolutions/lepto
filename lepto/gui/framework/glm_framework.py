import numpy as np 
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GridSearchCV

from lepto.standard.model.penalty import create_graph_continuous, create_graph_categorical
from lepto.standard.model.linear_model import GLMDiff
from lepto.behaviour.model.linear_demand import GLMDemand

def create_offset(transformer, categories_list):
    
    """
    Create the default offset‑beta structure for the GLM framework.

    Each variable is assigned a vector of NaNs (one per non-reference category),
    indicating that all coefficients are to be optimized unless modified later
    by the user.

    Parameters
    ----------
    transformer : object
        Transformer object containing lists `transformer.var_num` and
        `transformer.var_cat` defining the order of variables after encoding.
        It must correspond to the order in `categories_list`.

    categories_list : list of array-like
        List of category arrays produced by the preprocessing pipeline.
        For each variable, the first category corresponds to the reference,
        so only `len(categories_list[i]) - 1` coefficients are created.

    Returns
    -------
    offset_betas : dict
        Dictionary mapping:
            variable_name → 1D ndarray of shape (n_categories - 1,)
        filled with NaN values to indicate no fixed offsets.
    """

    offset_betas = {}
    for i, var in enumerate(transformer.var_num + transformer.var_cat) :
        offset_betas[var] = np.full(len(categories_list[i])-1, np.nan)
    return offset_betas

def create_graph_geographical(geographical_data, threshold_connection=20):
    
    """
    Create a geographical adjacency (penalty) matrix based on latitude–longitude distances, with quantile-based sparsification and mean-centering.

    The function computes the full pairwise Euclidean distance matrix between all
    geographic points, inverts the distances (1/distance), and then sparsifies the matrix
    by setting all values below the specified quantile to zero.
    The resulting matrix is then normalized so that its mean is 1 (if nonzero).

    Parameters
    ----------
    geographical_data : pandas.DataFrame
        DataFrame containing at least the columns:
            - var : Name of geographical variable (not used directly)
            - 'lat': latitude values
            - 'lon': longitude values
        Each row corresponds to a distinct geographic category.

    threshold_connection : int, default=20
        Number of top inverse distances to keep per row for sparsification.
        For example, 20 means only the top 20 inverse distances are kept nonzero.

    Returns
    -------
    graph : ndarray of shape (n, n)
        Symmetric matrix of inverse pairwise distances between all geographic points,
        sparsified below the given quantile and mean-centered to 1.
    """
    dists = pairwise_distances(np.column_stack([geographical_data['lat'].to_numpy(), geographical_data['lon'].to_numpy()]), metric='euclidean')
    # Avoid division by zero: set zeros to np.nan before inversion
    with np.errstate(divide='ignore', invalid='ignore'):
        inv_dists = np.where(dists > 0, 1.0 / dists, 0.0)
    # Sparsify: keep only the top N values per row, set the rest to zero
    inv_dists_sparse = np.zeros_like(inv_dists)
    threshold = np.minimum(threshold_connection, inv_dists.shape[0])
    for i in range(inv_dists.shape[0]):
        row = inv_dists[i]
        if np.count_nonzero(row) > threshold:
            # Get indices of the N largest values in the row
            idx = np.argpartition(row, -threshold)[-threshold:]
            inv_dists_sparse[i, idx] = row[idx]
        else:
            inv_dists_sparse[i] = row  # If less than threshold nonzeros, keep all
    inv_dists = inv_dists_sparse
    # Center so mean is 1 (avoid dividing by zero if all are zero)
    mean_val = inv_dists[inv_dists != 0].mean() if np.count_nonzero(inv_dists) > 0 else 1.0
    if mean_val != 0:
        inv_dists = inv_dists / mean_val
    # As float 32 to save memory, as these are just penalty weights and don't require high precision
    return inv_dists.astype(np.float32)

def create_graph_matrix(var_type, categories_var, geographical_data=None):
    
    """
    Build adjacency (penalty) matrices for each variable depending on its type.

    Variable types and corresponding graph structures:
        - 'continuous'   → linear chain graph (first‑order difference)
        - 'categorical'  → fully connected graph (all categories adjacent)
        - 'geographical' → distance‑based graph from geographical coordinates

    Parameters
    ----------
    var_type : dict
        Mapping:
            variable_name → {'continuous', 'categorical', 'geographical'}

    categories_var : dict
        Mapping:
            variable_name → list/array of categories for that variable
        Used to determine the size of the adjacency matrix.

    geographical_data : dict of pandas.DataFrame, optional
        Mapping:
            variable_name → DataFrame with columns ['lat', 'lon']
        Required only for variables of type 'geographical'.

    Returns
    -------
    adj_dict : dict
        Dictionary mapping:
            variable_name → adjacency matrix (ndarray)
        Each matrix encodes the structure needed for penalized GLM smoothing.
    """

    adj_dict = {}
    for var in var_type.keys() :
        if var_type[var] == 'continuous':
            adj_dict[var] = create_graph_continuous(len(categories_var[var]))
        elif var_type[var] == 'categorical':
            adj_dict[var] = create_graph_categorical(len(categories_var[var]))
        elif var_type[var] == 'geographical':
            adj_dict[var] = create_graph_geographical(geographical_data[var])
    return adj_dict


def create_categories(transformer, var_type, geographical_data):
    
    """
    Extract and adjust encoded category lists for each variable.

    Numeric and categorical variables use the categories produced by the pipeline's
    OneHotEncoder. Geographical variables override these categories with the list
    of unique geographic identifiers supplied by the user.

    Parameters
    ----------
    transformer : object
        Transformer object that contains:
            - transformer.transformer.named_steps['onehot'].categories_
            - transformer.var_num
            - transformer.var_cat
        The order of variables must match the order of category arrays.

    var_type : dict
        Mapping:
            variable_name → {'continuous', 'categorical', 'geographical'}

    geographical_data : dict of pandas.DataFrame
        Mapping:
            variable_name → DataFrame with a column `<var>` listing categories.
        Only used when `var_type[var] == 'geographical'`.

    Returns
    -------
    categories_list : list of array-like
        Updated list of categories, one array per variable, preserving order.
        For geographical variables, categories are replaced by the unique values
        in the geographic mapping.
    """

    categories_list = list(transformer._get_categories_var().values())
    for i, var in enumerate(transformer.var_num + transformer.var_cat):
        if var_type[var] == 'geographical':
            categories_list[i] = geographical_data[var][var].unique()
    return categories_list





class GLMFramework(BaseEstimator, RegressorMixin):
    
    """
    High‑level wrapper for graph‑penalized GLM estimation with lambda grid search.

    This class orchestrates:
    1. Construction of a `GLMDiff` model using:
        - an exponential-family distribution (Poisson, Gamma, Tweedie, Binomial),
        - an optional Tweedie power,
        - graph-based penalties per variable,
        - user‑provided offsets for partial or full coefficient fixing.
    2. Hyperparameter tuning of the regularization parameter `lam` using
        scikit‑learn's `GridSearchCV`.
    3. Storage, prediction, and re‑estimation (refit / rebase) of the selected GLM.

    Parameters
    ----------
    family : {'poisson', 'gamma', 'tweedie', 'binomial'}, default='poisson'
        Distribution family of the GLM, matching the exponential‑family assumption.

    tweedie_power : float, default=1.5
        Power parameter for the Tweedie variance function. Only used when
        `family='tweedie'`.

    lam_grid : array-like, default=np.linspace(1e-6, 10, 50)
        Sequence of regularization strengths over which to search.

    adj_matrix : dict or None
        Mapping:
            variable_name → adjacency matrix (ndarray)
        Each matrix defines the penalty graph for the variable. Required for fitting.

    offset_betas : dict or None
        Mapping:
            variable_name → 1d array-like of offsets
        Offsets allow fixing certain coefficients and optimizing others (NaNs).

    categories : 'auto' or dict, default='auto'
        Category structure used by `GLMDiff` for interpreting variable encodings.

    sparse_output : bool, default=False
        Whether to return sparse matrices for the transformed data.

    Attributes
    ----------
    grid_search : GridSearchCV
        Fitted grid search object after calling `fit()`.

    estimator : GLMDiff
        Best estimator selected by grid search (or updated by refit/rebase).

    is_fitted_ : bool
        Indicator that the model has been fitted.

    fit_intercept : bool
        Always True. Forwarded to `GLMDiff`.

    nbins : int
        Default number of bins used by `GLMDiff` for numeric smoothing.
    """

    def __init__(self,
                 family="poisson",
                 tweedie_power = 1.5,
                 lam_grid=np.linspace(1e-6, 10, 50),
                 adj_matrix=None,
                 offset_betas=None,
                 categories='auto',
                 sparse_output=False):
          
          self.fit_intercept = True
          self.nbins = 20
          self.family = family 
          self.tweedie_power = tweedie_power
          self.lam_grid = lam_grid
          self.adj_matrix = adj_matrix
          self.offset_betas = offset_betas
          self.categories = categories
          self.sparse_output = sparse_output

    def fit(self,
            X,
            y,
            sample_weight=None):
        
        """
        Fit the GLM framework using grid search over lambda values.

        Steps
        -----
        1. Aggregate offset vectors across variables and append an intercept offset.
        2. Build a penalty dictionary per variable using each adjacency matrix.
        3. Initialize a base `GLMDiff` estimator with:
            - distribution family,
            - Tweedie power (if applicable),
            - graph penalties,
            - offset coefficients,
            - category structure,
            - default lambda.
        4. Run `GridSearchCV` on parameter `lam` using `self.lam_grid`.
        5. Store the best estimator and mark the framework as fitted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix used by `GLMDiff`.

        y : array-like of shape (n_samples,)
            Target vector.

        sample_weight : array-like or None, default=None
            Optional per‑sample weights.

        Returns
        -------
        self : GLMFramework
            Fitted object.
        """
        # init GLM
        full_offset, penalty = GLMFramework.prepare_inputs(self.offset_betas, self.adj_matrix)
        # GLM
        glm = GLMDiff(
                family=self.family ,      
                tweedie_power=self.tweedie_power,      
                fit_intercept=True,
                nbins=self.nbins,               
                penalty_choice=penalty,
                offset_betas=full_offset,
                categories = self.categories,
                sparse_output=self.sparse_output
            )
        ## Grid serach CV --> best
        parameters = {'lam': self.lam_grid}
        self.grid_search = GridSearchCV(glm, parameters, refit=True)
        self.grid_search.fit(X=X,
            y=y,
            sample_weight=sample_weight)
        
        # Store model
        self.estimator = self.grid_search.best_estimator_
        self.is_fitted_ = True

    @staticmethod
    def prepare_inputs(offset_betas, adj_matrix):
        ## Offset
        full_offset = np.concatenate(list(offset_betas.values()))
        full_offset = np.append(full_offset, np.nan)
        ## Penalty
        penalty = {}
        for var in adj_matrix:
            penalty[var] = {'penalty':'graph', 'graph':adj_matrix[var]}

        return full_offset, penalty

    def predict(self, X):
        
        """
        Predict target values using the best estimator selected by grid search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input design matrix.

        Returns
        -------
        y_pred : ndarray
            Predicted values.

        Notes
        -----
        This method forwards directly to `self.estimator.predict(X)`.
        """

        self.estimator.predict(X)

    def refit(self, X, y, sample_weight, lam, adj_matrix, offset_betas):
        
        """
        Replace the internal estimator with a new `GLMDiff` fitted only with
        user‑specified parameters (without running grid search).

        This is used in the interactive application when the user adjusts:
        - lambda value,
        - penalty graph multipliers,
        - offsets (e.g., manual coefficient setting).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix used by `GLMDiff`.

        y : array-like of shape (n_samples,)
            Target vector.

        sample_weight : array-like or None, default=None
            Optional per‑sample weights.

        lam : float
            New regularization strength.

        adj_matrix : dict or None
            Mapping:
                variable_name → adjacency matrix (ndarray)
            Each matrix defines the penalty graph for the variable. Required for fitting.

        offset_betas : dict or None
            Mapping:
                variable_name → 1d array-like of offsets
            Offsets allow fixing certain coefficients and optimizing others (NaNs).

        Returns
        -------
        None
        """
        # Update
        self.adj_matrix = adj_matrix
        self.offset_betas = offset_betas
        # init GLM
        full_offset, penalty = GLMFramework.prepare_inputs(offset_betas, adj_matrix)

        # GLM
        glm = GLMDiff(
                family=self.family ,      
                tweedie_power=self.tweedie_power,      
                fit_intercept=True,
                nbins=self.nbins,               
                lam=lam,
                penalty_choice=penalty,
                offset_betas=full_offset,
                categories = self.categories,
                sparse_output=self.sparse_output)
        glm.fit(X, y, sample_weight)
        self.estimator = glm

    def rebase(self, X, y, sample_weight, new_betas):
        
        """
        Rebase the model by resetting the intercept offset to NaN and rebuilding
        the internal estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix used by `GLMDiff`.

        y : array-like of shape (n_samples,)
            Target vector.
        sample_weight : array-like or None, default=None
            Optional per‑sample weights.
        new_betas : dict or None
            Mapping:
                variable_name → 1d array-like of offsets
            Offsets allow fixing certain coefficients and optimizing others (NaNs).

        Steps
        -----
        1. Retrieve current penalty choices.
        2. Extract the model's current coefficient vector (betas).
        3. Force the intercept offset to NaN.
        4. Rebuild the estimator with the same lambda and penalty structure.

        Returns
        -------
        None
        """
        # Update
        betas_model = self.estimator.pipeline.named_steps['model'].betas
        # init GLM
        full_offset, penalty = GLMFramework.prepare_inputs(new_betas, self.adj_matrix)
        new_betas_map = np.where(np.isnan(full_offset), betas_model, full_offset)

        # GLM
        penalty_choice = self.estimator.penalty_choice
        # Intercept to rebase
        new_betas_map[-1] = np.nan
        glm = GLMDiff(
                family=self.family ,      
                tweedie_power=self.tweedie_power,      
                fit_intercept=True,
                nbins=self.nbins,               
                lam=self.estimator.lam,
                penalty_choice=penalty_choice,
                offset_betas=new_betas_map,
                categories = self.categories,
                sparse_output=self.sparse_output
            )
        glm.fit(X, y, sample_weight)
        self.estimator = glm


class GLMFrameworkBehaviour(BaseEstimator, RegressorMixin):
    
    """
    High‑level wrapper for graph‑penalized GLM estimation with lambda grid search.

    This class orchestrates:
    1. Construction of a `GLMDiff` model using:
        - an exponential-family distribution (Binomial),
        - graph-based penalties per variable,
        - user‑provided offsets for partial or full coefficient fixing.
    2. Hyperparameter tuning of the regularization parameter `lam` using
        scikit‑learn's `GridSearchCV`.
    3. Storage, prediction, and re‑estimation (refit / rebase) of the selected GLM.

    Parameters
    ----------
    family : {'poisson', 'gamma', 'tweedie', 'binomial'}, default='poisson'
        Distribution family of the GLM, matching the exponential‑family assumption.

    tweedie_power : float, default=1.5
        Power parameter for the Tweedie variance function. Only used when
        `family='tweedie'`.

    lam_grid : array-like, default=np.linspace(1e-6, 10, 50)
        Sequence of regularization strengths over which to search.

    lam_behaviour : float
        Regularization strengths for behaviour.

    adj_matrix_static : dict or None
        Mapping:
            variable_name → adjacency matrix (ndarray)
        Each matrix defines the penalty graph for the variable. Required for fitting.

    adj_matrix_dynamic : dict or None
        Mapping:
            variable_name → adjacency matrix (ndarray)
        Each matrix defines the penalty graph for the variable. Required for fitting.

    offset_betas_static : dict or None
        Mapping:
            variable_name → 1d array-like of offsets
        Offsets allow fixing certain coefficients and optimizing others (NaNs).

    offset_betas_dynamic : dict or None
        Mapping:
            variable_name → 1d array-like of offsets
        Offsets allow fixing certain coefficients and optimizing others (NaNs).

    categories_static : 'auto' or dict, default='auto'
        Category structure used by `GLMDiff` for interpreting variable encodings.

    categories_dynamic : 'auto' or dict, default='auto'
        Category structure used by `GLMDiff` for interpreting variable encodings.

    sparse_output : bool, default=False
        Whether to return sparse matrices for the transformed data.

    Attributes
    ----------
    grid_search : GridSearchCV
        Fitted grid search object after calling `fit()`.

    estimator : GLMDiff
        Best estimator selected by grid search (or updated by refit/rebase).

    is_fitted_ : bool
        Indicator that the model has been fitted.

    fit_intercept : bool
        Always True. Forwarded to `GLMDiff`.

    nbins : int
        Default number of bins used by `GLMDiff` for numeric smoothing.
    """

    def __init__(self,
                 var_glm_static,
                 var_glm_elasticity,
                 var_behaviour,
                 monotone_shape='increasing',
                 lam_grid=np.linspace(1e-6, 10, 50),
                 lam_behaviour=0,
                 adj_matrix_static=None,
                 offset_betas_static=None,
                 adj_matrix_dynamic=None,
                 offset_betas_dynamic=None,
                 categories_static='auto',
                 categories_dynamic='auto',
                 sparse_output=False):
          
          self.fit_intercept = True
          self.var_glm_static = var_glm_static
          self.var_glm_elasticity = var_glm_elasticity
          self.var_behaviour = var_behaviour
          self.monotone_shape = monotone_shape
          self.nbins = 20
          self.lam_behaviour = lam_behaviour 
          self.lam_grid = lam_grid
          self.adj_matrix_static = adj_matrix_static
          self.offset_betas_static = offset_betas_static
          self.categories_static = categories_static
          self.adj_matrix_dynamic = adj_matrix_dynamic
          self.offset_betas_dynamic = offset_betas_dynamic
          self.categories_dynamic = categories_dynamic
          self.sparse_output = sparse_output
          self.grid_search = None

    def fit(self,
            X,
            y,
            sample_weight=None):
        
        """
        Fit the GLM framework using grid search over lambda values.

        Steps
        -----
        1. Aggregate offset vectors across variables and append an intercept offset.
        2. Build a penalty dictionary per variable using each adjacency matrix.
        3. Initialize a base `GLMDiff` estimator with:
            - graph penalties,
            - offset coefficients,
            - category structure,
            - default lambda.
        4. Run `GridSearchCV` on parameter `lam` using `self.lam_grid`.
        5. Store the best estimator and mark the framework as fitted.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix used by `GLMDiff`.

        y : array-like of shape (n_samples,)
            Target vector.

        sample_weight : array-like or None, default=None
            Optional per‑sample weights.

        Returns
        -------
        self : GLMFrameworkBehaviour
            Fitted object.
        """
        # init GLM
        full_offset_static, penalty_static = GLMFrameworkBehaviour.prepare_inputs(self.offset_betas_static, self.adj_matrix_static)
        full_offset_dynamic, penalty_dynamic = GLMFrameworkBehaviour.prepare_inputs(self.offset_betas_dynamic, self.adj_matrix_dynamic)
        # GLM
        glm = GLMDemand(
                 self.var_glm_static,
                 self.var_glm_elasticity,
                 self.var_behaviour,
                 direction=self.monotone_shape,
                 fit_intercept=self.fit_intercept,
                 nbins=self.nbins,
                 lam_behaviour = self.lam_behaviour,
                 static_penalty_choice=penalty_static,
                 elasticity_penalty_choice=penalty_dynamic,
                 static_offset_betas=full_offset_static,
                 elasticity_offset_betas=full_offset_dynamic,
                 categories_static=self.categories_static,
                 categories_elasticity=self.categories_dynamic,
                 sparse_output=self.sparse_output
            )
        ## Grid serach CV --> best
        parameters = {'lam': self.lam_grid}
        self.grid_search = GridSearchCV(glm, parameters, refit=True)
        self.grid_search.fit(X=X,
            y=y,
            sample_weight=sample_weight)
        
        # Store model
        self.estimator = self.grid_search.best_estimator_
        self.is_fitted_ = True

    @staticmethod
    def prepare_inputs(offset_betas, adj_matrix):
        ## Offset
        full_offset = np.concatenate(list(offset_betas.values()))
        full_offset = np.append(full_offset, np.nan)
        ## Penalty
        penalty = {}
        for var in adj_matrix:
            penalty[var] = {'penalty':'graph', 'graph':adj_matrix[var]}

        return full_offset, penalty

    def predict(self, X):
        
        """
        Predict target values using the best estimator selected by grid search.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input design matrix.

        Returns
        -------
        y_pred : ndarray
            Predicted values.

        Notes
        -----
        This method forwards directly to `self.estimator.predict(X)`.
        """

        self.estimator.predict(X)

    def refit(self, X, y, sample_weight, lam, lam_behaviour, adj_matrix_static, offset_betas_static, adj_matrix_dynamic, offset_betas_dynamic):
        
        """
        Replace the internal estimator with a new `GLMDiff` fitted only with
        user‑specified parameters (without running grid search).

        This is used in the interactive application when the user adjusts:
        - lambda value,
        - penalty graph multipliers,
        - offsets (e.g., manual coefficient setting).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix used by `GLMDiff`.

        y : array-like of shape (n_samples,)
            Target vector.

        sample_weight : array-like or None, default=None
            Optional per‑sample weights.

        lam : float
            New regularization strength.

        lam_behaviour : float
            Regularization strengths for behaviour.

        adj_matrix_static : dict or None
            Mapping:
                variable_name → adjacency matrix (ndarray)
            Each matrix defines the penalty graph for the variable. Required for fitting.

        offset_betas_static : dict or None
            Mapping:
                variable_name → 1d array-like of offsets
            Offsets allow fixing certain coefficients and optimizing others (NaNs).

        adj_matrix_dynamic : dict or None
            Mapping:
                variable_name → adjacency matrix (ndarray)
            Each matrix defines the penalty graph for the variable. Required for fitting.

        offset_betas_dynamic : dict or None
            Mapping:
                variable_name → 1d array-like of offsets
            Offsets allow fixing certain coefficients and optimizing others (NaNs).

        Returns
        -------
        None
        """
        # Update
        self.adj_matrix_static = adj_matrix_static
        self.offset_betas_static = offset_betas_static
        self.adj_matrix_dynamic = adj_matrix_dynamic
        self.offset_betas_dynamic = offset_betas_dynamic
        # init GLM
        full_offset_static, penalty_static = GLMFrameworkBehaviour.prepare_inputs(self.offset_betas_static, self.adj_matrix_static)
        full_offset_dynamic, penalty_dynamic = GLMFrameworkBehaviour.prepare_inputs(self.offset_betas_dynamic, self.adj_matrix_dynamic)

        # GLM
        glm = GLMDemand(
                 self.var_glm_static,
                 self.var_glm_elasticity,
                 self.var_behaviour,
                 direction=self.monotone_shape,
                 fit_intercept=self.fit_intercept,
                 nbins=self.nbins,
                 lam = lam,
                 lam_behaviour = lam_behaviour,
                 static_penalty_choice=penalty_static,
                 elasticity_penalty_choice=penalty_dynamic,
                 static_offset_betas=full_offset_static,
                 elasticity_offset_betas=full_offset_dynamic,
                 categories_static=self.categories_static,
                 categories_elasticity=self.categories_dynamic,
                 sparse_output=self.sparse_output
            )
        glm.fit(X, y, sample_weight)
        self.estimator = glm

    def rebase(self, X, y, sample_weight, new_betas_static, new_betas_dynamic):
        
        """
        Rebase the model by resetting the intercept offset to NaN and rebuilding
        the internal estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Design matrix used by `GLMDiff`.

        y : array-like of shape (n_samples,)
            Target vector.
        sample_weight : array-like or None, default=None
            Optional per‑sample weights.
        new_betas_static : dict or None
            Mapping:
                variable_name → 1d array-like of offsets
            Offsets allow fixing certain coefficients and optimizing others (NaNs).
        new_betas_dynamic : dict or None
            Mapping:
                variable_name → 1d array-like of offsets
            Offsets allow fixing certain coefficients and optimizing others (NaNs).

        Steps
        -----
        1. Retrieve current penalty choices.
        2. Extract the model's current coefficient vector (betas).
        3. Force the intercept offset to NaN.
        4. Rebuild the estimator with the same lambda and penalty structure.

        Returns
        -------
        None
        """
        # Update
        betas_model_static = self.estimator.pipeline.named_steps['model'].coef_glm1_
        betas_model_dynamic = self.estimator.pipeline.named_steps['model'].coef_glm2_
        # init GLM
        full_offset_static, penalty_static = GLMFrameworkBehaviour.prepare_inputs(new_betas_static, self.adj_matrix_static)
        full_offset_dynamic, penalty_dynamic = GLMFrameworkBehaviour.prepare_inputs(new_betas_dynamic, self.adj_matrix_dynamic)
        new_betas_map_static = np.where(np.isnan(full_offset_static), betas_model_static, full_offset_static)
        new_betas_map_dynamic = np.where(np.isnan(full_offset_dynamic), betas_model_dynamic, full_offset_dynamic)

        # GLM
        penalty_choice_static = self.estimator.static_penalty_choice
        penalty_choice_dynamic = self.estimator.dynamic_penalty_choice
        # Intercept to rebase
        new_betas_map_static[-1] = np.nan
        new_betas_map_dynamic[-1] = np.nan
        glm = GLMDemand(
                 self.estimator.var_glm_static,
                 self.estimator.var_glm_elasticity,
                 self.estimator.var_behaviour,
                 direction=self.estimator.monotone_shape,
                 fit_intercept=self.estimator.fit_intercept,
                 nbins=self.nbins,
                 lam = self.estimator.lam,
                 lam_behaviour = self.estimator.lam_behaviour,
                 static_penalty_choice=penalty_choice_static,
                 elasticity_penalty_choice=penalty_choice_dynamic,
                 static_offset_betas=new_betas_map_static,
                 elasticity_offset_betas=new_betas_map_dynamic,
                 categories_static=self.estimator.categories_static,
                 categories_elasticity=self.estimator.categories_dynamic,
                 sparse_output=self.sparse_output
            )
        glm.fit(X, y, sample_weight)
        self.estimator = glm
