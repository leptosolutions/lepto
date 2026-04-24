import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer

from lepto.standard.model.penalty import categorical_matrix_graph, create_graph_continuous, create_graph_categorical


# def _detect_num_var(X):
#     return list(X.select_dtypes(include='number').columns.values)

# def _detect_cat_var(X):
#     return list(X.select_dtypes(exclude='number').columns.values)

NUMERIC_THRESHOLD = 20

def _detect_num_var(X):
    """
    Numeric variables with at least threshold unique values.
    """
    num_cols = X.select_dtypes(include="number")
    return [
        col
        for col in num_cols.columns
        if num_cols[col].nunique(dropna=True) >= NUMERIC_THRESHOLD
    ]


def _detect_cat_var(X):
    """
    Categorical variables:
    - non-numeric columns
    - numeric columns with fewer than threshold unique values
    """
    cat_cols = list(X.select_dtypes(exclude="number").columns)
    num_cols = X.select_dtypes(include="number")
    cat_from_num = [
        col
        for col in num_cols.columns
        if num_cols[col].nunique(dropna=True) < NUMERIC_THRESHOLD
    ]

    return cat_from_num + cat_cols


class AddIntercept(BaseEstimator, TransformerMixin):
    """ Add intercept to dataset

    Parameters
    -----------
    add_intercept : bool
        Add intercept to X matrix
    """

    def __init__(self, add_intercept=True):
        self.add_intercept = add_intercept

    def transform(self, X):
        """ Transform X

        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_features]

        Return
        --------
        X_new : array-like or sparse matrix, shape [n_samples, n_features+1]
        """
        if not self.add_intercept:
            return X
        n_samples = X.shape[0]
        if sp.issparse(X):
            intercept_col = sp.csr_matrix(np.ones((n_samples, 1), dtype=X.dtype))
            X_new = sp.hstack([X, intercept_col], format='csr')
        else:
            X_new = np.insert(X, X.shape[1], 1, axis=1)
        return X_new

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

class GLMData(BaseEstimator, TransformerMixin):
    
    """
    Prepares design matrix and penalty structure for Generalized Linear Models (GLMs)
    with optional structured penalties (continuous, categorical, graph-based).

    This transformer:
    - Handles preprocessing for numeric and categorical variables.
    - Applies binning for numeric features (for generalized lasso).
    - Encodes categorical variables using one-hot encoding.
    - Optionally adds an intercept column.
    - Constructs a block-diagonal penalty matrix `D` for structured regularization.

    Parameters
    ----------
    fit_intercept : bool, default=True
        Whether to add an intercept column to the transformed design matrix.
    nbins : int, default=20
        Number of bins for discretizing numeric variables.
        Use a large number of bins to allow flexible splitting.
    categories : 'auto' or list of lists, default='auto'
        Categories for one-hot encoding. Passed to `OneHotEncoder`.
    sparse_output : bool, default=False
        Whether to return sparse matrices for the transformed data.

    Attributes
    ----------
    transformer : sklearn.pipeline.Pipeline
        Full preprocessing pipeline combining imputation, binning, encoding, and intercept addition.
    var_num : list of str
        Names of numeric variables detected in input.
    var_cat : list of str
        Names of categorical variables detected in input.
    D : scipy.sparse.csr_matrix
        Global block-diagonal penalty matrix constructed after `fit`.
    D_list : list of ndarray
        Individual penalty matrices for each variable.
    D_list_size : list of int
        Sizes of each block in `D`.
    model_variables : list of str
        Combined list of all variables processed.
    mod_reference : str, default='first'
        Reference category handling for one-hot encoding.

    Notes
    -----
    - Compatible with scikit-learn pipelines.
    - Penalty matrix `D` is aligned with transformed feature space (including intercept if added).
    - Supports structured penalties for smoothness (continuous), pairwise differences (categorical),
    and graph-based constraints.

    Examples
    --------
    >>> glm_data = GLMData(fit_intercept=True, nbins=10)
    >>> X = pd.DataFrame({'age': [25, 40, 60], 'region': ['A', 'B', 'A']})
    >>> glm_data.fit(X)
    >>> X_transformed, D = glm_data.transform(X)
    >>> X_transformed.shape
    (3, glm_data.transformer[-1].n_features_in_)
    >>> D.shape
    (penalty_rows, X_transformed.shape[1])
    """

    def __init__(self, fit_intercept=True, nbins=20, categories="auto", sparse_output=False):
        self.fit_intercept = fit_intercept
        self.nbins = nbins
        self.model_variables = None
        self.mod_reference = 'first'
        self.categories = categories
        self.sparse_output = sparse_output
        self.D = None
        self.var_num = None
        self.var_cat = None
        self.D_list = None
        self.D_list_size = None
        self.sparse_threshold = 1 if self.sparse_output else 0

        # Transformation : numeric
        transform_data_num = Pipeline([('numimpute', SimpleImputer(missing_values=np.nan, strategy='mean')),
                                           ('numbin', KBinsDiscretizer(n_bins=self.nbins, encode='ordinal', strategy='quantile',
                                                                       quantile_method='averaged_inverted_cdf'))])
        # Tranformation : Categorical
        transform_data_car = Pipeline([('catimpute', SimpleImputer(missing_values=np.nan, strategy='most_frequent'))])

        col_transformer = ColumnTransformer([('trnum', transform_data_num, _detect_num_var),
                                             ('trcat', transform_data_car, _detect_cat_var)],
                                             sparse_threshold=self.sparse_threshold)

        add_inter = AddIntercept(add_intercept=self.fit_intercept)
        self.transformer = Pipeline([('col_transformer', col_transformer),
                                     ('onehot', OneHotEncoder(drop=self.mod_reference, dtype='uint8', handle_unknown='ignore', categories=self.categories, sparse_output=self.sparse_output)),
                                     ('add_intercept', add_inter)])

    def fit(self,
            X,
            y=None,
            penalty_choice=None):
        
        """
        Fit the preprocessing pipeline and construct the penalty matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.
        y : array-like of shape (n_samples,), optional
            Target variable (unused, included for compatibility).
        penalty_choice : dict, optional
            Dictionary specifying penalty type per variable:
            Example:
            {
                'age': {'penalty': 'continuous'},
                'region': {'penalty': 'graph', 'graph': adjacency_matrix}
            }
            where adjacency_matrix.shape == (k, k) for k categories.

        Returns
        -------
        self : GLMData
            Fitted transformer instance.

        Raises
        ------
        ValueError
            If penalty type is unknown.

        Notes
        -----
        - Detects numeric and categorical variables automatically.
        - Builds block-diagonal penalty matrix `D` aligned with transformed features.
        """
        # Fit transformer
        self.var_num = _detect_num_var(X)
        self.var_cat = _detect_cat_var(X)
        self.transformer.fit(X)

        # Create D
        self.D = self._build_D(penalty_choice=penalty_choice)
        
        self.is_fitted_ = True
        
        return self


    def transform(self, X):
        
        """
        Transform input data into GLM-ready design matrix and return penalty matrix.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input feature matrix.

        Returns
        -------
        X_transformed : ndarray of shape (n_samples, n_encoded_features)
            Preprocessed design matrix including one-hot encoding and optional intercept.
        D : scipy.sparse.csr_matrix
            Block-diagonal penalty matrix aligned with `X_transformed`.

        Examples
        --------
        >>> X_transformed, D = glm_data.transform(X)
        >>> X_transformed.shape
        (n_samples, n_encoded_features)
        >>> D.shape
        (penalty_rows, n_encoded_features)
        """
        # Transform X
        X_transform = self.transformer.transform(X)
        return X_transform, self.D

    def _build_D(self, penalty_choice):
        
        """
        Construct global block-diagonal penalty matrix from per-variable penalties.

        Parameters
        ----------
        penalty_choice : dict, optional
            Mapping of variable names to penalty specifications:
            - 'continuous' : first-order difference penalty.
            - 'categorical' : pairwise difference penalty.
            - 'graph' : graph-guided penalty using adjacency matrix.

        Returns
        -------
        D : scipy.sparse.csr_matrix
            Block-diagonal penalty matrix across all variables.
            Includes an extra column for intercept if `fit_intercept=True`.

        Raises
        ------
        ValueError
            If penalty type is unknown.

        Notes
        -----
        - If no penalty specified for a variable, a zero matrix is used for that block
        """

        # Init list
        D_list = list()
        penalty_choice = penalty_choice or {}

        # Get categories for size
        var_cats = self._get_categories_var()

        # Create Ds
        for var in (self.var_num + self.var_cat): # be careful depend on Pipeline
            # Create raw D
            if var in penalty_choice:
                if penalty_choice[var]['penalty']=='graph':
                    graph = penalty_choice[var]['graph']
                elif penalty_choice[var]['penalty']=='continuous':
                    graph = create_graph_continuous(len(var_cats[var]))
                elif penalty_choice[var]['penalty']=='categorical':
                    graph = create_graph_categorical(len(var_cats[var]))
                else: 
                    raise ValueError("Unknown penalty type for {}.".format(var))
                D_temp = categorical_matrix_graph(len(var_cats[var]), adj_matrix=graph)
            else:
                graph = create_graph_continuous(len(var_cats[var]))
                D_temp = categorical_matrix_graph(len(var_cats[var]), adj_matrix=graph) * 0
            D_list.append(D_temp)


        # Concat D
        D = sp.block_diag(D_list, format='csr')
        # Add column for intercept
        if self.fit_intercept:
            D = sp.hstack((D, np.zeros((D.shape[0], 1))), format='csr')

        # To dense if not sparse output
        if not self.sparse_output and sp.issparse(D):
            return np.asarray(D.todense()).astype(np.float32)

        return D.astype(np.float32)
    

    def _get_categories_var(self):
        """
        Retrieve category labels for each variable after preprocessing.

        Returns
        -------
        categories_dict : dict
            Mapping of variable names to their category labels:
            - Numeric variables: interval strings for bins.
            - Categorical variables: list of category names.

        Notes
        -----
        - Numeric intervals are derived from `KBinsDiscretizer.bin_edges_`.
        - Categorical categories are obtained from `OneHotEncoder.categories_`.

        Examples
        --------
        >>> glm_data._get_categories_var()
        {'age': ['(0, 10]', '(10, 20]', ...], 'region': ['A', 'B', 'C']}
        """
        # --- Numeric block ---
        num_bin = self.transformer.named_steps['col_transformer']['trnum'].named_steps['numbin']
        num_bin_labels = {}
        if num_bin is not None and hasattr(num_bin, "bin_edges_"): 
            for col_name, edges in zip(self.var_num, num_bin.bin_edges_):
                # Build interval labels like "(-inf, a]", "(a, b]", ..., "(z, +inf)"
                # KBinsDiscretizer by default creates closed intervals depending on strategy;
                # here we craft readable interval strings using edges.
                intervals = []
                for i in range(len(edges) - 1):
                    left = edges[i]
                    right = edges[i+1]
                    # Interval representation; adjust formatting as needed
                    intervals.append(f"({left:.6g}, {right:.6g}]")
                num_bin_labels[col_name] = np.array(intervals, dtype=object)

        # --- Categorical block ---
        cat_encoder =  self.transformer.named_steps['onehot']
        if cat_encoder is not None and hasattr(cat_encoder, "categories_"): 
            cat_categories_dict = {var: cats.tolist() for var, cats in zip(self.var_cat, cat_encoder.categories_[len(self.var_num):])}
        else:
            cat_categories_dict = {}
        # --- Merge ---
        all_categories_dict = {**num_bin_labels, **cat_categories_dict}
        return all_categories_dict
