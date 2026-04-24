from sklearn.base import BaseEstimator, TransformerMixin
from lepto.standard.model.transformers import GLMData

class GLMDemandData(BaseEstimator, TransformerMixin):
    """ GLM data prep

    Parameters
    ----------
    fit_intercept : bool
        Add intercept to X matrix

    nbins : int
        GenLasso cust numeric variable into bins\n
        Use a large number of bins in order to find the best splitting points

    categories_static : 'auto' or dict, default='auto'
        Category structure used by `GLMData` for interpreting variable encodings.

    categories_dynamic : 'auto' or dict, default='auto'
        Category structure used by `GLMData` for interpreting variable encodings.

    sparse_output : bool, default=False    
        Whether to return sparse matrices for the transformed data.
    """
    def __init__(self, 
                 var_glm_static,
                 var_glm_elasticity,
                 fit_intercept=True,
                 nbins=20,
                 categories_static="auto",
                 categories_elasticity="auto",
                 sparse_output=False):
        self.var_glm_static = var_glm_static
        self.var_glm_elasticity = var_glm_elasticity
        self.fit_intercept = fit_intercept
        self.nbins = nbins
        self.categories_static = categories_static
        self.categories_elasticity = categories_elasticity
        self.sparse_output = sparse_output
    
        self.static_transformer = GLMData(fit_intercept=self.fit_intercept,
                                          nbins=self.nbins,
                                          categories=self.categories_static,
                                          sparse_output=self.sparse_output)
        self.dynamic_transformer = GLMData(fit_intercept=self.fit_intercept,
                                           nbins=self.nbins,
                                           categories=self.categories_elasticity,
                                           sparse_output=self.sparse_output)

    def fit(self,
            X,
            price=None,
            y=None,
            static_penalty_choice=None,
            dynamic_penalty_choice=None):
        """

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]

        y : 1d array-like
            Ground truth (correct) labels

        price : 1d array-like
            Price

        static_penalty_choice : dict, optional
            Optional 
            Example: {'age': {penalty:'continous'}, 'region': {'penalty':'graph', 'graph':matrix}} where matrix.shape == (k, k).

        dynamic_penalty_choice : dict, optional
            Optional 
            Example: {'age': {penalty:'continous'}, 'region': {'penalty':'graph', 'graph':matrix}} where matrix.shape == (k, k).

        Return
        --------
        self
        """
        X_static = X[self.var_glm_static]
        X_dynamic = X[self.var_glm_elasticity]
        # Fit transformer
        self.static_transformer.fit(X_static, penalty_choice=static_penalty_choice)
        self.dynamic_transformer.fit(X_dynamic, penalty_choice=dynamic_penalty_choice)

        self.is_fitted_ = True
        
        return self

    def transform(self, X):
        """
        Tranform over X


        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]

        Return
        --------
        X_transform, X_transform, D, D: Matrix or Dataframe
            X_transform and matrix of constraint
        """
        # Transform X
        X_static_transform = self.static_transformer.transform(X)[0]
        X_dynamic_transform = self.dynamic_transformer.transform(X)[0]
        # D
        D_static_transform = self.static_transformer.transform(X)[1]
        D_dynamic_transform = self.dynamic_transformer.transform(X)[1]
        return X_static_transform, X_dynamic_transform, D_static_transform, D_dynamic_transform
