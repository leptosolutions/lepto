import pandas as pd
from lepto.standard.model.transformers import GLMData
from lepto.behaviour.model.transformers import GLMDemandData
class DataPreprocessor():
    """ High level preprocessor for GLM.

    This class wraps the preprocessing pipeline for GLM and behaviour GLM models.
    It handles variable selection, fitting, and transformation using the appropriate
    transformer (GLMData or GLMDemandData), and supports both dense and sparse output.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing all variables.
    variables : list of str
        List of variable names to use for modeling.
    sparse_output : bool, default=False
        Whether to use sparse output in the underlying transformer.

    Attributes
    ----------
    variables : list of str
        List of variable names used for modeling.
    transformer_data : GLMData or GLMDemandData
        The fitted transformer instance after calling run().
    df : pandas.DataFrame
        The input dataframe.
    sparse_output : bool
        Whether sparse output is enabled.

    Methods
    -------
    run()
        Fit the transformer and return the processed DataFrame with mapped categories.
    """
    def __init__(self,
                 df,
                 variables,
                 sparse_output=False):
        
        self.variables = variables
        self.transformer_data = None
        self.df = df
        self.sparse_output = sparse_output
    def run(self):

        X_raw = self.df[self.variables]
        self.transformer_data = GLMData(sparse_output=self.sparse_output)
        self.transformer_data.fit(X_raw)
        X_new = pd.DataFrame(self.transformer_data.transformer.named_steps['col_transformer'].transform(X_raw),
                             columns=self.transformer_data.var_num+self.transformer_data.var_cat)
        for var in self.transformer_data.var_num:
            categories_var = self.transformer_data._get_categories_var()
            X_new[var] = X_new[var].map({i: categories_var[var][i] for i in range(len(categories_var[var]))})

        return X_new

