import pytest
import numpy as np
import pandas as pd
from importlib.resources import files

from lepto.gui.framework.glm_framework import create_offset, create_graph_matrix, create_categories, GLMFramework
from lepto.gui.framework.data_preprocessor import DataPreprocessor

def test_framework():
    path = files("lepto.data") / "sample_standard.csv"
    df = pd.read_csv(path)
    X = df[["age", "region", "segment"]]
    y = df["y"].values
    w = df["exposure"].values

    # Create fake data lon and lat for example
    data_lat_long = pd.DataFrame({
        "region": ["north", "south", "east", "west"],
        "lat": [48.8566, 45.7640, 43.2965, 50.6292],
        "lon": [2.3522, 4.8357, 5.3698, 3.0573]
    })
    # Launch Data preparation
    var_for_model = ["age", "region", "segment"]
    preproc = DataPreprocessor(
                df=df,
                variables=var_for_model,
                sparse_output=True)
    X= preproc.run()
    # Create default values
    transformer_data = preproc.transformer_data
    ## Variables types and categories
    geographical_data_dict = {}
    geographical_data_dict["region"] = data_lat_long
    variable_types = {"age": "continuous", "region": "geographical", "segment": "categorical"}
    categories_list = create_categories(transformer_data, variable_types, geographical_data_dict)
    categories_dict = dict(zip(transformer_data.var_num + transformer_data.var_cat, categories_list))
    ## Default offset
    default_offset = create_offset(transformer_data, categories_list)
    ## Default adjacendy matrix
    categories_var = transformer_data._get_categories_var()
    adj = create_graph_matrix(variable_types, categories_var, geographical_data_dict)

    # Test adjacency matrix for region variable
    expected = np.array([[0.        , 0.91481066, 0.5735622 , 1.9020193 ],
       [0.91481066, 0.        , 1.4372178 , 0.70046884],
       [0.5735622 , 1.4372178 , 0.        , 0.47192112],
       [1.9020193 , 0.70046884, 0.47192112, 0.        ]])
    assert np.allclose(adj['region'], expected)

    # Test default offset for region variable
    expected = np.array([np.nan, np.nan, np.nan])
    assert np.allclose(default_offset['region'], expected, equal_nan=True)

    # Test categories for region variable
    expected = np.array(['north', 'south', 'east', 'west'])
    assert np.all(categories_dict['region']==expected)
