
import logging
import streamlit as st
import numpy as np
import pandas as pd
from lepto.gui.framework.glm_framework import create_offset, create_graph_matrix, create_categories, GLMFramework, GLMFrameworkBehaviour
from lepto.gui.utils.utils import is_symmetric, SHAPE_MAX_RENDER
from lepto.standard.model.linear_model import analyse_var
from lepto.standard.model.transformers import _detect_cat_var
from lepto.gui.utils.save_ui import save_controls

# LOGGER
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------ PAGE ------------------------
st.title("Part 2 — Model Setup")

# ------------------------SAVE ------------------------
with st.sidebar:
    save_controls()

# ------------------------ INIT STATE ------------------------
if "preproc" not in st.session_state:
    st.error("Please complete Part 1 first.")
    st.stop()

# Default values
transformer_data = st.session_state.preproc.transformer_data
variable_types = st.session_state.variable_types
## Create categories list for dataprep
categories_list = create_categories(transformer_data, variable_types, st.session_state.geographical_data_dict)
## Transform categories list to dict
st.session_state.categories_dict = dict(zip(transformer_data.var_num + transformer_data.var_cat, categories_list))
## Compute default offset all nan
default_offset = create_offset(transformer_data, categories_list)
## Compute default adj matrix depend on variable types
st.session_state.categories_var = transformer_data._get_categories_var()
adj = create_graph_matrix(st.session_state.variable_types, st.session_state.categories_var, st.session_state.geographical_data_dict)

# ---- Standard ------
if st.session_state.glm_type=="Standard":
    if "user_offset" not in st.session_state:
        # Store a per-variable numpy array (copy to avoid mutating defaults)
        st.session_state.user_offset = {
            v: np.array(default_offset[v], copy=True)
            for v in st.session_state.var_for_model
        }

    if "user_adj" not in st.session_state:
        # Store a per-variable matrix (copy to avoid mutating defaults)
        st.session_state.user_adj = {
            v: np.array(adj[v], copy=True)
            for v in st.session_state.var_for_model
        }
# ---- Behaviour ------
elif st.session_state.glm_type=="Behaviour":
    # The order of variables or preserved via GLMDemand
    st.session_state.categories_dict_static = {k: st.session_state.categories_dict[k] for k in _detect_cat_var(st.session_state.X[st.session_state.var_for_model_static]) if k in  st.session_state.categories_dict} 
    st.session_state.categories_dict_dynamic = {k: st.session_state.categories_dict[k] for k in _detect_cat_var(st.session_state.X[st.session_state.var_for_model_dynamic]) if k in  st.session_state.categories_dict} 
    categories_list_static = [values for values in st.session_state.categories_dict_static.values()]
    categories_list_dynamic = [values for values in st.session_state.categories_dict_dynamic.values()]
    default_offset_static = {k: default_offset[k] for k in st.session_state.var_for_model_static if k in  default_offset}
    default_offset_dynamic = {k: default_offset[k] for k in st.session_state.var_for_model_dynamic if k in  default_offset}
    adj_static = {k: adj[k] for k in st.session_state.var_for_model_static if k in  adj}
    adj_dynamic = {k: adj[k] for k in st.session_state.var_for_model_dynamic if k in  adj}


    if "user_offset_static" not in st.session_state:
        # Store a per-variable numpy array (copy to avoid mutating defaults)
        st.session_state.user_offset_static = {
            v: np.array(default_offset_static[v], copy=True)
            for v in st.session_state.var_for_model_static
        }

    if "user_offset_dynamic" not in st.session_state:
        # Store a per-variable numpy array (copy to avoid mutating defaults)
        st.session_state.user_offset_dynamic = {
            v: np.array(default_offset_dynamic[v], copy=True)
            for v in st.session_state.var_for_model_dynamic
        }

    if "user_adj_static" not in st.session_state:
        # Store a per-variable matrix (copy to avoid mutating defaults)
        st.session_state.user_adj_static = {
            v: np.array(adj_static[v], copy=True)
            for v in st.session_state.var_for_model_static
        }

    if "user_adj_dynamic" not in st.session_state:
        # Store a per-variable matrix (copy to avoid mutating defaults)
        st.session_state.user_adj_dynamic = {
            v: np.array(adj_dynamic[v], copy=True)
            for v in st.session_state.var_for_model_dynamic
        }

# Lambda min
def _sync_lambda_min():
    st.session_state["lambda_min"] = st.session_state["_lambda_min"]
if "lambda_min" not in st.session_state:
    st.session_state.lambda_min = 0.01
# Lambda max
def _sync_lambda_max():
    st.session_state["lambda_max"] = st.session_state["_lambda_max"]
if "lambda_max" not in st.session_state:
    st.session_state.lambda_max = 0.01
# N lambda
def _sync_n_lambdas():
    st.session_state["n_lambdas"] = st.session_state["_n_lambdas"]
if "n_lambdas" not in st.session_state:
    st.session_state.n_lambdas = 2
# Lambda behaviour
def _sync_lambda_behaviour():
    st.session_state["lambda_behaviour"] = st.session_state["_lambda_behaviour"]
if "lambda_behaviour" not in st.session_state:
    st.session_state.lambda_behaviour = 0.0
# Monotone shape
def _sync_monotone_shape():
    st.session_state["monotone_shape"] = st.session_state["_monotone_shape"]
if "monotone_shape" not in st.session_state:
    st.session_state.monotone_shape = 'increasing'
# Family
def _sync_family():
    st.session_state["family"] = st.session_state["_family"]
if "family" not in st.session_state:
    st.session_state.family = 'gaussian'
# Tweedie power
def _sync_tweedie_power():
    st.session_state["tweedie_power"] = st.session_state["_tweedie_power"]
if "tweedie_power" not in st.session_state:
    st.session_state.tweedie_power = 1.5
# divide target weight
def _sync_divide_target_weight():
    st.session_state["divide_target_weight"] = st.session_state["_divide_target_weight"]
if "divide_target_weight" not in st.session_state:
    st.session_state.divide_target_weight = True

# ------------------------ PARAMETER SETUP ------------------------
st.subheader("Model Parameters")

# Input parameters
## Penalisation
lambda_min = st.number_input("Lambda min", min_value=0.0, value=st.session_state['lambda_min'], step=None, format="%.5e", key='_lambda_min', on_change=_sync_lambda_min)
lambda_max = st.number_input("Lambda max", min_value=0.0, value=st.session_state['lambda_max'], step=None, format="%.5e", key='_lambda_max', on_change=_sync_lambda_max)
n_lambdas = st.number_input("Number of lambdas", min_value=1, value=st.session_state['n_lambdas'], key='_n_lambdas', on_change=_sync_n_lambdas)
if st.session_state.glm_type=="Behaviour":
    lambda_behaviour = st.number_input("Lambda behaviour", min_value=0.0, value=st.session_state['lambda_behaviour'], step=None, format="%.5e", key='_lambda_behaviour', on_change=_sync_lambda_behaviour)
    monotone_shape = st.selectbox("Monotonicity shape", 
                                  ["increasing", "decreasing"],
                                  index=["increasing", "decreasing"].index(st.session_state["monotone_shape"]), 
                                  key='_monotone_shape', on_change=_sync_monotone_shape)
## Family
if st.session_state.glm_type=="Standard":
    family = st.selectbox("Distribution family", 
                         ["gaussian", "poisson", "gamma", "tweedie", "binomial"],
                          index=["gaussian", "poisson", "gamma", "tweedie", "binomial"].index(st.session_state["family"]), 
                          key='_family', on_change=_sync_family)
elif st.session_state.glm_type=="Behaviour":
    family = "binomial"
if family=="tweedie":
    tweedie_power = st.number_input("Tweedie power", min_value=1.0, max_value=2.0, value=st.session_state['tweedie_power'], key='_tweedie_power', on_change=_sync_tweedie_power)
else:
    tweedie_power = 1.5
## Weight handling
divide_target_weight = st.checkbox("Divide target by weight during training?", value=st.session_state['divide_target_weight'], key='_divide_target_weight', on_change=_sync_divide_target_weight)

# ------------------------ VARIABLE DETAIL VISUALIZATION ------------------------
st.subheader("Variable Details")

var = st.selectbox("Choose variable to visualize", st.session_state.var_for_model)
fig = analyse_var(variable=st.session_state.X[var],
                y=st.session_state.y,
                sample_weights=st.session_state.sample_weight,
                preds=None,
                coef=None,
                var_name=var)
st.plotly_chart(fig, width="stretch")

# ---- Standard ------
if st.session_state.glm_type=="Standard":
    if st.session_state.user_adj[var].shape[0]<SHAPE_MAX_RENDER:
        st.subheader("Variable offset")
        st.write("""Define a custom offset for your variable. Fix some coefficients and let others be optimized (np.nan)""")
        # Offsets
        edited_offset = st.data_editor(
            pd.DataFrame({"modality": st.session_state.categories_dict[var][1:], "offset": st.session_state.user_offset[var]}),
            width="stretch",
            hide_index=True,
            key=f"offset_editor_{var}"
        )
        st.session_state.user_offset[var] = edited_offset["offset"].values

        st.subheader("Variable adjacency matrices")
        st.write("""Define a custom adjacency matrices for your variable (e.g., neighboring regions more similar; diagonal 0, off-diagonals encode adjacency weights).""")
        # Adjacency matrices
        edited_adj = st.data_editor(
            pd.DataFrame(st.session_state.user_adj[var], index=st.session_state.categories_dict[var], columns=st.session_state.categories_dict[var]),
            width="stretch",
            hide_index=False,
            key=f"adj_editor_{var}"
        )
        if is_symmetric(edited_adj):
            st.session_state.user_adj[var] = edited_adj.to_numpy()
        else:
            st.error("Adjacency matrices has to be symetric.")
    else:
        st.info(f"Offset and adjacency matrix editing is not available for variables with more than {SHAPE_MAX_RENDER} modalities to avoid performance issues.")

# ---- Behaviour ------
elif st.session_state.glm_type=="Behaviour":
    model_part = st.selectbox("Choose part of model", ["static", "elasticity"], index=0)
    # Offsets
    st.subheader("Variable offset")
    st.write("""Define a custom offset for your variable. Fix some coefficients and let others be optimized (np.nan)""")
    if (model_part=="static") & (var in st.session_state.var_for_model_static):
        if st.session_state.user_adj_static[var].shape[0]<SHAPE_MAX_RENDER:
            edited_offset_static = st.data_editor(
                pd.DataFrame({"modality": st.session_state.categories_dict_static[var][1:], "offset": st.session_state.user_offset_static[var]}),
                width="stretch",
                hide_index=True,
                key=f"offset_editor_static_{var}"
            )
            st.session_state.user_offset_static[var] = edited_offset_static["offset"].values
        else:
            st.info(f"Offset and adjacency matrix editing is not available for variables with more than {SHAPE_MAX_RENDER} modalities to avoid performance issues.")

    elif (model_part=="elasticity") & (var in st.session_state.var_for_model_dynamic):
        if st.session_state.user_adj_dynamic[var].shape[0]<SHAPE_MAX_RENDER:
            edited_offset_dynamic = st.data_editor(
                pd.DataFrame({"modality": st.session_state.categories_dict_dynamic[var][1:], "offset": st.session_state.user_offset_dynamic[var]}),
                width="stretch",
                hide_index=True,
                key=f"offset_editor_dynamic_{var}"
            )
            st.session_state.user_offset_dynamic[var] = edited_offset_dynamic["offset"].values
        else:
            st.info(f"Offset and adjacency matrix editing is not available for variables with more than {SHAPE_MAX_RENDER} modalities to avoid performance issues.")

    # Adjacency matrices
    st.subheader("Variable adjacency matrices")
    st.write("""Define a custom adjacency matrices for your variable (e.g., neighboring regions more similar; diagonal 0, off-diagonals encode adjacency weights).""")
    if (model_part=="static") & (var in st.session_state.var_for_model_static):
        if st.session_state.user_adj_static[var].shape[0]<SHAPE_MAX_RENDER:
            edited_adj_static = st.data_editor(
                pd.DataFrame(st.session_state.user_adj_static[var], index=st.session_state.categories_dict_static[var], columns=st.session_state.categories_dict_static[var]),
                width="stretch",
                hide_index=False,
                key=f"adj_editor_static_{var}"
            )
            if is_symmetric(edited_adj_static):
                st.session_state.user_adj_static[var] = edited_adj_static.to_numpy()
            else:
                st.error("Adjacency matrices has to be symetric.")

    elif (model_part=="elasticity") & (var in st.session_state.var_for_model_dynamic):
        if st.session_state.user_adj_dynamic[var].shape[0]<SHAPE_MAX_RENDER:
            edited_adj_dynamic = st.data_editor(
                pd.DataFrame(st.session_state.user_adj_dynamic[var], index=st.session_state.categories_dict_dynamic[var], columns=st.session_state.categories_dict_dynamic[var]),
                width="stretch",
                hide_index=False,
                key=f"adj_editor_dynamic_{var}"
            )
            if is_symmetric(edited_adj_dynamic):
                st.session_state.user_adj_dynamic[var] = edited_adj_dynamic.to_numpy()
            else:
                st.error("Adjacency matrices has to be symetric.")

# logger.info(st.session_state.user_adj_static)
# logger.info(st.session_state.user_adj_dynamic)
# ------------------------ MODEL LAUNCH ------------------------
if st.button("Launch model"):
    if divide_target_weight:
        st.session_state.y = st.session_state.y / st.session_state.sample_weight
    # ---- Standard ------
    if st.session_state.glm_type=="Standard":
        glm = GLMFramework(family=family,
                            tweedie_power=tweedie_power,
                            lam_grid=np.linspace(lambda_min, lambda_max, n_lambdas),
                            adj_matrix=st.session_state.user_adj,
                            offset_betas=st.session_state.user_offset,
                            categories = categories_list,
                            sparse_output=st.session_state.sparse)
        glm.fit(st.session_state.X_train, st.session_state.y_train, st.session_state.sample_weight_train)
    # ---- Behaviour ------
    elif st.session_state.glm_type=="Behaviour":
        glm = GLMFrameworkBehaviour(
            st.session_state.var_for_model_static,
            st.session_state.var_for_model_dynamic,
            st.session_state.behaviour_var,
            monotone_shape=monotone_shape,
            lam_grid=np.linspace(lambda_min, lambda_max, n_lambdas),
            lam_behaviour=lambda_behaviour,
            adj_matrix_static=st.session_state.user_adj_static,
            offset_betas_static=st.session_state.user_offset_static,
            adj_matrix_dynamic=st.session_state.user_adj_dynamic,
            offset_betas_dynamic=st.session_state.user_offset_dynamic,
            categories_static=categories_list_static,
            categories_dynamic=categories_list_dynamic,
            sparse_output=st.session_state.sparse)
    
        glm.fit(st.session_state.X_train, st.session_state.y_train, st.session_state.sample_weight_train)

    st.session_state.framework = glm
    st.success("Model successfully trained!", icon="🔥")

