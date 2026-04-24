
import streamlit as st
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
from lepto.gui.framework.utils import lift_chart  
from lepto.gui.utils.save_ui import save_controls, save_json, save_json_df
from lepto.gui.utils.maps import folium_colored_points
from lepto.gui.utils.utils import is_symmetric, SHAPE_MAX_RENDER

# ------------------------ PAGE ------------------------
st.title("Part 3 — Model Review")

# ------------------------ INIT STATE ------------------------
if "framework" not in st.session_state:
    st.error("Run a model first.")
    st.stop()

# ------------------------SAVE ------------------------
with st.sidebar:
    save_controls()
    save_json()
    save_json_df()

# ------------------------ INIT  ------------------------
glm_framework = st.session_state.framework
if "best_glm" not in st.session_state:
    st.session_state.best_glm = glm_framework.estimator

# new_lambda
def _sync_new_lambda():
    st.session_state["new_lambda"] = st.session_state["_new_lambda"]
if "new_lambda" not in st.session_state:
    st.session_state.new_lambda = st.session_state.best_glm.lam
    
# new_lambda_behaviour
def _sync_new_lambda_behaviour():
    st.session_state["new_lambda_behaviour"] = st.session_state["_new_lambda_behaviour"]
if st.session_state.glm_type=="Behaviour":
    if "new_lambda_behaviour" not in st.session_state:
        st.session_state.new_lambda_behaviour = st.session_state.best_glm.lam_behaviour

tab1, tab2 = st.tabs(["Global View", "Variable View"])

# ------------------------ GLOBAL ------------------------
with tab1:
    # Compute pred
    pred_train = st.session_state.best_glm.predict(st.session_state.X_train)
    pred_test = st.session_state.best_glm.predict(st.session_state.X_test)

    # Scores
    st.subheader("Grid search scores")
    st.dataframe(pd.DataFrame(glm_framework.grid_search.cv_results_), width="stretch")
    st.subheader("Current GLM score")
    st.dataframe(pd.DataFrame(
        {"mean-y-train": np.mean(st.session_state.y_train),
         "mean-pred-train": np.mean(pred_train),
         "mean-y-test": np.mean(st.session_state.y_test),
         "mean-pred-test": np.mean(pred_test),
         "lambda": st.session_state.best_glm.lam,
         "score-train": st.session_state.best_glm.score(st.session_state.X_train, st.session_state.y_train, st.session_state.sample_weight_train),
         "score-test": st.session_state.best_glm.score(st.session_state.X_test, st.session_state.y_test, st.session_state.sample_weight_test)
         }, index=[0]),
         width="stretch")

    # Lift chart
    st.subheader("Current GLM Lift Chart - Train")
    fig = lift_chart(st.session_state.y_train, pred_train)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Current GLM Lift Chart - Test")
    fig = lift_chart(st.session_state.y_test, pred_test)
    st.plotly_chart(fig, width="stretch")

    # Importance plot
    if st.session_state.glm_type=="Standard":
        st.subheader("Current GLM Variable Importance")
        fig = st.session_state.best_glm.variable_importance()
        st.plotly_chart(fig, width="stretch")
    elif st.session_state.glm_type=="Behaviour":
        st.subheader("Current GLM Variable Importance Static part")
        fig = st.session_state.best_glm.variable_importance(model="static")
        st.plotly_chart(fig, width="stretch")
        st.subheader("Current GLM Variable Importance Elasticity part")
        fig = st.session_state.best_glm.variable_importance(model="elasticity")
        st.plotly_chart(fig, width="stretch")

    # Monotonicity on Behaviour
    if st.session_state.glm_type=="Behaviour":
        if st.session_state.best_glm.lam_behaviour>0:
            st.subheader("Monotonicity constraint")
            if len(st.session_state.best_glm.model.rows_non_monotone)>0:
                st.warning("""Your monotonicty constraint is not respected, please increase Lambda Behaviour !""", icon="⚠️")
            else:
                st.success("""Your monotonicity constraint is respected !""", icon="🔥")
    
# ------------------------ VARIABLES ------------------------
with tab2:
    # ------------------------ RERUN ------------------------
    st.subheader("Rerun")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Relaunch model"):
            # ---- Standard ------
            if st.session_state.glm_type=="Standard":
                glm_framework.refit(st.session_state.X_train, st.session_state.y_train, st.session_state.sample_weight_train,
                                    lam=st.session_state.new_lambda, adj_matrix=st.session_state.user_adj, offset_betas=st.session_state.user_offset)
            
            # ---- Behaviour ------
            elif st.session_state.glm_type=="Behaviour":
                glm_framework.refit(st.session_state.X_train, st.session_state.y_train, st.session_state.sample_weight_train,
                                    lam=st.session_state.new_lambda, 
                                    lam_behaviour = st.session_state.new_lambda_behaviour, 
                                    adj_matrix_static=st.session_state.user_adj_static, 
                                    adj_matrix_dynamic=st.session_state.user_adj_dynamic, 
                                    offset_betas_static=st.session_state.user_offset_static, 
                                    offset_betas_dynamic=st.session_state.user_offset_dynamic)
            
            st.session_state.framework = glm_framework
            st.session_state.best_glm = glm_framework.estimator
            st.rerun()
            st.success("Model updated!", icon="🔥")

    with col2:
        if st.button("Rebase model"):
            # ---- Standard ------
            if st.session_state.glm_type=="Standard":
                glm_framework.rebase(st.session_state.X_train, st.session_state.y_train, st.session_state.sample_weight_train,
                                    new_betas=st.session_state.user_offset)
            # ---- Behaviour ------
            elif st.session_state.glm_type=="Behaviour":
                glm_framework.rebase(st.session_state.X_train, st.session_state.y_train, st.session_state.sample_weight_train,
                                    new_betas_static=st.session_state.user_offset_static,
                                    new_betas_dynamic=st.session_state.user_offset_dynamic)

            st.session_state.framework = glm_framework
            st.session_state.best_glm = glm_framework.estimator
            st.rerun()
            st.success("Model rebalanced!", icon="🔥")

    # ------------------------ Viz ------------------------
    st.subheader("Variable Visualization")
    # ---- Standard ------
    if st.session_state.glm_type=="Standard":
        var = st.selectbox("Choose variable to visualize", st.session_state.var_for_model)
        # Plot classical graph
        figvar = st.session_state.best_glm.plot(st.session_state.X_test, st.session_state.y_test, st.session_state.sample_weight_test, var, pred_test)
        st.plotly_chart(figvar, width="stretch")

        # Plot map graph
        ## Merge geographical data with coef
        if st.session_state.variable_types[var] == "geographical":
            show = st.checkbox("Display variable on maps", key=f"show_map_{var}")
            if show:
                # Retreive coeff
                betas_select = pd.DataFrame(list(st.session_state.best_glm.summary['coefficients'][var].items()), columns=['modality', 'coefficient'])
                betas_select = pd.concat([betas_select, 
                                        pd.DataFrame({"modality": list(set(st.session_state.categories_var[var]) - set(betas_select['modality'].values))[0],  "coefficient": 0}, index=[betas_select.shape[0]])]).sort_values(['modality'])

                betas_select_lat_lon = betas_select.merge(
                    st.session_state.geographical_data_dict[var],
                    left_on=["modality"],
                    right_on=[var],
                    how="left",
                )
                betas_select_lat_lon = betas_select_lat_lon[(betas_select_lat_lon['lat'].notna()) & (betas_select_lat_lon['lon'].notna())]
                ## Display map
                st_folium(folium_colored_points(betas_select_lat_lon), width=700, height=500)
    
    # ---- Behaviour ------
    elif st.session_state.glm_type=="Behaviour":
        model_part = st.selectbox("Choose part of model", ["static", "elasticity"], index=0)
        var = st.selectbox("Choose variable to visualize", st.session_state.var_for_model_static if model_part=="static" else st.session_state.var_for_model_dynamic)
        # Plot classical graph
        figvar = st.session_state.best_glm.plot(st.session_state.X_test, st.session_state.y_test, st.session_state.sample_weight_test, var, model_part, pred_test)
        st.plotly_chart(figvar, width="stretch")

        # Plot map graph
        ## Merge geographical data with coef
        if st.session_state.variable_types[var] == "geographical":
            show = st.checkbox("Display variable on maps", key=f"show_map_{var}")
            if show:
                # Retreive coeff
                betas_select = pd.DataFrame(list(st.session_state.best_glm.summary['coefficients'][model_part][var].items()), columns=['modality', 'coefficient'])
                if model_part=="static":
                    betas_select = pd.concat([betas_select, 
                                            pd.DataFrame({"modality": list(set(st.session_state.categories_var_static[var]) - set(betas_select['modality'].values))[0],  "coefficient": 0}, index=[betas_select.shape[0]])]).sort_values(['modality'])
                elif model_part=="elasticity":
                    betas_select = pd.concat([betas_select, 
                                            pd.DataFrame({"modality": list(set(st.session_state.categories_var_dynamic[var]) - set(betas_select['modality'].values))[0],  "coefficient": 0}, index=[betas_select.shape[0]])]).sort_values(['modality'])
                
                betas_select_lat_lon = betas_select.merge(
                    st.session_state.geographical_data_dict[var],
                    left_on=["modality"],
                    right_on=[var],
                    how="left",
                )
                ## Display map
                st_folium(folium_colored_points(betas_select_lat_lon), width=700, height=500)

    # ------------------------ EDIT PARAM ------------------------
    st.subheader("Edit model parameters")
    # ---- Standard ------
    if st.session_state.glm_type=="Standard":
        # Lambda
        st.number_input(f"Lambda", value=st.session_state.best_glm.lam, step=None, format="%.5e", key='_new_lambda', on_change=_sync_new_lambda)

        # Offsets
        if st.session_state.user_adj[var].shape[0]<SHAPE_MAX_RENDER:
            st.subheader("Variable offset")
            edited_offset = st.data_editor(
                pd.DataFrame({"modality": st.session_state.categories_dict[var][1:], "offset": st.session_state.user_offset[var]}),
                width="stretch",
                hide_index=True,
                key=f"offset_editor_{var}"
            )
            st.session_state.user_offset[var] = edited_offset["offset"].values

            # Adjacency
            st.subheader("Variable adjacency matrices")
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
            st.info(f"Offset and adjacency matrix edition is not available for variables with more than {SHAPE_MAX_RENDER} modalities.")

    # ---- Behaviour ------
    elif st.session_state.glm_type=="Behaviour":
        # Lambda
        st.number_input(f"Lambda", value=st.session_state.best_glm.lam, step=None, format="%.5e", key='_new_lambda', on_change=_sync_new_lambda)
        st.number_input(f"Lambda Behaviour", value=st.session_state.best_glm.lam_behaviour, step=None, format="%.5e", key='_new_lambda_behaviour', on_change=_sync_new_lambda_behaviour)

        # Offsets
        st.subheader("Variable offset")
        if (model_part=="static") & (var in st.session_state.var_for_model_static):
            if st.session_state.user_adj[var].shape[0]<SHAPE_MAX_RENDER:
                edited_offset_static = st.data_editor(
                    pd.DataFrame({"modality": st.session_state.categories_dict_static[var][1:], "offset": st.session_state.user_offset_static[var]}),
                    width="stretch",
                    hide_index=True,
                    key=f"offset_editor_{var}"
                )
                st.session_state.user_offset_static[var] = edited_offset_static["offset"].values
            else:
                st.info(f"Offset and adjacency matrix editing is not available for variables with more than {SHAPE_MAX_RENDER} modalities to avoid performance issues.")
        elif (model_part=="elasticity") & (var in st.session_state.var_for_model_dynamic):
            if st.session_state.user_adj[var].shape[0]<SHAPE_MAX_RENDER:
                edited_offset_dynamic = st.data_editor(
                    pd.DataFrame({"modality": st.session_state.categories_dict_dynamic[var][1:], "offset": st.session_state.user_offset_dynamic[var]}),
                    width="stretch",
                    hide_index=True,
                    key=f"offset_editor_{var}"
                )
                st.session_state.user_offset_dynamic[var] = edited_offset_dynamic["offset"].values
            else:
                st.info(f"Offset and adjacency matrix editing is not available for variables with more than {SHAPE_MAX_RENDER} modalities to avoid performance issues.")

        # Adjacency
        st.subheader("Variable adjacency matrices")
        if (model_part=="static") & (var in st.session_state.var_for_model_static):
            if st.session_state.user_adj_static[var].shape[0]<SHAPE_MAX_RENDER:
                edited_adj_static = st.data_editor(
                    pd.DataFrame(st.session_state.user_adj_static[var], index=st.session_state.categories_dict_static[var], columns=st.session_state.categories_dict_static[var]),
                    width="stretch",
                    hide_index=False,
                    key=f"adj_editor_{var}"
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
                    key=f"adj_editor_{var}"
                )
                if is_symmetric(edited_adj_dynamic):
                    st.session_state.user_adj_dynamic[var] = edited_adj_dynamic.to_numpy()
                else:
                    st.error("Adjacency matrices has to be symetric.")

