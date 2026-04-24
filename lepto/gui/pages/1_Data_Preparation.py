
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from lepto.gui.framework.utils import plot_distribution 
from lepto.gui.framework.data_preprocessor import DataPreprocessor
from lepto.gui.utils.save_ui import save_controls

st.title("Part 1 — Data Visualization & Target Definition")

# ------------------------SAVE ------------------------
with st.sidebar:
    save_controls()

# ------------------------Init ------------------------
if "df" not in st.session_state:
    st.error("Please upload a dataset.")
    st.stop()

# GLM type
def _sync_glm_type():
    st.session_state["glm_type"] = st.session_state["_glm_type"]
if "glm_type" not in st.session_state:
    st.session_state.glm_type = 'Standard'
# Target
def _sync_target():
    st.session_state["target"] = st.session_state["_target"]
if "target" not in st.session_state:
    st.session_state.target = st.session_state.df.columns[0]
# Weight
def _sync_weight():
    st.session_state["weight"] = st.session_state["_weight"]
if "weight" not in st.session_state:
    st.session_state.weight = "None"
# Behaviour
def _sync_behaviour_var():
    st.session_state["behaviour_var"] = st.session_state["_behaviour_var"]
if "behaviour_var" not in st.session_state:
    st.session_state.behaviour_var = st.session_state.df.columns[0]
# Var model
def _sync_var_for_model():
    st.session_state["var_for_model"] = st.session_state["_var_for_model"]
if "var_for_model" not in st.session_state:
    st.session_state.var_for_model = [st.session_state.df.columns[0]]
# Var static
def _sync_var_for_model_static():
    st.session_state["var_for_model_static"] = st.session_state["_var_for_model_static"]
if "var_for_model_static" not in st.session_state:
    st.session_state.var_for_model_static = [st.session_state.df.columns[0]]
# Var dynamic
def _sync_var_for_model_dynamic():
    st.session_state["var_for_model_dynamic"] = st.session_state["_var_for_model_dynamic"]
if "var_for_model_dynamic" not in st.session_state:
    st.session_state.var_for_model_dynamic = [st.session_state.df.columns[0]]
# Type
st.session_state.setdefault("variable_types", {})
def make_type_sync_cb(var: str, widget_key: str):
    def _cb():
        st.session_state["variable_types"][var] = st.session_state[widget_key]
    return _cb
# sparse output
def _sync_sparse():
    st.session_state["sparse"] = st.session_state["_sparse"]
if "sparse" not in st.session_state:
    st.session_state.sparse = False

# ------------------------ VARIABLE EXPLORATION ------------------------
st.subheader("Dataset explorer")
if "df" in st.session_state:
    df = st.session_state.df

    # ------------------------  EXPLORATION ------------------------
    selected_var = st.selectbox("Choose variable to visualize", df.columns)
    st.plotly_chart(plot_distribution(df[selected_var]), width="stretch")

    # ------------------------  DEFINE ------------------------
    # Type of GLM
    st.subheader("GLM type")
    st.session_state.glm_type = st.selectbox(
        "Type of GLM", ['Standard', 'Behaviour'],
        index=['Standard', 'Behaviour'].index(st.session_state["glm_type"]),
        key='_glm_type',
        on_change=_sync_glm_type)
    if st.session_state.glm_type=="Standard":
        st.write("""You are about to build a standard GLM, equation of predictor: """)
        # st.latex(r"""\hat{\beta}=\arg\min_{\beta}\left[-\sum_{i=1}^{n} \ell(y_i, x_i^\top \beta)+\lambda \lVert D\beta \rVert_2^2\right]""")
        st.latex(r"""\eta = g(X\beta)""")
    elif st.session_state.glm_type=="Behaviour":
        st.write("""You are about to build a behaviour GLM, equation of predictor: """)
        st.latex(r"""\eta = sigmoid(X\beta_1 + q X \beta_2 )""")
        # st.latex(r"""(\hat{\beta}_1,\hat{\beta}_2)=\arg\min_{\beta_1,\beta_2}\Bigg\{-\sum_{i=1}^n w_i\Big[y_i\log\sigma(z_i)+(1-y_i)\log(1-\sigma(z_i))\Big]+\lambda\Big(\|D_1\beta_1+d_1\|_2^2+\|D_2\beta_2+d_2\|_2^2\Big)\Bigg\}""")     
        # st.latex(r"""z_i = x_{1i}^\top\beta_1 + q_i\,(x_{2i}^\top\beta_2),\qquad\sigma(z)=\frac{1}{1+e^{-z}}""")

    
    # target
    st.subheader("Target Setup")
    st.session_state.target = st.selectbox("Target variable",
                                           df.columns,
                                           index=list(df.columns).index(st.session_state["target"]),
                                           key='_target',
                                           on_change=_sync_target)
    
    if df[st.session_state.target].isna().any():
        n_na = int(df[st.session_state.target].isna().sum())
        st.error(
            f"Target variable '{st.session_state.target}' contains {n_na} missing value(s). "
            "Please select a target without NA values."
        )
        st.stop()

    # Behaviour
    if st.session_state.glm_type=="Behaviour":
        st.subheader("Behaviour Setup")
        st.session_state.behaviour_var = st.selectbox("Behaviour variable",
                                                      df.columns,
                                                      index=list(df.columns).index(st.session_state["behaviour_var"]),
                                                      key='_behaviour_var',
                                                      on_change=_sync_behaviour_var)
        behaviour = df[st.session_state.behaviour_var].values
        st.session_state.behaviour_var = st.session_state.behaviour_var

    # Weight
    st.subheader("Weight Setup")
    OPTIONS_WEIGHT = ["None"] + list(df.columns)
    st.session_state.weight = st.selectbox("Weight variable (optional)", 
                                           OPTIONS_WEIGHT,
                                           index=OPTIONS_WEIGHT.index(st.session_state["weight"]),
                                           key='_weight',
                                           on_change=_sync_weight)
    if st.session_state.weight != "None":
        if df[st.session_state.weight].isna().any():
            n_na = int(df[st.session_state.weight].isna().sum())
            st.error(
                f"Weight variable '{st.session_state.target}' contains {n_na} missing value(s). "
                "Please select a weight variable without NA values."
            )
            st.stop()   

    # Define target and weight
    y = df[st.session_state.target].values
    if st.session_state.weight == "None":
        weight = None
        sample_weight =  np.ones(len(y))
    else: 
        sample_weight =  df[st.session_state.weight].values

    # variable selection
    st.subheader("Variables for the model")
    if st.session_state.glm_type=="Standard":
        st.session_state.var_for_model = st.multiselect("Select variables to include",
                                                        df.columns,
                                                        default=st.session_state["var_for_model"],
                                                        key='_var_for_model',
                                                        on_change=_sync_var_for_model)
    elif st.session_state.glm_type=="Behaviour":
        var_for_model_static = st.multiselect("Select variables to include in static model",
                                              df.columns,
                                              default=st.session_state["var_for_model_static"],
                                              key='_var_for_model_static',
                                              on_change=_sync_var_for_model_static)
        var_for_model_dynamic = st.multiselect("Select variables to include in elasticty model",
                                               df.columns,
                                               default=st.session_state["var_for_model_dynamic"],
                                               key='_var_for_model_dynamic',
                                               on_change=_sync_var_for_model_dynamic)
        st.session_state.var_for_model = list(set(var_for_model_static) | set(var_for_model_dynamic))
        st.session_state.var_for_model_static = var_for_model_static
        st.session_state.var_for_model_dynamic = var_for_model_dynamic

    # variable type selection
    st.subheader("Variable Types")
    variable_types = {}
    geographical_data_dict = {}
    for var in st.session_state.var_for_model:
        current_type = st.session_state["variable_types"].get(var, "continuous")
        variable_types[var] = st.selectbox(
            f"Type of '{var}'", ["continuous", "categorical", "geographical"],
            index=["continuous", "categorical", "geographical"].index(current_type),
            key=f"type_{var}",
            on_change=make_type_sync_cb(var, f"type_{var}")
        )

        if variable_types[var] == "geographical":
            geo_file = st.file_uploader(f"Upload geo mapping for {var}", type=["csv", "parquet.gzip"], key=f"geo_{var}")
            if geo_file:
                if geo_file.name.endswith(".csv"):
                    geo_df = pd.read_csv(geo_file)
                else:
                    geo_df = pd.read_parquet(geo_file)
                if (var not in geo_df.columns) | ('lat' not in geo_df.columns) | ('lon' not in geo_df.columns):
                    st.error(f"""Your geo_file must contain {var}, lat and lon""")
                geo_df = geo_df.sort_values(by=var)
                geographical_data_dict[var] = geo_df
                # Compute nb element not in geographical data
                element_not_in = list(set(df[var]) - set(geographical_data_dict[var][var]))
                if len(element_not_in)>0:
                    st.warning(f"""{len(element_not_in)} geograpical code from your dataset are not in your geo_file.""", icon="⚠️")

    # ------------------------ Other parameters ------------------------
    st.subheader("Other parameters")
    st.session_state.sparse = st.checkbox("Do you want to use sparse matrix (recommanded when using geographical data, but optimisation will be slower)?", value=st.session_state['sparse'], key='_sparse', on_change=_sync_sparse)

    # ------------------------ PREPROCESS BUTTON ------------------------
    if st.button("Launch data preprocessing"):
        # Process data
        preproc = DataPreprocessor(
            df=df,
            variables=st.session_state.var_for_model,
            sparse_output=st.session_state.sparse)
        X = preproc.run()

        if st.session_state.glm_type=="Behaviour":
            X[st.session_state.behaviour_var] = behaviour
 
        # Train, Test split
        X_train, X_test, y_train, y_test, sample_weight_train, sample_weight_test = train_test_split(X, y, sample_weight, test_size=0.2, random_state=42)
        
        st.session_state.X = X
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.preproc = preproc
        st.session_state.y = y
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.sample_weight = sample_weight
        st.session_state.sample_weight_train = sample_weight_train
        st.session_state.sample_weight_test = sample_weight_test
        st.session_state.variable_types = variable_types
        st.session_state.geographical_data_dict = geographical_data_dict
        st.success("Data preprocessing completed!", icon="🔥")
