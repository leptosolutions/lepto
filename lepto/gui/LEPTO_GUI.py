import os
import sys
import pickle
import runpy
import pandas as pd
import pyarrow.parquet as pq
import streamlit as st
from lepto.gui.utils.save_ui import save_controls

def main():
    st.set_page_config(page_title="GLM Builder", layout="wide")
    st.title("Get Started - Build your GLM model")

    # ------------------------SAVE ------------------------
    with st.sidebar:
        save_controls()

    # ------------------------ LOAD DATASET / LOAD PREVIOUS STATE ------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Upload file")
        uploaded_file = st.file_uploader(
            "Upload dataset (CSV or Parquet)",
            type=["csv", "parquet.gzip"],
            key="dataset_uploader",
        )

        if uploaded_file:
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_parquet(uploaded_file)

            st.session_state.df = df
            st.write("Dataset loaded:", df.shape)

            st.dataframe(df.head(10), hide_index=True, width="stretch")

    with col2:
        st.subheader("Or load previous session")
        state_file = st.file_uploader(
            "Load saved GLM state",
            type=["pkl"],
            key="state_uploader",
        )

        if state_file:
            loaded_state = pickle.load(state_file)
            for k, v in loaded_state.items():
                    st.session_state[k] = v
            st.success("State loaded successfully!")

    # ------------------------ NAVIGATE IN PAGES ------------------------
    st.write("""
    1. **Data preparation**  
    2. **Model setup**  
    3. **Model review**  
    """)


def run_app():
    """
    Console-script entry point.

    It re-invokes Streamlit as if you had typed:
        streamlit run <this_file>
    """
    script_path = os.path.abspath(__file__)
    sys.argv = ["streamlit", "run", script_path] + sys.argv[1:]
    runpy.run_module("streamlit", run_name="__main__")


if __name__ == "__main__":
    main()
