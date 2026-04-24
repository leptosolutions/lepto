
import streamlit as st
import pandas as pd
import pickle
import io
import json

def session_state_to_bytes():
    """Serialize the current session_state to bytes (pickle)."""
    buffer = io.BytesIO()
    pickle.dump(dict(st.session_state), buffer)
    buffer.seek(0)
    return buffer.getvalue()


def save_controls():
    """
    Display Save / Save As controls.
    """

    with st.container(border=True):
        st.markdown("### 💾 Save state")

        # --- SAVE (reuse last filename) ---
        new_name = st.text_input(
            "Save as",
            value="glm.pkl",
            label_visibility="collapsed",
        )
        data = session_state_to_bytes()
        st.download_button(
            label="Save",
            data=data,
            file_name=new_name,
            mime="application/octet-stream",
            use_container_width=True,
        )

def save_json():
    """
    Display Save As json.
    """
    with st.container(border=True):
        st.markdown("### 💾 Save model json")

        # --- SAVE AS ---
        new_name = st.text_input(
            "Save as",
            value="glm.json",
            label_visibility="collapsed",
        )

        json_str = json.dumps(st.session_state.framework.grid_search.best_estimator_.summary, indent=2, ensure_ascii=False)
        st.download_button(
                    label="Download model JSON",
                    data=json_str,
                    file_name=new_name,
                    mime="application/json",
                    use_container_width=True,
                )
        
def save_json_df():
    """
    Display Save As df.
    """
    with st.container(border=True):
        st.markdown("### 💾 Save model excel format")

        # --- SAVE AS ---
        new_name = st.text_input(
            "Save as",
            value="glm.xlsx",
            label_visibility="collapsed",
        )

        model_df = st.session_state.framework.grid_search.best_estimator_.compute_summary_df()
        if st.session_state.glm_type=="Standard":
            excel_bytes = dfs_to_excel_bytes(model_df, None)
        elif st.session_state.glm_type=="Behaviour":
            excel_bytes = dfs_to_excel_bytes(model_df['static'], model_df['elasticity'])
        st.download_button(
                    label="Download model excel",
                    data=excel_bytes,
                    file_name=new_name,
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )
            
def dfs_to_excel_bytes(df1: pd.DataFrame, df2: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    if df2 is None:
        with pd.ExcelWriter(output, engine=None) as writer:
            df1.to_excel(writer, index=False, sheet_name="coefficients", header=False)
    else:
        with pd.ExcelWriter(output, engine=None) as writer:
            df1.to_excel(writer, index=False, sheet_name="static", header=False)
            df2.to_excel(writer, index=False, sheet_name="elasticity", header=False)
    return output.getvalue()
