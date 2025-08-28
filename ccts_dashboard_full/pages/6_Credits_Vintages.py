
import streamlit as st, pandas as pd, plotly.express as px
from utils.data_loader import list_output_folder, load_csv_safe
st.set_page_config(page_title="Credits & Vintages • CCTS", page_icon="🎟️", layout="wide")
st.title("🎟️ Credits & Vintages")
path = st.session_state.get("__ccts_output_path__", "sample_output")
files = list_output_folder(path)
agents = load_csv_safe(files.get("agents"))
if not isinstance(agents, pd.DataFrame) or agents.empty:
    st.info("No agent_history.csv found."); st.stop()
credits_col = next((c for c in ["credits","credits_owned","total_credits"] if c in agents.columns), None)
firm_col = next((c for c in ["firm_id","firm","agent_id"] if c in agents.columns), None)
if credits_col and firm_col:
    st.subheader("Credits Owned (last available by firm)")
    df = agents[[firm_col, credits_col]].copy(); last = df.groupby(firm_col).tail(1)
    st.plotly_chart(px.bar(last.sort_values(credits_col, ascending=False), x=firm_col, y=credits_col), use_container_width=True)
else:
    st.info("Credits columns not found; showing raw table.")
    st.dataframe(agents.tail(1000), use_container_width=True)
