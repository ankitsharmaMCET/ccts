
import streamlit as st, pandas as pd, plotly.express as px
from utils.data_loader import list_output_folder, load_csv_safe
st.set_page_config(page_title="Abatement & Projects • CCTS", page_icon="🛠️", layout="wide")
st.title("🛠️ Abatement & Projects")
path = st.session_state.get("__ccts_output_path__", "sample_output")
files = list_output_folder(path)
agents = load_csv_safe(files.get("agents"))
if not isinstance(agents, pd.DataFrame) or agents.empty:
    st.info("No agent_history.csv found."); st.stop()
abatement_col = next((c for c in ["abatement","realized_abatement","abatement_realized"] if c in agents.columns), None)
firm_col = next((c for c in ["firm_id","firm","agent_id"] if c in agents.columns), None)
sector_col = next((c for c in ["sector"] if c in agents.columns), None)
if abatement_col and firm_col:
    st.subheader("Realized Abatement by Firm (last available)")
    df = agents[[firm_col, sector_col, abatement_col]].copy() if sector_col else agents[[firm_col, abatement_col]].copy()
    last = df.groupby(firm_col).tail(1)
    st.plotly_chart(px.bar(last.sort_values(abatement_col, ascending=False), x=firm_col, y=abatement_col, color=sector_col if sector_col else None), use_container_width=True)
else:
    st.info("Abatement column not found in agent history. Showing raw table.")
    st.dataframe(agents.tail(1000), use_container_width=True)
