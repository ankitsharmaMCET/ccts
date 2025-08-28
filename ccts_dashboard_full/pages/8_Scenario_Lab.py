
import streamlit as st, pandas as pd
from utils.data_loader import list_output_folder, load_csv_safe, compute_kpis
st.set_page_config(page_title="Scenario & What-If • CCTS", page_icon="🧪", layout="wide")
st.title("🧪 Scenario & What-If Lab")
col = st.columns(2)
with col[0]:
    st.subheader("Run A")
    path_a = st.text_input("Output folder A", value=st.session_state.get("__ccts_output_path__", "sample_output"), key="path_a")
with col[1]:
    st.subheader("Run B")
    path_b = st.text_input("Output folder B", value="sample_output_b", key="path_b")
files_a = list_output_folder(path_a); files_b = list_output_folder(path_b)
ts_a = load_csv_safe(files_a.get("timeseries")); an_a = load_csv_safe(files_a.get("annual"))
ts_b = load_csv_safe(files_b.get("timeseries")); an_b = load_csv_safe(files_b.get("annual"))
kpi_a = compute_kpis(ts_a, an_a); kpi_b = compute_kpis(ts_b, an_b)
kpis = ["carbon_price","volume","volatility","compliance_rate","emissions","abatement","penalties","trades"]
df = pd.DataFrame({"KPI": kpis, "Run A": [kpi_a.get(k) for k in kpis], "Run B": [kpi_b.get(k) for k in kpis]})
st.dataframe(df, use_container_width=True)
