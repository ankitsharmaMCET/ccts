
import streamlit as st, pandas as pd, plotly.express as px
from utils.data_loader import list_output_folder, load_csv_safe
st.set_page_config(page_title="Compliance & Policy • CCTS", page_icon="✅", layout="wide")
st.title("✅ Compliance & Policy")
path = st.session_state.get("__ccts_output_path__", "sample_output")
files = list_output_folder(path)
agents = load_csv_safe(files.get("agents"))
annual = load_csv_safe(files.get("annual"))
if isinstance(annual, pd.DataFrame) and not annual.empty:
    st.subheader("Annual View")
    st.dataframe(annual, use_container_width=True)
else:
    st.caption("No annual summary CSV.")
st.markdown("---")
st.subheader("Firms: Compliance Gap / Intensity vs Target")
if isinstance(agents, pd.DataFrame) and not agents.empty:
    cols = agents.columns
    gap_col = next((c for c in ["compliance_gap","gap","annual_gap"] if c in cols), None)
    intensity_col = next((c for c in ["intensity","emissions_intensity"] if c in cols), None)
    target_col = next((c for c in ["target_intensity","year_target"] if c in cols), None)
    firm_col = next((c for c in ["firm_id","firm","agent_id"] if c in cols), None)
    if gap_col and firm_col:
        df = agents[[firm_col, gap_col]].copy().groupby(firm_col).tail(1)
        st.plotly_chart(px.bar(df.sort_values(gap_col), x=firm_col, y=gap_col, title="Compliance Gap (last step)"), use_container_width=True)
    elif intensity_col and target_col and firm_col:
        df = agents[[firm_col, intensity_col, target_col]].copy().groupby(firm_col).tail(1)
        df["delta"] = df[intensity_col] - df[target_col]
        st.plotly_chart(px.bar(df.sort_values("delta"), x=firm_col, y="delta", title="Intensity – Target (last step)"), use_container_width=True)
    else:
        st.info("Agent history lacks gap or intensity/target fields. Showing raw table.")
        st.dataframe(agents.tail(1000), use_container_width=True)
else:
    st.caption("No agent_history.csv found.")
