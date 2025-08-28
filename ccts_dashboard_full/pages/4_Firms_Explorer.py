
import streamlit as st, pandas as pd, plotly.express as px
from utils.data_loader import list_output_folder, load_csv_safe
st.set_page_config(page_title="Firms Explorer • CCTS", page_icon="🏭", layout="wide")
st.title("🏭 Firms Explorer")
path = st.session_state.get("__ccts_output_path__", "sample_output")
files = list_output_folder(path)
agents = load_csv_safe(files.get("agents"))
if not isinstance(agents, pd.DataFrame) or agents.empty:
    st.info("No agent_history.csv found."); st.stop()
firm_col = next((c for c in ["firm_id","firm","agent_id"] if c in agents.columns), None)
sector_col = next((c for c in ["sector"] if c in agents.columns), None)
with st.sidebar:
    st.header("Filters")
    sector = st.selectbox("Sector", sorted(agents[sector_col].dropna().unique()) if sector_col else [])
    firms = sorted(agents.loc[agents[sector_col]==sector, firm_col].unique()) if sector_col else sorted(agents[firm_col].unique())
    firm = st.selectbox("Firm", firms if firms else [])
if firm:
    df = agents[agents[firm_col]==firm].copy()
    st.subheader(f"Firm: {firm}")
    cols = st.columns(3)
    for label, possibles in [("Production","production units prod".split()),("Emissions","emissions total_emissions".split()),("Intensity","intensity emissions_intensity".split())]:
        series = next((c for c in possibles if c in df.columns), None)
        if series:
            idx = 0 if label=='Production' else 1 if label=='Emissions' else 2
            cols[idx].plotly_chart(px.line(df, y=series, title=label), use_container_width=True)
    cols2 = st.columns(2)
    for label, possibles in [("Cash (₹)","cash cash_on_hand balance".split()),("Credits Owned","credits credits_owned total_credits".split())]:
        series = next((c for c in possibles if c in df.columns), None)
        if series:
            cols2[0 if label.startswith('Cash') else 1].plotly_chart(px.area(df, y=series, title=label), use_container_width=True)
    st.markdown("#### Raw Records (last 200 rows)")
    st.dataframe(df.tail(200), use_container_width=True)
else:
    st.info("Pick a sector and firm in the sidebar.")
