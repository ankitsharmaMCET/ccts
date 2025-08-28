import streamlit as st
import plotly.express as px
from utils.data_loader import list_output_folder, load_csv_safe, load_excel_sheets, compute_kpis

st.set_page_config(page_title="CCTS Dashboard", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")
st.title("🌿 CCTS Dashboard")
st.caption("Bottom-up Carbon Credit Trading Simulator — Full Dashboard")

# ---- Sidebar ----
with st.sidebar:
    st.header("Run Selector")
    path = st.text_input(
        "Output folder path",
        value="sample_output",
        help="Point to a folder containing the exported CSV/JSON/Excel from your simulation."
    )

    files = list_output_folder(path)
    if not files:
        st.warning("Folder not found. Update the path.")
    else:
        ok = [k for k, v in files.items() if v is not None]
        st.success(f"Found: {', '.join(ok) if ok else 'no known files'}")

    st.markdown("---")
    st.page_link("pages/0_Run_Model.py", label="Run Model", icon="🚀")
    st.page_link("pages/1_Overview.py", label="Overview", icon="🏠")
    st.page_link("pages/2_Market_Monitor.py", label="Market Monitor", icon="📈")
    st.page_link("pages/3_Compliance_Policy.py", label="Compliance & Policy", icon="✅")
    st.page_link("pages/4_Firms_Explorer.py", label="Firms Explorer", icon="🏭")
    st.page_link("pages/5_Abatement_Projects.py", label="Abatement & Projects", icon="🛠️")
    st.page_link("pages/6_Credits_Vintages.py", label="Credits & Vintages", icon="🎟️")
    st.page_link("pages/7_Transactions.py", label="Transactions & Order Flow", icon="🔁")
    st.page_link("pages/8_Scenario_Lab.py", label="Scenario & What-If", icon="🧪")
    st.page_link("pages/9_Exports.py", label="Exports", icon="📦")

# Make selected output path available to all pages
st.session_state["__ccts_output_path__"] = path

# ---- Landing KPI strip ----
timeseries = load_csv_safe(files.get("timeseries"))
annual = load_csv_safe(files.get("annual"))
kpis = compute_kpis(timeseries, annual)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Carbon Price (₹)", f"{kpis['carbon_price']:.2f}" if kpis['carbon_price'] is not None else "—")
c2.metric("Volume", f"{kpis['volume']:.2f}" if kpis['volume'] is not None else "—")
c3.metric("Volatility", f"{kpis['volatility']:.4f}" if kpis['volatility'] is not None else "—")
c4.metric("Compliance Rate", f"{kpis['compliance_rate']*100:.1f}%" if kpis['compliance_rate'] is not None else "—")
c5.metric("Emissions (tCO₂e)", f"{kpis['emissions']:.0f}" if kpis['emissions'] is not None else "—")
c6.metric("Penalties (₹)", f"{kpis['penalties']:.0f}" if kpis['penalties'] is not None else "—")

st.markdown("### Quick Trends")
if timeseries is not None and not timeseries.empty:
    price_col = next((c for c in ["carbon_price", "price", "mid"] if c in timeseries.columns), None)
    vol_col = next((c for c in ["volume", "market_volume", "traded_volume"] if c in timeseries.columns), None)
    cols = st.columns(2)
    if price_col:
        cols[0].plotly_chart(px.line(timeseries.tail(250), y=price_col, title="Carbon Price"), use_container_width=True)
    if vol_col:
        cols[1].plotly_chart(px.bar(timeseries.tail(250), y=vol_col, title="Market Volume"), use_container_width=True)
else:
    st.info("No timeseries found. Use the sample_output folder for a demo.")
