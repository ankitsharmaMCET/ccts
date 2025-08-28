
import streamlit as st, plotly.express as px, pandas as pd
from utils.data_loader import list_output_folder, load_csv_safe, load_excel_sheets, compute_kpis
st.set_page_config(page_title="Overview • CCTS", page_icon="🏠", layout="wide")
st.title("🏠 Overview")
path = st.session_state.get("__ccts_output_path__", "sample_output")
files = list_output_folder(path)
timeseries = load_csv_safe(files.get("timeseries"))
annual = load_csv_safe(files.get("annual"))
kpis = compute_kpis(timeseries, annual)
c1,c2,c3,c4,c5,c6 = st.columns(6)
c1.metric("Carbon Price (₹)", f"{kpis['carbon_price']:.2f}" if kpis['carbon_price'] is not None else "—")
c2.metric("Volume", f"{kpis['volume']:.2f}" if kpis['volume'] is not None else "—")
c3.metric("Volatility", f"{kpis['volatility']:.4f}" if kpis['volatility'] is not None else "—")
c4.metric("Compliance Rate", f"{kpis['compliance_rate']*100:.1f}%" if kpis['compliance_rate'] is not None else "—")
c5.metric("Emissions (tCO₂e)", f"{kpis['emissions']:.0f}" if kpis['emissions'] is not None else "—")
c6.metric("Penalties (₹)", f"{kpis['penalties']:.0f}" if kpis['penalties'] is not None else "—")
st.markdown("### Trends")
if isinstance(timeseries, pd.DataFrame) and not timeseries.empty:
    price_col = next((c for c in ["carbon_price","price","mid"] if c in timeseries.columns), None)
    vol_col = next((c for c in ["volume","market_volume","traded_volume"] if c in timeseries.columns), None)
    cols = st.columns(2)
    if price_col: cols[0].plotly_chart(px.line(timeseries, y=price_col, title="Carbon Price"), use_container_width=True)
    if vol_col: cols[1].plotly_chart(px.area(timeseries, y=vol_col, title="Market Volume"), use_container_width=True)
else:
    st.info("No timeseries found.")
st.markdown("### Annual Summary")
if isinstance(annual, pd.DataFrame) and not annual.empty:
    st.dataframe(annual, use_container_width=True)
else:
    st.caption("No annual summary CSV found.")
