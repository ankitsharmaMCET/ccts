
import streamlit as st, plotly.express as px, pandas as pd
from utils.data_loader import list_output_folder, load_csv_safe, load_json_safe
st.set_page_config(page_title="Market Monitor • CCTS", page_icon="📈", layout="wide")
st.title("📈 Market Monitor")
path = st.session_state.get("__ccts_output_path__", "sample_output")
files = list_output_folder(path)
ts = load_csv_safe(files.get("timeseries"))
ob = load_json_safe(files.get("orderbook"))
col1, col2 = st.columns(2)
if isinstance(ts, pd.DataFrame) and not ts.empty:
    price_col = next((c for c in ["carbon_price","price","mid"] if c in ts.columns), None)
    vol_col = next((c for c in ["volume","market_volume","traded_volume"] if c in ts.columns), None)
    if price_col: col1.plotly_chart(px.line(ts.tail(250), y=price_col, title="Carbon Price (last 250 steps)"), use_container_width=True)
    if vol_col: col2.plotly_chart(px.bar(ts.tail(250), y=vol_col, title="Traded Volume (last 250 steps)"), use_container_width=True)
else:
    st.info("No timeseries available.")
st.markdown("### Order Book Snapshot")
if isinstance(ob, dict) and (ob.get("bids") or ob.get("asks")):
    cb1, cb2 = st.columns(2)
    cb1.subheader("Bids (Top 10)"); cb1.table(ob.get("bids", [])[:10])
    cb2.subheader("Asks (Top 10)"); cb2.table(ob.get("asks", [])[:10])
else:
    st.caption("No or empty orderbook snapshot JSON.")
