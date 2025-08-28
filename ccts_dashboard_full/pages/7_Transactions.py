
import streamlit as st, pandas as pd, plotly.express as px
from utils.data_loader import list_output_folder, load_csv_safe
st.set_page_config(page_title="Transactions & Order Flow • CCTS", page_icon="🔁", layout="wide")
st.title("🔁 Transactions & Order Flow")
path = st.session_state.get("__ccts_output_path__", "sample_output")
files = list_output_folder(path)
tx = load_csv_safe(files.get("transactions"))
if isinstance(tx, pd.DataFrame) and not tx.empty:
    cols = tx.columns
    qty_col = next((c for c in ["qty","quantity","size","volume"] if c in cols), None)
    price_col = next((c for c in ["price","trade_price"] if c in cols), None)
    step_col = next((c for c in ["step","time","t"] if c in cols), None)
    if qty_col and step_col:
        st.plotly_chart(px.bar(tx, x=step_col, y=qty_col, title="Order Flow (by step)"), use_container_width=True)
    if price_col and step_col:
        st.plotly_chart(px.scatter(tx, x=step_col, y=price_col, size=qty_col, title="Trade Prices"), use_container_width=True)
    st.markdown("### Trades (last 500)")
    st.dataframe(tx.tail(500), use_container_width=True)
else:
    st.info("No transactions CSV found.")
