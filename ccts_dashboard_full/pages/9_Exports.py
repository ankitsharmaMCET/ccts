
import streamlit as st
from utils.data_loader import list_output_folder
st.set_page_config(page_title="Exports • CCTS", page_icon="📦", layout="wide")
st.title("📦 Exports")
path = st.session_state.get("__ccts_output_path__", "sample_output")
files = list_output_folder(path)
st.write("Download raw artifacts:")
for key, p in files.items():
    if p and p.exists():
        with open(p, "rb") as f:
            st.download_button(label=f"Download {p.name}", data=f, file_name=p.name)
    else:
        st.caption(f"Missing: {key}")
