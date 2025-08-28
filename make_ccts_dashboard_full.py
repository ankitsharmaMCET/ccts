# Creates a full Streamlit dashboard for your CCTS outputs, including pages,
# utils, theme, Docker files, and a sample dataset so it runs instantly.
# Usage (PowerShell):
#   python .\make_ccts_dashboard_full.py

from pathlib import Path
import json, textwrap, math, os
import numpy as np
import pandas as pd

ROOT = Path("ccts_dashboard_full")
PAGES = ROOT / "pages"
UTILS = ROOT / "utils"
ASSETS = ROOT / "assets"
CFGDIR = ROOT / ".streamlit"
SAMPLE = ROOT / "sample_output"

for d in (PAGES, UTILS, ASSETS, CFGDIR, SAMPLE):
    d.mkdir(parents=True, exist_ok=True)

# ---------- requirements ----------
(ROOT / "requirements.txt").write_text(textwrap.dedent("""
streamlit>=1.36
pandas>=2.1
numpy>=1.26
plotly>=5.22
duckdb>=1.0.0
pyarrow>=15.0
openpyxl>=3.1
"""), encoding="utf-8")

# ---------- theme ----------
(CFGDIR / "config.toml").write_text(textwrap.dedent("""
[theme]
base="light"
primaryColor="#166534"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f1f5f9"
textColor="#0f172a"
font="sans serif"

[server]
enableStaticServing = true
"""), encoding="utf-8")

# ---------- utils/schema & helpers ----------
(UTILS / "schema.py").write_text(textwrap.dedent(r"""
TIMESERIES_COLS = {
    "price": ["carbon_price","price","mid"],
    "volume": ["volume","market_volume","traded_volume"],
    "volatility": ["volatility","rolling_volatility","sigma"],
    "trades": ["trades","num_trades","trade_count"],
    "step": ["step","t","time"],
}
AGENT_COLS = {
    "firm": ["firm_id","firm","agent_id"],
    "sector": ["sector","industry"],
    "production": ["production","prod","output_units"],
    "emissions": ["emissions","total_emissions"],
    "intensity": ["intensity","emissions_intensity"],
    "cash": ["cash","cash_on_hand","balance"],
    "credits": ["credits","credits_owned","total_credits"],
    "abatement": ["abatement","realized_abatement","abatement_realized"],
    "step": ["step","t","time"],
}
ANNUAL_COLS = {
    "year": ["year"],
    "compliance_rate": ["compliance_rate","compliant_pct","compliance_percent"],
    "emissions": ["total_emissions","emissions"],
    "abatement": ["total_abatement","abatement"],
    "penalties": ["penalties_paid","total_penalties","penalties"],
    "volume": ["market_volume","volume"],
    "avg_price": ["avg_price","mean_price"],
}
TX_COLS = {
    "step": ["step","t","time"],
    "buyer": ["buyer","buyer_id"],
    "seller": ["seller","seller_id"],
    "qty": ["qty","quantity","size","volume"],
    "price": ["price","trade_price"],
}
"""), encoding="utf-8")

(UTILS / "helpers.py").write_text(textwrap.dedent(r"""
from typing import Optional, List
import pandas as pd
def pick_col(df: pd.DataFrame, options: List[str]) -> Optional[str]:
    for c in options:
        if c in df.columns:
            return c
    return None
def last_value(df: pd.DataFrame, col: str):
    return None if df.empty or col not in df.columns else df[col].iloc[-1]
def pct(x, y):
    if x is None or y in (None, 0): return None
    try: return 100.0 * (x / y - 1.0)
    except Exception: return None
"""), encoding="utf-8")

(UTILS / "data_loader.py").write_text(textwrap.dedent(r"""
import json
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import streamlit as st

DEFAULT_FILES = {
    "timeseries": "timeseries.csv",
    "agents": "agent_history.csv",
    "transactions": "transactions.csv",
    "annual": "annual_summary.csv",
    "orderbook": "orderbook_snapshots.json",
    "excel": "enhanced_ccts_simulation_results.xlsx",
    "summary": "summary.txt",
}

@st.cache_data(show_spinner=False, ttl=60)
def list_output_folder(path_str: str) -> Dict[str, Path]:
    base = Path(path_str).expanduser()
    files = {}
    if not base.exists():
        return files
    for key, name in DEFAULT_FILES.items():
        p = base / name
        files[key] = p if p.exists() else None
    # tolerant globs
    if files.get("transactions") is None:
        cand = list(base.glob("*transactions*.csv"))
        files["transactions"] = cand[0] if cand else None
    if files.get("annual") is None:
        cand = list(base.glob("*annual*summary*.csv"))
        files["annual"] = cand[0] if cand else None
    if files.get("timeseries") is None:
        cand = list(base.glob("*timeseries*.csv"))
        files["timeseries"] = cand[0] if cand else None
    if files.get("agents") is None:
        cand = list(base.glob("*agent*history*.csv"))
        files["agents"] = cand[0] if cand else None
    if files.get("orderbook") is None:
        cand = list(base.glob("*orderbook*snapshot*.json"))
        files["orderbook"] = cand[0] if cand else None
    return files

@st.cache_data(show_spinner=False, ttl=60)
def load_csv_safe(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path and path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            try:
                return pd.read_excel(path)
            except Exception:
                return None
    return None

@st.cache_data(show_spinner=False, ttl=60)
def load_json_safe(path: Optional[Path]):
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

@st.cache_data(show_spinner=False, ttl=60)
def load_excel_sheets(path: Optional[Path]):
    out = {}
    if path and path.exists():
        try:
            xls = pd.ExcelFile(path)
            for s in xls.sheet_names:
                out[s] = xls.parse(s)
        except Exception:
            pass
    return out

def compute_kpis(timeseries: Optional[pd.DataFrame], annual: Optional[pd.DataFrame]) -> dict:
    kpis = {"carbon_price":None,"volume":None,"volatility":None,"compliance_rate":None,"emissions":None,"abatement":None,"penalties":None,"trades":None}
    if isinstance(timeseries, pd.DataFrame) and not timeseries.empty:
        for col in ["carbon_price","price","mid"]:
            if col in timeseries.columns: kpis["carbon_price"]=timeseries[col].iloc[-1]; break
        for col in ["volume","market_volume","traded_volume"]:
            if col in timeseries.columns: kpis["volume"]=timeseries[col].iloc[-1]; break
        for col in ["volatility","rolling_volatility","sigma"]:
            if col in timeseries.columns: kpis["volatility"]=timeseries[col].iloc[-1]; break
        for col in ["trades","num_trades","trade_count"]:
            if col in timeseries.columns: kpis["trades"]=int(timeseries[col].iloc[-1]); break
    if isinstance(annual, pd.DataFrame) and not annual.empty:
        def pick(names): 
            for n in names:
                if n in annual.columns: return annual[n].iloc[-1]
            return None
        kpis["compliance_rate"]=pick(["compliance_rate","compliant_pct","compliance_percent"])
        kpis["emissions"]=pick(["total_emissions","emissions"])
        kpis["abatement"]=pick(["total_abatement","abatement"])
        kpis["penalties"]=pick(["penalties_paid","total_penalties","penalties"])
    return kpis
"""), encoding="utf-8")

# ---------- app ----------
(ROOT / "app.py").write_text(textwrap.dedent(r"""
import streamlit as st
import plotly.express as px
from utils.data_loader import list_output_folder, load_csv_safe, load_excel_sheets, compute_kpis

st.set_page_config(page_title="CCTS Dashboard", page_icon="🌿", layout="wide", initial_sidebar_state="expanded")
st.title("🌿 CCTS Dashboard")
st.caption("Bottom-up Carbon Credit Trading Simulator — Full Dashboard")

with st.sidebar:
    st.header("Run Selector")
    path = st.text_input("Output folder path", value="sample_output", help="Point to a folder containing the exported CSV/JSON/Excel from your simulation.")
    files = list_output_folder(path)
    if not files:
        st.warning("Folder not found. Update the path.")
    else:
        ok = [k for k,v in files.items() if v is not None]
        st.success(f"Found: {', '.join(ok) if ok else 'no known files'}")

    st.markdown("---")
    st.page_link("pages/1_Overview.py", label="Overview", icon="🏠")
    st.page_link("pages/2_Market_Monitor.py", label="Market Monitor", icon="📈")
    st.page_link("pages/3_Compliance_Policy.py", label="Compliance & Policy", icon="✅")
    st.page_link("pages/4_Firms_Explorer.py", label="Firms Explorer", icon="🏭")
    st.page_link("pages/5_Abatement_Projects.py", label="Abatement & Projects", icon="🛠️")
    st.page_link("pages/6_Credits_Vintages.py", label="Credits & Vintages", icon="🎟️")
    st.page_link("pages/7_Transactions.py", label="Transactions & Order Flow", icon="🔁")
    st.page_link("pages/8_Scenario_Lab.py", label="Scenario & What-If", icon="🧪")
    st.page_link("pages/9_Exports.py", label="Exports", icon="📦")

st.session_state["__ccts_output_path__"] = path
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

st.markdown("### Quick Trends")
if timeseries is not None and not timeseries.empty:
    price_col = next((c for c in ["carbon_price","price","mid"] if c in timeseries.columns), None)
    vol_col = next((c for c in ["volume","market_volume","traded_volume"] if c in timeseries.columns), None)
    cols = st.columns(2)
    if price_col:
        cols[0].plotly_chart(px.line(timeseries.tail(250), y=price_col, title="Carbon Price"), use_container_width=True)
    if vol_col:
        cols[1].plotly_chart(px.bar(timeseries.tail(250), y=vol_col, title="Market Volume"), use_container_width=True)
else:
    st.info("No timeseries found. Use the sample_output folder for a demo.")
"""), encoding="utf-8")

# ---------- pages ----------
pages = {
"1_Overview.py": r"""
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
""",
"2_Market_Monitor.py": r"""
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
""",
"3_Compliance_Policy.py": r"""
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
""",
"4_Firms_Explorer.py": r"""
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
""",
"5_Abatement_Projects.py": r"""
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
""",
"6_Credits_Vintages.py": r"""
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
""",
"7_Transactions.py": r"""
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
""",
"8_Scenario_Lab.py": r"""
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
""",
"9_Exports.py": r"""
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
""",
}
for fname, code in pages.items():
    (PAGES / fname).write_text(textwrap.dedent(code), encoding="utf-8")

# ---------- sample data so the app runs instantly ----------
np.random.seed(7)
steps = 200
ts = pd.DataFrame({
    "step": np.arange(steps),
    "carbon_price": np.round(4500 + np.cumsum(np.random.normal(0, 15, steps)), 2),
    "volume": np.abs(np.random.normal(100, 25, steps)).round(2),
    "volatility": np.abs(np.random.normal(0.02, 0.005, steps)).round(4),
    "trades": np.random.poisson(12, steps),
})
ts.to_csv(SAMPLE / "timeseries.csv", index=False)

firms = ["POW_01","POW_02","STE_01","CEM_01","CHE_01","ALU_01"]
sectors = {"POW_01":"Power","POW_02":"Power","STE_01":"Steel","CEM_01":"Cement","CHE_01":"Chemicals","ALU_01":"Aluminum"}
rows = []
for f in firms:
    base_prod = np.random.randint(40000, 120000)
    intensity = np.random.uniform(0.6, 2.0)
    cash0 = np.random.randint(2_000_000, 8_000_000)
    credits0 = np.random.randint(0, 8000)
    for s in range(steps):
        prod = base_prod * (1 + 0.05*np.sin(2*math.pi * s/50.0)) + np.random.normal(0, 1000)
        em_int = max(0.1, intensity + np.random.normal(0, 0.02))
        emissions = max(0.0, prod * em_int / 1e4)
        cash = cash0 + np.random.normal(0, 20000) - 50* s
        credits = max(0, credits0 + int(np.random.normal(0, 20)))
        abatement = max(0, np.random.normal(200, 30))
        rows.append({
            "step": s, "firm_id": f, "sector": sectors[f],
            "production": round(prod,2),
            "emissions": round(emissions,2),
            "intensity": round(em_int,3),
            "cash": round(cash,2),
            "credits_owned": credits,
            "abatement": abatement,
        })
pd.DataFrame(rows).to_csv(SAMPLE / "agent_history.csv", index=False)

annual = pd.DataFrame({
    "year": [2025, 2026],
    "compliance_rate": [0.88, 0.93],
    "total_emissions": [1.2e6, 1.1e6],
    "total_abatement": [2.5e5, 2.8e5],
    "penalties_paid": [5.2e7, 4.6e7],
    "market_volume": [2.1e5, 2.4e5],
    "avg_price": [4800, 4900],
})
annual.to_csv(SAMPLE / "annual_summary.csv", index=False)

tx = pd.DataFrame({
    "step": np.random.choice(range(steps), size=1000),
    "buyer": np.random.choice(firms, size=1000),
    "seller": np.random.choice(firms, size=1000),
    "qty": np.abs(np.random.normal(50, 20, 1000)).round(2),
    "price": np.round(np.random.normal(4800, 60, 1000), 2),
})
tx.to_csv(SAMPLE / "transactions.csv", index=False)

orderbook = { "bids": [[4790+i, int(100-5*i)] for i in range(10)],
              "asks": [[4810+i, int(100-5*i)] for i in range(10)] }
(SAMPLE / "orderbook_snapshots.json").write_text(json.dumps(orderbook, indent=2), encoding="utf-8")
(SAMPLE / "summary.txt").write_text("Sample run summary: all systems nominal.", encoding="utf-8")

# ---------- Docker support ----------
(ROOT / "Dockerfile").write_text(textwrap.dedent("""
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app
EXPOSE 8501
ENV STREAMLIT_SERVER_PORT=8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
"""), encoding="utf-8")

(ROOT / "docker-compose.yml").write_text(textwrap.dedent("""
version: "3.9"
services:
  ccts_dashboard:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./sample_output:/app/sample_output
      - ./external_output:/app/external_output
    environment:
      - STREAMLIT_SERVER_PORT=8501
"""), encoding="utf-8")

