
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
