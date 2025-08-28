# =====================================================================
# UTILS: dataclasses & helpers (aligned with market/model/analysis)
# =====================================================================

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Any
import math
import pandas as pd


# ----------------------------
# Data structures
# ----------------------------

@dataclass
class MarketOrder:
    agent: Optional[object]     # None for market maker / MSR
    quantity: float
    timestamp: int
    order_type: str             # 'market' | 'limit'
    price: Optional[float]      # None for market orders


@dataclass
class Transaction:
    step: int
    buyer_id: str
    seller_id: str
    quantity: float
    price: float
    transaction_type: str       # 'market_execution' | 'secondary' | 'MSR' | etc.


@dataclass
class ComplianceRecord:
    year: int
    gap: float
    penalty: float
    is_compliant: bool


# ----------------------------
# Helpers
# ----------------------------

def safe_divide(numer: float, denom: float, default: float = 0.0) -> float:
    try:
        if denom == 0 or (isinstance(denom, (int, float)) and abs(denom) < 1e-15):
            return default
        return float(numer) / float(denom)
    except Exception:
        return default


def safe_get_column(df: pd.DataFrame, col: str, default: Any) -> pd.Series:
    """Return df[col] if present; otherwise a Series filled with default of the same length."""
    try:
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            return pd.Series([], dtype=type(default))
        if col in df.columns:
            return df[col]
        return pd.Series([default] * len(df), index=df.index)
    except Exception:
        # Fallback empty series
        return pd.Series([], dtype=type(default))
