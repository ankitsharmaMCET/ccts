
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
