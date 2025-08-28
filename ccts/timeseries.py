# =====================================================================
# TIME-SERIES EXPORTS & PLOTS (drop-in)
# =====================================================================

import os
import logging
from pathlib import Path
from typing import Optional, Sequence, Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .model import CCTSModel

logger = logging.getLogger(__name__)


# ---------------------------
# Helpers
# ---------------------------

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _has_cols(df: pd.DataFrame, cols: Sequence[str]) -> bool:
    return all(c in df.columns for c in cols)

def _safe_plot_series(ax, df: pd.DataFrame, col: str, title: str, ylabel: str):
    if col in df.columns:
        df[col].plot(ax=ax)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        return True
    else:
        ax.set_title(f"{title} (missing: {col})")
        return False


# ---------------------------
# Public API
# ---------------------------

def export_time_series(model: CCTSModel,
                       filepath: str = "output/timeseries.csv") -> str:
    """
    Export step-level model time series from DataCollector.model_reporters.
    """
    try:
        ts = model.datacollector.get_model_vars_dataframe()
        out = Path(filepath)
        _ensure_dir(out.parent)
        ts.to_csv(out, index=True)
        logger.info(f"[timeseries] Model time series -> {out}")
        return str(out)
    except Exception as e:
        logger.error(f"[timeseries] export_time_series failed: {e}", exc_info=True)
        return ""


def export_agent_histories(model: CCTSModel,
                           filepath: str = "output/agent_history.csv") -> str:
    """
    Export full agent trajectories from DataCollector.agent_reporters.
    MultiIndex (Step, AgentID) will be preserved in CSV index.
    """
    try:
        df_agents = model.datacollector.get_agent_vars_dataframe()
        out = Path(filepath)
        _ensure_dir(out.parent)
        df_agents.to_csv(out, index=True)
        logger.info(f"[timeseries] Agent histories -> {out}")
        return str(out)
    except Exception as e:
        logger.error(f"[timeseries] export_agent_histories failed: {e}", exc_info=True)
        return ""


def plot_key_timeseries(model: CCTSModel,
                        output_dir: str = "output",
                        fname: str = "timeseries_overview.png") -> Optional[str]:
    """
    Create a compact 3-panel figure:
      (1) Carbon Price
      (2) Compliance Rate
      (3) Emissions vs Cumulative Abatement
    Handles missing columns gracefully.
    """
    try:
        df = model.datacollector.get_model_vars_dataframe().copy()
        outdir = _ensure_dir(output_dir)

        fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

        _safe_plot_series(axes[0], df, "Carbon_Price", "Carbon Price Evolution", "₹/tCO₂e")
        _safe_plot_series(axes[1], df, "Compliance_Rate", "Compliance Rate Over Time", "Share")

        # Panel 3: Emissions vs Cumulative Abatement
        third_has_any = False
        if "Total_Emissions" in df.columns:
            df["Total_Emissions"].plot(ax=axes[2], label="Total Emissions")
            third_has_any = True
        if "Total_Abated_Tons_Cumulative" in df.columns:
            df["Total_Abated_Tons_Cumulative"].plot(ax=axes[2], label="Cumulative Abatement")
            third_has_any = True

        axes[2].set_title("Emissions vs Abatement")
        axes[2].set_ylabel("tCO₂e")
        if third_has_any:
            axes[2].legend()

        plt.tight_layout()
        outpath = outdir / fname
        plt.savefig(outpath, dpi=160, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"[timeseries] Key timeseries plot -> {outpath}")
        return str(outpath)
    except Exception as e:
        logger.error(f"[timeseries] plot_key_timeseries failed: {e}", exc_info=True)
        return None


def plot_sector_trends(agent_history_csv: str | None = None,
                       agent_df: pd.DataFrame | None = None,
                       value_col: str = "Annual_Emissions",
                       output_dir: str = "output",
                       fname: str = "sector_trends.png") -> Optional[str]:
    """
    Plot sector totals over time from agent histories.
    You can pass either a path to the exported agent_history.csv or a preloaded DataFrame.

    value_col options that commonly exist in your model:
      - "Annual_Emissions" (preferred)
      - "Annual_Production"
      - "Abated_Tons_Cumulative"
      - "Credits_Owned"
    """
    try:
        if agent_df is None:
            if agent_history_csv is None:
                logger.warning("[timeseries] plot_sector_trends called without data")
                return None
            # The exported CSV preserves the MultiIndex as columns; load with index_col for Step/AgentID
            df = pd.read_csv(agent_history_csv)
        else:
            df = agent_df.copy()

        if "Step" not in df.columns:
            # If CSV was written with MultiIndex, columns may include 'Step'
            # Ensure 'Step' exists
            possible = [c for c in df.columns if c.lower() == "step"]
            if possible:
                df.rename(columns={possible[0]: "Step"}, inplace=True)

        if "Sector" not in df.columns:
            logger.warning("[timeseries] Sector column not found; skipping sector plot")
            return None
        if value_col not in df.columns:
            logger.warning(f"[timeseries] {value_col} not found; skipping sector plot")
            return None

        outdir = _ensure_dir(output_dir)

        g = (df.groupby(["Step", "Sector"])[value_col]
               .sum()
               .unstack(fill_value=0)
               .sort_index())

        fig, ax = plt.subplots(figsize=(12, 6))
        g.plot(ax=ax)
        ax.set_title(f"Sector {value_col} Over Time")
        ax.set_ylabel(value_col)
        ax.set_xlabel("Step")
        ax.legend(loc="best", fontsize=9, ncol=2)
        plt.tight_layout()
        outpath = outdir / fname
        plt.savefig(outpath, dpi=160, bbox_inches="tight")
        plt.close(fig)

        logger.info(f"[timeseries] Sector trend plot -> {outpath}")
        return str(outpath)
    except Exception as e:
        logger.error(f"[timeseries] plot_sector_trends failed: {e}", exc_info=True)
        return None


def export_market_orderbook_snapshots(model: CCTSModel,
                                      filepath: str = "output/orderbook_snapshots.json") -> str:
    """
    Export market order book snapshots captured per step (if available).
    market.order_book_history is populated in IndianCarbonMarket.step().
    """
    try:
        history = getattr(model.market, "order_book_history", None)
        if not history:
            logger.info("[timeseries] No order_book_history found; skipping.")
            return ""
        out = Path(filepath)
        _ensure_dir(out.parent)
        pd.Series(history).to_json(out, orient="records")
        logger.info(f"[timeseries] Order book snapshots -> {out}")
        return str(out)
    except Exception as e:
        logger.error(f"[timeseries] export_market_orderbook_snapshots failed: {e}", exc_info=True)
        return ""


def summarize_compliance_trajectory(model: CCTSModel) -> Dict[str, float]:
    """
    Quick stats for compliance rate trajectory: start, end, min, max, trend.
    """
    try:
        df = model.datacollector.get_model_vars_dataframe()
        if "Compliance_Rate" not in df.columns or df.empty:
            return {}
        s = df["Compliance_Rate"].dropna()
        if s.empty:
            return {}
        trend = float(s.iloc[-1] - s.iloc[0])
        return {
            "start": float(s.iloc[0]),
            "end": float(s.iloc[-1]),
            "min": float(s.min()),
            "max": float(s.max()),
            "trend": trend
        }
    except Exception as e:
        logger.error(f"[timeseries] summarize_compliance_trajectory failed: {e}", exc_info=True)
        return {}
