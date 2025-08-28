# =====================================================================
# CCTS SIMULATION RUNNER (CSV-first, dashboard-compatible)
# =====================================================================
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

# --- CCTS modules ---
import ccts.config as config
from ccts.config import (
    config_factory,
    validate_config,
    merge_configs,
    load_config_from_file,
)
from ccts.model import CCTSModel
from ccts.analysis import analyze_results, generate_summary_report
from ccts.timeseries import (
    export_time_series,
    export_agent_histories,
    plot_key_timeseries,
    plot_sector_trends,
    export_market_orderbook_snapshots,
)

logger = logging.getLogger(__name__)


# =====================================================================
# I/O helpers
# =====================================================================

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _resolve_path(user_supplied: Optional[str]) -> Optional[Path]:
    """
    Resolve a path provided by the user. If it's absolute or exists as-is, keep it.
    Otherwise, try to resolve relative to ./input/<user_supplied>.
    """
    if not user_supplied:
        return None
    p = Path(user_supplied)
    if p.exists():
        return p
    alt = Path("input") / user_supplied
    if alt.exists():
        return alt
    # Return the raw path (error will be raised by the consumer if it truly doesn't exist)
    return p


def load_firms_csv(csv_path: Optional[str]) -> pd.DataFrame:
    """
    Load firm data from CSV. The model expects at minimum:
      firm_id, sector, baseline_production, baseline_emissions_intensity,
      cash_on_hand, revenue_per_unit.
    Optional columns are filled with sensible defaults.
    """
    if not csv_path:
        raise FileNotFoundError("No firms CSV provided. Please pass --firms-csv PATH")

    path = _resolve_path(csv_path)
    if not path or not path.exists():
        raise FileNotFoundError(f"Firms CSV not found at {csv_path}")

    df = pd.read_csv(path)

    required = [
        "firm_id",
        "sector",
        "baseline_production",
        "baseline_emissions_intensity",
        "cash_on_hand",
        "revenue_per_unit",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Firms CSV is missing required columns: {missing}")

    # Sensible defaults for optionals used by agents/market:
    if "is_covered" not in df.columns:
        df["is_covered"] = True
    if "risk_aversion" not in df.columns:
        df["risk_aversion"] = 0.5
    if "technology_readiness" not in df.columns:
        df["technology_readiness"] = 0.5
    if "max_production_change_percent" not in df.columns:
        df["max_production_change_percent"] = 0.2
    if "variable_production_cost" not in df.columns:
        df["variable_production_cost"] = 0.0

    return df


def transactions_to_dataframe(model: CCTSModel) -> pd.DataFrame:
    """Convert the market's transactions log to a DataFrame (aligned with utils.Transaction)."""
    try:
        tx = getattr(model.market, "transactions_log", [])
        if not tx:
            return pd.DataFrame(
                columns=["step", "buyer_id", "seller_id", "quantity", "price", "transaction_type"]
            )

        return pd.DataFrame(
            [
                {
                    "step": t.step,
                    "buyer_id": t.buyer_id,
                    "seller_id": t.seller_id,
                    "quantity": t.quantity,
                    "price": t.price,
                    "transaction_type": t.transaction_type,
                }
                for t in tx
            ]
        )
    except Exception as e:
        logger.error(f"Failed to convert transactions log: {e}", exc_info=True)
        return pd.DataFrame(
            columns=["step", "buyer_id", "seller_id", "quantity", "price", "transaction_type"]
        )


def save_excel_outputs(
    output_path: str,
    model: CCTSModel,
    results: Dict[str, Any],
    transactions_df: pd.DataFrame,
    initial_firms_df: pd.DataFrame,
) -> Optional[str]:
    """Save model/agent vars, results, and transactions to an Excel workbook."""
    try:
        out = Path(output_path)
        _ensure_dir(out.parent)

        model_df = model.datacollector.get_model_vars_dataframe()
        agent_df = model.datacollector.get_agent_vars_dataframe()
        summary_text = generate_summary_report(results)

        with pd.ExcelWriter(out, engine="xlsxwriter") as writer:
            model_df.to_excel(writer, sheet_name="Model_Vars")
            agent_df.to_excel(writer, sheet_name="Agent_Vars")
            if transactions_df is not None and not transactions_df.empty:
                transactions_df.to_excel(writer, sheet_name="Transactions", index=False)

            # Results tabs
            annual = results.get("annual_summary")
            if isinstance(annual, list):
                pd.DataFrame(annual).to_excel(writer, sheet_name="Annual_Summary", index=False)

            sector = results.get("sector_analysis")
            if isinstance(sector, dict) and sector:
                pd.DataFrame.from_dict(sector, orient="index").to_excel(writer, sheet_name="Sector_Analysis")

            market = results.get("market_analysis")
            if isinstance(market, dict) and market:
                # keep simple numeric keys; participation_analysis is a list
                market_copy = {k: v for k, v in market.items() if not isinstance(v, list)}
                if market_copy:
                    pd.DataFrame([market_copy]).to_excel(writer, sheet_name="Market_Analysis", index=False)
                if isinstance(market.get("participation_analysis"), list):
                    pd.DataFrame(market["participation_analysis"]).to_excel(
                        writer, sheet_name="Participation", index=False
                    )

            econ = results.get("economic_analysis")
            if isinstance(econ, dict) and econ:
                pd.DataFrame([econ]).to_excel(writer, sheet_name="Economic_Summary", index=False)

            comp = results.get("compliance_summary")
            if isinstance(comp, pd.DataFrame):
                comp.to_excel(writer, sheet_name="Compliance_Summary", index=False)

            pd.DataFrame({"Summary": [summary_text]}).to_excel(
                writer, sheet_name="Summary_Text", index=False
            )

            # Persist the input firms sheet for traceability
            if initial_firms_df is not None and not initial_firms_df.empty:
                initial_firms_df.to_excel(writer, sheet_name="Input_Firms", index=False)

        logger.info(f"Excel results saved -> {out}")
        return str(out)
    except Exception as e:
        logger.error(f"Failed to save Excel outputs: {e}", exc_info=True)
        return None


# =====================================================================
# Dashboard compatibility helpers (CSV-first canonicalization)
# =====================================================================

def _canon_timeseries_csv(csv_path: Path, float_format: Optional[str] = None) -> None:
    """Rename variant columns -> canonical dashboard names."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    lower = {c.lower(): c for c in df.columns}
    ren: Dict[str, str] = {}

    # step
    for k in ("step", "t", "time"):
        if k in lower:
            ren[lower[k]] = "step"
            break
    # price
    for k in ("carbon_price", "price", "mid"):
        if k in lower:
            ren[lower[k]] = "carbon_price"
            break
    # volume
    for k in ("volume", "market_volume", "traded_volume"):
        if k in lower:
            ren[lower[k]] = "volume"
            break
    # volatility
    for k in ("volatility", "rolling_volatility", "sigma"):
        if k in lower:
            ren[lower[k]] = "volatility"
            break
    # trades
    for k in ("trades", "num_trades", "trade_count"):
        if k in lower:
            ren[lower[k]] = "trades"
            break

    if ren:
        df.rename(columns=ren, inplace=True)

    # keep relevant columns if present
    keep = [c for c in ["step", "carbon_price", "volume", "volatility", "trades"] if c in df.columns]
    if keep:
        df = df[keep]

    df.to_csv(csv_path, index=False, float_format=float_format)


def _canon_agent_history_csv(csv_path: Path, float_format: Optional[str] = None) -> None:
    """Map agent history to canonical columns for the dashboard."""
    if not csv_path.exists():
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    lower = {c.lower(): c for c in df.columns}
    ren: Dict[str, str] = {}

    # step
    for k in ("step", "t", "time"):
        if k in lower:
            ren[lower[k]] = "step"
            break
    # firm id
    for k in ("firm_id", "firm", "agent_id"):
        if k in lower:
            ren[lower[k]] = "firm_id"
            break
    # sector
    if "sector" in lower:
        ren[lower["sector"]] = "sector"
    # production
    for k in ("production", "annual_production", "prod", "units", "output_units"):
        if k in lower:
            ren[lower[k]] = "production"
            break
    # emissions (include common variants)
    for k in ("emissions", "total_emissions", "annual_emissions"):
        if k in lower:
            ren[lower[k]] = "emissions"
            break
    # intensity
    for k in ("intensity", "emissions_intensity"):
        if k in lower:
            ren[lower[k]] = "intensity"
            break
    # cash
    for k in ("cash", "cash_on_hand", "balance"):
        if k in lower:
            ren[lower[k]] = "cash"
            break
    # credits
    for k in ("credits_owned", "credits", "total_credits"):
        if k in lower:
            ren[lower[k]] = "credits_owned"
            break
    # abatement
    for k in ("abatement", "realized_abatement", "abatement_realized"):
        if k in lower:
            ren[lower[k]] = "abatement"
            break

    if ren:
        df.rename(columns=ren, inplace=True)

    # ensure required columns exist
    for col in ["step", "firm_id", "sector", "production", "emissions", "intensity"]:
        if col not in df.columns:
            df[col] = np.nan

    keep = ["step", "firm_id", "sector", "production", "emissions", "intensity"]
    keep += [c for c in ["cash", "credits_owned", "abatement"] if c in df.columns]
    df = df[keep]

    df.to_csv(csv_path, index=False, float_format=float_format)


def _write_transactions_csv(df: pd.DataFrame, out_path: Path, float_format: Optional[str] = None) -> Optional[str]:
    """Write transactions.csv with canonical names."""
    if df is None or df.empty:
        return None
    tx = df.rename(
        columns={
            "buyer_id": "buyer",
            "seller_id": "seller",
            "quantity": "qty",
            "trade_price": "price",
            "t": "step",
            "time": "step",
        }
    )
    keep = [c for c in ["step", "buyer", "seller", "qty", "price", "transaction_type"] if c in tx.columns]
    if not keep:
        return None
    tx = tx[keep]
    out = out_path / "transactions.csv"
    tx.to_csv(out, index=False, float_format=float_format)
    return str(out)


def _write_annual_summary_csv(results: Dict[str, Any], out_path: Path, float_format: Optional[str] = None) -> Optional[str]:
    """Write annual_summary.csv from results['annual_summary'] if available."""
    annual = results.get("annual_summary")
    if isinstance(annual, list) and annual:
        df = pd.DataFrame(annual).copy()
        # rename common variants
        df.rename(
            columns={
                "compliant_pct": "compliance_rate",
                "compliance_percent": "compliance_rate",
                "emissions": "total_emissions",
                "abatement": "total_abatement",
                "total_penalties": "penalties_paid",
                "volume": "market_volume",
                "mean_price": "avg_price",
            },
            inplace=True,
        )
        out = out_path / "annual_summary.csv"
        df.to_csv(out, index=False, float_format=float_format)
        return str(out)
    return None


# =====================================================================
# Main simulation routine
# =====================================================================

def run_simulation(
    firms_csv: str,
    scenario: Optional[str],
    custom_config_path: Optional[str],
    out_dir: str,
    steps: Optional[int] = None,
    seed: Optional[int] = None,
    log_level: str = "INFO",
    excel_filename: str = "enhanced_ccts_simulation_results.xlsx",
    no_excel: bool = False,
    float_format: Optional[str] = None,
) -> Dict[str, Any]:
    # Logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Starting CCTS simulation...")

    # Load firms
    firms_df = load_firms_csv(firms_csv)
    initial_firms_df = firms_df.copy()

    # Determine configuration
    if custom_config_path:
        cfg_path = _resolve_path(custom_config_path)
        logger.info(f"Loading custom configuration from {cfg_path}")
        custom_config = load_config_from_file(str(cfg_path) if cfg_path else custom_config_path)
        base_cfg = merge_configs(config.DEFAULT_CONFIG, custom_config)
    else:
        logger.info(f"No custom config specified; using scenario: {scenario}")
        base_cfg = config_factory.create_config(scenario=scenario)

    if steps is not None:
        base_cfg["num_steps"] = int(steps)
    if seed is not None:
        base_cfg["random_seed"] = int(seed)

    cfg = validate_config(base_cfg)
    logger.info(
        f"Scenario: {scenario} | Steps: {cfg['num_steps']} | Start Year: {cfg.get('start_year', 'N/A')}"
    )

    # Build market params (aligned with IndianCarbonMarket._load_market_parameters)
    MARKET_KEYS = {
        "market_stability_reserve_credits",
        "market_stability_reserve_fund",
        "credit_validity",
        "penalty_rate",
        "min_price",
        "max_price",
        "stability_reserve",
        "verification_cost",
        "minimum_trade_size",
        "order_expiry_steps",
        "maximum_order_size",
        "partial_fill_allowed",
        "price_collar_active",
        "circuit_breaker_threshold",
        "clearing_frequency",
        "banking_allowed",
        "borrowing_allowed",
        "price_sensitivity",
        "base_volatility",
        "max_daily_change",
        "liquidity_spread",
        "MarketStabilityReserve_impact",
        "MarketStabilityReserve_participation",
    }
    market_params = {k: v for k, v in cfg.items() if k in MARKET_KEYS}

    # Create and run model
    model = CCTSModel(
        firm_data=firms_df,
        market_params=market_params,
        model_config=cfg,
        seed=cfg.get("random_seed"),
    )

    num_steps = int(cfg["num_steps"])
    for _ in range(num_steps):
        model.step()

    # Collect outputs
    transactions_df = transactions_to_dataframe(model)
    agent_df = model.datacollector.get_agent_vars_dataframe()
    results = analyze_results(model, agent_df, transactions_df, initial_firms_df)

    # Outputs
    out_path = _ensure_dir(out_dir)

    # Time-series exports & plots (via shared helpers)
    ts_csv = export_time_series(model, filepath=str(out_path / "timeseries.csv"))
    agent_hist_csv = export_agent_histories(model, filepath=str(out_path / "agent_history.csv"))
    plot_key_timeseries(model, output_dir=str(out_path), fname="timeseries_overview.png")
    if agent_hist_csv:
        # NOTE: your agent history column used in plot_sector_trends is 'Annual_Emissions' — keep as-is
        plot_sector_trends(
            agent_history_csv=agent_hist_csv,
            output_dir=str(out_path),
            value_col="Annual_Emissions",
            fname="sector_emissions.png",
        )

    # Order book snapshots (if captured)
    export_market_orderbook_snapshots(model, filepath=str(out_path / "orderbook_snapshots.json"))

    # Canonicalize CSVs & write additional CSVs for dashboard
    try:
        if ts_csv:
            _canon_timeseries_csv(Path(ts_csv), float_format=float_format)
        if agent_hist_csv:
            _canon_agent_history_csv(Path(agent_hist_csv), float_format=float_format)
    except Exception as e:
        logger.warning(f"CSV canonicalization failed: {e}")

    # Transactions -> transactions.csv
    try:
        tx_csv = _write_transactions_csv(transactions_df, out_path, float_format=float_format)
        if tx_csv:
            logger.info(f"Transactions CSV saved -> {tx_csv}")
    except Exception as e:
        logger.warning(f"Failed to write transactions.csv: {e}")

    # Annual summary -> annual_summary.csv
    try:
        annual_csv = _write_annual_summary_csv(results, out_path, float_format=float_format)
        if annual_csv:
            logger.info(f"Annual summary CSV saved -> {annual_csv}")
    except Exception as e:
        logger.warning(f"Failed to write annual_summary.csv: {e}")

    # Excel workbook (optional)
    excel_path = out_path / excel_filename
    if not no_excel:
        save_excel_outputs(
            output_path=str(excel_path),
            model=model,
            results=results,
            transactions_df=transactions_df,
            initial_firms_df=initial_firms_df,
        )

    # Summary.txt
    summary_txt = generate_summary_report(results)
    with open(out_path / "summary.txt", "w", encoding="utf-8") as f:
        f.write(summary_txt)
    logger.info(f"Summary text saved -> {out_path / 'summary.txt'}")

    return {
        "results": results,
        "excel": (str(excel_path) if not no_excel else None),
        "timeseries_csv": str(out_path / "timeseries.csv"),
        "agent_history_csv": str(out_path / "agent_history.csv"),
        "transactions_csv": (str(out_path / "transactions.csv") if (out_path / "transactions.csv").exists() else None),
        "annual_summary_csv": (str(out_path / "annual_summary.csv") if (out_path / "annual_summary.csv").exists() else None),
        "summary_txt": str(out_path / "summary.txt"),
    }


# =====================================================================
# CLI
# =====================================================================

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the CCTS simulation.")
    p.add_argument(
        "--firms-csv",
        type=str,
        required=True,
        help="Path to firms CSV (absolute/relative). If not found, './input/<path>' will be tried.",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default="baseline",
        choices=config_factory.list_scenarios(),
        help="Scenario to run if no custom config file is specified.",
    )
    p.add_argument(
        "--custom-config",
        type=str,
        default=None,
        help="Optional path to a custom config file (e.g., config.json). If not found, './input/<path>' will be tried. Overrides --scenario.",
    )
    p.add_argument("--steps", type=int, default=None, help="Override number of steps.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    p.add_argument("--out-dir", type=str, default="output", help="Output directory.")
    p.add_argument(
        "--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level."
    )
    p.add_argument(
        "--excel-filename", type=str, default="enhanced_ccts_simulation_results.xlsx",
        help="Excel output filename."
    )
    p.add_argument(
        "--no-excel", action="store_true",
        help="Skip Excel; write CSV/JSON only."
    )
    p.add_argument(
        "--float-format", type=str, default=None,
        help="Optional float format for CSVs, e.g. '%%.2f'."
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    run_simulation(
        firms_csv=args.firms_csv,
        scenario=args.scenario,
        custom_config_path=args.custom_config,
        out_dir=args.out_dir,
        steps=args.steps,
        seed=args.seed,
        log_level=args.log_level,
        excel_filename=args.excel_filename,
        no_excel=args.no_excel,
        float_format=args.float_format,
    )


if __name__ == "__main__":
    main()
