# =====================================================================
# ENHANCED ANALYSIS FUNCTIONS (CORRECTED VERSION)
# =====================================================================

import logging
from typing import Dict
import pandas as pd
import numpy as np

from .model import CCTSModel
from .utils import safe_divide, safe_get_column

logger = logging.getLogger(__name__)

# ---------------------------
# Public entry point
# ---------------------------
def analyze_results(
    model: CCTSModel,
    agent_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    initial_firms_df: pd.DataFrame,
) -> Dict:
    """Enhanced comprehensive results analysis."""
    try:
        if agent_df is None or agent_df.empty:
            return {"error": "No agent data available for analysis"}

        # Normalize index/columns
        if "Step" in agent_df.index.names:
            final_step = agent_df.index.get_level_values("Step").max()
            agent_df_reset = agent_df.reset_index()
        else:
            # assume first index level is Step
            final_step = agent_df.index.max() if hasattr(agent_df.index, "max") else agent_df["Step"].max()
            agent_df_reset = agent_df.reset_index()
            if "Step" not in agent_df_reset.columns:
                agent_df_reset = agent_df_reset.rename(columns={agent_df_reset.columns[0]: "Step"})

        # Merge static firm data (sector, etc.)
        merge_key_right = "firm_id" if "firm_id" in initial_firms_df.columns else ("Firm_ID" if "Firm_ID" in initial_firms_df.columns else None)
        merge_key_left = "Firm_ID" if "Firm_ID" in agent_df_reset.columns else ("AgentID" if "AgentID" in agent_df_reset.columns else None)

        if merge_key_right and merge_key_left:
            try:
                merged_agent_data = pd.merge(
                    agent_df_reset,
                    initial_firms_df,
                    left_on=merge_key_left,
                    right_on=merge_key_right,
                    how="left",
                    suffixes=("", "_initial"),
                )
            except Exception as e:
                logger.warning(f"Merge failed, continuing with agent data only: {e}")
                merged_agent_data = agent_df_reset.copy()
        else:
            merged_agent_data = agent_df_reset.copy()

        final_agent_data = merged_agent_data[merged_agent_data["Step"] == final_step].copy()
        if final_agent_data.empty:
            return {"error": "Final agent data is empty after filtering"}

        results = {}
        results.update(_analyze_compliance(final_agent_data, merged_agent_data, final_step))
        results.update(_analyze_market_performance(merged_agent_data, transactions_df))
        results.update(_analyze_sector_performance(final_agent_data))
        results["market_concentration"] = _analyze_market_concentration(transactions_df)
        results.update(_analyze_abatement_effectiveness(final_agent_data))
        results.update(_analyze_economic_impacts(final_agent_data))
        results.update(_analyze_annual_summary(merged_agent_data, transactions_df, model))
        results.update(_analyze_model_performance(model))
        return results

    except Exception as e:
        logger.error(f"Results analysis error: {e}", exc_info=True)
        return {"error": str(e)}

# ---------------------------
# Components
# ---------------------------
def _analyze_compliance(final_agent_data: pd.DataFrame, all_agent_data: pd.DataFrame, final_step: int) -> Dict:
    """Analyze compliance performance with intensity-consistent calculations."""
    try:
        is_cov = safe_get_column(final_agent_data, "Is_Covered", False)
        covered = final_agent_data[is_cov == True].copy()
        if covered.empty:
            return {"compliance_analysis": "No covered entities found"}

        annual_em = safe_get_column(covered, "Annual_Emissions", 0)
        annual_prod = safe_get_column(covered, "Annual_Production", 0)
        target_I = safe_get_column(covered, "Current_Target_Intensity", 0)
        credits = safe_get_column(covered, "Credits_Owned", 0)

        allowed_em = target_I * annual_prod
        gross_gap = annual_em - allowed_em
        net_gap = (gross_gap - credits).clip(lower=0)

        compliant = net_gap <= 1e-6
        total_gross = gross_gap.clip(lower=0).sum()
        total_net = net_gap.sum()

        actual_I = safe_divide(annual_em.sum(), annual_prod.sum())
        target_I_avg = target_I.mean()

        summary = pd.DataFrame(
            {
                "Status": ["Compliant", "Non-Compliant"],
                "Count": [int(compliant.sum()), int((~compliant).sum())],
                "Percentage": [float(compliant.mean() * 100), float((1 - compliant.mean()) * 100)],
            }
        )

        bank_by_sector = None
        if "Sector" in covered.columns:
            bank_by_sector = (
                covered.groupby("Sector")["Credits_Owned"]
                .agg(Total_Credits="sum", Avg_Credits="mean", Firm_Count="count")
                .round(2)
                .to_dict("index")
            )

        return {
            "compliance_summary": summary,
            "covered_entities_count": int(len(covered)),
            "total_gross_compliance_gap": float(total_gross),
            "total_net_compliance_gap": float(total_net),
            "average_compliance_gap": float(net_gap.mean()),
            "overall_compliance_rate": float(compliant.mean()),
            "actual_intensity": float(actual_I),
            "target_intensity_average": float(target_I_avg),
            "intensity_gap": float(actual_I - target_I_avg),
            "banking_analysis": bank_by_sector,
        }
    except Exception as e:
        logger.error(f"Compliance analysis error: {e}", exc_info=True)
        return {"compliance_analysis_error": str(e)}

def _analyze_market_performance(agent_df: pd.DataFrame, transactions_df: pd.DataFrame) -> Dict:
    """Analyze carbon market performance."""
    try:
        out = {}
        if transactions_df is not None and not transactions_df.empty and {"quantity", "price"} <= set(transactions_df.columns):
            q = transactions_df["quantity"]
            p = transactions_df["price"]
            out.update(
                {
                    "total_trading_volume": float(q.sum()),
                    "total_transaction_value": float((q * p).sum()),
                    "total_transactions": int(len(transactions_df)),
                    "average_trade_size": float(q.mean()),
                    "average_trade_price": float(p.mean()),
                    "min_trade_price": float(p.min()),
                    "max_trade_price": float(p.max()),
                    "median_trade_price": float(p.median()),
                    "price_volatility": float(p.std() / p.mean()) if p.mean() > 0 else 0.0,
                    "trade_size_distribution": q.describe().to_dict(),
                }
            )
        else:
            out.update(
                {
                    "total_trading_volume": 0.0,
                    "total_transaction_value": 0.0,
                    "total_transactions": 0,
                    "message": "No valid transaction data available",
                }
            )

        if "Step" in agent_df.columns:
            steps = []
            for s in sorted(agent_df["Step"].unique()):
                sd = agent_df[agent_df["Step"] == s]
                need = safe_get_column(sd, "Credits_Needed", 0)
                surp = safe_get_column(sd, "Credits_Surplus", 0)
                steps.append(
                    {
                        "step": int(s),
                        "buy_participation": float((need > 0).mean()),
                        "sell_participation": float((surp > 0).mean()),
                        "total_participation": float(((need > 0) | (surp > 0)).mean()),
                    }
                )
            out["participation_analysis"] = steps

        return {"market_analysis": out}
    except Exception as e:
        logger.error(f"Market analysis error: {e}", exc_info=True)
        return {"market_analysis_error": str(e)}

# E:\model_2\ccts\analysis.py

def _analyze_sector_performance(final_agent_data: pd.DataFrame) -> Dict:
    """Analyze performance by sector with safe column access."""
    try:
        if "Sector" not in final_agent_data.columns or final_agent_data["Sector"].isnull().all():
            return {"sector_analysis": "No sector data available"}

        # Pick available columns for each metric
        candidates = {
            "Emissions": ["Annual_Emissions", "Current_Emissions", "Emissions"],
            "Production": ["Annual_Production", "Current_Production", "Production"],
            "Abatement": ["Abated_Tons_Cumulative", "Abated_Tons", "Total_Abatement"],
            "Credits": ["Credits_Owned", "Credits", "Total_Credits"],
            "Cash": ["Cash", "Available_Cash", "Current_Cash"],
            "Compliance": ["Is_Compliant", "Compliant", "Compliance_Status"],
            "Profit": ["Current_Profit", "Profit"],
            "Monthly_Abatement_Cost": ["Monthly_Abatement_Cost", "AbatementCost"], # <- Added
            "Monthly_Carbon_Cost": ["Monthly_Carbon_Cost", "CarbonCost"], # <- Added
            "Monthly_Variable_Cost": ["Monthly_Variable_Cost", "VariableCost"] # <- Added
        }
        
        chosen = {}
        for k, cols in candidates.items():
            chosen[k] = next((c for c in cols if c in final_agent_data.columns), None)

        group = final_agent_data.groupby("Sector")
        results = {}
        for sector, g in group:
            row = {"firm_count": int(len(g))}
            for metric, col in chosen.items():
                if col is None:
                    continue
                s = g[col]
                if metric == "Compliance":
                    # boolean or non-zero indicator
                    rate = float(s.mean()) if s.dtype == bool else float((s != 0).mean())
                    row["Compliance_rate"] = rate
                else:
                    row[f"{metric}_total"] = float(s.sum())
                    row[f"{metric}_average"] = float(s.mean())
                    row[f"{metric}_std"] = float(s.std())
            results[sector] = row
        return {"sector_analysis": results}
    except Exception as e:
        logger.error(f"Sector analysis error: {e}", exc_info=True)
        return {"sector_analysis_error": str(e)}

def _analyze_abatement_effectiveness(final_agent_data: pd.DataFrame) -> Dict:
    """Analyze abatement project effectiveness."""
    try:
        col = next(
            (c for c in ["Abated_Tons_Cumulative", "Abated_Tons", "Abatement", "Total_Abatement"] if c in final_agent_data.columns),
            None,
        )
        if col is None:
            return {"abatement_analysis": "No abatement data available"}
        s = final_agent_data[col].fillna(0)
        out = {
            "total_abatement": float(s.sum()),
            "average_abatement_per_firm": float(s.mean()),
            "median_abatement": float(s.median()),
            "firms_with_abatement": int((s > 0).sum()),
            "abatement_participation_rate": float((s > 0).mean()),
            "max_abatement": float(s.max()),
            "abatement_distribution": s.describe().to_dict(),
        }
        if "Sector" in final_agent_data.columns:
            sec = final_agent_data.groupby("Sector")[col].agg(["sum", "mean", "count"]).round(2)
            sec.columns = ["Total", "Average", "Firms"]
            out["abatement_by_sector"] = sec.to_dict("index")
        return {"abatement_analysis": out}
    except Exception as e:
        logger.error(f"Abatement analysis error: {e}", exc_info=True)
        return {"abatement_analysis_error": str(e)}

def _analyze_market_concentration(transactions_df: pd.DataFrame) -> Dict:
    if transactions_df.empty:
        return {}

    total_volume = transactions_df['quantity'].sum()

    # Group by buyer and sum volume
    buyer_volume = transactions_df.groupby('buyer_id')['quantity'].sum().sort_values(ascending=False)
    seller_volume = transactions_df.groupby('seller_id')['quantity'].sum().sort_values(ascending=False)

    # Calculate concentration (e.g., top 3 firms)
    top_buyers = (buyer_volume.head(3) / total_volume * 100).to_dict()
    top_sellers = (seller_volume.head(3) / total_volume * 100).to_dict()

    return {
        "top_3_buyers_pct_volume": top_buyers,
        "top_3_sellers_pct_volume": top_sellers
    }

def _analyze_economic_impacts(final_agent_data: pd.DataFrame) -> Dict:
    """Analyze economic impacts with robust column handling."""
    try:
        metrics = {
            "profit": ["Current_Profit", "Profit", "Total_Profit"],
            "cash": ["Cash", "Available_Cash", "Current_Cash"],
            "revenue": ["Revenue", "Total_Revenue", "Annual_Revenue"],
        }
        chosen = {k: next((c for c in cols if c in final_agent_data.columns), None) for k, cols in metrics.items()}
        if not any(chosen.values()):
            return {"economic_analysis": "No economic data available"}

        out = {}
        for k, col in chosen.items():
            if col is None:
                continue
            s = final_agent_data[col].fillna(0)
            out.update(
                {
                    f"total_{k}": float(s.sum()),
                    f"average_{k}": float(s.mean()),
                    f"median_{k}": float(s.median()),
                    f"{k}_distribution": s.describe().to_dict(),
                }
            )
        if chosen.get("cash"):
            s = final_agent_data[chosen["cash"]].fillna(0)
            out.update(
                {
                    "firms_with_positive_cash": int((s > 0).sum()),
                    "cash_positive_rate": float((s > 0).mean()),
                    "firms_in_financial_distress": int((s <= 0).sum()),
                }
            )
        return {"economic_analysis": out}
    except Exception as e:
        logger.error(f"Economic analysis error: {e}", exc_info=True)
        return {"economic_analysis_error": str(e)}

def _analyze_annual_summary(agent_df: pd.DataFrame, transactions_df: pd.DataFrame, model: CCTSModel) -> Dict:
    """Annual summary by calendar year using model.start_year/steps_per_year; robust firm key."""
    try:
        if agent_df.empty or "Step" not in agent_df.columns:
            return {"annual_summary": "No agent data for annual summary"}

        steps_per_year = int(getattr(model, "steps_per_year", 12))
        start_year = int(getattr(model, "start_year", 1))

        df = agent_df.copy()
        df["YearIndex"] = (df["Step"] // steps_per_year).astype(int)
        df["CalendarYear"] = start_year + df["YearIndex"]

        # Choose grouping key
        agent_key = "Firm_ID" if "Firm_ID" in df.columns else ("AgentID" if "AgentID" in df.columns else None)
        if agent_key is None:
            df["_agent_row"] = df.groupby(["CalendarYear"]).cumcount()
            agent_key = "_agent_row"

        annual_rows = []
        for yr in sorted(df["CalendarYear"].unique()):
            yr_df = df[df["CalendarYear"] == yr]

            def last_per_agent(col):
                return yr_df.groupby(agent_key)[col].last() if col in yr_df.columns else None

            totals = {}
            for name, candidates in {
                "Total_Production": ["Annual_Production", "Total_Annual_Production"],
                "Total_Emissions": ["Annual_Emissions", "Total_Annual_Emissions"],
                "Total_Abatement": ["Abated_Tons_Cumulative", "Abated_Tons", "Total_Abatement"],
                "Total_Credits": ["Credits_Owned", "Credits"],
            }.items():
                series = next((last_per_agent(c) for c in candidates if last_per_agent(c) is not None), None)
                totals[name] = float(series.sum()) if (series is not None and not series.empty) else 0.0

            prof = next((last_per_agent(c) for c in ["Current_Profit", "Profit"] if last_per_agent(c) is not None), None)
            avg_profit = float(prof.mean()) if (prof is not None and not prof.empty) else 0.0

            annual_rows.append(
                {
                    "CalendarYear": int(yr),
                    "Total_Production": totals["Total_Production"],
                    "Total_Emissions": totals["Total_Emissions"],
                    "Total_Abatement": totals["Total_Abatement"],
                    "Average_Profit": avg_profit,
                    "Total_Credits": totals["Total_Credits"],
                }
            )

        # Merge market stats by calendar year
        if transactions_df is not None and not transactions_df.empty and "step" in transactions_df.columns:
            tx = transactions_df.copy()
            tx["YearIndex"] = (tx["step"] // steps_per_year).astype(int)
            tx["CalendarYear"] = start_year + tx["YearIndex"]
            tx_summary = (
                tx.groupby("CalendarYear")
                .agg(Total_Volume=("quantity", "sum"), Average_Price=("price", "mean"))
                .reset_index()
            )
            annual_df = pd.DataFrame(annual_rows).merge(tx_summary, on="CalendarYear", how="left").fillna(0)
            annual_rows = annual_df.to_dict("records")

        return {"annual_summary": annual_rows}
    except Exception as e:
        logger.error(f"Annual summary analysis error: {e}", exc_info=True)
        return {"annual_summary_error": str(e)}

def _analyze_model_performance(model: CCTSModel) -> Dict:
    """Analyze overall model performance."""
    try:
        model_summary = model.get_model_summary()
        market_summary = model.market.get_market_summary()

        total_agents = len(model.schedule.agents)
        covered_agents = len([a for a in model.schedule.agents if getattr(a, "is_covered_entity", False)])

        perf = {
            "simulation_steps_completed": int(model.schedule.steps),
            "simulation_years_completed": int(model.schedule.steps // model.steps_per_year),
            "total_agents": int(total_agents),
            "covered_entities": int(covered_agents),
            "coverage_rate": safe_divide(covered_agents, total_agents),
            "final_carbon_price": float(model.market.carbon_price),
            "price_volatility": float(getattr(model.market, "volatility_index", 0.0)),
            "market_liquidity": int(len(model.market.buy_orders) + len(model.market.sell_orders)),
            "total_transactions": int(len(model.market.transactions_log)),
            "market_efficiency": safe_divide(len(model.market.transactions_log), model.schedule.steps),
        }
        return {"model_performance": perf, "model_summary": model_summary, "market_summary": market_summary}
    except Exception as e:
        logger.error(f"Model performance analysis error: {e}", exc_info=True)
        return {"model_performance_error": str(e)}

# ---------------------------
# Pretty report (optional)
# ---------------------------
def generate_summary_report(results: Dict) -> str:
    """Human-readable summary report."""
    try:
        lines = []
        lines += ["=" * 60, "CCTS SIMULATION ANALYSIS REPORT", "=" * 60]

        final_price = results.get("model_performance", {}).get("final_carbon_price", 0)
        compliance_rate = results.get("overall_compliance_rate", 0) * 100
        total_abatement = results.get("abatement_analysis", {}).get("total_abatement", 0)
        total_tx = results.get("market_analysis", {}).get("total_transactions", 0)
        years = results.get("model_performance", {}).get("simulation_years_completed", 0)

        lines.append(
            f"\nThis simulation ran for {years} years. Final carbon price: ₹{final_price:.2f}. "
            f"Compliance rate: {compliance_rate:.1f}%. "
            f"Total abatement: {total_abatement:.2f} tCO₂e. "
            f"Transactions: {total_tx}."
        )

        if "model_performance" in results:
            mp = results["model_performance"]
            lines += [
                "\nSIMULATION OVERVIEW:",
                f"- Years Completed: {mp.get('simulation_years_completed', 'N/A')}",
                f"- Total Agents: {mp.get('total_agents', 'N/A')}",
                f"- Covered Entities: {mp.get('covered_entities', 'N/A')}",
                f"- Final Carbon Price: ₹{mp.get('final_carbon_price', 0):.2f}",
            ]

        if "overall_compliance_rate" in results:
            lines += [
                "\nCOMPLIANCE PERFORMANCE:",
                f"- Overall Compliance Rate: {results['overall_compliance_rate']*100:.1f}%",
                f"- Total Compliance Gap: {results.get('total_net_compliance_gap', 0):.2f} t",
                f"- Actual vs Target Intensity Gap: {results.get('intensity_gap', 0):.4f}",
            ]

        if "market_analysis" in results:
            ma = results["market_analysis"]
            lines += [
                "\nMARKET PERFORMANCE:",
                f"- Total Trading Volume: {ma.get('total_trading_volume', 0):.2f} credits",
                f"- Total Transactions: {ma.get('total_transactions', 0)}",
                f"- Average Trade Price: ₹{ma.get('average_trade_price', 0):.2f}",
                f"- Price Volatility: {ma.get('price_volatility', 0)*100:.1f}%",
            ]

        if "abatement_analysis" in results:
            aa = results["abatement_analysis"]
            lines += [
                "\nABATEMENT EFFECTIVENESS:",
                f"- Total Abatement: {aa.get('total_abatement', 0):.2f} t",
                f"- Participation Rate: {aa.get('abatement_participation_rate', 0)*100:.1f}%",
                f"- Average per Firm: {aa.get('average_abatement_per_firm', 0):.2f} t",
            ]

        if "economic_analysis" in results:
            ea = results["economic_analysis"]
            if "total_profit" in ea or "cash_positive_rate" in ea:
                lines += ["\nECONOMIC IMPACTS:"]
                if "total_profit" in ea:
                    lines.append(f"- Total Profit: ₹{ea['total_profit']:.2f}")
                if "cash_positive_rate" in ea:
                    lines.append(f"- Firms with Positive Cash: {ea['cash_positive_rate']*100:.1f}%")

        # Per-sector table
        sa = results.get("sector_analysis", {})
        if isinstance(sa, dict) and sa:
            lines += ["\nPER-SECTOR PERFORMANCE:"]
            hdr = f"{'Sector':<15}{'Firms':>6}{'Compliant(%)':>15}{'Abatement(t)':>15}{'Avg Profit(₹)':>15}"
            lines += [hdr, "-" * len(hdr)]
            for sec, data in sa.items():
                firms = data.get("firm_count", 0)
                comp = data.get("Compliance_rate", 0) * 100
                abate = data.get("Abatement_total", 0)
                avg_profit = data.get("Profit_average", data.get("Current_Profit_average", 0))
                lines.append(f"{sec:<15}{firms:>6}{comp:>15.1f}{abate:>15.2f}{avg_profit:>15.2f}")

        lines.append("=" * 60)
        return "\n".join(lines)
    except Exception as e:
        logger.error(f"Summary report generation error: {e}", exc_info=True)
        return f"Error generating summary report: {str(e)}"
