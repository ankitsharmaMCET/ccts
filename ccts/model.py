# =====================================================================
# ENHANCED CCTS MODEL - CORRECTED VERSION (synchronized with agent/market)
# =====================================================================

import logging
import pandas as pd
import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from typing import Dict, List, Any
from .market import IndianCarbonMarket
from .agent import FirmAgent
from .utils import safe_divide

logger = logging.getLogger(__name__)


class CCTSModel(Model):
    """Enhanced CCTS Model with advanced features, monitoring, and reproducibility."""

    def __init__(self, firm_data: pd.DataFrame, market_params: Dict, model_config: Dict, seed: int = None):
        super().__init__()

        # ------------------------------------------------------------------
        # Reproducibility
        # ------------------------------------------------------------------
        if seed is not None:
            try:
                self.random.seed(seed)
                np.random.seed(int(seed))
            except Exception:
                pass

        # ------------------------------------------------------------------
        # Config + aliases (agents may read self.config)
        # ------------------------------------------------------------------
        self.model_config = model_config or {}
        self.config = self.model_config  # alias for compatibility
        self.start_year = int(self.model_config.get("start_year", 2024))
        self.steps_per_year = int(self.model_config.get("steps_per_year", 12))
        # Keep an always-up-to-date calendar year attribute
        self.current_year = self.start_year

        # Expose annual_reduction_rate so agents can fall back when a year target is missing
        try:
            self.annual_reduction_rate = float(self.model_config.get("annual_reduction_rate", 0.0))
        except Exception:
            self.annual_reduction_rate = 0.0

        # ------------------------------------------------------------------
        # Core components
        # ------------------------------------------------------------------
        self.schedule = RandomActivation(self)
        self.coverage_threshold = float(self.model_config.get('coverage_threshold', 25000.0))

        initial_price = float(self.model_config.get('initial_carbon_price', self.model_config.get('min_price', 5000.0)))
        self.market = IndianCarbonMarket(self, market_params, initial_price)

        # Covered sectors from input (default True ensures presence)
        try:
            self.covered_sectors = list(firm_data[firm_data['is_covered'] == True]['sector'].unique())
        except Exception:
            self.covered_sectors = list(firm_data['sector'].unique())

        self._create_agents(firm_data)
        self._setup_data_collection()
        self.yearly_summary_history: List[Dict[str, Any]] = []

        self.model_stats = {
            'total_trades': 0,
            'total_volume': 0.0,
            'total_abatement': 0.0,
            'compliance_rate': 0.0
        }

        logger.info(
            f"CCTS Model initialized with {len(self.schedule.agents)} agents, "
            f"{len(self.covered_sectors)} covered sectors"
        )

    # ------------------------------------------------------------------
    # Agents
    # ------------------------------------------------------------------
    def _create_agents(self, firm_data: pd.DataFrame) -> None:
        """Create firm agents from data with better error handling"""
        firm_records = firm_data.to_dict('records')
        created_agents = 0

        for i, data in enumerate(firm_records):
            try:
                required_fields = [
                    'firm_id', 'sector', 'baseline_production', 'baseline_emissions_intensity',
                    'cash_on_hand', 'revenue_per_unit'
                ]

                missing_fields = [field for field in required_fields if field not in data or pd.isna(data[field])]
                if missing_fields:
                    logger.warning(f"Agent {i} missing required fields: {missing_fields}, skipping")
                    continue

                # Pass full row; FirmAgent handles year-specific targets internally.
                agent = FirmAgent(i, self, data)
                self.schedule.add(agent)
                created_agents += 1

            except Exception as e:
                logger.error(f"Error creating agent {i} (firm_id: {data.get('firm_id', 'unknown')}): {e}")
                continue

        logger.info(f"Created {created_agents} firm agents out of {len(firm_records)} records")

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------
    def _setup_data_collection(self) -> None:
        """Setup comprehensive data collection with safe attribute access"""

        def safe_get_attr(agent, attr_name, default=0.0):
            return getattr(agent, attr_name, default)

        def safe_get_list_last(agent, attr_name, default=0.0):
            lst = getattr(agent, attr_name, [])
            return lst[-1] if lst else default
        
        def _calculate_total_penalties(model):
            return sum(sum(rec.penalty for rec in getattr(a, 'penalty_history', [])) for a in model.schedule.agents)


        model_reporters = {
            "Carbon_Price": lambda m: m.market.carbon_price,
            "Compliance_Rate": lambda m: m._calculate_compliance_rate(),
            "Total_Emissions": lambda m: sum(safe_get_attr(a, 'current_emissions', 0) for a in m.schedule.agents),
            "Total_Abated_Tons_Cumulative": lambda m: sum(safe_get_attr(a, 'abated_tonnes_cumulative', 0) for a in m.schedule.agents),
            "Total_Abated_Tons_Step": lambda m: sum(safe_get_attr(a, 'abated_tonnes_this_step', 0) for a in m.schedule.agents),
            "Total_Production": lambda m: sum(safe_get_attr(a, 'current_production', 0) for a in m.schedule.agents),
            "Total_Annual_Emissions": lambda m: sum(safe_get_attr(a, 'total_annual_emissions', 0) for a in m.schedule.agents),
            "Total_Annual_Production": lambda m: sum(safe_get_attr(a, 'total_annual_production', 0) for a in m.schedule.agents),
            "Market_Volume": lambda m: m.market.daily_volume,
            "Price_Volatility": lambda m: getattr(m.market, "volatility_index", 0.0),
            "Total_Transactions": lambda m: len(m.market.transactions_log),
            "Market_Depth": lambda m: len(m.market.buy_orders) + len(m.market.sell_orders),
            "Buy_Orders": lambda m: len(m.market.buy_orders),
            "Sell_Orders": lambda m: len(m.market.sell_orders),
            "Price_Spread": lambda m: m.market.market_stats.get('average_spread', 0.0),
            "market_stability_reserve_fund": lambda m: m.market.market_stability_reserve_fund,
            "market_stability_reserve_credits": lambda m: m.market.market_stability_reserve_credits,
            "Total_Penalties": _calculate_total_penalties,
        }

        agent_reporters = {
            "Firm_ID": lambda a: safe_get_attr(a, 'firm_id', 'unknown'),
            "Sector": lambda a: safe_get_attr(a, 'sector', 'unknown'),
            "Emissions": lambda a: safe_get_attr(a, 'current_emissions', 0),
            "Current_Production": lambda a: safe_get_attr(a, 'current_production', 0),
            "Annual_Emissions": lambda a: safe_get_attr(a, 'total_annual_emissions', 0),
            "Annual_Production": lambda a: safe_get_attr(a, 'total_annual_production', 0),
            "Monthly_Revenue": lambda a: safe_get_list_last(a, 'revenue_history', 0),
            "Monthly_Abatement_Cost": lambda a: safe_get_list_last(a, 'abatement_cost_history', 0),
            "Monthly_Carbon_Cost": lambda a: safe_get_list_last(a, 'carbon_cost_history', 0),
            "Monthly_Variable_Cost": lambda a: safe_get_list_last(a, 'variable_cost_history', 0),
            "Total_Penalties": lambda a: sum(rec.penalty for rec in getattr(a, 'penalty_history', [])),
            "Cash": lambda a: safe_get_attr(a, 'cash', 0),
            "Credits_Owned": lambda a: (a.total_credits_owned() if hasattr(a, "total_credits_owned") else safe_get_attr(a, 'credits_owned', 0)),
            "Abated_Tons_Cumulative": lambda a: safe_get_attr(a, 'abated_tonnes_cumulative', 0),
            "Is_Covered": lambda a: safe_get_attr(a, 'is_covered_entity', False),
            "Is_Compliant": lambda a: safe_get_attr(a, 'is_compliant', True),
            "Expected_Price": lambda a: safe_get_attr(a, 'expected_price', 0),
            "Risk_Aversion": lambda a: safe_get_attr(a, 'risk_aversion', 0.5),
            "Market_Confidence": lambda a: safe_get_attr(a, 'market_confidence', 0.5),
            "Current_Profit": lambda a: safe_get_list_last(a, 'profit_history', 0),
            "Compliance_Gap": lambda a: a.calculate_compliance_gap() if hasattr(a, 'calculate_compliance_gap') else 0,
            "Current_Target_Intensity": lambda a: safe_get_attr(a, 'current_target_intensity', 0),
            "Credits_Needed": lambda a: safe_get_attr(a, '_last_credits_needed', 0),
            "Credits_Surplus": lambda a: safe_get_attr(a, '_last_credits_surplus', 0),
            "Current_Target_Year": lambda a: (a._current_sim_year() if hasattr(a, "_current_sim_year")
                                              else int(a.model.start_year + (a.model.schedule.steps // a.model.steps_per_year))),
            "Current_Profit": lambda a: safe_get_list_last(a, 'profit_history', 0),
            "Annual_Revenue": lambda a: safe_get_list_last(a, 'revenue_history', 0),
            "Annual_Abatement_Cost": lambda a: safe_get_list_last(a, 'abatement_cost_history', 0),
            "Annual_Carbon_Cost": lambda a: safe_get_list_last(a, 'carbon_cost_history', 0),
            "Annual_Variable_Cost": lambda a: safe_get_list_last(a, 'variable_cost_history', 0), 
            "Abated_Tons_Cumulative": lambda a: safe_get_attr(a, 'abated_tonnes_cumulative', 0),                                 
        }

        self.datacollector = DataCollector(
            model_reporters=model_reporters,
            agent_reporters=agent_reporters
        )

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _calculate_compliance_rate(self) -> float:
        """Calculate current compliance rate safely"""
        try:
            covered_agents = [a for a in self.schedule.agents if getattr(a, 'is_covered_entity', False)]
            if not covered_agents:
                return 1.0
            compliant_agents = [a for a in covered_agents if getattr(a, 'is_compliant', True)]
            return safe_divide(len(compliant_agents), len(covered_agents), default=1.0)
        except Exception as e:
            logger.error(f"Compliance rate calculation error: {e}")
            return 0.0

    def _calculate_average_profit(self) -> float:
        """Calculate average profit across all agents"""
        try:
            profits = []
            for agent in self.schedule.agents:
                profit_history = getattr(agent, 'profit_history', [])
                if profit_history:
                    profits.append(profit_history[-1])
            return float(np.mean(profits)) if profits else 0.0
        except Exception as e:
            logger.error(f"Average profit calculation error: {e}")
            return 0.0

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self) -> None:
        """Enhanced model step; preserves year-end tallies before market compliance reset."""
        try:
            # Keep current_year in sync at the beginning of the tick
            self.current_year = self.start_year + (self.schedule.steps // self.steps_per_year)
            # Agents act (update production/emissions, place orders)
            self.schedule.step()

            # Determine if this tick completes a compliance year
            year_end = (self.schedule.steps % self.steps_per_year) == 0

            if year_end:
                # IMPORTANT: collect BEFORE market.step() so annual tallies are captured pre-reset
                self.datacollector.collect(self)
                # Now let the market clear and run year-end compliance/minting+expiry
                self.market.step()
                # Log the finished year's summary using the record we just collected
                self._log_annual_summary()
            else:
                # Normal tick: market clears first, then we collect
                self.market.step()
                self.datacollector.collect(self)
            # Update current_year again after market/compliance (defensive)
            self.current_year = self.start_year + (self.schedule.steps // self.steps_per_year)

            # Update aggregate model stats after the tick
            self._update_model_statistics()

        except Exception as e:
            logger.error(f"Model step {self.schedule.steps} error: {e}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _update_model_statistics(self) -> None:
        """Update comprehensive model statistics safely"""
        try:
            self.model_stats['total_trades'] = len(self.market.transactions_log)
            self.model_stats['total_volume'] = sum(t.quantity for t in self.market.transactions_log)
            self.model_stats['total_abatement'] = sum(getattr(a, 'abated_tonnes_cumulative', 0) for a in self.schedule.agents)
            self.model_stats['compliance_rate'] = self._calculate_compliance_rate()
        except Exception as e:
            logger.error(f"Model statistics update error: {e}")

    def _log_annual_summary(self) -> None:
        """Log annual summary statistics using data collected at the final step of the completed year."""
        try:
            year_end_step = self.schedule.steps  # this step just completed the year
            year_start_step = year_end_step - self.steps_per_year + 1

            # Label the completed year
            year = self.start_year + (year_end_step // self.steps_per_year) - 1

            # Use the PRE-reset record we collected earlier
            full_agent_data = self.datacollector.get_agent_vars_dataframe()
            if full_agent_data.index.get_level_values('Step').isin([year_end_step]).any():
                last_step_data = full_agent_data.xs(year_end_step, level='Step')
            else:
                logger.warning(f"No agent data found for step {year_end_step}, skipping annual summary.")
                return

            total_emissions = last_step_data.get('Annual_Emissions', pd.Series(dtype=float)).sum()
            total_production = last_step_data.get('Annual_Production', pd.Series(dtype=float)).sum()
            total_abatement = last_step_data.get('Abated_Tons_Cumulative', pd.Series(dtype=float)).sum()
            avg_intensity = safe_divide(total_emissions, total_production)

            annual_transactions = [
                t for t in self.market.transactions_log
                if year_start_step <= t.step <= year_end_step
            ]
            annual_volume = sum(t.quantity for t in annual_transactions)
            annual_trades = len(annual_transactions)

            annual_summary_data = {
                'year': year,
                'carbon_price': self.market.carbon_price,
                'total_annual_emissions': float(total_emissions),
                'total_annual_production': float(total_production),
                'total_abatement': float(total_abatement),
                'average_intensity': float(avg_intensity),
                'compliance_rate': float(self._calculate_compliance_rate()),
                'total_trades_this_year': int(annual_trades),
                'market_volume_this_year': float(annual_volume),
                'msr_fund_balance': float(self.market.market_stability_reserve_fund),
                'msr_credit_balance': float(self.market.market_stability_reserve_credits)
            }

            self.yearly_summary_history.append(annual_summary_data)

            logger.info(f"=== YEAR {year} SUMMARY ===")
            logger.info(f"Carbon Price: ₹{self.market.carbon_price:.2f}")
            logger.info(f"Total Annual Emissions: {total_emissions:.2f} tonnes")
            logger.info(f"Total Annual Production: {total_production:.2f} units")
            logger.info(f"Total Abatement: {total_abatement:.2f} tonnes")
            logger.info(f"Average Intensity: {avg_intensity:.4f} tonnes/unit")
            logger.info(f"Compliance Rate: {self._calculate_compliance_rate():.1%}")
            logger.info(f"Total Trades: {annual_trades}")
            logger.info(f"Market Volume: {annual_volume:.2f} credits")
            logger.info(f"market_stability_reserve_fund: ₹{self.market.market_stability_reserve_fund:.2f}")
            logger.info(f"market_stability_reserve_credits: {self.market.market_stability_reserve_credits:.2f} credits")
            logger.info("=" * 30)

        except Exception as e:
            logger.error(f"Annual summary logging error: {e}")

    def get_model_summary(self) -> Dict:
        """Get a comprehensive model summary for analysis"""
        try:
            return {
                'steps_completed': int(self.schedule.steps),
                'years_completed': int(self.schedule.steps // self.steps_per_year),
                'final_carbon_price': float(self.market.carbon_price),
                'total_agents': int(len(self.schedule.agents)),
                'covered_agents': int(len([a for a in self.schedule.agents if getattr(a, 'is_covered_entity', False)])),
                'model_stats': self.model_stats.copy(),
                'market_stats': self.market.market_stats.copy()
            }
        except Exception as e:
            logger.error(f"Model summary generation error: {e}")
            return {'error': str(e)}
