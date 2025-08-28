# ENHANCED FIRM AGENT WITH REAL-WORLD CONSTRAINTS (corrected)
# =====================================================================

import logging
import random
import numpy as np
import traceback
from mesa import Agent
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import json 
from .utils import MarketOrder, ComplianceRecord, Transaction, safe_divide

logger = logging.getLogger(__name__)

# Numerical tolerance for near-zero comparisons (prevents float-dust spam)
EPS = 1e-9


class FirmAgent(Agent):
    """Enhanced Indian CCTS-compliant firm agent with advanced decision-making and real-world constraints."""

    # Class-level constants for easy tuning (can be overridden by config at init)
    CASH_BUFFER_RATIO = 0.2
    RISK_PREMIUM_MULTIPLIER = 0.05
    URGENCY_PREMIUM_CAP = 0.1
    RISK_DISCOUNT_MULTIPLIER = 0.03
    VOLUME_DISCOUNT_CAP = 0.08
    MAX_CREDITS_MULTIPLIER = 100
    PROFIT_HISTORY_MAX_LEN = 12
    DECISION_HISTORY_MAX_LEN = 12
    PRICE_FORECAST_MAX_LEN = 6
    PRODUCTION_STABILITY_PENALTY_FACTOR = 0.01

    # derive from model for consistency
    @property
    def STEPS_PER_YEAR(self):
        return getattr(self.model, "steps_per_year", 12)

    # Real-world constraint parameters (defaults; bound to config in __init__)
    MAX_PROJECTS_PER_YEAR = 2
    ABATEMENT_BUDGET_RATIO = 0.3
    DIMINISHING_RETURNS_FACTOR = 0.85
    SCALING_PENALTY_FACTOR = 0.015
    MIN_CREDIT_RESERVE_RATIO = 0.2
    PROJECT_IMPLEMENTATION_VARIABILITY = 0.3

    def __init__(self, unique_id: int, model: 'CCTSModel', initial_data: Dict):
        super().__init__(unique_id, model)

        self._initialize_core_attributes(initial_data)
        self._initialize_compliance_attributes(initial_data)
        self._initialize_behavioral_attributes(initial_data)
        self._initialize_state_tracking()

        # Bind behavior knobs to config (so scenarios actually change behavior)
        cfg = getattr(self.model, "config", {}) or {}
        try:
            self.MAX_PROJECTS_PER_YEAR = int(cfg.get("max_projects_per_year", self.MAX_PROJECTS_PER_YEAR))
            self.ABATEMENT_BUDGET_RATIO = float(cfg.get("abatement_budget_ratio", self.ABATEMENT_BUDGET_RATIO))
            self.DIMINISHING_RETURNS_FACTOR = float(cfg.get("diminishing_returns_factor", self.DIMINISHING_RETURNS_FACTOR))
            self.SCALING_PENALTY_FACTOR = float(cfg.get("scaling_penalty_factor", self.SCALING_PENALTY_FACTOR))
            self.MIN_CREDIT_RESERVE_RATIO = float(cfg.get("min_credit_reserve_ratio", self.MIN_CREDIT_RESERVE_RATIO))
            self.PROJECT_IMPLEMENTATION_VARIABILITY = float(cfg.get("project_implementation_variability", self.PROJECT_IMPLEMENTATION_VARIABILITY))
        except Exception as e:
            logger.warning(f"Config binding error for Firm {self.firm_id}: {e}")

        # Real-world constraints initialization
        self.projects_in_progress: List[Dict] = []  # {project_data: dict, completion_step: int}
        self.annual_abatement_budget = 0.0
        self.last_production_change = 0.0
        self.credit_reserve_ratio = self.MIN_CREDIT_RESERVE_RATIO

        logger.debug(f"Firm {self.firm_id} initialized with baseline emissions: {self.baseline_emissions:.2f}")

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------
    def _initialize_core_attributes(self, data: Dict) -> None:
        """Initialize core firm attributes with validation"""
        self.firm_id = data['firm_id']
        self.sector = data['sector']
        self.baseline_production = float(data['baseline_production'])
        self.baseline_emissions_intensity = float(data['baseline_emissions_intensity'])

        # Year-specific target intensities from firms.csv
        # Preferred: target_intensity_YYYY ; Back-compat: target_emissions_intensity_YYYY
        self.year_targets: Dict[int, float] = {}
        for key, value in data.items():
            if not isinstance(key, str):
                continue
            if key.startswith("target_intensity_") or key.startswith("target_emissions_intensity_"):
                try:
                    year = int(key.split("_")[-1])
                    if value is not None and str(value) != "":
                        self.year_targets[year] = float(value)
                except Exception:
                    logger.warning(f"Invalid target year column '{key}' for firm {self.firm_id}")

        self._sorted_target_years = sorted(self.year_targets.keys())

        self.macc = json.loads(data.get('macc', '[]'))
        self.risk_aversion = max(0.1, min(0.9, float(data.get('risk_aversion', 0.5))))
        self.cash = max(0.0, float(data['cash_on_hand']))
        self.revenue_per_unit = float(data['revenue_per_unit'])
        self.max_production_change_percent = float(data.get('max_production_change_percent', 0.2))
        self.variable_production_cost = float(data.get('variable_production_cost', 0))

        self.min_production = self.baseline_production * (1 - self.max_production_change_percent)
        self.max_production = self.baseline_production * (1 + self.max_production_change_percent)
        self.baseline_emissions = self.baseline_production * self.baseline_emissions_intensity

        if self.baseline_production <= 0 or self.revenue_per_unit <= 0:
            raise ValueError(f"Firm {self.firm_id}: Production and revenue must be positive")

    def _initialize_compliance_attributes(self, data: Dict) -> None:
        """Initialize CCTS compliance attributes"""
        self.is_covered_entity = bool(data.get('is_covered', False))
        self.coverage_threshold = self.model.coverage_threshold
        # Credit storage by vintage year
        self.credit_vintages: Dict[int, float] = {}
        self.penalty_history: List[float] = []
        self.is_compliant = True
        self.verification_cost = getattr(self.model.market, "verification_cost", 0.0)
        # Annual aggregates for the current compliance year
        self.total_annual_emissions = 0.0
        self.total_annual_production = 0.0
        # Borrowing / carryover deficit (principal only; interest applied at next year)
        self.borrowed_credits = 0.0
        self.noncompliance_streak = 0


    def _initialize_behavioral_attributes(self, data: Dict) -> None:
        """Initialize behavioral and learning attributes"""
        self.learning_rate = max(0.0, min(1.0, float(data.get('learning_rate', 0.1))))
        self.technology_readiness = max(0.0, min(1.0, float(data.get('technology_readiness', 0.5))))
        self.strategy_memory: List[Dict] = []
        self.market_confidence = 0.5
        self.strategic_planning_horizon = 12
        self.adaptive_behavior = True

    def _initialize_state_tracking(self) -> None:
        """Initialize state tracking variables"""
        self.current_production = self.baseline_production
        self.current_emissions = self.baseline_emissions
        # Keep a scalar mirror but derive from vintages when possible
        self.credits_owned = 0.0
        self.abated_tonnes_cumulative = 0.0
        self.abated_tonnes_this_step = 0.0
        self.completed_projects: List[Dict] = []
        self.revenue_history: List[float] = []
        self.variable_cost_history: List[float] = []
        self.carbon_cost_history: List[float] = []
        self.abatement_cost_history: List[float] = []
        self.compliance_gap_history: List[float] = []

        # Initialize target intensity for the starting year
        self.current_target_intensity = self._target_intensity_for_year(self._current_sim_year())

        self.profit_history: List[float] = []
        self.decision_history: List[Dict] = []
        self.expected_price = self.model.market.carbon_price
        self.price_forecast: List[float] = []

        self._last_credits_needed = 0.0
        self._last_credits_surplus = 0.0
        self._total_revenue = 0.0
        self._total_costs = 0.0

    # -------------------------------------------------------------
    # Target intensity logic (year-specific with safe fallback)
    # -------------------------------------------------------------
    def _current_sim_year(self) -> int:
        """Return the calendar year currently simulated (completed/label year)."""
        try:
            return int(self.model.start_year + (self.model.schedule.steps // self.model.steps_per_year))
        except Exception:
            return int(self.model.start_year)

    def _target_intensity_for_year(self, year: int) -> float:
        """Pick the target intensity for a given year.
        Priority: exact match → nearest past target year → fallback to baseline or annual reduction rule.
        """
        # Exact match
        if year in self.year_targets:
            return self.year_targets[year]
        # Nearest past
        if self._sorted_target_years:
            past = [y for y in self._sorted_target_years if y <= year]
            if past:
                return self.year_targets[max(past)]
        # Fallback: use annual_reduction_rate if present, else baseline
        try:
            r = float(
                getattr(self.model, "annual_reduction_rate", 0.0)
                or getattr(self.model, "config", {}).get("annual_reduction_rate", 0.0)
                or 0.0
            )
            years_since_start = max(0, year - int(self.model.start_year))
            return self.baseline_emissions_intensity * ((1.0 - r) ** years_since_start)
        except Exception:
            return self.baseline_emissions_intensity

    # Public wrappers used by market/compliance
    def get_target_intensity(self, year: int) -> float:
        return float(self._target_intensity_for_year(year))

    def get_annual_totals(self, year: int) -> Tuple[float, float]:
        """Return (Q, E) = (verified production, verified emissions) for given year.
        Note: we track only the current compliance-year aggregates; this function
        assumes it is called at year-end for the current year.
        """
        return float(self.total_annual_production), float(self.total_annual_emissions)
    

    # NEW: Return “verified” totals (with measurement uncertainty if enabled)
    def get_verified_annual_totals(self, year: int) -> Tuple[float, float]:
        Q = float(self.total_annual_production)
        E = float(self.total_annual_emissions)
        cfg = getattr(self.model, "config", {}) or {}
        mode = str(cfg.get("measurement_uncertainty_mode", "normal")).lower()
        sigma = float(cfg.get("measurement_uncertainty", 0.0))
        if sigma > 0 and mode == "normal":
            # multiplicative noise on E only (typical MRV handling)
            eps = self.model.random.gauss(0.0, sigma)
            E = max(0.0, E * (1.0 + eps))
        return Q, E

    # NEW: optional annual decay on banked vintages (haircut)
    def apply_banking_decay(self, rate: float) -> None:
        if rate <= 0.0 or not getattr(self, "credit_vintages", None):
            return
        for y in list(self.credit_vintages.keys()):
            self.credit_vintages[y] *= max(0.0, 1.0 - rate)
        self._recompute_credits_owned()

    # -----------------------------------------------------------------
    # Project mechanics
    # -----------------------------------------------------------------
    def _check_project_completions(self) -> None:
        """Check and complete ongoing projects"""
        completed_indices = []
        for idx, project in enumerate(self.projects_in_progress):
            if self.model.schedule.steps >= project['completion_step']:
                # Apply abatement
                abatement = project['project_data']['abatement_potential']
                self.abated_tonnes_cumulative += abatement
                self.completed_projects.append(project['project_data'])
                completed_indices.append(idx)
                logger.info(f"Firm {self.firm_id} completed project {project['project_data']['name']}")

        # Remove completed projects in reverse order
        for idx in sorted(completed_indices, reverse=True):
            del self.projects_in_progress[idx]

    def _apply_diminishing_returns(self, project: Dict) -> Dict:
        """Reduce benefits of similar projects"""
        project_type = project.get('type', 'general')
        similar_count = sum(1 for p in self.completed_projects
                            if p.get('type', 'general') == project_type)

        if similar_count > 0:
            reduction_factor = self.DIMINISHING_RETURNS_FACTOR ** similar_count
            project = project.copy()
            project['abatement_potential'] *= reduction_factor
            logger.debug(f"Applying {reduction_factor:.2f} DR reduction to {project['name']}")

        return project

    def _has_similar_project(self, project: Dict) -> bool:
        """Check if similar projects exist"""
        project_type = project.get('type', 'general')
        return any(p.get('type', 'general') == project_type
                   for p in self.completed_projects)

    # -----------------------------------------------------------------
    # Credits & compliance
    # -----------------------------------------------------------------
    def _recompute_credits_owned(self) -> float:
        """Keep scalar credits_owned consistent with vintage books."""
        total = float(sum(self.credit_vintages.values())) if hasattr(self, "credit_vintages") else 0.0
        self.credits_owned = total
        return total

    def total_credits_owned(self) -> float:
        """
        Sum of all valid credits currently owned across vintages.
        Assumes self.credit_vintages is a dict: {year: qty}.
        If you have an expiry rule, ensure expired vintages are pruned before this is called.
        """
        try:
            return float(sum(max(0.0, qty) for qty in self.credit_vintages.values()))
        except Exception:
            return 0.0

    def total_banked_credits(self, exclude_year: int) -> float:
        """Credits with vintage strictly before exclude_year (banked from past years)."""
        try:
            return float(sum(max(0.0, qty) for y, qty in self.credit_vintages.items() if int(y) < int(exclude_year)))
        except Exception:
            return 0.0

    def prune_expired_vintages(self, current_year: int, validity_years: int) -> None:
        """
        Remove expired vintages in-place.
        Convention: credits minted in year y are valid THROUGH y+validity_years.
        They expire starting y+validity_years+1.
        """
        try:
            valid_through = {y: (y + validity_years) for y in list(self.credit_vintages.keys())}
            for y, qty in list(self.credit_vintages.items()):
                if current_year > valid_through[y]:
                    self.credit_vintages.pop(y, None)
        except Exception:
            pass

    def surrender_credits_fifo(self, required: float) -> float:
        """Surrender credits FIFO by vintage; returns amount actually surrendered."""
        if required <= 1e-12:
            return 0.0
        surrendered = 0.0
        for v in sorted(list(self.credit_vintages.keys())):
            if surrendered >= required - 1e-12:
                break
            take = min(self.credit_vintages[v], required - surrendered)
            if take > 0:
                self.credit_vintages[v] -= take
                surrendered += take
                if self.credit_vintages[v] <= 1e-12:
                    del self.credit_vintages[v]
        self._recompute_credits_owned()
        return surrendered

    def expire_credits(self, current_year: int) -> float:
        """Expire old credits and return the amount expired"""
        expired_credits = 0.0
        expired_vintages = []
        validity = getattr(self.model.market, "credit_validity", None)
        if validity is None:
            validity = int(getattr(self.model, "credit_lifetime_years", 2))

        for vintage_year in list(self.credit_vintages.keys()):
            if current_year > vintage_year + int(validity):
                expired_credits += self.credit_vintages[vintage_year]
                expired_vintages.append(vintage_year)

        for vintage in expired_vintages:
            del self.credit_vintages[vintage]

        self._recompute_credits_owned()

        if expired_credits > 0:
            logger.debug(f"Firm {self.firm_id}: {expired_credits} credits expired")

        return expired_credits

    def calculate_compliance_gap(self) -> float:
        """
        Calculate compliance gap using cumulative annual data.
        Positive => deficit (needs to buy); Negative => surplus (post-mint supply at year-end).
        """
        if not self.is_covered_entity:
            return 0.0

        # Ensure current target reflects the current year
        self.current_target_intensity = self._target_intensity_for_year(self._current_sim_year())

        allowed_emissions = float(self.current_target_intensity) * float(self.total_annual_production)
        gross_gap = float(self.total_annual_emissions) - allowed_emissions

        # Treat tiny gaps as zero (float noise)
        return 0.0 if abs(gross_gap) <= EPS else gross_gap

    def update_coverage_status(self) -> None:
        """Update coverage status based on sector and baseline emissions threshold"""
        self.is_covered_entity = (
            self.sector in self.model.covered_sectors and
            self.baseline_emissions >= self.coverage_threshold
        )

    def get_available_projects(self) -> List[Dict]:
        """Return list of abatement projects not yet completed or in progress"""
        completed_names = {p['name'] for p in self.completed_projects}
        in_progress_names = {p['project_data']['name'] for p in self.projects_in_progress}
        return [proj for proj in self.macc
                if proj['name'] not in completed_names
                and proj['name'] not in in_progress_names]

    def implement_abatement_portfolio(self, portfolio: List[Dict]) -> float:
        """Implement abatement projects with time delays and constraints"""
        total_abatement = 0.0

        for project in portfolio:
            abatement_potential = project.get('abatement_potential', 0)
            cost_per_tonne = project.get('cost_per_tonne', 0)
            total_cost = abatement_potential * cost_per_tonne

            if self.cash >= total_cost:
                # Implementation time with variability
                base_time = project.get('implementation_time', 6)  # Default 6 months
                variability = random.uniform(-self.PROJECT_IMPLEMENTATION_VARIABILITY,
                                             self.PROJECT_IMPLEMENTATION_VARIABILITY)
                implementation_time = max(3, int(base_time * (1 + variability)))

                completion_step = self.model.schedule.steps + implementation_time

                self.projects_in_progress.append({
                    'project_data': project.copy(),
                    'completion_step': completion_step,
                    'cost_incurred': total_cost
                })

                # Pay upfront
                self.cash -= total_cost
                logger.info(
                    f"Firm {self.firm_id} started {project['name']}: "
                    f"Completion in {implementation_time} months"
                )
            else:
                logger.debug(f"Firm {self.firm_id} cannot afford project {project['name']}")
                break

        return total_abatement

    # -----------------------------------------------------------------
    # Expectations and decisions
    # -----------------------------------------------------------------
    def update_price_expectations(self) -> None:
        """Update expected carbon price using recent market price history and adjustments"""
        try:
            price_history = self.model.market.price_history
            if len(price_history) > 3:
                weights = np.exp(np.linspace(-1, 0, min(5, len(price_history))))  # EW weights
                recent_prices = [p for p in price_history[-len(weights):] if p > 0]
                if recent_prices:
                    self.expected_price = np.average(recent_prices, weights=weights[-len(recent_prices):])
                else:
                    self.expected_price = self.model.market.carbon_price
            else:
                self.expected_price = max(1.0, self.model.market.carbon_price)

            risk_adjustment = (self.risk_aversion - 0.5) * 0.1
            self.expected_price *= (1 + risk_adjustment)

            volatility = getattr(self.model.market, 'volatility_index', 0.0)
            confidence_adjustment = (1 - volatility) * self.market_confidence * 0.05
            self.expected_price *= (1 + confidence_adjustment)

            self.expected_price = max(1.0, self.expected_price)

            self.price_forecast.append(self.expected_price)
            if len(self.price_forecast) > self.PRICE_FORECAST_MAX_LEN:
                self.price_forecast = self.price_forecast[-self.PRICE_FORECAST_MAX_LEN:]

        except Exception as e:
            logger.error(f"Price expectation update error for {self.firm_id}: {e}")
            self.expected_price = max(1.0, self.model.market.carbon_price)

    def select_optimal_abatement_portfolio(self) -> Tuple[List[Dict], float]:
        """Select abatement projects with real-world constraints"""
        available_projects = self.get_available_projects()
        if not available_projects:
            return [], 0.0

        # Apply constraints
        active_projects = len(self.completed_projects) + len(self.projects_in_progress)
        if active_projects >= self.MAX_PROJECTS_PER_YEAR:
            logger.debug(f"Firm {self.firm_id} reached project limit ({self.MAX_PROJECTS_PER_YEAR})")
            return [], 0.0

        # Calculate annual budget (once per year)
        if self.model.schedule.steps % self.model.steps_per_year == 0:
            self.annual_abatement_budget = self.cash * self.ABATEMENT_BUDGET_RATIO

        sorted_projects = sorted(available_projects, key=lambda x: x.get('cost_per_tonne', float('inf')))
        optimal_portfolio: List[Dict] = []
        total_cost = 0.0
        total_abatement = 0.0

        for project in sorted_projects:
            # Diminishing returns for similar projects
            if self._has_similar_project(project):
                project = self._apply_diminishing_returns(project)

            abatement_potential = project.get('abatement_potential', 0)
            project_cost = abatement_potential * project.get('cost_per_tonne', float('inf'))

            # Check NPV and constraints
            if self._calculate_project_npv(project) > 0:
                # Check budget constraint
                if total_cost + project_cost > self.annual_abatement_budget:
                    logger.debug(f"Firm {self.firm_id} abatement budget exceeded")
                    break

                # Check project limit
                if len(optimal_portfolio) >= (self.MAX_PROJECTS_PER_YEAR - active_projects):
                    logger.debug(f"Firm {self.firm_id} project limit reached")
                    break

                optimal_portfolio.append(project)
                total_cost += project_cost
                total_abatement += abatement_potential

        portfolio_npv = (total_abatement * self.expected_price) - total_cost
        return optimal_portfolio, portfolio_npv

    def _calculate_project_npv(self, project: Dict) -> float:
        """Calculate Net Present Value of an abatement project"""
        try:
            cost_per_tonne = project.get('cost_per_tonne', float('inf'))
            abatement_potential = project.get('abatement_potential', 0)

            if cost_per_tonne == float('inf') or abatement_potential <= 0:
                return -float('inf')

            implementation_success_rate = self.technology_readiness * 0.8 + 0.2
            risk_adjusted_price = self.expected_price * implementation_success_rate

            benefits = abatement_potential * risk_adjusted_price
            costs = abatement_potential * cost_per_tonne

            return benefits - costs

        except Exception as e:
            logger.error(f"NPV calculation error for {self.firm_id}: {e}")
            return -float('inf')

    def optimize_production_decision(self) -> float:
        """Optimize production level with scaling costs and cash constraints"""
        try:
            max_affordable_cash = max(0, self.cash - (self.cash * self.CASH_BUFFER_RATIO))

            # EFFECTIVE INTENSITY CALCULATION:
            # Abatement reduces baseline intensity by: abated_tonnes / baseline_production
            # This intensity reduction applies to any production level
            # Current emissions = current_production * (baseline_intensity - intensity_reduction)


            # Per-unit production cost should include full per-unit carbon cost
            estimated_cost_per_unit = self.variable_production_cost + (effective_intensity * self.expected_price)
            if estimated_cost_per_unit > 0:
                constrained_max_production = min(
                    self.max_production,
                    safe_divide(max_affordable_cash, estimated_cost_per_unit * self.STEPS_PER_YEAR, self.max_production)
                )
            else:
                constrained_max_production = self.max_production

            constrained_max_production = max(self.min_production, constrained_max_production)

            # Production scaling penalty function
            def scaling_penalty(production):
                """Quadratic penalty for large production changes"""
                baseline = self.baseline_production
                change_ratio = abs(production - baseline) / baseline

                # No penalty for changes < 10%
                if change_ratio <= 0.1:
                    return 0

                # Quadratic penalty for larger changes
                excess_change = change_ratio - 0.1
                return self.SCALING_PENALTY_FACTOR * (excess_change ** 2) * production * self.revenue_per_unit

            def profit_function(production_array: np.ndarray) -> float:
                prod = float(production_array[0])

                revenue = prod * self.revenue_per_unit
                variable_costs = prod * self.variable_production_cost

                # Ensure target reflects the current year for this step
                self.current_target_intensity = self._target_intensity_for_year(self._current_sim_year())

                # Calculate effective intensity for this production level
                if prod > 0:
                    abatement_intensity_reduction = safe_divide(
                        self.abated_tonnes_cumulative, 
                        self.baseline_production, 
                        0.0
                    )
                    effective_intensity = max(0.0, self.baseline_emissions_intensity - abatement_intensity_reduction)
                    emissions_step = prod * effective_intensity
                else:
                    emissions_step = 0.0
                allowed_step = self.current_target_intensity * prod
                excess_emissions = max(0.0, emissions_step - allowed_step)
                carbon_costs = excess_emissions * self.expected_price

                # Add scaling penalty
                penalty = scaling_penalty(prod)

                profit = revenue - variable_costs - carbon_costs - penalty

                production_change = abs(prod - self.baseline_production) / self.baseline_production
                stability_penalty = production_change * self.risk_aversion * revenue * self.PRODUCTION_STABILITY_PENALTY_FACTOR

                return -(profit - stability_penalty)  # Minimize negative profit

            bounds = [(self.min_production, constrained_max_production)]
            initial_guess = [min(constrained_max_production, self.current_production)]

            result = minimize(
                profit_function,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                options={'ftol': 1e-6, 'maxiter': 100}
            )

            if result.success:
                optimized_production = float(result.x[0])
                return max(self.min_production, min(constrained_max_production, optimized_production))
            else:
                logger.warning(f"Production optimization failed for {self.firm_id}, using constrained baseline")
                return min(constrained_max_production, self.baseline_production)

        except Exception as e:
            logger.error(f"Production optimization error for {self.firm_id}: {e}")
            return self.baseline_production

    # -----------------------------------------------------------------
    # Trading (strategy + order placement)
    # -----------------------------------------------------------------
    def calculate_trading_strategy(self) -> Tuple[float, float]:
        """
        Calculate trading strategy with proper credit availability checks.

        Changes:
        - BUY: net out already-owned credits before sizing purchases, so we don't over-buy.
        - SELL: sell only already-owned credits (banked), with a reserve; no pre-mint leakage.
        - Optional hygiene: prune expired vintages if helper exists.
        """
        if not self.is_covered_entity:
            # Keep analysis helpers consistent for uncovered entities
            self._last_credits_needed = 0.0
            self._last_credits_surplus = 0.0
            return 0.0, 0.0

        # Optional hygiene: prune expired vintages before counting owned credits
        try:
            if hasattr(self, "prune_expired_vintages"):
                current_year = int(getattr(self.model, "current_year", 0))
                validity_years = int(getattr(self.model, "vintage_validity_years", 2))
                if current_year > 0:
                    self.prune_expired_vintages(current_year=current_year, validity_years=validity_years)
        except Exception:
            # Never fail trading logic because of expiry pruning
            pass

        try:
            compliance_gap = self.calculate_compliance_gap()  # >0 deficit (need to buy), <0 surplus (can sell)

            credits_needed = 0.0
            credits_surplus = 0.0

            if compliance_gap > 0.0:
                # --- BUY CASE ---
                # Net out already-owned credits before buying (prevents over-buying).
                try:
                    owned = float(self.total_credits_owned())
                except Exception:
                    owned = 0.0
                net_deficit = float(max(0.0, compliance_gap - owned))
                credits_needed = net_deficit

            elif compliance_gap < 0.0:
                # --- SELL CASE ---
                # Only consider already-owned (banked) credits; no pre-mint leakage.
                # If a banked-only helper exists, prefer it; otherwise fall back to total owned.
                try:
                    if hasattr(self, "total_banked_credits") and hasattr(self, "model") and hasattr(self.model, "current_year"):
                        available_for_sale = float(self.total_banked_credits(exclude_year=int(self.model.current_year)))
                    else:
                        available_for_sale = float(self.total_credits_owned())
                except Exception:
                    available_for_sale = 0.0

                # Apply reserve to what we actually own
                reserve_amount = max(0.0, available_for_sale * float(self.credit_reserve_ratio))
                sellable_credits = max(0.0, available_for_sale - reserve_amount)

                # Actual surplus we can list for sale is bounded by owned/banked supply
                credits_surplus = sellable_credits

                # Update reserve ratio based on market confidence (bounded)
                try:
                    if float(self.market_confidence) > 0.7:
                        self.credit_reserve_ratio = max(float(self.MIN_CREDIT_RESERVE_RATIO), float(self.credit_reserve_ratio) - 0.05)
                    else:
                        self.credit_reserve_ratio = min(0.5, float(self.credit_reserve_ratio) + 0.05)
                except Exception:
                    # Keep whatever reserve ratio we already had
                    pass

                # Clamp to safe bounds [MIN, 0.5]
                self.credit_reserve_ratio = max(float(self.MIN_CREDIT_RESERVE_RATIO), min(0.5, float(self.credit_reserve_ratio)))

            # Store for diagnostics/plots
            self._last_credits_needed = float(credits_needed)
            self._last_credits_surplus = float(credits_surplus)

            return float(credits_needed), float(credits_surplus)

        except Exception:
            # Fail-safe: never crash the step due to trading logic
            self._last_credits_needed = 0.0
            self._last_credits_surplus = 0.0
            return 0.0, 0.0

    def place_market_orders(self, credits_needed: float, credits_surplus: float) -> None:
        """Place buy and sell orders in the market with improved validation"""
        if not self.is_covered_entity:
            return

        min_cash_buffer = self.CASH_BUFFER_RATIO * self.cash

        try:
            # Handle buy orders
            if credits_needed > 0:
                max_affordable = safe_divide(
                    max(0, self.cash - min_cash_buffer),
                    self.expected_price
                )

                order_quantity = min(credits_needed, max_affordable)

                if order_quantity > self.model.market.minimum_trade_size:
                    base_price = self.expected_price
                    risk_premium = self.risk_aversion * base_price * self.RISK_PREMIUM_MULTIPLIER
                    urgency_premium = min(
                        safe_divide(order_quantity, self.model.market.market_stability_reserve_credits, 0.0),
                        self.URGENCY_PREMIUM_CAP
                    ) * base_price

                    limit_price = base_price + risk_premium + urgency_premium
                    limit_price = max(self.model.market.min_price, min(self.model.market.max_price, limit_price))

                    self.model.market.place_order('buy', self, order_quantity, limit_price)
                else:
                    logger.debug(f"Firm {self.firm_id} has insufficient cash for buy order ({self.cash:.2f})")

            # Handle sell orders with enhanced validation
            if credits_surplus > 0:
                # Only sell what we truly own (after reserve), and only *banked* (pre-current-year) credits
                if hasattr(self, "model") and hasattr(self.model, "current_year"):
                    banked = self.total_banked_credits(exclude_year=int(self.model.current_year))
                else:
                    banked = self.total_credits_owned()
                actual_available = min(credits_surplus, banked)

                # Apply minimum trade size check
                if actual_available > self.model.market.minimum_trade_size:
                    base_price = self.expected_price
                    risk_discount = self.risk_aversion * base_price * self.RISK_DISCOUNT_MULTIPLIER
                    volume_discount = min(
                        safe_divide(actual_available, self.model.market.market_stability_reserve_credits, 0.0),
                        self.VOLUME_DISCOUNT_CAP
                    ) * base_price

                    limit_price = base_price - risk_discount - volume_discount
                    limit_price = max(self.model.market.min_price, min(self.model.market.max_price, limit_price))

                    success = self.model.market.place_order('sell', self, actual_available, limit_price)
                    if success:
                        logger.debug(f"Firm {self.firm_id} placed sell order: {actual_available:.2f} credits @ ₹{limit_price:.2f}")
                    else:
                        logger.warning(f"Firm {self.firm_id} failed to place sell order for {actual_available:.2f} credits")
                else:
                    if credits_surplus > 0:
                        logger.debug(
                            f"Firm {self.firm_id} surplus too small to trade: "
                            f"calc={credits_surplus:.6f}, banked={banked:.6f}, "
                            f"min_trade={self.model.market.minimum_trade_size:.6f}"
                        )

        except Exception as e:
            logger.error(f"Order placement error for {self.firm_id}: {e}")

    # -----------------------------------------------------------------
    # Learning / adaptation
    # -----------------------------------------------------------------
    def learn_from_experience(self) -> None:
        """Adjust behavior based on recent performance"""
        if not self.adaptive_behavior or len(self.profit_history) < 3:
            return

        try:
            recent_profits = self.profit_history[-3:]
            profit_trend = np.mean(np.diff(recent_profits))

            if profit_trend > 0:
                self.market_confidence = min(1.0, self.market_confidence + 0.02)
                self.risk_aversion = max(0.1, self.risk_aversion - 0.01)
            else:
                self.market_confidence = max(0.0, self.market_confidence - 0.02)
                self.risk_aversion = min(0.9, self.risk_aversion + 0.01)

            if len(self.decision_history) >= 2:
                recent_decisions = self.decision_history[-2:]
                if all(d.get('npv_abate', 0) > 0 for d in recent_decisions):
                    self.technology_readiness = min(1.0, self.technology_readiness + 0.01)

        except Exception as e:
            logger.error(f"Learning error for {self.firm_id}: {e}")

    # -----------------------------------------------------------------
    # Credit vintage helpers
    # -----------------------------------------------------------------
    def add_credits_to_vintage(self, quantity: float, vintage_year: int) -> None:
        """Add credits to the appropriate vintage year"""
        if vintage_year not in self.credit_vintages:
            self.credit_vintages[vintage_year] = 0.0
        self.credit_vintages[vintage_year] += float(quantity)
        self._recompute_credits_owned()

    # Diagnostic utility — optional but helpful
    def debug_credit_balance(self) -> Dict:
        """Diagnostic method to understand credit balance issues"""
        try:
            compliance_gap = self.calculate_compliance_gap()

            debug_info = {
                'firm_id': self.firm_id,
                'step': self.model.schedule.steps,
                'credits_owned': self.total_credits_owned(),
                'compliance_gap': compliance_gap,
                'annual_emissions': self.total_annual_emissions,
                'annual_production': self.total_annual_production,
                'target_intensity': self.current_target_intensity,
                'allowed_emissions': float(self.current_target_intensity) * float(self.total_annual_production),
                'credit_vintages': dict(self.credit_vintages) if hasattr(self, 'credit_vintages') else {},
                'abated_cumulative': self.abated_tonnes_cumulative,
                'last_credits_needed': self._last_credits_needed,
                'last_credits_surplus': self._last_credits_surplus,
                'credit_reserve_ratio': self.credit_reserve_ratio
            }

            # Log only if the apparent surplus truly exceeds owned credits (beyond EPS)
            if (debug_info['last_credits_surplus'] - debug_info['credits_owned']) > EPS:
                logger.warning(f"CREDIT DISCREPANCY for Firm {self.firm_id}:")
                for key, value in debug_info.items():
                    logger.warning(f"  {key}: {value}")

            return debug_info

        except Exception as e:
            logger.error(f"Debug credit balance error for {self.firm_id}: {e}")
            return {'error': str(e)}

    # -----------------------------------------------------------------
    # Step
    # -----------------------------------------------------------------
    def step(self) -> None:
        """Execute one simulation step with project tracking"""
        try:
            # 0) Project completions first
            self._check_project_completions()

            # 0.a) Ensure target intensity is up-to-date for this step/year
            self.current_target_intensity = self._target_intensity_for_year(self._current_sim_year())

            # 1) Update price expectations
            self.update_price_expectations()

            # 2) Choose & implement abatement portfolio (projects) — credits added on completion
            portfolio, npv = self.select_optimal_abatement_portfolio()
            abatement_costs_this_step = 0.0
            if portfolio:
                abatement_costs_this_step = sum(
                    p['abatement_potential'] * p['cost_per_tonne'] for p in portfolio
                )
                self.implement_abatement_portfolio(portfolio)

            # 3) Optimize production
            production = self.optimize_production_decision()
            self.current_production = production

            # 4) Update emissions
            # EFFECTIVE INTENSITY CALCULATION:
            # Abatement reduces baseline intensity by: abated_tonnes / baseline_production
            # This intensity reduction applies to any production level
            # Current emissions = current_production * (baseline_intensity - intensity_reduction)
            # Calculate effective intensity using current production
            # This gives us the true current intensity after abatement
            if production > 0:
                # Scale abated tonnes to current production level to get intensity reduction
                abatement_intensity_reduction = safe_divide(
                    self.abated_tonnes_cumulative, 
                    self.baseline_production, 
                    0.0
                )
                effective_intensity = max(0.0, self.baseline_emissions_intensity - abatement_intensity_reduction)
            else:
                effective_intensity = self.baseline_emissions_intensity
            self.current_emissions = production * effective_intensity

            # 5) Update annual aggregates
            self.total_annual_emissions += self.current_emissions
            self.total_annual_production += self.current_production

            # 6) Trading need/surplus + order placement
            credits_needed, credits_surplus = self.calculate_trading_strategy()
            self.place_market_orders(credits_needed, credits_surplus)

            # 7) Compliance helpers (for DataCollector & analysis)
            gap_now = self.calculate_compliance_gap()
            if gap_now > 0:
                owned_now = self.total_credits_owned()
                self._last_credits_needed = max(0.0, gap_now - owned_now)
                self._last_credits_surplus = 0.0
            else:
                self._last_credits_needed = 0.0
                banked_now = self.total_banked_credits(exclude_year=self._current_sim_year())
                self._last_credits_surplus = banked_now  # banked only, no pre-mint leakage
            self.is_compliant = (self._last_credits_needed <= EPS)

            # Keep a history of the raw gap
            self.compliance_gap_history.append(gap_now)

            # 8) Financials (operating profit; trade cashflows posted in market)
            revenue = production * self.revenue_per_unit
            variable_costs = production * self.variable_production_cost

            # Optional accrual proxy (guarded to avoid double-counting with trades)
            use_proxy = bool(getattr(self.model, "config", {}).get("use_proxy_carbon_costs_in_profit", False))
            expected_carbon_costs = 0.0
            if use_proxy:
                monthly_allowed = self.current_target_intensity * self.current_production
                excess_emissions = max(0.0, self.current_emissions - monthly_allowed)
                expected_carbon_costs = excess_emissions * self.expected_price

            monthly_profit = revenue - variable_costs - abatement_costs_this_step - expected_carbon_costs

            self.cash += monthly_profit
            self.profit_history.append(monthly_profit)
            self.revenue_history.append(revenue)
            self.variable_cost_history.append(variable_costs)
            self.carbon_cost_history.append(expected_carbon_costs)
            self.abatement_cost_history.append(abatement_costs_this_step)

            self.decision_history.append({
                'step': self.model.schedule.steps,
                'npv_abate': npv,
                'decision': 'optimize'
            })

            if len(self.profit_history) > self.PROFIT_HISTORY_MAX_LEN:
                self.profit_history = self.profit_history[-self.PROFIT_HISTORY_MAX_LEN:]
            if len(self.decision_history) > self.DECISION_HISTORY_MAX_LEN:
                self.decision_history = self.decision_history[-self.DECISION_HISTORY_MAX_LEN:]

            # 9) Adaptation
            self.learn_from_experience()

            # Optional diagnostics
            if self._last_credits_surplus > 0:
                self.debug_credit_balance()

        except Exception as e:
            logger.error(f"Step error for Firm {self.firm_id}: {e}")
            traceback.print_exc()

    # -----------------------------------------------------------------
    # Optional hook: call this after annual compliance to hard-reset tallies
    # -----------------------------------------------------------------
    def end_of_year_reset(self) -> None:
        """Optional hard reset of annual tallies (call from market after compliance)."""
        # Zero annual tallies
        self.total_annual_emissions = 0.0
        self.total_annual_production = 0.0
        self.abated_tonnes_this_step = 0.0
        # Reset annual budget
        self.annual_abatement_budget = 0.0
        # Roll targets forward for the new year
        try:
            # At end-of-year, schedule.steps is a multiple of steps_per_year
            self.current_target_intensity = self._target_intensity_for_year(self._current_sim_year())
        except Exception:
            # Keep previous targets on failure
            pass
