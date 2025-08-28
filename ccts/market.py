# =====================================================================
# ENHANCED CARBON MARKET (CORRECTED VERSION)
# =====================================================================

import logging
import random
import numpy as np
from typing import Dict, List, Tuple, Optional
from .utils import MarketOrder, Transaction, ComplianceRecord, safe_divide

logger = logging.getLogger(__name__)

def _agent_name(agent):
    """Get agent name safely"""
    if agent is None:
        return "MarketStabilityReserve"
    return getattr(agent, "firm_id", f"Agent_{getattr(agent, 'unique_id', 'unknown')}")

class IndianCarbonMarket:
    """Indian CCTS-compliant carbon market with enhanced error handling"""

    def __init__(self, model: 'CCTSModel', market_params: Dict, initial_price: float):
        self.model = model
        self.carbon_price = max(1.0, initial_price)
        self.buy_orders: List[MarketOrder] = []
        self.sell_orders: List[MarketOrder] = []
        self.transactions_log: List[Transaction] = []
        self.price_history = [self.carbon_price]
        self.volatility_index = 0.0
        self.daily_volume = 0.0
        self.steps_per_year = getattr(self.model, "steps_per_year", 12)
        self.order_book_history = []
        # prevent multiple emergency injections within the same step
        self._emergency_last_step = -1
        try:
            seed_val = self.model.random.randint(0, 2**31 - 1)
            random.seed(seed_val)
        except Exception:
            pass

        self._load_market_parameters(market_params)

        self.market_stats = {
            'total_volume': 0.0,
            'average_spread': 0.0,
            'market_depth': 0.0,
            'price_volatility': 0.0
        }

        logger.info(f"Carbon market initialized with price: ₹{self.carbon_price:.2f}")

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------
    def _load_market_parameters(self, market_params: Dict) -> None:
        """Load and validate market parameters with safe defaults"""
        try:
            self.order_expiry_steps = int(market_params.get("order_expiry_steps", 12))
            self.maximum_order_size = float(market_params.get("maximum_order_size", 1e9))
            self.price_collar_active = bool(market_params.get("price_collar_active", True))
            self.circuit_breaker_threshold = float(market_params.get("circuit_breaker_threshold", 0.25))
            self.clearing_frequency = int(market_params.get("clearing_frequency", 1))
            self.partial_fill_allowed = bool(market_params.get("partial_fill_allowed", True))
            self.banking_allowed = bool(market_params.get("banking_allowed", True))
            self.borrowing_allowed = bool(market_params.get("borrowing_allowed", False))
            self.market_stability_reserve_credits = max(0.0, float(market_params.get('market_stability_reserve_credits', 1000)))
            self.market_stability_reserve_fund = max(0.0, float(market_params.get('market_stability_reserve_fund', 1000000)))
            self.credit_validity = max(1, int(market_params.get('credit_validity', 5)))
            self.penalty_rate = max(0.0, float(market_params.get('penalty_rate', 2000)))
            self.min_price = max(1.0, float(market_params.get('min_price', 3000)))
            self.max_price = max(self.min_price * 2, float(market_params.get('max_price', 50000)))
            self.stability_reserve = max(0.0, min(1.0, float(market_params.get('stability_reserve', 0.1))))
            self.verification_cost = max(0.0, float(market_params.get('verification_cost', 50)))
            self.minimum_trade_size = max(0.0, float(market_params.get('minimum_trade_size', 1e-6)))
            self.MarketStabilityReserve_participation = bool(market_params.get('MarketStabilityReserve_participation', True))
            self.params = {
                'price_sensitivity': max(0.0, min(1.0, float(market_params.get('price_sensitivity', 0.05)))),
                'base_volatility': max(0.0, min(1.0, float(market_params.get('base_volatility', 0.01)))),
                'max_daily_change': max(0.01, min(0.5, float(market_params.get('max_daily_change', 0.15)))),
                'liquidity_spread': max(0.001, min(0.1, float(market_params.get('liquidity_spread', 0.02)))),
                'MarketStabilityReserve_impact': max(0.0, min(1.0, float(market_params.get('MarketStabilityReserve_impact', 0.5))))
            }

            self.carbon_price = max(self.min_price, min(self.max_price, self.carbon_price))

        except (KeyError, ValueError) as e:
            logger.error(f"Error loading market parameters: {e}")
            self._set_default_parameters()

    def _set_default_parameters(self) -> None:
        """Set safe default parameters"""
        self.market_stability_reserve_credits = 1000.0
        self.market_stability_reserve_fund = 1000000.0
        self.credit_validity = 5
        self.penalty_rate = 2000.0
        self.min_price = 3000.0
        self.max_price = 50000.0
        self.stability_reserve = 0.1
        self.verification_cost = 50.0
        self.minimum_trade_size = 1e-6
        self.order_expiry_steps = 12
        self.MarketStabilityReserve_participation = True
        self.params = {
            'price_sensitivity': 0.05,
            'base_volatility': 0.01,
            'max_daily_change': 0.15,
            'liquidity_spread': 0.02,
            'MarketStabilityReserve_impact': 0.5
        }

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------
    def _expire_orders(self) -> None:
        """Drop stale orders based on order_expiry_steps."""
        try:
            if int(self.order_expiry_steps) <= 0:
                return
            cutoff = self.model.schedule.steps - int(self.order_expiry_steps)
            self.buy_orders  = [o for o in self.buy_orders  if getattr(o, "timestamp", 0) > cutoff]
            self.sell_orders = [o for o in self.sell_orders if getattr(o, "timestamp", 0) > cutoff]
        except Exception as e:
            logger.error(f"Order expiry failed: {e}")

    def place_order(self, order_type: str, agent: Optional['FirmAgent'], quantity: float,
                    limit_price: Optional[float] = None) -> bool:
        """Place buy/sell order with enhanced validation"""
        if quantity <= self.minimum_trade_size:
            logger.warning(f"Invalid order quantity (below minimum trade size): {quantity:.6f} by {_agent_name(agent)}")
            return False
        
        if quantity > self.maximum_order_size:
            quantity = self.maximum_order_size  # hard-cap oversized orders

        if agent is not None:
            if order_type == 'buy':
                px = (limit_price if limit_price is not None else self.carbon_price)
                required_cash = quantity * px
                if getattr(agent, 'cash', 0) < required_cash:
                    logger.debug(f"Agent {_agent_name(agent)} insufficient cash for buy order")
                    return False
            elif order_type == 'sell':
                # Only allow if owned credits cover it (banked)
                owned = getattr(agent, 'total_credits_owned', None)
                total_owned = owned() if callable(owned) else getattr(agent, 'credits_owned', 0.0)
                if total_owned < quantity:
                    logger.debug(f"Agent {_agent_name(agent)} insufficient credits for sell order")
                    return False

        order_kind = 'limit' if limit_price is not None else 'market'
        
        if limit_price is not None:
            limit_price = max(self.min_price, min(self.max_price, limit_price))

        order = MarketOrder(
            agent=agent,
            quantity=float(quantity),
            timestamp=self.model.schedule.steps,
            order_type=order_kind,
            price=limit_price
        )

        # Validate market maker inventory/budget
        if agent is None:
            if order_type == 'buy' and limit_price:
                required_cash = order.quantity * limit_price
                if self.market_stability_reserve_fund < required_cash:
                    logger.warning(f"Market maker insufficient cash: {self.market_stability_reserve_fund:.2f} < {required_cash:.2f}")
                    return False
            if order_type == 'sell' and self.market_stability_reserve_credits < order.quantity:
                logger.warning(f"Market maker insufficient credits: {self.market_stability_reserve_credits:.2f} < {order.quantity:.2f}")
                return False

        if order_type == 'buy':
            self.buy_orders.append(order)
        else:
            self.sell_orders.append(order)

        logger.debug(f"Order placed: {order_type} {quantity:.2f} credits by {_agent_name(agent)} "
                    f"(type={order.order_type}, price={order.price})")
        return True

    def _sort_orders(self) -> None:
        """Sort orders with proper priority"""
        def buy_key(order):
            is_market = 0 if order.order_type == 'market' else 1
            price_key = -(order.price if order.price is not None else self.max_price * 2)
            return (is_market, price_key, order.timestamp)

        def sell_key(order):
            is_market = 0 if order.order_type == 'market' else 1
            price_key = order.price if order.price is not None else 0
            return (is_market, price_key, order.timestamp)

        self.buy_orders.sort(key=buy_key)
        self.sell_orders.sort(key=sell_key)

    # ------------------------------------------------------------------
    # Compliance
    # ------------------------------------------------------------------
    def annual_compliance_check(self, current_year: int) -> None:
        """Annual intensity-based compliance; FIFO surrender, penalties, and MINTING of surplus credits."""
        try:
            non_compliant = 0
            cfg = getattr(self.model, "config", {}) or {}
            borrow_on = bool(getattr(self, "borrowing_allowed", False))
            borrow_cap = float(cfg.get("borrowing_limit_ratio", 0.0))
            borrow_r = float(cfg.get("borrowing_interest_rate", 0.0))
            decay_rate = float(cfg.get("banking_decay_rate", 0.0))
            pen_escal = float(cfg.get("penalty_escalation_factor", 0.0))


            for agent in self.model.schedule.agents:
                if not getattr(agent, "is_covered_entity", False):
                    continue

                # Optional haircut on banked vintages before we do anything else
                if decay_rate > 0 and hasattr(agent, "apply_banking_decay"):
                    agent.apply_banking_decay(decay_rate)

                # Gather (verified) annual totals & target
                try:
                   # Prefer verified totals (adds MRV noise if configured)
                    Q, E = (agent.get_verified_annual_totals(current_year)
                            if hasattr(agent, "get_verified_annual_totals")
                            else agent.get_annual_totals(current_year))
                except Exception:
                    Q = float(getattr(agent, "total_annual_production", 0.0))
                    E = float(getattr(agent, "total_annual_emissions", 0.0))
                try:
                    tau = agent.get_target_intensity(current_year)
                except Exception:
                    tau = float(getattr(agent, "current_target_intensity", 0.0))

                allowed = tau * Q
                gap = E - allowed  # +ve deficit, -ve surplus

                if gap > 0:
                    # DEFICIT: surrender FIFO first
                    try:
                        owned = agent.total_credits_owned() if hasattr(agent, "total_credits_owned") else getattr(agent, "credits_owned", 0.0)
                        need = min(gap, owned)
                        surrendered = agent.surrender_credits_fifo(need) if hasattr(agent, "surrender_credits_fifo") else 0.0
                    except Exception:
                        surrendered = 0.0

                    net_gap = max(0.0, gap - surrendered)

                    # Optional: allow borrowing BEFORE penalty
                    if borrow_on and net_gap > 1e-9 and borrow_cap > 0.0:
                        cap_abs = max(0.0, allowed * borrow_cap)  # τ·Q·ratio
                        borrow_amount = min(net_gap, cap_abs)
                        if borrow_amount > 0:
                            # Record as principal to repay next year with interest
                            agent.borrowed_credits += float(borrow_amount) * (1.0 + borrow_r)
                            net_gap -= borrow_amount

                    if net_gap > 1e-9:
                        # Penalty with escalation for consecutive noncompliance
                        streak = int(getattr(agent, "noncompliance_streak", 0))
                        escalator = (1.0 + pen_escal * max(0, streak))
                        penalty = net_gap * float(self.penalty_rate) * max(1.0, escalator)



                        agent.cash = float(getattr(agent, "cash", 0.0)) - penalty
                        if not hasattr(agent, 'penalty_history'):
                            agent.penalty_history = []
                        agent.penalty_history.append(ComplianceRecord(
                            year=current_year, gap=net_gap, penalty=penalty, is_compliant=False
                        ))
                        agent.is_compliant = False
                        agent.noncompliance_streak = int(getattr(agent, "noncompliance_streak", 0)) + 1
                        non_compliant += 1
                    else:
                        agent.is_compliant = True
                        agent.noncompliance_streak = 0

                else:
                    # SURPLUS: mint credits (vintage = current_year)
                    surplus = abs(gap)
                    if self.banking_allowed and surplus > 1e-9:
                        agent.add_credits_to_vintage(surplus, current_year)
                    agent.is_compliant = True
                    agent.noncompliance_streak = 0

                # Verification fee (if any)
                vc = float(getattr(self, "verification_cost", 0.0))
                if vc > 0:
                    agent.cash = float(getattr(agent, "cash", 0.0)) - vc

                # Optional: reset per-year tallies after compliance
                if hasattr(agent, "end_of_year_reset"):
                    agent.end_of_year_reset()

            if non_compliant > 0:
                logger.info(f"Annual compliance {current_year}: {non_compliant} non-compliant entities")
            else:
                logger.info(f"Annual compliance {current_year}: all compliant")

        except Exception as e:
            logger.error(f"Error during annual compliance check: {e}")

    # ------------------------------------------------------------------
    # Trade execution
    # ------------------------------------------------------------------
    def execute_trade(self, buyer: Optional['FirmAgent'], seller: Optional['FirmAgent'],
                      quantity: float, price: float) -> float:
        """Enhanced trade execution with vintage tracking"""
        # Only execute trade if quantity is above the minimum trade size
        if quantity <= self.minimum_trade_size:
            logger.debug(f"Trade quantity {quantity:.6f} is below minimum trade size, skipping trade.")
            return 0.0

        try:
            buyer_id = _agent_name(buyer)
            seller_id = _agent_name(seller)

            price = max(self.min_price, min(self.max_price, price))
            transaction_value = quantity * price
            # Use the model's start year for current year calculation
            current_year = self.model.start_year + (self.model.schedule.steps // self.steps_per_year)

            if buyer is not None:
                buyer_cash = getattr(buyer, 'cash', 0.0)
                if buyer_cash >= transaction_value:
                    buyer.cash = buyer_cash - transaction_value
                    buyer.add_credits_to_vintage(quantity, current_year)
                else:
                    logger.warning(f"Buyer {buyer_id} insufficient cash: {buyer_cash:.2f} < {transaction_value:.2f}")
                    return 0.0
            else:
                if self.market_stability_reserve_fund >= transaction_value:
                    self.market_stability_reserve_fund -= transaction_value
                    self.market_stability_reserve_credits += quantity
                else:
                    logger.warning(f"Market maker insufficient cash: {self.market_stability_reserve_fund:.2f} < {transaction_value:.2f}")
                    return 0.0

            if seller is not None:
                owned = seller.total_credits_owned() if hasattr(seller, "total_credits_owned") else getattr(seller, "credits_owned", 0.0)
                if owned >= quantity:
                    seller.cash = getattr(seller, "cash", 0.0) + transaction_value
                    self._remove_credits_from_vintages(seller, quantity)
                else:
                    logger.warning(f"Seller {seller_id} insufficient credits: {owned:.2f} < {quantity:.2f}")
                    return 0.0
            else:
                if self.market_stability_reserve_credits >= quantity:
                    self.market_stability_reserve_fund += transaction_value
                    self.market_stability_reserve_credits -= quantity
                else:
                    logger.warning(f"Market maker insufficient credits: {self.market_stability_reserve_credits:.2f} < {quantity:.2f}")
                    return 0.0

            transaction = Transaction(
                step=self.model.schedule.steps,
                buyer_id=buyer_id,
                seller_id=seller_id,
                quantity=quantity,
                price=price,
                transaction_type='market_execution'
            )
            self.transactions_log.append(transaction)

            self.daily_volume += quantity
            self.market_stats['total_volume'] += quantity

            logger.info(f"Trade executed: {quantity:.2f} credits @ ₹{price:.2f} ({buyer_id} ← {seller_id})")
            return transaction_value

        except Exception as e:
            logger.error(f"Trade execution error: {e}")
            return 0.0

    def _remove_credits_from_vintages(self, seller: 'FirmAgent', quantity: float) -> None:
        """Remove credits from seller's vintages using FIFO"""
        try:
            if not hasattr(seller, 'credit_vintages') or not seller.credit_vintages:
                return

            credits_to_remove = float(quantity)
            for year in sorted(list(seller.credit_vintages.keys())):
                if credits_to_remove <= 1e-9:
                    break
                available = float(seller.credit_vintages.get(year, 0.0))
                take = min(available, credits_to_remove)
                seller.credit_vintages[year] = available - take
                credits_to_remove -= take
                if seller.credit_vintages[year] <= 1e-9:
                    del seller.credit_vintages[year]

            # keep scalar mirror in sync if present
            if hasattr(seller, "_recompute_credits_owned"):
                seller._recompute_credits_owned()

        except Exception as e:
            logger.error(f"Error removing credits from vintages: {e}")

    # ------------------------------------------------------------------
    # MSR
    # ------------------------------------------------------------------
    def MarketStabilityReserve_intervention(self) -> None:
        """Enhanced market maker intervention"""
        try:
            current_spread = self._calculate_bid_ask_spread()
            target_spread = self.carbon_price * self.params['liquidity_spread']
            
            should_intervene = (
                current_spread == float('inf') or 
                current_spread > target_spread * 2 or
                self.volatility_index > self.params.get('base_volatility', 0.01) * 3
            )

            if should_intervene:
                buy_price = max(self.min_price, self.carbon_price * (1 - target_spread))
                sell_price = min(self.max_price, self.carbon_price * (1 + target_spread))
                self._inject_liquidity_at_spread(buy_price, sell_price)

        except Exception as e:
            logger.error(f"Market maker intervention error: {e}")

    def _calculate_bid_ask_spread(self) -> float:
        """Calculate current bid-ask spread safely"""
        try:
            if not self.buy_orders or not self.sell_orders:
                return float('inf')

            best_bid = None
            for order in self.buy_orders:
                if order.price is not None:
                    best_bid = order.price
                    break

            best_ask = None
            for order in self.sell_orders:
                if order.price is not None:
                    best_ask = order.price
                    break

            if best_bid is None or best_ask is None or best_ask <= 0:
                return float('inf')

            return (best_ask - best_bid) / best_ask

        except Exception:
            return float('inf')

    def _inject_liquidity_at_spread(self, buy_price: float, sell_price: float) -> None:
        """Inject liquidity when market is illiquid"""
        try:
            base_liquidity = min(self.market_stability_reserve_credits * 0.05, 50.0)
            liquidity_amount = max(1.0, base_liquidity)

            if self.market_stability_reserve_fund >= buy_price * liquidity_amount and buy_price >= self.min_price:
                self.place_order('buy', None, liquidity_amount, buy_price)
                logger.info(f"market_stability_reserve injected buy liquidity: {liquidity_amount:.2f} @ ₹{buy_price:.2f}")

            if self.market_stability_reserve_credits >= liquidity_amount and sell_price <= self.max_price:
                self.place_order('sell', None, liquidity_amount, sell_price)
                logger.info(f"market_stability_reserve injected sell liquidity: {liquidity_amount:.2f} @ ₹{sell_price:.2f}")

        except Exception as e:
            logger.error(f"Liquidity injection error: {e}")

    # ------------------------------------------------------------------
    # Metrics & price
    # ------------------------------------------------------------------
    def calculate_market_metrics(self) -> Tuple[float, float, float]:
        """Calculate comprehensive market metrics safely"""
        try:
            covered_buy = sum(
                order.quantity for order in self.buy_orders
                if order.agent is not None and getattr(order.agent, "is_covered_entity", False)
            )
            covered_sell = sum(
                order.quantity for order in self.sell_orders
                if order.agent is not None and getattr(order.agent, "is_covered_entity", False)
            )

            total_orders = covered_buy + covered_sell
            if total_orders == 0:
                return 0.0, 0.0, 0.0

            imbalance_ratio = safe_divide(covered_buy - covered_sell, total_orders)
            liquidity_index = safe_divide(
                min(covered_buy, covered_sell),
                max(covered_buy, covered_sell, 1.0)
            )
            market_depth = total_orders

            return imbalance_ratio, liquidity_index, market_depth

        except Exception as e:
            logger.error(f"Market metrics calculation error: {e}")
            return 0.0, 0.0, 0.0

    def update_price_volatility(self) -> float:
        """Enhanced volatility calculation"""
        try:
            if len(self.price_history) < 2:
                self.volatility_index = self.params.get('base_volatility', 0.01)
                return self.volatility_index

            valid_prices = [p for p in self.price_history if p > 0]
            if len(valid_prices) < 2:
                self.volatility_index = self.params.get('base_volatility', 0.01)
                return self.volatility_index

            log_returns = []
            for i in range(1, len(valid_prices)):
                if valid_prices[i-1] > 0 and valid_prices[i] > 0:
                    log_returns.append(np.log(valid_prices[i] / valid_prices[i-1]))

            if len(log_returns) < 2:
                self.volatility_index = self.params.get('base_volatility', 0.01)
                return self.volatility_index

            returns_std = np.std(log_returns)
            annualized_vol = returns_std * np.sqrt(self.steps_per_year)
            
            base_vol = self.params.get('base_volatility', 0.01)
            self.volatility_index = max(base_vol, min(annualized_vol, 1.0))

            return self.volatility_index

        except Exception as e:
            logger.error(f"Volatility calculation error: {e}")
            self.volatility_index = self.params.get('base_volatility', 0.01)
            return self.volatility_index

    def _update_carbon_price(self, imbalance_ratio: float) -> None:
        """Stable price adjustment with circuit breakers"""
        try:
            if len(self.price_history) == 0:
                return

            base_adjustment = self.params.get('price_sensitivity', 0.05) * np.tanh(imbalance_ratio * 2)

            damping_factor = max(0.1, 1 - self.volatility_index)
            base_adjustment *= damping_factor

            if hasattr(self.model, "random"):
                rand_val = self.model.random.random()
                if rand_val < 0.05:
                    shock = self.model.random.gauss(0, 0.02)
                    base_adjustment += shock
            else:
                if random.random() < 0.05:
                    base_adjustment += random.gauss(0, 0.02)

            max_change = self.params.get('max_daily_change', 0.15)
            total_adjustment = max(-max_change, min(max_change, base_adjustment))

            old_price = self.carbon_price
            new_price = old_price * (1 + total_adjustment)
            
            new_price = max(self.min_price, min(self.max_price, new_price))

            if len(self.price_history) >= 5:
                recent_avg = np.mean(self.price_history[-5:])
                circuit_limit = float(self.circuit_breaker_threshold)  # e.g., 0.25
                
                upper_limit = recent_avg * (1 + circuit_limit)
                lower_limit = recent_avg * (1 - circuit_limit)
                
                if new_price > upper_limit:
                    new_price = upper_limit
                    logger.warning(f"Price circuit breaker triggered (upper): {old_price:.2f} -> {new_price:.2f}")
                elif new_price < lower_limit:
                    new_price = lower_limit
                    logger.warning(f"Price circuit breaker triggered (lower): {old_price:.2f} -> {new_price:.2f}")

            self.carbon_price = new_price
            self.price_history.append(self.carbon_price)

            if len(self.price_history) > 100:
                self.price_history = self.price_history[-100:]

            if abs(new_price - old_price) > old_price * 0.1:
                logger.info(f"Significant price change: ₹{old_price:.2f} -> ₹{new_price:.2f} "
                          f"(imbalance: {imbalance_ratio:.3f})")

        except Exception as e:
            logger.error(f"Price update error: {e}")

    def _update_market_statistics(self) -> None:
        """Update market statistics safely"""
        try:
            self.market_stats.update({
                'average_spread': self._calculate_bid_ask_spread(),
                'market_depth': len(self.buy_orders) + len(self.sell_orders),
                'price_volatility': self.volatility_index,
                'total_volume': sum(t.quantity for t in self.transactions_log)
            })
        except Exception as e:
            logger.error(f"Market statistics update error: {e}")

    # ------------------------------------------------------------------
    # Order book capture
    # ------------------------------------------------------------------
    def _capture_order_book(self) -> Dict:
        """Captures the current state of the order book."""
        buy_prices = sorted(list(set(o.price for o in self.buy_orders if o.price is not None)), reverse=True)
        sell_prices = sorted(list(set(o.price for o in self.sell_orders if o.price is not None)))
    
        buy_levels = {
            price: sum(o.quantity for o in self.buy_orders if o.price == price)
            for price in buy_prices
        }
        sell_levels = {
            price: sum(o.quantity for o in self.sell_orders if o.price == price)
            for price in sell_prices
        }
    
        return {
            'step': self.model.schedule.steps,
            'buy_side': buy_levels,
            'sell_side': sell_levels
        }

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def process_market_orders(self) -> None:
        """Process market orders against limit orders"""
        try:
            trades_executed = 0
            total_volume = 0.0

            self._sort_orders()

            market_buys = [o for o in self.buy_orders if o.order_type == 'market']
            for buy_order in market_buys[:]:
                while buy_order.quantity > self.minimum_trade_size and self.sell_orders:
                    best_sell = self.sell_orders[0]
                    if best_sell.order_type != 'limit' or best_sell.price is None:
                        break
                    
                    trade_quantity = min(buy_order.quantity, best_sell.quantity)
                    trade_price = best_sell.price

                    transaction_value = self.execute_trade(
                        buy_order.agent, best_sell.agent, trade_quantity, trade_price
                    )
                    
                    if transaction_value > 0:
                        trades_executed += 1
                        total_volume += trade_quantity
                        
                        buy_order.quantity -= trade_quantity
                        best_sell.quantity -= trade_quantity
                        
                        if best_sell.quantity <= self.minimum_trade_size:
                            self.sell_orders.remove(best_sell)
                            self._sort_orders()
                    else:
                        break

                if buy_order.quantity <= self.minimum_trade_size:
                    if buy_order in self.buy_orders:
                        self.buy_orders.remove(buy_order)
            
            market_sells = [o for o in self.sell_orders if o.order_type == 'market']
            for sell_order in market_sells[:]:
                while sell_order.quantity > self.minimum_trade_size and self.buy_orders:
                    best_buy = self.buy_orders[0]
                    if best_buy.order_type != 'limit' or best_buy.price is None:
                        break
                    
                    trade_quantity = min(sell_order.quantity, best_buy.quantity)
                    trade_price = best_buy.price

                    transaction_value = self.execute_trade(
                        best_buy.agent, sell_order.agent, trade_quantity, trade_price
                    )
                    
                    if transaction_value > 0:
                        trades_executed += 1
                        total_volume += trade_quantity
                        
                        sell_order.quantity -= trade_quantity
                        best_buy.quantity -= trade_quantity
                        
                        if best_buy.quantity <= self.minimum_trade_size:
                            self.buy_orders.remove(best_buy)
                            self._sort_orders()
                    else:
                        break

                if sell_order.quantity <= self.minimum_trade_size:
                    if sell_order in self.sell_orders:
                        self.sell_orders.remove(sell_order)

            if trades_executed > 0:
                logger.info(f"Executed {trades_executed} market trades, volume: {total_volume:.2f}")

        except Exception as e:
            logger.error(f"Market order processing error: {e}")

    def match_limit_orders(self) -> None:
        """Match limit orders with price-time priority"""
        try:
            self._sort_orders()
            trades_executed = 0
            total_volume = 0.0
            
            orders_to_remove = {'buy': [], 'sell': []}

            while self.buy_orders and self.sell_orders:
                best_buy = self.buy_orders[0]
                best_sell = self.sell_orders[0]

                if best_buy.order_type == 'market' or best_sell.order_type == 'market':
                    break

                if best_buy.price is None or best_sell.price is None:
                    break

                if best_buy.price >= best_sell.price:
                    trade_price = (best_buy.price + best_sell.price) / 2.0
                    trade_quantity = min(best_buy.quantity, best_sell.quantity)

                    if trade_quantity <= self.minimum_trade_size:
                        break
                        
                    transaction_value = self.execute_trade(
                        best_buy.agent, best_sell.agent, trade_quantity, trade_price
                    )
                    
                    if transaction_value > 0:
                        trades_executed += 1
                        total_volume += trade_quantity

                        best_buy.quantity -= trade_quantity
                        best_sell.quantity -= trade_quantity

                        if best_buy.quantity <= self.minimum_trade_size:
                            orders_to_remove['buy'].append(best_buy)
                        if best_sell.quantity <= self.minimum_trade_size:
                            orders_to_remove['sell'].append(best_sell)
                    else:
                        break
                else:
                    break

            for order in orders_to_remove['buy']:
                if order in self.buy_orders:
                    self.buy_orders.remove(order)
            for order in orders_to_remove['sell']:
                if order in self.sell_orders:
                    self.sell_orders.remove(order)
            
            if trades_executed > 0:
                logger.info(f"Executed {trades_executed} limit trades, volume: {total_volume:.2f}")

        except Exception as e:
            logger.error(f"Limit order matching error: {e}")

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------
    def step(self) -> None:
        """
        Enhanced market step with comprehensive processing.
        Run annual compliance at the end of the compliance year (after the last step of the year).
        """
        try:
            self.daily_volume = 0.0

            # Purge stale orders every step
            self._expire_orders()

            # Year-end housekeeping (run when current step completes a year)
            if (self.model.schedule.steps % self.steps_per_year) == 0:
                current_year = self.model.start_year + ((self.model.schedule.steps) // self.steps_per_year) - 1

                # Expire vintages older than validity for the completed year
                for agent in self.model.schedule.agents:
                    if hasattr(agent, 'expire_credits'):
                        agent.expire_credits(current_year)

                # Run compliance check for the completed year (surrender/penalize or mint surplus)
                self.annual_compliance_check(current_year)

                # Repay borrowed credits right after compliance is evaluated for the year that just ended:
                # Those debts mature at the NEXT compliance, so we enforce repayment from any newly
                # minted/surplus or via immediate buy orders at current price (as a simple mechanism).
                for agent in self.model.schedule.agents:
                    debt = float(getattr(agent, "borrowed_credits", 0.0))
                    if debt <= 1e-9:
                        continue
                    # Try to surrender from vintages first
                    surrendered = 0.0
                    if hasattr(agent, "surrender_credits_fifo"):
                        surrendered = agent.surrender_credits_fifo(min(debt, agent.total_credits_owned()))
                    remaining = max(0.0, debt - surrendered)
                    if remaining > 1e-9:
                        # Force immediate market purchase (at current price) if cash allows
                        px = max(self.min_price, min(self.max_price, self.carbon_price))
                        needed_cash = remaining * px
                        if getattr(agent, "cash", 0.0) >= needed_cash:
                            # “Internalized” fill: move cash -> vintages, then surrender
                            agent.cash -= needed_cash
                            agent.add_credits_to_vintage(remaining, current_year+1)
                            if hasattr(agent, "surrender_credits_fifo"):
                                agent.surrender_credits_fifo(remaining)
                            remaining = 0.0
                        else:
                            logger.warning(f"{getattr(agent,'firm_id','?')} cannot fully repay borrowed credits ({remaining:.2f})")
                    agent.borrowed_credits = remaining


            # Market operations happen at every step
            if self.MarketStabilityReserve_participation:
                self.MarketStabilityReserve_intervention()
        
            self._sort_orders()
            self.process_market_orders()
            self.match_limit_orders()

            imbalance_ratio, liquidity_index, market_depth = self.calculate_market_metrics()
            self._update_carbon_price(imbalance_ratio)
            self.update_price_volatility()
            self._update_market_statistics()

            # Stress intervention
            if self.carbon_price > 0.9 * self.max_price or (len(self.transactions_log) == 0 and self.model.schedule.steps > 12):
                if self._emergency_last_step != self.model.schedule.steps:
                    self._emergency_last_step = self.model.schedule.steps
                    logger.warning("Market stress detected - initiating emergency intervention.")
                    self._emergency_intervention()

            self._update_market_statistics()
            self.order_book_history.append(self._capture_order_book())

        except Exception as e:
            logger.error(f"Market step error: {e}")

    # ------------------------------------------------------------------
    # Emergency
    # ------------------------------------------------------------------
    def _emergency_intervention(self) -> None:
        """Emergency intervention during market stress"""
        try:
            emergency_liquidity = min(self.market_stability_reserve_credits * 0.2, 200.0)
            
            if emergency_liquidity > 10.0:
                mid_price = self.carbon_price
                buy_price = max(self.min_price, mid_price * 0.95)
                sell_price = min(self.max_price, mid_price * 1.05)
                
                if self.market_stability_reserve_fund >= buy_price * emergency_liquidity:
                    self.place_order('buy', None, emergency_liquidity, buy_price)
                    logger.warning(f"Emergency buy liquidity: {emergency_liquidity:.2f} @ ₹{buy_price:.2f}")
                
                if self.market_stability_reserve_credits >= emergency_liquidity:
                    self.place_order('sell', None, emergency_liquidity, sell_price)
                    logger.warning(f"Emergency sell liquidity: {emergency_liquidity:.2f} @ ₹{sell_price:.2f}")

        except Exception as e:
            logger.error(f"Emergency intervention error: {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    def get_market_summary(self) -> Dict:
        """Get comprehensive market summary"""
        try:
            return {
                'current_price': self.carbon_price,
                'price_history_length': len(self.price_history),
                'total_transactions': len(self.transactions_log),
                'daily_volume': self.daily_volume,
                'volatility_index': self.volatility_index,
                'buy_orders_count': len(self.buy_orders),
                'sell_orders_count': len(self.sell_orders),
                'market_stability_reserve_fund': self.market_stability_reserve_fund,
                'market_stability_reserve_credits': self.market_stability_reserve_credits,
                'market_stats': self.market_stats.copy()
            }
        except Exception as e:
            logger.error(f"Market summary error: {e}")
            return {'error': str(e)}
