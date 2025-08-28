
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
