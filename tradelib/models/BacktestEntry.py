from datetime import datetime
from trading_platform import TradingPlatform
from models.Portfolio import Portfolio
from tradelib_logger import logger
from tradelib_utils import round_to_step
from models.Option import Option
from models.OptionDetailed import OptionDetailed
from tradelib_trade_utils import get_atm_option, get_expiry_date
from tradelib_global_constants import date_time_format, expiry_info

_logger = logger.getLogger("backtest_entry")

class BacktestEntry:
    def __init__(self, timestamp: datetime, trade_done: bool, portfolio: Portfolio, trading_platform: TradingPlatform) -> None:
        self.timestamp = timestamp
        self.trade_done = trade_done
        self.spot = trading_platform.getSpot(timestamp)
        nearest_expiry = get_expiry_date(timestamp, expiry_info, trading_platform, logger)

        curr_atm = None
        curr_atm_ce: OptionDetailed = None

        try:
            curr_atm_ce = get_atm_option(trading_platform, trading_platform.underlying, self.spot, "CE", nearest_expiry, trading_platform.steps, timestamp, _logger)
        except:
            curr_atm_ce = None

        curr_atm_pe: OptionDetailed = None
        try:
            curr_atm_pe = get_atm_option(trading_platform, trading_platform.underlying, self.spot, "PE", nearest_expiry, trading_platform.steps, timestamp, _logger)
        except:
            curr_atm_pe = None

        if curr_atm_ce == None and curr_atm_pe == None:
            _logger.critical("No ATM found")
        elif (curr_atm_pe == None) ^ (curr_atm_ce == None):
            curr_atm = curr_atm_pe if curr_atm_pe != None else curr_atm_ce
        else:
            if abs(curr_atm_ce.strike - self.spot) < abs(curr_atm_pe.strike - self.spot):
                curr_atm = curr_atm_ce
            else:
                curr_atm = curr_atm_pe

        self.atmIv = curr_atm.sigma if curr_atm != None else None
        portfolio_greeks = portfolio.getPortfolioGreeks(timestamp)
        self.delta = portfolio_greeks.delta
        self.gamma = portfolio_greeks.gamma
        self.vega = portfolio_greeks.vega
        self.theta = portfolio_greeks.theta
        self.sigma = portfolio_greeks.sigma
        self.total_contracts = portfolio.getContractSize()
        self.cash = portfolio.cash.value
        self.instrumentsM2m = portfolio.getMarkToMarket(timestamp)
        self.pval = self.cash + self.instrumentsM2m

        _logger.info(f"Inside BacktestEntry - timestamp : {timestamp}, ATM-IV: {self.atmIv}, Spot: {self.spot}")