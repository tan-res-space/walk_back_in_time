from datetime import datetime, date, timedelta, time
from tradelib_logger import logger
from algo_driver import AlgoDriver
from models.Portfolio import Portfolio
from trading_platform.BacktestTradingPlatform import BacktestTradingPlatform
from trading_platform._TradingPlatform import TradingPlatform
from tradelib_global_constants import date_format
from tradelib_utils import is_data_available_for_date, get_holiday_list
from tradelib_trade_utils import is_trading_date
import copy
from strategies._strategy import TradeStrategy
from typing import List
from strategies.strategy_components._strategy_component import StrategyComponent
from strategies.strategy_factory import StrategyFactory

class BacktestDriver:
    def __init__(self, start_date: date, end_date: date, trade_start_time: time, trade_end_time: time, data_dir: str, output_dir: str,strategy_name: str, strategy_components: List[StrategyComponent]=[], risk_free_rate:float=None, steps:float=None, underlying:str=None, dividend:str=None) -> None:
        self.trading_platform = BacktestTradingPlatform(data_dir, risk_free_rate, steps, underlying, dividend)
        self.output_dir = output_dir
        self.logger = logger.getLogger("backtest_driver")
        self.portfolio = Portfolio("INR", self.trading_platform, self.output_dir)
        self.algo_driver = AlgoDriver(strategy_name, self.output_dir,self.trading_platform, self.portfolio, trade_start_time, trade_end_time, strategy_components)
        self.start_date = start_date
        self.end_date = end_date
        self.holiday_list = get_holiday_list(self.start_date.year)


    def single_process_driver(self):
        start_date = copy.copy(self.start_date)
        end_date = copy.copy(self.end_date)

        if start_date > end_date:
            self.logger.critical("end_date is earlier than start_date")
    
        while start_date <= end_date:
            if is_trading_date(date=start_date, trading_platform=self.trading_platform):
            # if is_data_available_for_date(self.trading_platform.option_data_dir, start_date, self.trading_platform.underlying):
                self.logger.info(f'{"-"*20} trading day: {start_date.strftime(date_format)} {"-"*20}')
                self.algo_driver.drive(start_date)
            else:
                if start_date.weekday() in [5, 6]:
                    self.logger.info(f"date: {start_date.strftime(date_format)} is a weekend")
                elif start_date in self.holiday_list:
                    self.logger.info(f"date: {start_date.strftime(date_format)} is holiday")
                else:
                    self.logger.critical(f"trading day data not available for: {start_date.strftime(date_format)}")
            print(f'{"-"*20} processed: {start_date.strftime(date_format)} {"-"*20}')
            start_date = start_date + timedelta(days=1)