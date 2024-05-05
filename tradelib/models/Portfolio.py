from typing import Dict, List
from models.Trade import Trade
from models.Cash import Cash
from models.Instrument import Instrument
from models.OptionDetailed import OptionDetailed
from models.Greeks import Greeks
from datetime import datetime
from tradelib.trading_platform._TradingPlatform import TradingPlatform
import csv
import copy
from models.Option import Option
import os
from tradelib_logger import logger
from tradelib_global_constants import allowed_fields_in_portfolio_dump, client_name, initial_cash
from tradelib_utils import get_portfolio_filename, serialiseInstrument
from tradelib_blackscholes_utils import getInstrumentDetailsWithBlackscholes

_logger = logger.getLogger("portfolio")

class Portfolio:
    def __init__(self, currency: str, trading_platform: TradingPlatform, output_dir: str) -> None:
        self.ins_map: Dict[str, Trade] = {}
        self.cash = Cash(initial_cash=initial_cash)
        self.currency: str = currency
        self.trading_platform = trading_platform
        self.portfolio_dir = os.path.join(output_dir, 'portfolio')
        os.makedirs(self.portfolio_dir, exist_ok=True)

    def clearPortfolio(self, timestamp: datetime):
        clear_list = []
        for key, val in self.ins_map.items():
            if val.instrument.instrument_type == "option":
                ins: Option = val.instrument
                if ins.expiry == timestamp.date():
                    clear_list.append(key)

        for clear_id in clear_list:
            del self.ins_map[clear_id]
        
        _logger.info("unwinded all expiring instruments")

    
    def saveToCSV(self, timestamp: datetime):
        '''
        saves current state of portfolio in a csv
        '''
        file_path = os.path.join(self.portfolio_dir, get_portfolio_filename(timestamp, client_name))
        with open(file_path, "w") as fp:
            writer = csv.writer(fp)
            writer.writerow(['instrument_id', 'instrument_id_key', 'instrument_object', 'position', 'current price', 'value'])
            writer.writerow([self.cash.instrument_type, self.cash.idKey(), serialiseInstrument(self.cash, False, allowed_fields_in_portfolio_dump), self.cash.value, 1, self.cash.value])
            for key, val in self.ins_map.items():
                try:
                    detailedIns = getInstrumentDetailsWithBlackscholes(val.instrument, timestamp, self.trading_platform, _logger)
                    if detailedIns.instrument_type == "option":
                        price = detailedIns.mid
                        id = detailedIns.id
                    writer.writerow([id, val.instrument.idKey(), serialiseInstrument(val.instrument, False,allowed_fields_in_portfolio_dump), val.position, price, val.position*price])
                except Exception as e:
                    _logger.critical(e)

    def updatePortfolio(self, tradeList: List[Trade], timestamp: datetime) -> None:
        '''
        Updates portfolio with tradelist
        '''
        _logger.debug(f"Porfolio state before update:")
        greeks = self.getPortfolioGreeks(timestamp)
        _logger.debug(f"delta: {greeks.delta}, gamma: {greeks.gamma}, vega: {greeks.vega}, theta: {greeks.theta}, sigma: {greeks.sigma}, contract_size: {self.getContractSize()}")
        cashChange = 0
        for trade in tradeList:
            if trade.position == 0:
                continue
            
            if trade.instrument.idKey() in self.ins_map.keys():
                self.ins_map[trade.instrument.idKey()].position = self.ins_map[trade.instrument.idKey()].position + trade.position
                if self.ins_map[trade.instrument.idKey()].position == 0:
                    del self.ins_map[trade.instrument.idKey()]
            else:
                self.ins_map[trade.instrument.idKey()] = copy.copy(trade)
            
            if trade.instrument.detailed == False:
                detailedIns = self.trading_platform.getInstrumentDetails(trade.instrument, timestamp)
            else:
                detailedIns = trade.instrument
            if detailedIns.instrument_type == "option":
                price = detailedIns.mid
            cashChange += -(price * trade.position)

        self.cash.addCash(cashChange)
        _logger.info("updated portfolio")
        _logger.debug(f"Porfolio state after update:")
        greeks = self.getPortfolioGreeks(timestamp)
        _logger.debug(f"delta: {greeks.delta}, gamma: {greeks.gamma}, vega: {greeks.vega}, theta: {greeks.theta}, sigma: {greeks.sigma}, contract_size: {self.getContractSize()}")
        self.getPortfolioGreeks(timestamp)

    def getContractSize(self):
        position = 0
        for key, val in self.ins_map.items():
            position += abs(val.position)

        return position

    def getMarkToMarket(self, timestamp):
        total_m2m = 0
        for key, val in self.ins_map.items():
            try:
                insDetailed: OptionDetailed = getInstrumentDetailsWithBlackscholes(val.instrument, timestamp, self.trading_platform, _logger)
                total_m2m += insDetailed.mid*val.position
            except Exception as e:
                _logger.critical(e)
        return total_m2m

    def getPortfolioGreeks(self, timestamp: datetime):
        '''
        Fetches the portfolio greeks
        '''
        greeks = Greeks()
        for key, val in self.ins_map.items():
            try:
                insDetailed: OptionDetailed = getInstrumentDetailsWithBlackscholes(val.instrument, timestamp, self.trading_platform, _logger)
                greeks.delta += insDetailed.delta * val.position
                greeks.theta += insDetailed.theta * val.position
                greeks.gamma += insDetailed.gamma * val.position
                greeks.vega += insDetailed.vega * val.position
                greeks.sigma += insDetailed.sigma * val.position
            except Exception as e:
                _logger.critical(e)
        return greeks
