from datetime import datetime, date, time
from typing import List, Dict
from trading_platform._TradingPlatform import TradingPlatform
from tradelib.models.Instrument import Instrument
from tradelib.models.OptionDetailed import OptionDetailed
from tradelib.models.Option import Option
from tradelib.models.Trade import Trade
import os
import pandas as pd
from tradelib_logger import logger
from tradelib_global_constants import date_format, DIVIDEND_MAP, data_dir
from tradelib_utils import get_options_intraday_filename, get_expiry_dates_on_weekday, round_to_step, get_expiry_from_pre_process, get_trading_days_from_pre_process, get_spot_intraday_filename,get_options_intraday_filename_2, is_file_exist
import copy

# TODO: partial exceution allow or not
# TODO: allow underlying object pass

_logger = logger.getLogger("trading platform")
class BacktestTradingPlatform(TradingPlatform):
    def __init__(self, option_data_dir: str, risk_free_rate: float=None, steps: float=None,underlying: str=None, dividend: float=None) -> None:
        self.option_data_dir = option_data_dir
        super().__init__(underlying, dividend, steps, risk_free_rate)
        self.intraday_options_df: pd.Dataframe = None
        self.lowest_level_options_df: pd.DataFrame = None

        self.intraday_options_dict: dict = {}
        self.lowest_level_options_dict: dict = {}

        self.spot_date = None
        self.intraday_spot_df = pd.DataFrame()
        self.spot = None
        self.spot_datetime = None

        self.date: date = None
        self.timestamp: datetime = None
        self.expiry_date = None
        self.cache: Dict[str, Instrument] = {} # this map caches the data from searching dataframe
        self.expiry_list = get_expiry_from_pre_process(data_dir=data_dir)
        self.trading_date_list = get_trading_days_from_pre_process(data_dir=data_dir)

# Sumegh
    def __checkAndReinitialiseDF(self, timestamp:datetime, expiry_date: date) -> None:
        '''
        Reinitialised the intraday DF if the date is changed.
        '''
        date_curr: date = timestamp.date()

        if date_curr != self.date:
            del self.intraday_options_dict
            self.intraday_options_dict = {}

            self.date = date_curr

        if expiry_date not in self.intraday_options_dict.keys():
            try:
                filepath = os.path.join(self.option_data_dir, get_options_intraday_filename_2(self.underlying, timestamp, expiry_date))
                if is_file_exist(filepath):
                    df = pd.read_csv(filepath)

                    df['Date Time'] = pd.to_datetime(df['Date Time'])
                    df.set_index('Date Time', inplace=True)

                    self.intraday_options_dict[expiry_date] = df

                else:
                    self.intraday_options_dict[expiry_date] = pd.DataFrame()
                    raise Exception(f"Data not exist for tradingday: {date_curr} expiryday: {expiry_date}")
                    # df = pd.DataFrame()

            except Exception as e:
                raise Exception(e)

        
        del self.lowest_level_options_df
        if len(self.intraday_options_dict[expiry_date]) != 0:
            try:
                self.lowest_level_options_df = self.intraday_options_dict[expiry_date].loc[timestamp]

            except Exception as e:
                self.lowest_level_options_df = pd.DataFrame()
                raise Exception(f"Data not found at time {timestamp} for expiry {expiry_date} :: {e}")

        else:
            self.lowest_level_options_df = pd.DataFrame()

# Sumegh
    def __getOptionDetailedFromDF(self, ins: Option, timestamp: datetime)-> OptionDetailed:
        '''
        Fetches the detailed option values for a timestamp from the intraday dataframe

        None:
        All of the Exceptions are taken care by utils file.
        '''
        insDetailed = None

        if len(self.lowest_level_options_df) == 0:
            raise Exception(f"Data not available for the trading date {timestamp} and expiry date {ins.expiry.strftime(date_format)}.")

        insdf = self.lowest_level_options_df[(self.lowest_level_options_df['Strike'] == int(ins.strike)) & (self.lowest_level_options_df['ExpiryDate'] == ins.expiry.strftime(date_format)) & (self.lowest_level_options_df['Type'] == ins.opt_type) & (self.lowest_level_options_df['Instrument'] == ins.underlying)]
        
        # insdf = self.intraday_options_df.loc[timestamp][(self.intraday_options_df.loc[timestamp]['Strike'] == int(ins.strike)) & (self.intraday_options_df.loc[timestamp]['ExpiryDate'] == ins.expiry.strftime("%Y-%m-%d")) & (self.intraday_options_df.loc[timestamp]['Type'] == ins.opt_type) & (self.intraday_options_df.loc[timestamp]['Instrument'] == ins.underlying)]
        
        if len(insdf) == 0:
            raise Exception(f"No data found in market for instrument {ins.idKey()} for the {timestamp}.")

        
        insData = insdf.loc[timestamp]
        insDetailed = OptionDetailed(timestamp, insData['ExchToken'], insData['Strike'], pd.to_datetime(insData['ExpiryDateTime']).date(), insData['Instrument'], insData['Type'], insData['BidPrice'], insData['AskPrice'], insData['Delta'], insData['Theta'], insData['Gamma'], insData['Vega'], insData['Sigma'], insData['Spot'])
        # insDetailed = OptionDetailed(timestamp, insData['ExchToken'], insData['Strike'], pd.to_datetime(insData['ExpiryDateTime']).date(), insData['Instrument'], insData['Type'], insData['BidPrice'], insData['AskPrice'], insData['Delta'], insData['Theta'], insData['Gamma'], insData['Vega'], insData['Sigma'], self.getSpot(timestamp=timestamp))

        return insDetailed

# Sumegh
    def __getInstrumentDetailsFromDF(self, ins: Instrument, timestamp: datetime) -> Instrument:
        '''
        Fetches detailed instrument values from intraday dataframe

        None:
        All of the Exceptions are taken care by utils file.
        '''
        insDetailed = None

        try:
            if ins.instrument_type == "option":
                insDetailed: Instrument = self.__getOptionDetailedFromDF(ins, timestamp)

            if insDetailed == None:
                raise Exception("For instrument type data fetcher not implemented.")
        except Exception as e:
            raise e

        return insDetailed

# Sumegh
    def getAllInstrumentDetailsTable(self, timestamp: datetime, expiry_date: date) -> pd.DataFrame:
        '''
        For a timestamp and underlying returns all of the available options in a dataframe
        '''
        self.__checkAndReinitialiseDF(timestamp, expiry_date)
        return self.lowest_level_options_df

# Sumegh
    # TODO: implement this to make code faster
    def getInstrumentDetailsList(self, insList: List[Instrument], timestamp: datetime, expiry_date: date) -> List[Instrument]:
        self.__checkAndInvalidateCache(timestamp, expiry_date)
        pass

# Sumegh Not used
    def __getSpotFromDF(self) -> float:
        '''
        Fetches the spot price for a timestamp and underlying
        '''
        spot = None
        spot = round(float(self.lowest_level_options_df['Spot'].values[0]),2)
        return spot

# Sumegh
    def __checkAndReinitialiseSpotDF(self, timestamp) -> None:
        curr_date = timestamp.date()

        if curr_date != self.spot_date:
            del self.intraday_spot_df

            filepath = os.path.join(self.option_data_dir, "Spot", get_spot_intraday_filename(self.underlying, curr_date))
            self.intraday_spot_df = pd.read_csv(filepath)

            self.intraday_spot_df['Date Time'] = pd.to_datetime(self.intraday_spot_df['Date Time'])
            self.intraday_spot_df.set_index('Date Time', inplace=True)

            self.spot_date = curr_date

# Sumegh
    def getInstrumentDetails(self, ins: Instrument, timestamp: datetime) -> Instrument:
        '''
        Fetches details of an instrument for a timestamp and instrument
        '''
        self.__checkAndInvalidateCache(timestamp, expiry_date=ins.expiry)

        if ins.idKey() not in self.cache.keys():
            self.cache[ins.idKey()] = self.__getInstrumentDetailsFromDF(ins, timestamp)
        
        return self.cache[ins.idKey()]

# Sumegh
    def getSpot(self, timestamp: datetime) -> float:
        '''
        Fetches spot for a timestamp
        '''
        self.__checkAndReinitialiseSpotDF(timestamp)

        if timestamp != self.spot_datetime:
            self.spot = round(self.intraday_spot_df.loc[timestamp]['Spot'], 2)

            self.spot_datetime = timestamp
        
        return self.spot

        # self.__checkAndReinitialiseSpotDF(timestamp)

        # if "SPOT" not in self.cache.keys():
        #     self.cache["SPOT"] = self.__getSpotFromDF()

        # return self.cache["SPOT"]


    def submitOrder(self, tradeList: List[Trade]):
        return tradeList

# Sumegh
    def __checkAndInvalidateCache(self, timestamp:datetime, expiry_date: date, forceful: bool = False):
        '''
        invalidates caches after checking timestamp
        '''
        if forceful or timestamp != self.timestamp:# or expiry_date != self.expiry_date:
            self.cache.clear()
            self.timestamp = timestamp

        self.__checkAndReinitialiseDF(timestamp, expiry_date)
        
            # self.expiry_date = expiry_date

# Not used in any other module
    def getNearestExpiries(self, timestamp: datetime) -> List[date]:
        if (len(self.expiry_list) == 0):
            return []
        while timestamp.date() > self.expiry_list[0]:
            if (len(self.expiry_list) == 0):
                return []
            self.expiry_list.pop(0)
        return copy.copy(self.expiry_list)


    def get_expiry_list(self):
        return self.expiry_list

