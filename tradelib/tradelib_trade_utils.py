from trading_platform import TradingPlatform
from datetime import date, datetime, time, timedelta
from typing import List, Tuple
from models.Option import Option
from models.OptionDetailed import OptionDetailed
from logging import Logger
from tradelib_utils import moneyness_percentage, round_to_step
from tradelib_global_constants import HARD_LIMIT, unwind_time
import pandas as pd
from tradelib_utils import get_theoretical_date, get_unwind_date_for_an_expiry

import warnings

# Ignore the SettingWithCopyWarning
warnings.filterwarnings("ignore")


def get_nearest_strike_option(trading_platform: TradingPlatform, price: float, _logger: Logger, opt_type:str, nearest_expiry_date: date, underlying: str, timestamp: datetime) -> OptionDetailed:
    opt: OptionDetailed = None
    i = 0
    steps = trading_platform.steps
    rounded_price = round_to_step(price, steps)
    while opt == None and i <= HARD_LIMIT:
        i += 1
        try:
            if (i == 1):
                try:
                    opt = trading_platform.getInstrumentDetails(Option(strike=rounded_price, expiry=nearest_expiry_date, underlying=underlying, opt_type=opt_type), timestamp)
                except:
                    opt = None
                
                if opt == None:
                    _logger.debug(f"{opt_type} Options with strike {rounded_price} not found")
            else:
                upper_price = rounded_price+steps*(i-1)
                lower_price = rounded_price-steps*(i-1)
                try:
                    opt1: OptionDetailed = trading_platform.getInstrumentDetails(Option(upper_price, nearest_expiry_date, underlying, opt_type), timestamp)
                except:
                    opt1 = None

                try:
                    opt2: OptionDetailed = trading_platform.getInstrumentDetails(Option(lower_price, nearest_expiry_date, underlying, opt_type), timestamp)
                except:
                    opt2 = None

                if (opt1 == None) ^ (opt2 == None):
                    opt = opt1 if opt1 != None else opt2
                    not_found_strike = upper_price + lower_price - opt.strike
                    _logger.debug(f"{opt_type} Option with strike {opt.strike} found, but with strike {not_found_strike} not found")
                elif opt1 == None and opt2 == None:
                    _logger.debug(f"{opt_type} Options with strikes {upper_price}, {lower_price} not found")
                    continue
                else:
                    _logger.debug(f"{opt_type} Options with strikes {opt1.strike}, {opt2.strike} found")
                    if abs(opt1.strike - price) < abs(opt2.strike - price):
                        opt = opt1
                    else:
                        opt = opt2
                _logger.info(f"{opt_type} Option with strike {opt.strike} chosen as closest to price {price}")
        except Exception as e:
            _logger.critical("something went wrong with function: get_nearest_strike_option (tradelib/tradelib_trade_utils)")
            raise e
    if opt != None:
        return opt
    else:
        _logger.warning(f"No {opt_type} Option near {price} found with price variation: {steps*(i-1)}")
        raise Exception((f"No {opt_type} Option near {price} found with price variation: {steps*(i-1)}"))


def get_atm_option(trading_platform: TradingPlatform, underlying: str, spot:float, opt_type: str, nearest_expiry_date: date, steps:float, timestamp: datetime, _logger: Logger) -> OptionDetailed:
    try:
        atm = get_nearest_strike_option(trading_platform, spot, _logger, opt_type, nearest_expiry_date, underlying, timestamp)
        _logger.debug(f"ATM {opt_type} Found, Price: {atm.mid}, Moneyness Percentage: {atm.moneyness}, strike: {atm.strike}")
        return atm
    except Exception as e:
        _logger.critical(f"No ATM {opt_type} found")
        return None

def get_static_otm_option(trading_platform: TradingPlatform, 
                          underlying: str, 
                          atm_strike:int, 
                          outstrike: float, 
                          steps:float, 
                          tolerance: bool, 
                          strict_tolerance: bool, 
                          opt_type: str, 
                          nearest_expiry_date: date,
                          timestamp: datetime, 
                          _logger: Logger) -> Tuple[OptionDetailed, OptionDetailed]:
    cal_sign = 1 if opt_type == "PE" else -1
    otm_strike = round_to_step(atm_strike - (cal_sign)*atm_strike*(outstrike/100), steps)

    otm = None
    step_to_look = 0
    while otm == None and cal_sign*(otm_strike + (cal_sign)*step_to_look) < cal_sign*atm_strike: #TODO
        if strict_tolerance and (step_to_look/otm_strike) * 100 > tolerance:
            _logger.debug(f"otm {opt_type} not found within tolerance({tolerance}%), and strict tolerance is on")
            break

        try:
            otm: OptionDetailed = trading_platform.getInstrumentDetails(Option(strike=otm_strike+(cal_sign)*step_to_look, expiry=nearest_expiry_date, underlying=underlying, opt_type=opt_type), timestamp)
        except:
            try:
                if (step_to_look == 0):
                    raise Exception
                _logger.debug(f"OTM {opt_type} not found, with strike: {otm_strike+(cal_sign)*step_to_look}")
                otm: OptionDetailed = trading_platform.getInstrumentDetails(Option(strike=otm_strike-(cal_sign)*step_to_look, expiry=nearest_expiry_date, underlying=underlying, opt_type=opt_type), timestamp)
            except:
                _logger.debug(f"OTM {opt_type} not found, with strike: {otm_strike-(cal_sign)*step_to_look}")
                _logger.debug(f"OTM {opt_type} not found, with step_to_look: {step_to_look}")
                step_to_look += steps
                continue

    if otm != None:
        _logger.info(f"OTM {opt_type} Found, Strike: {otm.strike} Price: {otm.mid} Moneyness Percentage: {otm.moneyness}, tolerance: {round(abs(otm_strike-otm.strike)/otm_strike, 2)}")
    else:
        _logger.debug(f"OTM {opt_type} not found")

    return otm

# TODO: check tolerance should be added or not
def get_nearest_strike_option_towards_atm(trading_platform: TradingPlatform, strike: float, _logger: Logger, opt_type:str, nearest_expiry_date: date, underlying: str, timestamp: datetime, atm_strike: float) -> OptionDetailed:
    steps = trading_platform.steps
    cal_sign = 1 if opt_type == "PE" else -1
    opt = None
    step_to_look = steps
    while opt == None and cal_sign*(strike + (cal_sign)*step_to_look) < cal_sign*atm_strike: #TODO
        try:
            opt: OptionDetailed = trading_platform.getInstrumentDetails(Option(strike=strike+(cal_sign)*step_to_look, expiry=nearest_expiry_date, underlying=underlying, opt_type=opt_type), timestamp)
        except:
            try:
                _logger.debug(f"Option {opt_type} not found, with strike: {strike+(cal_sign)*step_to_look}")
                opt: OptionDetailed = trading_platform.getInstrumentDetails(Option(strike=strike-(cal_sign)*step_to_look, expiry=nearest_expiry_date, underlying=underlying, opt_type=opt_type), timestamp)
            except:
                _logger.debug(f"Option {opt_type} not found, with strike: {strike-(cal_sign)*step_to_look}")
                _logger.debug(f"Option {opt_type} not found, with step_to_look: {step_to_look}")
                step_to_look += steps
                continue

    if opt != None:
        _logger.info(f"Nearset Option {opt_type} Found, Strike: {opt.strike} Price: {opt.mid} Moneyness Percentage: {opt.moneyness}, tolerance: {round(abs(strike-opt.strike)/strike, 2)}")
    else:
        _logger.debug(f"No nearest Option {opt_type} found near {strike}, have crossed ATM, thus skipping.")

    return opt


def get_delta_based_otm_option(trading_platform: TradingPlatform, 
                          underlying: str, 
                          delta_outstrike: float, 
                          tolerance: int, 
                          strict_tolerance: bool, 
                          opt_type: str, 
                          nearest_expiry_date: date,
                          timestamp: datetime, 
                          _logger: Logger) -> OptionDetailed:

    cal_sign = 1 if opt_type == "CE" else -1

    data = trading_platform.getAllInstrumentDetailsTable(timestamp=timestamp, expiry_date=nearest_expiry_date)

    data = data.loc[(data['Type']==opt_type)]
    # if we want to use tolerance
    if strict_tolerance:
        delta_upper_limit = cal_sign * (delta_outstrike/100) + (tolerance/100)
        delta_lower_limit = cal_sign * (delta_outstrike/100) - (tolerance/100)

        data = data.loc[(data['Delta']>=delta_lower_limit) & (data['Delta']<=delta_upper_limit)]

    data.loc[:, "Delta Difference"] = abs(abs(data['Delta']) - (delta_outstrike/100)).values

    try:
        searched_data = data.sort_values(by="Delta Difference").iloc[0]

        otm: OptionDetailed = OptionDetailed(timestamp=timestamp, 
                                                id=searched_data['ExchToken'], 
                                                strike=searched_data['Strike'], 
                                                expiry=nearest_expiry_date, 
                                                underlying=underlying, 
                                                opt_type=opt_type, 
                                                bid=searched_data['BidPrice'], 
                                                ask=searched_data['AskPrice'], 
                                                delta=searched_data['Delta'], 
                                                theta=searched_data['Theta'], 
                                                gamma=searched_data['Gamma'], 
                                                vega=searched_data['Vega'], 
                                                sigma=searched_data['Sigma'], 
                                                spot=searched_data['Spot']
                                                )

    except:
        otm = None


    if otm != None:
        _logger.info(f"OTM {otm.opt_type} Found, Strike: {otm.strike} Price: {otm.mid} Moneyness Percentage: {otm.moneyness} Delta: {otm.delta}")
    else:
        _logger.debug(f"OTM {opt_type} not found")

    return otm


def get_actual_expiry_dates(theoretical_dates_dict: dict, trading_platform: TradingPlatform):
    # get explist
    expiry_date_list = trading_platform.get_expiry_list()

    actual_expiry_date_dict = {}

    for theo_day in theoretical_dates_dict.keys():
        collected_list = []

        for date in theoretical_dates_dict[theo_day]:
            if date in expiry_date_list:
                collected_list.append(date)

            else:
                collected_list.append(None)

        actual_expiry_date_dict[theo_day] = collected_list

    return actual_expiry_date_dict

def get_early_expiry_date(theoretical_expiry_date: date, current_date: date, trading_platform: TradingPlatform, _logger: Logger):
    
    # get explist
    expiry_date_list = trading_platform.get_expiry_list()
    # expiry_date_list = expiry_date_set

    if theoretical_expiry_date in expiry_date_list:
        return theoretical_expiry_date
    
    else:
        # print(f"{date} is not a expiry dates. Searching for a nearest expiry.")
        counter = 0
        date1 = theoretical_expiry_date
        while counter < 7:
            date1 = date1 - timedelta(days=1)

            if date1 < current_date:
                _logger.info(f"Found no expiry date For {theoretical_expiry_date}")
                # collected_list.append(None)
                return None

            if date1 in expiry_date_list:
                _logger.info(f"Found nearest expiry date For {theoretical_expiry_date} as {date1}")
                # is_run = False

                return date1

            counter += 1
            
        _logger.critical(f"Found no expiry date For {theoretical_expiry_date}")
        return None


def get_late_expiry_date(theoretical_expiry_date, trading_platform: TradingPlatform, _logger: Logger):

    # get explist
    expiry_date_list = trading_platform.get_expiry_list()
    # expiry_date_list = expiry_date_set

    if theoretical_expiry_date in expiry_date_list:
        return theoretical_expiry_date
    
    else:
        # print(f"{date} is not a expiry dates. Searching for a nearest expiry.")
        counter = 0
        date1 = theoretical_expiry_date
        while counter < 7:
            date1 = date1 + timedelta(days=1)

            if date1 in expiry_date_list:
                _logger.info(f"Found nearest expiry date For {theoretical_expiry_date} as {date1}")

                return date1

            counter += 1

        _logger.critical(f"Found no expiry date For {theoretical_expiry_date}")
        return None


def is_trading_date(date: date, trading_platform: TradingPlatform) -> bool:
    return date in trading_platform.trading_date_list

def get_expiry_date(timestamp: datetime, expiry_info, trading_platform, _logger):

    theoretical_dates = get_theoretical_date(info_list=expiry_info, current_date=timestamp.date())
    actual_expiry_dates = get_actual_expiry_dates(theoretical_dates_dict=theoretical_dates, trading_platform = trading_platform)

    exp_list = []
    for day in actual_expiry_dates.keys():
        # print(i)
        for j, date in enumerate(actual_expiry_dates[day]):
            if date == None:
                
                theory_date = theoretical_dates[day][j]
                _logger.info(f"{theory_date} ({day}) is not an expiry going for early expiry.")

                exp_date = get_early_expiry_date(theory_date, timestamp.date(), trading_platform=trading_platform, _logger = _logger)

                # HOT FIX for Banknifty 2023. Need to change the code.
                if (exp_date == None) & (theory_date == timestamp.date()):
                    _logger.info(f"{theory_date} ({day}) is not an expiry going for late expiry.")

                    exp_date = get_late_expiry_date(theory_date, trading_platform=trading_platform, _logger=_logger)

                if exp_date != None:
                    _logger.info(f"Found early expiry for {theory_date} ({day}) is {exp_date} ({exp_date.strftime('%A')})")

                exp_list.append(exp_date)
                
            else:
                exp_list.append(date)

    if len(exp_list) <= 1:
        return exp_list[0]

    unwind_date = get_unwind_date_for_an_expiry(expiry_date=exp_list[0])
    if timestamp >= datetime.combine(unwind_date, unwind_time):
        return exp_list[1]

    return exp_list[0]