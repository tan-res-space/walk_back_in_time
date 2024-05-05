'''
This module contains the utility functions required
'''

from datetime import datetime,timedelta, date
from typing import Tuple
from tradelib.tradelib_global_constants import date_format_no_dash, date_time_format_no_dash, ROOT_DIR, date_time_format, date_format, data_dir, meta_data_file_name
from tradelib.models.Instrument import Instrument
import copy
from typing import List
from math import ceil, floor
import json
import re
import os
import sys

from tradelib_logger import logger
_logger = logger.getLogger("tradelib_utils")


def get_weekday_dates_year(year:int, dayname: str) -> List[date]:
    start_date = date(year, 1, 1)

    while start_date.strftime("%A").upper() != dayname.upper():
        start_date += timedelta(days=1)

    weekday_list: List[date] = []
    while start_date.year == year:
        weekday_list.append(start_date)
        start_date += timedelta(days=7)

    return weekday_list

def get_date_from_data_file(filename: str, underlying: str) -> date:
    try:
        return datetime.strptime(filename, f'{underlying}_%Y%m%d_Intraday_Preprocessed.csv').date()
    except Exception:
        return None
    
def get_last_weekday(year):
    itr_date = date(year, 12, 1)

    while itr_date.day in [5, 6]:
        itr_date.day -= timedelta(days=1)
    
    return itr_date
    
def is_csv_empty(file_path):
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
        second_line = f.readline().strip()
    if second_line:
        return False
    else:
        return True

def get_holiday_list(year, file_path=None) -> List[date]:
    holiday_list_folder = os.path.join(ROOT_DIR, 'data', 'holiday_lists')
    if file_path == None:
        file_path = os.path.join(holiday_list_folder, f'holidays_{year}.json')
    
    try:
        with open(file_path, 'r') as fp:
            date_list = json.load(fp)
            date_list = list(map(lambda x: datetime.strptime(x, date_time_format).date(), date_list))
            return date_list
    except Exception as e:
        raise Exception(f"Please add holiday list for year {year} in {holiday_list_folder}")


def is_data_available_for_date(data_dir: str, _date: date, underlying: str):
    filename = get_options_intraday_fileName(underlying, _date)
    file_list = os.listdir(data_dir)

    if filename in file_list:
        if not is_csv_empty(os.path.join(data_dir, filename)):
            return True

    return False

def is_data_available_for_date_expiry(data_dir: str, underlying: str, _date: date, _expiry_date: date):
    
    return os.path.exists(os.path.join(data_dir, get_options_intraday_filename_2(underlying=underlying, timestamp=_date, expiry_date=_expiry_date)))

def is_file_exist(filepath: str) -> bool:
    return os.path.exists(filepath)

def get_underlying_from_data_dir(data_dir: str):
    data_files = list(filter(lambda filename: re.match( r"\w+_\d{8}_Intraday_Preprocessed.csv", filename), os.listdir(data_dir)))

    if (len(data_files) == 0):
        raise Exception("No files found in data dir")

    underlying = data_files[0].split('_')[0]
    return underlying

def get_data_available_dates(data_dir):
    data_files = list(filter(lambda filename: re.match( r"\w+_\d{8}_Intraday_Preprocessed.csv", filename), os.listdir(data_dir)))
    data_available_dates:List[date] = []

    if (len(data_files) == 0):
        raise Exception("No files found in data dir")

    underlying = data_files[0].split('_')[0]

    for data_file in data_files:
        data_file_date = get_date_from_data_file(data_file, underlying)
        if data_file_date != None:
            if not is_csv_empty(os.path.join(data_dir, data_file)):
                data_available_dates.append(data_file_date)
    
    if (len(data_available_dates) == 0):
        raise Exception("No data found in data dir")
    
    return data_available_dates

def get_expiry_dates_on_weekday(data_dir: str, dayname: str):
    data_available_dates = get_data_available_dates(data_dir)

    year = data_available_dates[0].year
    weekday_dates = get_weekday_dates_year(year, dayname)
    last_weekday = get_last_weekday(year)
    if last_weekday not in weekday_dates:
        weekday_dates.append(last_weekday)
    expiry_dates:List[date] = []
    for _date in weekday_dates:
        while ((len(expiry_dates) > 0) and _date > expiry_dates[-1]) or _date.year == year:
            if _date in data_available_dates:
                expiry_dates.append(_date)
                break
            else:
                _date -= timedelta(days=1)
        
    return expiry_dates


def is_component_execution_time(skip_counter:int, skip_count:int) -> Tuple[bool, int]:
    '''
    utility function to decide whether current time is for taking actions

    Parameters:
    skip_counter (int): the counter which is keeping track of the skip
    skip_count (int): how many time steps to skip

    Reurns:
    is_it_time,skip_counter (bool,int): ({True => take action,False => skip},{altered counter})
    '''
    # check whether skip counter is 0
    is_it_time = False
    if skip_counter == skip_count:
        # when skip counter is 0 return True
        is_it_time = True
        # update the skip counter to actual skip trade count
        skip_counter = 1
    else:
        # reduce the counter
        skip_counter += 1

    return (is_it_time, skip_counter)

def round_to_step(price: float, steps: float):
    floor_strike = floor(price/steps)*steps
    ceil_strike = ceil(price/steps)* steps

    if abs(ceil_strike- price) < abs(floor_strike - price):
        return ceil_strike
    else:
        return floor_strike

def moneyness_percentage(strike: float, spot: float, opt_type: str) -> float:
    # _logger.info(f"Inside tradelib_utils {spot}, {strike}, {opt_type}, moneyness={strike/spot-1}")
    if opt_type == "PE":
        return round(((spot-strike)/spot) * 100, 2)
    elif opt_type == "CE":
        return round(((strike-spot)/spot) * 100 , 2)
    else:
        raise Exception("Illegal Option type")
    
def round_to_lot_size(input_qty:float, lot_size, round_type: str="ROUND"):
    '''
    This function rounds or floors the trade quantity to respective lot size.
    This operation is done here to ensure that whichever module creates a new Trade object the lot_size is applied properly.
    
    Logic:
        The quantity is floored or rounded to the nearest multiple of respective lot size.
        e.g. if quantity is 39 and lot_size is 15, then 
                for round it will be rounded to 45
                for floor it will be floored to 30

    '''
    tmp_qty = abs(input_qty)
    qty_sign = 1 if input_qty >= 0 else -1 #input_qty//tmp_qty
    lot_round_strategy = round_type
    lot_size = lot_size
    out_qty = 0
    if lot_round_strategy == 'FLOOR':
        floor_val = tmp_qty//lot_size
        out_qty = floor_val * lot_size
    elif lot_round_strategy == 'ROUND':
        floor_val = tmp_qty//lot_size * lot_size
        ceil_val = floor_val + lot_size
        mid_val = round((ceil_val + floor_val)/2,1)
        if tmp_qty >= mid_val:
            out_qty = ceil_val
        else:
            out_qty = floor_val
    elif lot_round_strategy == 'CEIL':
        floor_val = tmp_qty // lot_size
        frac_part = tmp_qty - floor_val*lot_size

        out_qty = floor_val
        # if there is fractional part 
        if (frac_part) > 0.001: # floating point equals are errorprone so using delta of 0.001 as positions are rounded to 2 least significant digits
            out_qty += 1
        out_qty = out_qty*lot_size
    elif lot_round_strategy == 'EXACT':
        out_qty = tmp_qty
    else:
        raise Exception("ROUNDING STRATEGY ISN'T CONFIGURED IN PARAMS PROPERLY")
    return out_qty * qty_sign


def get_portfolio_filename(timestamp: datetime, client_name: str):
    '''
    generates name of the portfolio dump csv
    '''
    return f"{client_name}_portfolio_{timestamp.strftime(date_format_no_dash)}.csv"

def get_options_intraday_fileName(underlying: str, timestamp: date):
    '''
    Generates the intraday filename for an Option file
    '''
    return f"{underlying}_{timestamp.strftime(date_format_no_dash)}_Intraday_Preprocessed.csv"

def serialiseInstrument(ins: Instrument, allow_all: bool=True, allowed_fields: List[str]=[]):
    '''
    Serialises object to store as a string
    '''
    ins_dict = copy.copy(ins.__dict__)
    keys = list(ins_dict.keys())
    if not allow_all:
        for key in keys:
            if key not in allowed_fields:
                del ins_dict[key]
    if 'expiry' in ins_dict:
        ins_dict['expiry'] = ins_dict['expiry'].strftime(date_format)
    if 'timestamp' in ins_dict:
        ins_dict['timestamp'] = ins_dict['timestamp'].strftime(date_time_format)
    return ins_dict

def get_options_intraday_filename(underlying: str, timestamp: datetime):
    '''
    Generates the intraday filename for an Option file
    '''
    date_str = timestamp.strftime(date_format_no_dash)
    return f"{underlying}_{date_str}_Intraday_Preprocessed.csv"

def get_options_intraday_filename_2(underlying: str, timestamp: datetime, expiry_date: date):
    '''
    Generates the intraday filename for an Option file
    '''
    date_str = timestamp.strftime(date_format_no_dash)
    expiry_date_str = expiry_date.strftime(date_format_no_dash)
    return f"{underlying}_{date_str}_{expiry_date_str}_Intraday_Preprocessed.csv"

def get_spot_intraday_filename(underlying: str, timestamp: datetime):
    '''
    Generates the intraday filename for an Option file
    '''
    date_str = timestamp.strftime(date_format_no_dash)
    return f"{underlying}_{date_str}_Intraday_Spot.csv"

def get_blotter_filename(start_timestamp: datetime, end_timestamp: datetime):
    return f"blotter_{start_timestamp.strftime(date_time_format_no_dash)}_{end_timestamp.strftime(date_time_format_no_dash)}.csv"

def get_backtest_filename(_date: date):
    return f"{_date.strftime(date_format_no_dash)}.csv"

def get_expiry_from_pre_process(data_dir: str, meta_data_file_name: str = "meta_data.json"):
    meta_data_file = os.path.join(data_dir, meta_data_file_name)

    file = open(meta_data_file)
    data = json.load(file)

    expiry_date_set = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d").date(), data['Expiry_dates']))

    return expiry_date_set

def get_trading_days_from_pre_process(data_dir: str, meta_data_file_name: str = "meta_data.json"):
    meta_data_file = os.path.join(data_dir, meta_data_file_name)

    file = open(meta_data_file)
    data = json.load(file)

    trading_date_set = list(map(lambda x: datetime.strptime(x, "%Y-%m-%d").date(), data['Trading_dates']))

    return trading_date_set


def get_theoretical_date(info_list: list[tuple], current_date:date):
    """
    Calculate theoretical future dates based on a list of weekdays and occurrences.

    Args:
        info_list (List[Tuple[str, int]]): A list of tuples, each containing a weekday string and
            the desired number of occurrences of that weekday.
        current_date (date): The starting date for calculating theoretical dates.

    Returns:
        Dict[str, List[date]]: A dictionary where keys are weekdays and values are lists of
            datetime.date objects representing theoretical dates.

    Example:
        info_list = [('Thursday', 2), ("Friday", 1)]
        current_date = datetime(year=2023, month=9, day=4).date()

        output = {
                    'Thursday': [datetime.date(2023, 9, 7),datetime.date(2023, 9, 14)],
                    'Friday': [datetime.date(2023, 9, 8)]
                }
    """
    # TODO: use a precomputed theoretical list to get the dates.
    # TODO: dont use 1 day delta once a date is found use 7 day delta

    current_date_copy = current_date
    end_date = datetime(year=current_date.year, month=12, day=31).date()

    theoretical_day_dict = {}
    for info in info_list:
        day_on_expiry = info[0]
        number_of_expiry = info[1]

        date_list = []
        while current_date_copy <= end_date:
            if current_date_copy.strftime("%A") == day_on_expiry:
                date_list.append(current_date_copy)

            if len(date_list) == number_of_expiry:
                break

            current_date_copy += timedelta(days=1)

        theoretical_day_dict[day_on_expiry] = date_list
        current_date_copy = current_date

    return theoretical_day_dict


def get_unwind_date_for_an_expiry(expiry_date, meta_data_file_name=meta_data_file_name, data_dir=data_dir):
    unwind_dict = json.load(open(os.path.join(data_dir, meta_data_file_name)))['Unwind_dates']

    return datetime.strptime(unwind_dict[datetime.strftime(expiry_date, date_format)], date_format).date()


def get_exception_line_no():
    
    _,_, exception_traceback = sys.exc_info()
    line_number = exception_traceback.tb_lineno
    
    return line_number