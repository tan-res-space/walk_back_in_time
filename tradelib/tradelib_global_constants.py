import os, json
import numpy as np
from datetime import timedelta, date, time, datetime

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(ROOT_DIR, 'outputs')
DISABLE_DEBUG = False

### BASIC CONFIGURATION OF BACKTEST ENGINE
allowed_fields_in_portfolio_dump = ['instrument_type', 'strike', 'expiry', 'opt_type']

initial_cash = 0.0
start_date = date(2020, 1, 1)
end_date = date(2020, 12, 31)
trade_start_time = time(9, 20, 0)
trade_end_time = time(15, 30, 0)
unwind_time= time(15, 16, 0)
underlying = "NIFTY"
interval = timedelta(minutes=1)
strategy_to_execute = 'CONSTANT_RISK_CONDOR_STRATEGY' # "DELTA_CONDOR_STRATEGY" # "DELTA_CONDOR_STRATEGY"
delta_otm = False # true if using delta otm else false - mainly for chartpacks
date_time_format = "%Y-%m-%d %H:%M:%S"
date_format = "%Y-%m-%d"
date_format_no_dash = "%Y%m%d"
date_time_format_no_dash = "%Y%m%d%H%M%S"
tolerance = 10
strict_condor = True
strict_tolerance = True
take_later_expiry_hedge = False
delta_threshold = 0.01
RISK_FREE_RATE = 0.02
exchange_start_time = time(9, 15, 0)
exchange_end_time = time(15, 30, 0)
avg_trade_days_in_year = 252
number_of_holidays = 16
HARD_LIMIT = 20

gamma_cleanup = False
gamma_hedge = False

trade_interval_time = 5 # in minutes
hedge_interval_time = 2 # in minutes
unwind_time_before = 14 # in minutes

OTM_outstrike = 5

expiry_info = [('Thursday', 2)]

# underlying map need
steps = 50
unit_size = 50
lot_size = 50

### Client configuration
client_name = "EIS"

### Folder configuration
data_dir = "/home/cloudcraftz/Downloads/Constant Risk Data/2020"
meta_data_file_name = 'meta_data.json'

## delta
delta_otm_tolerance = 50
delta_outstrike = 5

### UnderlyingConfiuration (include in underlying map)
DIVIDEND_MAP = {
    "BANKNIFTY": 0.0078,
    "NIFTY": 0.0138
}

STEPS_MAP = {
    "BANKNIFTY": 100,
    "NIFTY": 50
}

# output folder
output_dir_folder = os.path.join(OUTPUT_DIR, str(start_date.year) + "_" + underlying + "_" + strategy_to_execute)
# output_dir_folder = os.path.join(OUTPUT_DIR, str(start_date.year) + "_" + underlying +strategy_to_execute + "_" + "20240207055834")

# ===== Constant Risk Parameters ==========
rampup_period = 375 * 3 
rampup_days = 3
rollover_days = 2
non_rollover_days = 3
total_trade_time_in_day = 375
rollover_period = 375 * 2
non_rollover_period = 375 * 3
expiry_weight_vector = np.array([0.5, 0.5])
option_legs = 4 # 4 for condor
# =========================================


# =============================================================================== Chartpack =================================================================================

# BANKNIFTY Exception EXP Dates
BNF_EXP_DICT = {
    "2023-09-07":"2023-09-06",
    "2023-09-14":"2023-09-13",
    "2023-09-21":"2023-09-20",

    "2023-10-06":"2023-10-05",
    "2023-10-13":"2023-10-12",
    "2023-10-20":"2023-10-19",

    "2023-11-02":"2023-11-01",
    "2023-11-09":"2023-11-08",
    "2023-11-16":"2023-11-15",
    "2023-11-23":"2023-11-22",

    "2023-12-07":"2023-12-06",
    "2023-12-14":"2023-12-13",
    "2023-12-21":"2023-12-20",
    "2023-12-28":"2023-12-27"
}

expiry_day_of_week = 3 # 3 => Thursday
holiday_list_store = "holiday_lists/"
currency = "INR"

# load holiday list
HOLIDAY_LIST = []
years = np.arange(start_date.year, end_date.year)
for year in years:
    try:
        with open(f"{holiday_list_store}holidays_{year}.json", 'r') as f:
            HOLIDAY_LIST.extend(json.load(f))
    except Exception as ex:
        print(f"error while loading holiday list for year {year}")
        raise ex
    
# Name of the strategy
custom_strategy_name = "CONSTANT RISK STRATEGY - BASELINE"
asset_class = "Options"
chart_title = "Constant Risk Condor Strategy"
contract_type = "WEEKLY"
strategy = "CONDOR"
expiry_types = "Two Front Weeks"
margin_calculation = False

# month mapper
month_mapper = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct", 11:"Nov", 12:"Dec"}


# baseline files
baseline_blotters = ""
baseline_backtests = ""