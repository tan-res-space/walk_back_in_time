from tradelib.backtest_driver import BacktestDriver
from datetime import datetime
import os

from tradelib.tradelib_global_constants import trade_start_time, trade_end_time, data_dir, OUTPUT_DIR, date_time_format_no_dash, start_date, end_date, strategy_to_execute, underlying

# start_date = start_date
# end_date = end_date
# strategy_to_execute = strategy_to_execute

output_dir = os.path.join(OUTPUT_DIR, str(start_date.year) + "_" + underlying + "_" + strategy_to_execute)

# try DI or resource manager
# print(start_date, end_date, trade_start_time, trade_end_time, data_dir, output_dir, strategy_to_execute)
backtest_driver = BacktestDriver(start_date, end_date, trade_start_time, trade_end_time, data_dir, output_dir, strategy_to_execute)
backtest_driver.single_process_driver()