from datetime import date
from functools import partial
from multiprocessing import Pool

from utils.screenerUtils import load_tickers, make_df
from backtesting.sample_backtests import HammerReversals, HammerReversalsWithKeltner, Hold
from backtesting import run_backtests


tickers = load_tickers('lists/tsx_cleaned.txt')
start = date(2018, 1, 1)
end = date.today()

backtests = [HammerReversals, HammerReversalsWithKeltner, Hold]
run_backtests(start, end, tickers[:50], *backtests)