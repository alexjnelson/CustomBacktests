from datetime import date

from utils.screenerUtils import load_tickers
from backtesting.sample_backtests import HammerReversals, HammerReversalsWithKeltner, Hold
from backtesting import run_backtests


tickers = load_tickers('lists/tsx_cleaned.txt')
start = date(2016, 1, 1)
end = date.today()

backtests = [HammerReversals, HammerReversalsWithKeltner, Hold]
run_backtests(start, end, tickers, *backtests)
