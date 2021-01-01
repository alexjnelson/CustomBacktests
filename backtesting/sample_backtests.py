if __name__ == '__main__':
    from os.path import dirname
    from sys import path
    path.append(dirname(dirname(__file__)))

import datetime as dt
from typing import Union

from pandas import DataFrame, Timestamp, concat

from backtesting import Backtest


class HammerReversals(Backtest):
    """
    This class is a sample of how to override the Backtest class. Buy when a hammer candlestick appears after a 3-day 
    downtrend, and sell when a hanging man candlestick appears after a 3-day uptrend
    """
    def __init__(self, df):
        from utils.addIndicators import hammer
        self.df = hammer(df)
        self.df = hammer(df, bullish=False)
        self._reset()

    def _reset(self):
        super()._reset()
        self._stoploss = 0

    def _trigger_long(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        if (c := df.index.get_loc(date)) == 0:
            return False
        yesterday = df.index[c - 1]

        # check for a hammer candlestick (by default, the hammer indicator is only made after a 3-day downtrend) and reversal confirmation
        if df.loc[yesterday, 'Hammer'] and df.loc[date, 'Close'] >= df.loc[yesterday, 'Close']:
            return True
        else:
            return False

    def _trigger_short(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        if (c := df.index.get_loc(date)) == 0:
            return False
        yesterday = df.index[c - 1]

        # check for a hanging candlestick (by default, only appears after a 3-day downtrend) and reversal confirmation
        if df.loc[yesterday, 'Hanging'] and df.loc[date, 'Close'] <= df.loc[yesterday, 'Close']:
            return True
        else:
            return False


class HammerReversalsWithKeltner(HammerReversals):
    """
    This class overrides the HammerReversals backtest and shows how to use custom buy/sell prices (as opposed to the default of
    always trading at the 'Adj Close). Buys at the same indicator as HammerReversals, but uses the 'Close' price. Sells when the 
    price reaches the 90th percentile of the Keltner channel. Since this sell trigger can likely be performed with a stop order,
    it is appropriate to use the exact price rather than the default 'Adj Close' price.
    """
    def _reset(self):
        super()._reset()
        self._triggered = False

    def _trigger_long(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        return df.loc[date, 'Close'] if super()._trigger_long(df, date) else None  # either None or False can be returned to indicate no-buy

    def _trigger_short(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        trigger_price = (df.loc[date, 'Kelt Upper'] - df.loc[date, 'Kelt Lower']) * 0.9 + df.loc[date, 'Kelt Lower']
        if df.loc[date, 'Close'] <= self._stoploss or df.loc[date, 'High'] >= trigger_price:
            return trigger_price if df.loc[date, 'High'] >= trigger_price else df.loc[date, 'Close']


class Hold(Backtest):
    def _trigger_long(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        return True
    
    def _trigger_short(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        return date == df.index[-1]

if __name__ == '__main__':
    from utils.screenerUtils import make_df
    tests = [HammerReversals, HammerReversalsWithKeltner, Hold]
    start = dt.date(2016, 1, 1)
    end = dt.date.today()
    df = make_df(start, end, 'qqq')[1]

    for t in tests:
        test = t(df)
        print (f'--\n{test.__class__.__name__}')
        print(test.backtest_long_only())
