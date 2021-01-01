from typing import Union

from pandas import DataFrame, Timestamp


class Backtest:
    """
    An abstract class to create backtests on a single security. Initialized with a pandas DataFrame in yfinance format.
    Backtests can be defined by overriding the long and short triggers; built-in test functions can be used
    to test the defined strategy, or a custom test can be created. Calculates descriptive stats about a backtest,
    including returns, batting average, and descriptions of gains/losses.

    Args:
        df: The pandas dataframe with data about the equity. Uses a Timestamp index, consistent with the yfinance library dataframes.
    """
    def _trigger_long(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        """
        This method should be overridden to identify that a long position should taken on the given date (e.g. purchase an equity)
        Users can analyze the whole dataframe, but must return "True" to take a long position only on the given date.
        Return "False" or "None" to decline to take a long position.
        --
        ADVANCED USAGE:
        By default, returning "True" will trigger a long position at the 'Adj Close' price. The user may enter the position at
        a different price by returning a numerical value from this method. Caution is advised because this module will not verify
        that it was possible to enter at the given price. 

        Args:
            df: The pandas dataframe with data about the equity. Uses a Timestamp index, consistent with the yfinance library dataframes.
            date: The Timestamp index of the date being evaluated in the dataframe.

        Returns:
            Returns a boolean value, where "True" means to take a long position, and "False" means don't take a long position

        Raises:
            TO DO
        """
        raise NotImplementedError('You must override the "_trigger_long" method')

    def _trigger_short(self, df: DataFrame, date: Timestamp) -> Union[bool, float]:
        """
        This method should be overridden to identify that a short position should taken on the given date (e.g. purchase an equity)
        Users can analyze the whole dataframe, but must return "True" to take a short position only on the given date.
        Return "False" or "None" to decline to take a short position.
        --
        ADVANCED USAGE:
        By default, returning "True" will trigger a short position at the 'Adj Close' price. The user may enter the position at
        a different price by returning a numerical value from this method. Caution is advised because this module will not verify
        that it was possible to enter at the given price. 

        Args:
            df: The pandas dataframe with data about the equity. Uses a Timestamp index, consistent with the yfinance library dataframes.
            date: The Timestamp index of the date being evaluated in the dataframe.

        Returns:
            Returns a boolean value, where "True" means to take a short position, and "False" means don't take a short position

        Raises:
            TO DO
        """
        raise NotImplementedError('You must override the "_trigger_short" method')

    def _reset(self):
        self._total_return = 1.0
        self._buy_price = None
        self._sell_price = None
        self._gains = []
        self._losses = []

    def _buy(self, price):
        try:
            self._buy_price = float(price)
        except (TypeError, ValueError):
            raise ValueError('Buy price must be a number')

    def _sell(self, price):
        try:
            self._sell_price = float(price)
        except (TypeError, ValueError):
            raise ValueError('Sell price must be a number')

    def _calculate_gain(self, reset_prices=True):
        result = (self._sell_price - self._buy_price) / self._buy_price + 1
        self._total_return *= result

        if result > 1:
            self._gains.append(result)
        elif result < 1:
            self._losses.append(result)

        if reset_prices:
            self._buy_price = None
            self._sell_price = None

    def get_stats(self):
        """
        Returns a dict containing summary statistics about gains that were saved to an instance of a backtest.

        Args:
            N/A

        Returns:
            dict : {'totalReturn': float, 'battingAvg': float, 'largestGain': float, 'largestLoss': float, 'avgGain': float, 'avgLoss': float}
        """
        n_gains = len(self._gains)
        n_losses = len(self._losses)
        total_trades = n_gains + n_losses
        return {
            'totalReturn': self._total_return,
            'battingAvg': None if total_trades == 0 else n_gains / total_trades,
            'largestGain':None if n_gains == 0 else max(self._gains),
            'largestLoss': None if n_losses == 0 else min(self._losses),
            'avgGain': None if n_gains == 0 else sum(self._gains) / n_gains,
            'avgLoss': None if n_losses == 0 else sum(self._losses) / n_losses
        }

    def backtest_long_only(self):
        """
        This method is used to backtest long-only equity trading strategies. A security is purchased on dates when the 
        '_trigger_long' method returns True (or a non-zero numeric value), and subsequently sold when the '_trigger_short' method
        returns True (or a non-zero numeric value). Note that the trigger methods are called every time the current long position
        makes it possible - that is, when holding a long position, only the '_short_trigger' method is called and vice-versa. 

        Args:
            N/A

        Returns:
            A dict containing summary statistics of the backtest, as defined in the 'get_stats' method

        Raises:
            TO DO
        """
        pos = False
        for date in self.df.index:
            if not pos and (long_trigger := self._trigger_long(self.df, date)):
                pos = True
                self._buy(self.df.loc[date, 'Adj Close'] if type(long_trigger) is bool else long_trigger)
            elif pos and (short_trigger := self._trigger_short(self.df, date)):
                pos = False
                self._sell(self.df.loc[date, 'Adj Close'] if type(short_trigger) is bool else short_trigger)
                self._calculate_gain()
        return self.get_stats()

    def __init__(self, df):
        self.df = df
        self._reset()
