from functools import partial
from multiprocessing import Pool
from typing import Union
from bisect import insort
import numpy as np
import datetime as dt

from pandas import DataFrame, Timestamp, concat
from CustomBacktests.utils.screenerUtils import make_df


class Backtest:
    """
    An abstract class to create backtests on a single security. Initialized with a pandas DataFrame in yfinance format.
    Backtests can be defined by overriding the long and short triggers; built-in test functions can be used
    to test the defined strategy, or a custom test can be created. Calculates descriptive stats about a backtest,
    including returns, batting average, and descriptions of gains/losses. The default testing method can be specified by
    setting the "run" attribute to the desired method. By default, this is 'backtest_long_only.'

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
        By default, returning "True" will trigger a long position at the self.defaultSourcePrice price. The user may enter the position at
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
        By default, returning "True" will trigger a short position at the self.defaultSourcePrice price. The user may enter the position at
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
        self._entry_price = None
        self._exit_price = None
        self._gains = []
        self._losses = []

    def _enter_long(self, price):
        try:
            self._entry_price = float(price)
        except (TypeError, ValueError):
            raise ValueError('Buy price must be a number')

    def _exit_long(self, price):
        try:
            self._exit_price = float(price)
        except (TypeError, ValueError):
            raise ValueError('Sell price must be a number')

    def _enter_short(self, price):
        try:
            self._entry_price = -float(price)
        except (TypeError, ValueError):
            raise ValueError('Sell price must be a number')

    def _exit_short(self, price):
        try:
            self._exit_price = -float(price)
        except (TypeError, ValueError):
            raise ValueError('Sell price must be a number')

    def _calculate_gain(self, reset_prices=True):
        result = (self._exit_price - self._entry_price) / abs(self._entry_price) + 1
        self._total_return *= result

        if result > 1:
            self._gains.append(result)
        elif result < 1:
            self._losses.append(result)

        if reset_prices:
            self._entry_price = None
            self._exit_price = None

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
        try:
            cagr = self._total_return ** (1 / ((self.df.index[-1] - self.df.index[0]).days / 365.25)) - 1
        except AttributeError:
            try:
                # if the index is in integer timestamps instead of pandas timestamps
                start = dt.datetime.fromtimestamp(self.df.index[0])
                end = dt.datetime.fromtimestamp(self.df.index[-1])
                cagr = self._total_return ** (1 / ((end - start).days / 365.25)) - 1
            except Exception:
                cagr = None

        return {
            'totalReturn': self._total_return - 1,
            'cagr': cagr,
            'battingAvg': None if total_trades == 0 else n_gains / total_trades,
            'largestGain': 0 if n_gains == 0 else max(self._gains) - 1,
            'largestLoss': 0 if n_losses == 0 else min(self._losses) - 1,
            'avgGain': 0 if n_gains == 0 else sum(self._gains) / n_gains - 1,
            'avgLoss': 0 if n_losses == 0 else sum(self._losses) / n_losses - 1,
            'n_gains': n_gains,
            'n_losses': n_losses
        }

    def _is_bool(self, value):
        return type(value) is bool or type(value) is np.bool_

    def _backtest_long_only(self):
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
                self._enter_long(self.df.loc[date, self.default_src_price] if self._is_bool(long_trigger) else long_trigger)
            elif pos and (short_trigger := self._trigger_short(self.df, date)):
                pos = False
                self._exit_long(self.df.loc[date, self.default_src_price] if self._is_bool(short_trigger) else short_trigger)
                self._calculate_gain()
        return self.get_stats()

    def _backtest_long_short(self):
        """
        This method is used to backtest equity trading strategies that go long on a buy signal and short on a sell signal. A security is purchased on dates when the 

        Args:
            N/A

        Returns:
            A dict containing summary statistics of the backtest, as defined in the 'get_stats' method

        Raises:
            TO DO
        """
        long_pos = False
        short_pos = False
        for date in self.df.index:
            if (long_trigger := self._trigger_long(self.df, date)):
                if short_pos:
                    short_pos = False
                    self._exit_short(self.df.loc[date, self.default_src_price] if self._is_bool(long_trigger) else long_trigger)
                    self._calculate_gain()
                if not long_pos:
                    long_pos = True
                    self._enter_long(self.df.loc[date, self.default_src_price] if self._is_bool(long_trigger) else long_trigger)
            if (short_trigger := self._trigger_short(self.df, date)):
                if long_pos:
                    long_pos = False
                    self._exit_long(self.df.loc[date, self.default_src_price] if self._is_bool(short_trigger) else short_trigger)
                    self._calculate_gain()
                if not short_pos:
                    short_pos = True
                    self._enter_short(self.df.loc[date, self.default_src_price] if self._is_bool(short_trigger) else short_trigger)
        return self.get_stats()

    def run(self):
        """
        This method should be overwritten to be the preferred testing method.

        Args:
            N/A

        Returns:
            A dict containing summary statistics of the backtest as returned by the designated backtest method
        """
        raise NotImplementedError

    def __init__(self, df, default_src_price='Adj Close'):
        self.df = df
        self.default_src_price = default_src_price
        self._reset()


def run_backtests(start, end, ticker_list, *backtests):
    """
    Runs backtests on multiple tickers over the specified time period. Tickers should be passed as a list and 
    backtest classes should be passed (not instances). Outputs results to a csv file.

    Args:
        start: A starting date for the period to test over
        ticker_list: A list of ticker strings; often these lists can be made with the 'utils.screenerUtils.load_tickers' function
        *backtests: The backtest classes to be 

    Returns:
        None

    Raises:
        ValueError if no backtests are passed
        TypeError if backtests aren't a subclass of Backtest or if tickers aren't strings
    """
    if len(backtests) == 0:
        raise ValueError('You must pass at least one backtest class')
    if not all([issubclass(bt, Backtest) for bt in backtests]):
        raise TypeError('All backtests must inherit the Backtest class')
    if any([type(t) != str for t in ticker_list]):
        raise TypeError('All tickers must be strings')

    pool = Pool(4)
    f = partial(make_df, start, end)

    results = {
        bt.__name__: {
            'avg_return': 0,
            'batting_avg': None,
            'n_gains': 0,
            'n_losses': 0,
            'avg_gain': 0,
            'avg_loss': 0,
            'largest_gain': 1,
            'largest_loss': 1,
            'top_n': [],
            'bot_n': [],
            'top_n_avg_return': 1,
            'bot_n_avg_return': 1
        }
        for bt in backtests
    }
    n_selected = len(ticker_list) // 5 + 1  # used to select the 20th and 80th percentiles of stocks
    max_name = max([len(bt.__name__) for bt in backtests])
    cur_best = None
    cur_best_res = 0

    try:
        for c, df_pair in enumerate(pool.imap_unordered(f, ticker_list)):
            if df_pair is None or len(df_pair[1]) == 0:
                continue
            t, df = df_pair

            for bt in backtests:
                bt_res = results[bt.__name__]
                res = bt(df).run()

                bt_res['avg_return'] += res['totalReturn'] * (res['n_gains'] + res['n_losses'])

                bt_res['n_gains'] += res['n_gains']
                bt_res['n_losses'] += res['n_losses']

                bt_res['avg_gain'] += res['avgGain'] * res['n_gains']
                bt_res['avg_loss'] += res['avgLoss'] * res['n_losses']

                bt_res['largest_gain'] = res['totalReturn'] if res['totalReturn'] > bt_res['largest_gain'] else bt_res['largest_gain']
                bt_res['largest_loss'] = res['totalReturn'] if res['totalReturn'] < bt_res['largest_loss'] else bt_res['largest_loss']

                pair = (res['totalReturn'], t)
                if len(bt_res['top_n']) < n_selected:  # both lists will fill simultaneously since they both start empty
                    insort(bt_res['top_n'], pair)
                    insort(bt_res['bot_n'], pair)
                # once they are both minimally fully, a new incoming result can only possibly fall into one list
                elif res['totalReturn'] > min(bt_res['top_n'])[0]:
                    bt_res['top_n'].pop(0)
                    insort(bt_res['top_n'], pair)
                elif res['totalReturn'] < max(bt_res['bot_n'])[0]:
                    bt_res['bot_n'].pop(-1)
                    insort(bt_res['bot_n'], pair)

            cur_best = None
            cur_best_res = 0
            for k, v in results.items():
                total_trades = v['n_gains'] + v['n_losses']
                if total_trades and v['avg_return'] / total_trades > cur_best_res:
                    cur_best = k
                    cur_best_res = v['avg_return'] / total_trades
            print(f'{c + 1} / {len(ticker_list)} complete. Current best: {cur_best} at {cur_best_res:.3f}' + ' ' * max_name, end='\r')
        print(f'\nComplete. Best strategy: {cur_best} at {cur_best_res:.3f}')
    except:  # if execution ends early, doesn't overwrite old save files, but saves to _backup.csv files instead
        backup_res = {}
        for bt, bt_res in results.items():
            backup_res[bt + '_incomplete'] = bt_res
        results = backup_res
        raise
    finally:
        for bt, bt_res in results.items():
            n_gains = bt_res['n_gains']
            n_losses = bt_res['n_losses']
            total_trades = n_gains + n_losses

            bt_res['avg_return'] /= total_trades if total_trades != 0 else 1
            bt_res['batting_avg'] = n_gains / total_trades if total_trades != 0 else None
            bt_res['n_gains'] = n_gains
            bt_res['n_losses'] = n_losses
            bt_res['avg_gain'] /= n_gains if n_gains != 0 else 1
            bt_res['avg_loss'] /= n_losses if n_losses != 0 else 1
            bt_res['top_n_avg_return'] = sum([ret[0] for ret in bt_res['top_n']]) / n_selected
            bt_res['bot_n_avg_return'] = sum([ret[0] for ret in bt_res['bot_n']]) / n_selected

            try:
                top_n = DataFrame([t[1] for t in reversed(bt_res['top_n'])], columns=['top_n'])
                bot_n = DataFrame([t[1] for t in bt_res['bot_n']], columns=['bot_n'])
                out_df = DataFrame(bt_res).drop(columns=['top_n', 'bot_n']).loc[0]
                out_df = concat([out_df, top_n, bot_n]).replace(np.nan, '')
                out_df.to_csv(f'results/{bt}.csv')
            except KeyError:
                continue
