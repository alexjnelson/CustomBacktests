import datetime as dt
import enum
import math

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None


class Cols(enum.Enum):
    OPEN = 'Open'
    HIGH = 'High'
    LOW = 'Low'
    CLOSE = 'Close'
    ADJCLOSE = 'Adj Close'


# returns an exponential moving average
def EMA(df, period=20, sourceCol=Cols.ADJCLOSE.value, colName=None):
    if len(df) < period:
        raise ValueError('Not enough data')
    if not (type(df) == pd.DataFrame and type(period) == int):
        raise TypeError
    colName = "EMA_{}".format(period) if colName is None else colName

    df[colName] = df[sourceCol].ewm(span=period, adjust=False).mean()
    return df


# returns a simple moving average
def SMA(df, period=20, sourceCol=None, colName=None):
    if len(df) < period:
        raise ValueError('Not enough data')

    if not (type(df) == pd.DataFrame and type(period) == int):
        raise TypeError
    colName = 'SMA_{}'.format(period) if colName is None else colName

    if sourceCol is None:
        df[colName] = TP(df)['TP'].rolling(window=period).mean()
        del df['TP']
    else:
        df[colName] = df[sourceCol].rolling(window=period).mean()

    return df


# returns a stock's daily typical price
def TP(df, colName='TP'):
    if not type(df) == pd.DataFrame:
        raise TypeError

    df[colName] = (df[Cols.HIGH.value] + df[Cols.LOW.value] + df[Cols.CLOSE.value]) / 3
    return df


# returns the average true range volatility indicator
def ATR(df, period=14, colName='ATR'):
    if len(df) < period:
        raise ValueError('Not enough data')

    if not (type(df) == pd.DataFrame and type(period) == int):
        raise TypeError

    if colName in df.columns:
        return df

    tr = {'TR': []}
    for n in range(len(df.index)):
        if n == 0:
            continue

        tr['TR'].append(max(df[Cols.HIGH.value][n] - df[Cols.LOW.value][n], abs(df[Cols.HIGH.value]
                                                                                [n] - df[Cols.CLOSE.value][n - 1]), abs(df[Cols.LOW.value][n] - df[Cols.CLOSE.value][n - 1])))

    df = df[1:]
    tr = pd.DataFrame(data=tr, index=df.index)
    tr[colName] = tr['TR'].rolling(window=period).mean()
    del tr['TR']
    return pd.concat([df, tr], axis=1)[period:]


# the keltner channels don't line up 100% perfectly with MarketWatch's keltner's of same params
def Keltner(df, emaPeriod=20, atrPeriod=20, atrMult=2, sourceCol=Cols.ADJCLOSE.value, colNameUpper='Kelt Upper', colNameLower='Kelt Lower'):
    if len(df) < max([emaPeriod, atrPeriod]):
        raise ValueError('Not enough data')
    if not (type(df) == pd.DataFrame and type(emaPeriod) == int and type(atrPeriod) == int and type(atrMult) == int):
        raise TypeError

    req = EMA(df, emaPeriod, sourceCol=sourceCol, colName='EMA_{} kelt'.format(emaPeriod))
    req['ATR kelt'] = ATR(df, atrPeriod, colName='ATR kelt')['ATR kelt']
    df[colNameUpper] = req['EMA_{} kelt'.format(
        emaPeriod)] + req['ATR kelt'] * atrMult
    df[colNameLower] = req['EMA_{} kelt'.format(
        emaPeriod)] - req['ATR kelt'] * atrMult
    del req['ATR kelt']
    del df['EMA_{} kelt'.format(emaPeriod)]
    return df


# the bollinger bands don't line up 100% perfectly with MarketWatch's bands of same params
def Bollinger(df, period=20, nDeviations=2, sourceCol=None, colNameUpper='Bol Upper', colNameLower='Bol Lower'):
    if len(df) < period:
        raise ValueError('Not enough data')

    if not (type(df) == pd.DataFrame and type(period) == int and type(nDeviations) == int):
        raise TypeError

    req = SMA(df, period, sourceCol=sourceCol, colName='SMA_{} bol'.format(period))
    if sourceCol is None:
        req['STD bol'] = TP(df, colName='TP bol')[
            'TP bol'].rolling(window=period).std()
        del df['TP bol']
    else:
        req['STD bol'] = df[sourceCol].rolling(window=period).std()

    df[colNameUpper] = req['SMA_{} bol'.format(
        period)] + nDeviations * req['STD bol']
    df[colNameLower] = req['SMA_{} bol'.format(
        period)] - nDeviations * req['STD bol']
    del req['STD bol']
    del df['SMA_{} bol'.format(period)]
    return df


# returns whether the stock is currently in a TTM squeeze (bollinger bands are between keltner channels)
def TTMSqueeze(df, emaPeriod=20, atrPeriod=20, atrMult=2, bollingerPeriod=20, nDeviations=2, sourceCol=None, colName='TTMSqueeze'):
    if len(df) < max([emaPeriod, atrPeriod, bollingerPeriod]):
        raise ValueError('Not enough data')

    if not type(df) == pd.DataFrame or any(type(x) != int for x in [emaPeriod, atrPeriod, atrMult, bollingerPeriod, nDeviations]):
        raise TypeError

    req = Keltner(df, emaPeriod, atrPeriod, atrMult, sourceCol=Cols.ADJCLOSE.value if sourceCol is None else sourceCol,
                  colNameUpper='Kelt Upper ttm', colNameLower='Kelt Lower ttm')
    req[['Bol Upper ttm', 'Bol Lower ttm']] = Bollinger(df, bollingerPeriod, nDeviations, sourceCol=sourceCol, colNameUpper='Bol Upper ttm', colNameLower='Bol Lower ttm')[
        ['Bol Upper ttm', 'Bol Lower ttm']]

    df[colName] = (req['Bol Lower ttm'] >= req['Kelt Lower ttm']) & (
        req['Bol Upper ttm'] <= req['Kelt Upper ttm'])
    del df['Kelt Upper ttm'], df['Kelt Lower ttm'], df['Bol Upper ttm'], df['Bol Lower ttm']
    return df


# returns the relative strength index (technical indicator)
def RSI(df, period=14, sourceCol=Cols.ADJCLOSE.value, colName='RSI'):
    if len(df) < period:
        raise ValueError('Not enough data')

    if not (type(df) == pd.DataFrame and type(period) == int):
        raise TypeError

    df['Gain'] = -(df[sourceCol].rolling(window=2).sum() -
                   2 * df[sourceCol])
    df['AvgGain'] = np.nan
    df['AvgLoss'] = np.nan
    df[colName] = np.nan

    with np.errstate(divide='ignore', invalid='ignore'):
        temp = df.iloc[:period]
        gainFilter = temp['Gain'] > 0
        lossFilter = temp['Gain'] < 0
        df['AvgGain'][period - 1] = temp[gainFilter]['Gain'].sum() / period
        df['AvgLoss'][period - 1] = temp[lossFilter]['Gain'].sum() / period
        df[colName][period - 1] = 100 - 100 / \
            (1 + df['AvgGain'][period - 1] / -df['AvgLoss'][period - 1])

        for n in range(len(df.index)):
            if n < period:
                continue
            df['AvgGain'][n] = (df['AvgGain'][n - 1] * (period - 1) +
                                (df['Gain'][n] if df['Gain'][n] > 0 else 0)) / period
            df['AvgLoss'][n] = (df['AvgLoss'][n - 1] * (period - 1) +
                                (df['Gain'][n] if df['Gain'][n] < 0 else 0)) / period
            df[colName][n] = 100 - 100 / (1 + df['AvgGain'][n] / -df['AvgLoss'][n])

    df = df[period - 1:]
    del df['Gain']
    del df['AvgGain']
    del df['AvgLoss']
    del temp
    return df

# returns whether the stock is the highest it's been in the specified number of days


def recentHigh(df, days=260, sourceCol=Cols.ADJCLOSE.value, colName=None):
    if len(df) < days:
        raise ValueError('Not enough data')

    colName = 'High_{}'.format(days) if colName is None else colName
    df[colName] = ''
    for i in df.index[1:]:
        index = df.index.get_loc(i)
        df[colName][i] = df[sourceCol][index] == max(
            df[sourceCol][0:index]) if index < days else df[sourceCol][index] == max(df[sourceCol][index - days + 1:index + 1])
    return df


# returns whether the stock is the lowest it's been in the specified number of days
def recentLow(df, days=260, sourceCol=Cols.ADJCLOSE.value, colName=None):
    if len(df) < days:
        raise ValueError('Not enough data')

    colName = 'Low_{}'.format(days) if colName is None else colName
    df[colName] = ''
    for i in df.index[1:]:
        index = df.index.get_loc(i)
        df[colName][i] = df[sourceCol][index] == min(
            df[sourceCol][0:index]) if index < days else df[sourceCol][index] == min(df[sourceCol][index - days + 1:index + 1])
    return df


# returns True while the short-emas are above the long-emas
def shortOverLong(df, shortEmas=[8], longEmas=[21], withTriggers=False, sourceCol=Cols.ADJCLOSE.value, colName=None):
    if type(shortEmas) != list:
        shortEmas = [shortEmas]
    if type(longEmas) != list:
        longEmas = [longEmas]

    if len(df) < max(shortEmas + longEmas):
        raise ValueError('Not enough data')

    colName = 'SoL'if colName is None else colName

    shortLabels = []
    longLabels = []
    for p in shortEmas:
        label = 'EMA_{}_SoL'.format(p)
        df = EMA(df, p, sourceCol=sourceCol, colName=label)
        shortLabels.append(label)
    for p in longEmas:
        label = 'EMA_{}_SoL'.format(p)
        df = EMA(df, p, sourceCol=sourceCol, colName=label)
        longLabels.append(label)

    df['longmax'] = df[longLabels].max(axis=1)
    df['shortmin'] = df[shortLabels].min(axis=1)
    df[colName] = df['shortmin'] > df['longmax']

    if withTriggers:
        df = goldenCross(df, shortEmas, longEmas, sourceCol=sourceCol, colName=colName + '_Gold')
        df = deathCross(df, shortEmas, longEmas, sourceCol=sourceCol, colName=colName + '_Death')

    try:
        for label in shortLabels + longLabels:
            del df[label]
        del df['longmax'], df['shortmin']
    except KeyError:
        pass
    return df


# identifies when emas crossover; cross can either be golden (short passes long) or death (long passes short)
def _crossover(df, crossType, shortEmas, longEmas, sourceCol, colName):
    if len(df) < max(shortEmas + longEmas):
        raise ValueError('Not enough data')

    if type(crossType) != str:
        raise ValueError
    elif crossType.lower() == 'golden':
        golden = True
    elif crossType.lower() == 'death':
        golden = False
    else:
        raise ValueError

    df = shortOverLong(df, shortEmas, longEmas, sourceCol=sourceCol, colName='SoL_cr')
    for i in range(1, len(df)):
        date = df.index[i]
        prev = df.index[i - 1]
        df.loc[date, colName] = df.loc[date, 'SoL_cr'] and not df.loc[prev,
                                                                      'SoL_cr'] if golden else not df.loc[date, 'SoL_cr'] and df.loc[prev, 'SoL_cr']
    del df['SoL_cr']
    return df


def goldenCross(df, shortEmas=[12], longEmas=[26], sourceCol=Cols.ADJCLOSE.value, colName=None):
    return _crossover(df, 'Golden', shortEmas, longEmas, sourceCol, 'Golden' if colName is None else colName)


def deathCross(df, shortEmas=[12], longEmas=[26], sourceCol=Cols.ADJCLOSE.value, colName=None):
    return _crossover(df, 'Death', shortEmas, longEmas, sourceCol, 'Death' if colName is None else colName)


# selltrigger based on RSI trends indicating the end of an uptrend. works best if the data given begins well before
# the time period to predict so the downtrend or uptrend can be identified
def failureSwings(df, rsiPeriod=14, initialUptrend=None, sourceCol=Cols.ADJCLOSE.value, colName=None):
    colName = 'FS Uptrend?' if colName is None else colName
    df = RSI(df, rsiPeriod, sourceCol, 'RSI_FS')
    df[colName] = ''

    df = observeTrend(df, sourceCol='RSI_FS',
                      initialUptrend=initialUptrend, colName=colName)
    del df['RSI_FS']
    return df


def MACD(df, withHistogram=True, withTriggers=False, withTrends=False, sourceCol=Cols.ADJCLOSE.value, colName=None):
    colName = 'MACD' if colName is None else colName

    df = EMA(df, 12, sourceCol=sourceCol, colName='EMA_12 MACD')
    df = EMA(df, 26, sourceCol=sourceCol, colName='EMA_26 MACD')

    df[colName] = df['EMA_12 MACD'] - df['EMA_26 MACD']
    df[colName + '_Sig'] = df[colName].ewm(span=9, adjust=False).mean()
    df[colName + '_Hist'] = df[colName] - df[colName + '_Sig']
    if withTrends:
        df[colName + ' Uptrend?'] = df[colName + '_Hist'] > 0

    if withTriggers:
        for i in range(1, len(df)):
            if withTriggers:
                date = df.index[i]
                prev = df.index[i - 1]
                df.loc[date, colName + '_Gold'] = df.loc[date, colName +
                                                         '_Hist'] >= 0 and df.loc[prev, colName + '_Hist'] < 0
                df.loc[date, colName + '_Death'] = df.loc[date, colName +
                                                          '_Hist'] <= 0 and df.loc[prev, colName + '_Hist'] > 0

    if not withHistogram:
        del df[colName + '_Hist']
    del df['EMA_12 MACD'], df['EMA_26 MACD']
    return df


def observeTrend(df, sourceCol, initialUptrend=None, trendlines=False, colName=None, lineColName=None):
    colName = (sourceCol + ' Uptrend?') if colName is None else colName
    lineColName = (
        colName + ' Trendline') if lineColName is None else lineColName

    df[colName] = False
    df = shortOverLong(df, 12, 26, colName='SoL_OBS')
    if trendlines:
        df[lineColName] = np.nan
        reg = LinearRegression()

    # false if not given an initial trend
    trendIdentified = initialUptrend is not None
    # in a downtrend, "peak" really means "trough"
    first_peak = df[sourceCol][0]
    second_peak = first_peak
    fail_point = first_peak
    uptrend = initialUptrend if initialUptrend is not None else df['SoL_OBS'][0]
    downtrend = not uptrend
    plotted_peaks = []

    for i in range(1, len(df)):
        obs = df[sourceCol][i]

        if uptrend and obs > first_peak or downtrend and obs < first_peak:
            first_peak = obs
            second_peak = obs
            fail_point = obs
        elif uptrend and obs <= fail_point and second_peak < first_peak:
            uptrend = False
            downtrend = True
            trendIdentified = True
            plotted_peaks = []
        elif downtrend and obs >= fail_point and second_peak > first_peak:
            uptrend = True
            downtrend = False
            trendIdentified = True
            plotted_peaks = []
        elif uptrend and obs <= fail_point or downtrend and obs >= fail_point:
            fail_point = obs
        else:
            second_peak = obs
            if trendlines:
                plotted_peaks.append(first_peak)

        if trendIdentified:
            df[colName][i] = bool(uptrend)
            if trendlines and len(plotted_peaks) > 0:
                indices = [i for i in range(len(plotted_peaks))]
                reg.fit(np.array(indices).reshape(-1, 1),
                        np.array(plotted_peaks).reshape(-1, 1))
                df[lineColName][i] = reg.predict(
                    np.array(indices[-1]).reshape(1, -1))[0][0]
            elif trendlines:
                try:
                    df[lineColName][i] = df[lineColName][i - 1]
                except IndexError:
                    pass

    del df['SoL_OBS']
    return df


def volumeForce(df, with_temp=True, colName='VF'):
    if len(df) < 3:
        raise ValueError('Not enough data')
    df[colName] = np.nan
    df['T'] = np.nan
    df['dm'] = np.nan
    df['cm'] = np.nan

    j = df.index[0]
    i = df.index[1]
    h = df.index[2]
    df.loc[i, 'T'] = 1 if (df.loc[i, Cols.HIGH.value] + df.loc[i, Cols.LOW.value] + df.loc[i, Cols.CLOSE.value]) > (
        df.loc[j, Cols.HIGH.value] + df.loc[j, Cols.LOW.value] + df.loc[j, Cols.CLOSE.value]) else -1
    df.loc[i, 'dm'] = df.loc[i, Cols.HIGH.value] - df.loc[i, Cols.LOW.value]
    # in first calculation (at index[2]), uses that day's dm instead of previous day's cm, so set previous day cm to that day's dm here
    df.loc[i, 'cm'] = df.loc[h, Cols.HIGH.value] - df.loc[h, Cols.LOW.value]
    j = i

    for i in df.index[2:]:
        df.loc[i, 'T'] = 1 if (df.loc[i, Cols.HIGH.value] + df.loc[i, Cols.LOW.value] + df.loc[i, Cols.CLOSE.value]) > (
            df.loc[j, Cols.HIGH.value] + df.loc[j, Cols.LOW.value] + df.loc[j, Cols.CLOSE.value]) else -1
        if with_temp:
            df.loc[i, 'dm'] = df.loc[i, Cols.HIGH.value] - df.loc[i, Cols.LOW.value]
            df.loc[i, 'cm'] = (df.loc[j, 'cm'] + df.loc[i, 'dm']) if df.loc[i,
                                                                            'T'] == df.loc[j, 'T'] else (df.loc[i, 'dm'] + df.loc[j, 'dm'])
            df.loc[i, colName] = df.loc[i, 'Volume'] * \
                ((2 * (df.loc[i, 'dm'] / df.loc[i, 'cm'] - 1))
                 if df.loc[i, 'cm'] != 0 else -2) * df.loc[i, 'T'] * 100
        else:
            df.loc[i, colName] = df.loc[i, 'Volume'] * df.loc[i, 'T'] * 100
        j = i

    del df['T'], df['dm'], df['cm']
    return df


def klinger(df, with_temp=True):
    df = volumeForce(df, with_temp=with_temp, colName='VF_Klinger')
    df = EMA(df, period=34, sourceCol='VF_Klinger', colName='klinger34')
    df = EMA(df, period=55, sourceCol='VF_Klinger', colName='klinger55')

    df['KO'] = df['klinger34'] - df['klinger55']
    df = EMA(df, period=13, sourceCol='KO', colName='KO Sig')
    df['KO Divg'] = df['KO'] - df['KO Sig']

    del df['VF_Klinger'], df['klinger34'], df['klinger55']
    return df


def shiftedSqueeze(df, maxPercent=0.01, emaPeriod=20, atrPeriod=20, atrMult=2, bollingerPeriod=20, nDeviations=2, colName='Squeeze'):
    if len(df) < max([emaPeriod, atrPeriod, bollingerPeriod]):
        raise ValueError('Not enough data')

    if not type(df) == pd.DataFrame or any(type(x) != int for x in [emaPeriod, atrPeriod, atrMult, bollingerPeriod, nDeviations]):
        raise TypeError

    req = Keltner(df, emaPeriod, atrPeriod, atrMult,
                  colNameUpper='Kelt Upper sq', colNameLower='Kelt Lower sq')
    req[['Bol Upper sq', 'Bol Lower sq']] = Bollinger(df, bollingerPeriod, nDeviations, colNameUpper='Bol Upper sq', colNameLower='Bol Lower sq')[
        ['Bol Upper sq', 'Bol Lower sq']]

    # if the bollinger belts are closer together than the keltner channels or within a certain % of the keltner's tightness, identifies squeeze
    df[colName] = ((req['Bol Upper sq'] - req['Bol Lower sq']) - (req['Kelt Upper sq'] -
                                                                  req['Kelt Lower sq'])) / (req['Bol Upper sq'] - req['Bol Lower sq']) <= maxPercent
    del df['Kelt Upper sq'], df['Kelt Lower sq'], df['Bol Upper sq'], df['Bol Lower sq']
    return df


def upwardsChannel(df, maxPercent=0.01, emaPeriod=20, atrPeriod=20, atrMult=2, bollingerPeriod=20, nDeviations=2, colName='Up Channel'):
    if len(df) < max([emaPeriod, atrPeriod, bollingerPeriod]):
        raise ValueError('Not enough data')

    if not type(df) == pd.DataFrame or any(type(x) != int for x in [emaPeriod, atrPeriod, atrMult, bollingerPeriod, nDeviations]):
        raise TypeError

    df = shiftedSqueeze(df, maxPercent=maxPercent, emaPeriod=emaPeriod, atrPeriod=atrPeriod,
                        atrMult=atrMult, bollingerPeriod=bollingerPeriod, nDeviations=nDeviations, colName='sq ch')
    df = EMA(df, emaPeriod, colName='ema ch')

    df[colName] = False
    in_squeeze = False
    for i in df.index:
        # squeeze starts
        if not in_squeeze and df.loc[i, 'sq ch']:
            in_squeeze = True
            start = i
            prev = i
        # while still in squeeze, checks if channel is increasing or decreasing. considers squeeze to be over if it's the last day
        elif in_squeeze and df.loc[i, 'sq ch'] and i != df.index[-1]:
            prev = i
        # when squeeze is over, checks if channel increased more than decreased
        # if in squeeze in the last day of the df, checks if the channel was pointed upwards and if so, also includes the last day in the upwards channel
        elif in_squeeze and df.loc[i, 'sq ch']:
            if df.loc[i, 'ema ch'] > df.loc[start, 'ema ch']:
                df.loc[start:i, colName] = True
            else:
                df.loc[start:prev, colName] = False
        elif in_squeeze:
            if df.loc[prev, 'ema ch'] > df.loc[start, 'ema ch']:
                df.loc[start:prev, colName] = True
            else:
                df.loc[start:prev, colName] = False
            in_squeeze = False

    del df['ema ch'], df['sq ch']
    return df


# makes hammer or hanging man indicators (bullish=True for hammer) to indicate trend reversals. if inverted, max_body_distance is from low instead of high
def hammer(df, trend_days=3, max_real_body_ratio=0.5, max_body_distance=0.2, bullish=True, inverted=False, colName=None):
    if not 0 < max_real_body_ratio <= 0.5 or not 0 < max_body_distance < 1 or trend_days < 0:
        raise ValueError

    colName = colName if colName is not None else 'Hammer' if bullish else 'Hanging'

    df[colName] = False
    for c, i in enumerate(df.index):
        # checks if the last 'trend_days' days of candles are all decreasing; if not, it is not a hammer
        break_flag = True
        if c > trend_days:
            for n in range(trend_days):
                break_flag = True
                if bullish and df.loc[df.index[c - n - 1], Cols.CLOSE.value] >= df.loc[df.index[c - n - 2], Cols.CLOSE.value]:
                    break
                if not bullish and df.loc[df.index[c - n - 1], Cols.CLOSE.value] <= df.loc[df.index[c - n - 2], Cols.CLOSE.value]:
                    break
                break_flag = False
        if break_flag:
            continue

        opening = df.loc[i, Cols.OPEN.value]
        close = df.loc[i, Cols.CLOSE.value]
        high = df.loc[i, Cols.HIGH.value]
        low = df.loc[i, Cols.LOW.value]
        # checks if the real body size (dif. b/w open and close) is 50% (body ratio) smaller than the high-low range
        # and whether the distance between the high and the top of the real body (bigger of open or close) is less than
        # 20% (max distance) of the high-low range. if inverted, checks distance between lower of open/close and the shadow low
        df.loc[i, colName] = abs(opening - close) <= ((high - low) * max_real_body_ratio) and ((
            high - max(opening, close)) < ((high - low) * max_body_distance)) if not inverted else ((
                min(opening, close)) - low < ((high - low) * max_body_distance))
    return df


def BollingerWidth(df, period=20, nDeviations=2, sourceCol=None, colName='BBW'):
    df = SMA(df, period, sourceCol=sourceCol, colName='BBW bol_mid')
    df = Bollinger(df, period, nDeviations, sourceCol, 'BBW bol_upper', 'BBW bol_lower')
    df[colName] = (df['BBW bol_upper'] - df['BBW bol_lower']) / df['BBW bol_mid']
    del df['BBW bol_upper'], df['BBW bol_lower'], df['BBW bol_mid']
    return df


def BBWP(df, period=21, lookback=255, smaPeriod=8, sourceCol=Cols.ADJCLOSE.value):
    df = BollingerWidth(df, period, sourceCol=sourceCol, colName='BBWP BBW')
    # calc number of values less than the current value, then divide by total number of values in the lookback
    # to get the percentile for the given BBW
    df['BBWP'] = df['BBWP BBW'].rolling(lookback + 1).apply(lambda s: s[s < s.iloc[-1]].shape[0]) / lookback
    df = SMA(df, smaPeriod, 'BBWP', 'BBWP_SMA')
    del df['BBWP BBW']
    return df


def tripleThreat(df, rsi_crossover=5, rsi_retest=3, rsi_instant=12, rsi_oversold=35, rsi_overbought=65, bbwp_confirm=4, ema_period=50, rsi_period=14, rsi_ma_period=7, bbwp_period=21, bbwp_lookback=255, bbwp_sma_period=8, sourceCol=Cols.ADJCLOSE.value):
    df = EMA(df, ema_period, sourceCol=sourceCol, colName='Confirmation_MA')
    df = RSI(df, rsi_period, sourceCol=sourceCol)
    df = SMA(df, rsi_ma_period, sourceCol='RSI', colName='RSI_MA')
    df = BBWP(df, bbwp_period, bbwp_lookback, bbwp_sma_period, sourceCol=sourceCol)

    cUp = 'crossed up'
    rUp = 'retested up'
    rBuy = 'retest buy'
    rsiBuyZone = 'rsi buy zone'
    rsiBuySignal = 'rsi buy signal'

    cDown = 'crossed down'
    rDown = 'retested down'
    rSell = 'retest sell'
    rsiSellZone = 'rsi sell zone'
    rsiSellSignal = 'rsi sell signal'

    confirmBBWP = 'bbwp confirm'

    fwdBuyConfirm = 'fwd buy confirm'
    allowBuy = 'allow buy'
    fullBuySignal = 'full buy signal'
    confirmedBuyZone = 'confirmed buy zone'
    confirmedBuySignal = 'confirmed buy signal'

    fwdSellConfirm = 'fwd sell confirm'
    allowSell = 'allow sell'
    fullSellSignal = 'full sell signal'
    confirmedSellZone = 'confirmed sell zone'
    confirmedSellSignal = 'confirmed sell signal'

    maBuyZone = 'ma buy zone'
    maBuyConfirm = 'ma buy confirm'
    maFullBuyZone = 'ma full buy zone'
    maBuySignal = 'ma buy signal'

    maSellZone = 'ma sell zone'
    maSellConfirm = 'ma sell confirm'
    maFullSellZone = 'ma full sell zone'
    maSellSignal = 'ma sell signal'


    df[cUp] = False
    df[rUp] = False
    df[rBuy] = False
    df[rsiBuyZone] = False
    df[rsiBuySignal] = False

    df[cDown] = False
    df[rDown] = False
    df[rSell] = False
    df[rsiSellZone] = False
    df[rsiSellSignal] = False

    df[confirmBBWP] = False

    df[fwdBuyConfirm] = False
    df[allowBuy] = False
    df[fullBuySignal] = False
    df[confirmedBuyZone] = False
    df[confirmedBuySignal] = False

    df[fwdSellConfirm] = False
    df[allowSell] = False
    df[fullSellSignal] = False
    df[confirmedSellZone] = False
    df[confirmedSellSignal] = False

    df[maBuyZone] = False
    df[maBuyConfirm] = False
    df[maFullBuyZone] = False
    df[maBuySignal] = False

    df[maSellZone] = False
    df[maSellConfirm] = False
    df[maFullSellZone] = False
    df[maSellSignal] = False
    

    for c, i in enumerate(df.index[1:]):
        prev = df.loc[df.index[c]]
        rsi = df.loc[i, 'RSI']
        rsi_ma = df.loc[i, 'RSI_MA']
        bbwp = df.loc[i, 'BBWP']
        bbwp_ma = df.loc[i, 'BBWP_SMA']
        price = df.loc[i, sourceCol]
        ma = df.loc[i, 'Confirmation_MA']

        # start checking for upwards retests when the RSI has crossed above the MA by the rsi_crossover and has not fallen to retest_threshold below the MA
        df.loc[i, cUp] = (prev[cUp] and not (rsi_ma - rsi > rsi_retest)) or (rsi - rsi_ma > rsi_crossover)
        # keep of track of when a retest occured (rsi exceeds MA by less than the retest threshold), until the RSI falls below the MA by the threshold. rsi crossed up must be true, because this already checks if the rsi has fallen below the MA by the threshold to cancel the signal
        df.loc[i, rUp] = df.loc[i, cUp] and (prev[rUp] or (rsi - rsi_ma < rsi_retest))
        # if there has been a retest and the rsi exceeds the MA by the crossover threshold again, trigger a buy signal. also trigger a buy signal if the RSI exceeds the MA by the "rsi instant" threshold
        df.loc[i, rBuy] = (df.loc[i, rUp] and (rsi - rsi_ma > rsi_crossover)) or (rsi - rsi_ma > rsi_instant)

        # now do the same for the sell side
        df.loc[i, cDown] = (prev[cDown] and not (rsi - rsi_ma > rsi_retest)) or (rsi_ma - rsi > rsi_crossover)
        df.loc[i, rDown] = df.loc[i, cDown] and (prev[rDown] or (rsi_ma - rsi < rsi_retest))
        df.loc[i, rSell] = (df.loc[i, rDown] and (rsi_ma - rsi > rsi_crossover)) or (rsi_ma - rsi > rsi_instant)

        # a zone is when a directional signal was given, but it carries over from previous zones unless a contra signal is given
        df.loc[i, rsiBuyZone] = (df.loc[i, rBuy] or prev[rsiBuyZone]) and not df.loc[i, rSell]
        df.loc[i, rsiSellZone] = (df.loc[i, rSell] or prev[rsiSellZone]) and not df.loc[i, rBuy]
        # create a signal when the zone switches
        df.loc[i, rsiBuySignal] = df.loc[i, rsiBuyZone] and prev[rsiSellZone]
        df.loc[i, rsiSellSignal] = df.loc[i, rsiSellZone] and prev[rsiBuyZone]

        # if RSI is oversold in a Buy zone, confirm the most recent Buy signal. if RSI is oversold in a Sell zone, confirm the next Buy signal.
        df.loc[i, fwdBuyConfirm] = False
        # carry over the previous confirmation unless entering a sell zone for the first time
        df.loc[i, fwdBuyConfirm] = (prev[fwdBuyConfirm] and not df.loc[i, rsiSellSignal]) or rsi <= rsi_oversold

        # if RSI is overbought in a Sell zone, confirm the most recent Sell signal. if RSI is overbought in a Buy zone, confirm the next Sell signal.
        df.loc[i, fwdSellConfirm] = False
        # carry over the previous confirmation unless entering a buy zone for the first time
        df.loc[i, fwdSellConfirm] = (prev[fwdSellConfirm] and not df.loc[i, rsiBuySignal]) or rsi >= rsi_overbought

        # a long position may only be entered if the RSI MA is no longer below the oversold level, and RSI is now above the MA
        df.loc[i, allowBuy] = rsi > rsi_ma and rsi_ma > rsi_oversold
        # a short position may only be entered if the RSI MA is no longer above the overbought level, and RSI is now below the MA
        df.loc[i, allowSell] = rsi < rsi_ma and rsi_ma < rsi_overbought

        # confirm any signal if BBWP > the BBWP confirmation threshold
        df.loc[i, confirmBBWP] = bbwp - bbwp_ma > bbwp_confirm

        # trade on confirmed signals
        df.loc[i, fullBuySignal] = df.loc[i, rsiBuyZone] and (df.loc[i, fwdBuyConfirm] or df.loc[i, confirmBBWP]) and df.loc[i, allowBuy]
        df.loc[i, fullSellSignal] = df.loc[i, rsiSellZone] and (df.loc[i, fwdSellConfirm] or df.loc[i, confirmBBWP]) and df.loc[i, allowSell]

        # only enter a buy or sell zone if there was confirmation with either the RSI or BBWP, and a position is allowed based on the relative placement of the RSI and its MA (allow_buy or allow_sell)
        df.loc[i, confirmedBuyZone] = (prev[confirmedBuyZone] or df.loc[i, fullBuySignal]) and not df.loc[i, fullSellSignal]
        df.loc[i, confirmedSellZone] = (prev[confirmedSellZone] or df.loc[i, fullSellSignal]) and not df.loc[i, fullBuySignal]

        df.loc[i, confirmedBuySignal] = df.loc[i, confirmedBuyZone] and prev[confirmedSellZone]
        df.loc[i, confirmedSellSignal] = df.loc[i, confirmedSellZone] and prev[confirmedBuyZone]

        # EMAs and EMA signals
        df.loc[i, maBuyZone] = df.loc[i, confirmedBuyZone] and (price > ma)
        df.loc[i, maSellZone] = df.loc[i, confirmedSellZone] and (price < ma)

        df.loc[i, maBuyConfirm] = df.loc[i, maBuyZone] and not prev[maBuyZone]
        df.loc[i, maSellConfirm] = df.loc[i, maSellZone] and not prev[maSellZone]

        df.loc[i, maFullBuyZone] = (prev[maFullBuyZone] or df.loc[i, maBuyConfirm]) and not df.loc[i, maSellConfirm]
        df.loc[i, maFullSellZone] = (prev[maFullSellZone] or df.loc[i, maSellConfirm]) and not df.loc[i, maBuyConfirm]

        df.loc[i, maBuySignal] = df.loc[i, maFullBuyZone] and not prev[maFullBuyZone]
        df.loc[i, maSellSignal] = df.loc[i, maFullSellZone] and not prev[maFullSellZone]

    # delete all supplemental cols except for the signals and indicators
    del df[cUp]
    del df[rUp]
    del df[rBuy]
    del df[rsiBuyZone]

    del df[cDown]
    del df[rDown]
    del df[rSell]
    del df[rsiSellZone]

    del df[confirmBBWP]

    del df[fwdBuyConfirm]
    del df[allowBuy]
    del df[fullBuySignal]
    del df[confirmedBuyZone]

    del df[fwdSellConfirm]
    del df[allowSell]
    del df[fullSellSignal]
    del df[confirmedSellZone]

    del df[maBuyZone]
    del df[maBuyConfirm]
    del df[maFullBuyZone]

    del df[maSellZone]
    del df[maSellConfirm]
    del df[maFullSellZone]
    return df
