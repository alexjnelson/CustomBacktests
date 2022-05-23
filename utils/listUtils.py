import datetime as dt
import re
from functools import partial
from multiprocessing import Pool

from CustomBacktests.utils.screenerUtils import load_tickers, make_df


def makeTickerFile(inf, outf, ext=''):
    """
    Formats a file of tickers for use in this module. Input file must be formatted such that
    the tickers are the first characters on each row. If the tickers require an extension (e.g. TSE stocks
    must end in '.to'), the extension can be passed to this function.


    Args:
        inf: string or path referencing the input file
        outf: string or path referencing the output file
        ext: the extension to be appended to each ticker

    Returns:
        None

    Raises:
        TO DO
    """
    with open(inf) as file:
        tickers = set()
        for row in file:
            tickers.add(re.match(r'[\w\.]+', row).group(0).replace('.', '-'))

    with open(outf, 'w+') as file:
        for t in tickers:
            file.write(f'{t}.{ext}\n')


def removeBadTickers(inf='lists/tsx.txt', outf=None, start=None, end=None):
    """
    Certain tickers may not have enough data for the desired period, so this function can be used
    to filter them out.


    Args:
        inf: string or path referencing the input file
        outf: string or path referencing the output file
        start: the datetime date at the start of the desired period (inclusive)
        end: the datetime date at the end of the desired period (inclusive)

    Returns:
        None

    Raises:
        TO DO
    """
    tickers = load_tickers(inf)
    start = dt.date(2018, 1, 1) if start is None else start
    end = dt.date.today() if end is None else end
    inf_no_ext = inf.split('.')[0]
    outf = f'{inf_no_ext}_cleaned.txt' if outf is None else outf
    updated = []

    p = Pool(4)
    f = partial(make_df, start, end)

    for df in p.imap_unordered(f, tickers):
        if df is not None and len(df[1]) > 0:
            updated.append(df[0])

    with open(outf, 'w+') as file:
        for t in updated:
            file.write(t + '\n')
