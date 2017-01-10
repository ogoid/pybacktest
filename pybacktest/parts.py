# coding: utf8

# part of pybacktest package: https://github.com/ematvey/pybacktest

""" Essential functions for translating signals into trades.
Usable both in backtesting and production.

"""

import pandas


try:
    import numba
    @numba.guvectorize(
        ['void(int32[:], int32[:], int32[:], int32[:], int32[:], int32[:])',
         'void(int64[:], int64[:], int64[:], int64[:], int64[:], int64[:])',
        ], '(m),(m),(m),(m),()->(m)')
    def _stp_vec(long_en, long_ex, short_en, short_ex, pos, res):
        for i in range(long_en.shape[0]):
            exit_long  = pos[0] > 0 and long_ex[i]
            exit_short = pos[0] < 0 and short_ex[i]
            pos[0] = 0 if exit_long or exit_short else pos[0]
            pos[0] = long_en[i] - short_en[i] if pos[0] == 0 else pos[0]
            res[i] = pos[0]

    def signals_to_positions(signals, init_pos=0, mask=('Buy','Sell','Short','Cover')):
        """
        Translate signal dataframe into positions series (trade prices aren't
        specified.
        WARNING: In production, override default zero value in init_pos with
        extreme caution.
        """
        long_en, long_ex, short_en, short_ex = mask
        ps = pandas.Series(0., index=signals.index)
        _stp_vec(signals[long_en], signals[long_ex],
                 signals[short_en], signals[short_ex],
                 init_pos, ps.values)
        return ps[ps != ps.shift()]

except ImportError:
    def signals_to_positions(signals, init_pos=0,
                             mask=('Buy', 'Sell', 'Short', 'Cover')):
        """
        Translate signal dataframe into positions series (trade prices aren't
        specified.
        WARNING: In production, override default zero value in init_pos with
        extreme caution.
        """
        long_en, long_ex, short_en, short_ex = mask
        pos = init_pos
        ps = pandas.Series(0., index=signals.index)
        for t, sig in signals.iterrows():
            # check exit signals
            if pos != 0:  # if in position
                if pos > 0 and sig[long_ex]:  # if exit long signal
                    pos -= sig[long_ex]
                elif pos < 0 and sig[short_ex]:  # if exit short signal
                    pos += sig[short_ex]
            # check entry (possibly right after exit)
            if pos == 0:
                if sig[long_en]:
                    pos += sig[long_en]
                elif sig[short_en]:
                    pos -= sig[short_en]
            ps[t] = pos
        return ps[ps != ps.shift()]



def trades_to_equity(trd):
    """
    Convert trades dataframe (cols [vol, price, pos]) to equity diff series
    """

    def _cmp_fn(x):
        if x > 0:
            return 1
        elif x < 0:
            return -1
        else:
            return 0

    psig = trd.pos.apply(_cmp_fn)
    closepoint = psig != psig.shift()
    e = (trd.vol * trd.price).cumsum()[closepoint] - \
        (trd.pos * trd.price)[closepoint]
    e = e.diff()
    e = e.reindex(trd.index).fillna(value=0)
    e[e != 0] *= -1
    return e


def extract_frame(dataobj, ext_mask, int_mask):
    df = {}
    for f_int, f_ext in zip(int_mask, ext_mask):
        obj = dataobj.get(f_ext)
        if isinstance(obj, pandas.Series):
            df[f_int] = obj
        else:
            df[f_int] = None
    if any([isinstance(x, pandas.Series) for x in list(df.values())]):
        return pandas.DataFrame(df)
    return None


class Slicer(object):
    def __init__(self, target, obj):
        self.target = target
        self.__len__ = obj.__len__

    def __getitem__(self, x):
        return self.target(x)
