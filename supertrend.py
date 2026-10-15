"""
The desk's supertrend, as recovered in wm-STstrat (validated there against the
AA report's SCMO_0043 ledger). Reproduced here, not re-derived:

  * bars:        weekly (W-FRI) or monthly, built from daily OHLC with resample_ohlc
  * ATR:         5-bar LINEAR-WEIGHTED true range, weights 1/15 .. 5/15 (newest heaviest)
                 ("lw"); "ema" = plain EMA span 5; "rma" = Wilder's
  * bands:       (H + L) / 2  +/-  multiplier x ATR, multiplier 2
  * state:       +1 (buy) when close crosses above the upper band from a sell
                 state; -1 (sell) when close crosses below the lower band from a
                 buy state; bands ratchet in the direction of the trend

Once the data files are in place, checks.py pins the weekly SC100 flip dates to
the parent repo's ledger so any drift in this copy is caught.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def resample_ohlc(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    o = df["open"].resample(freq).first()
    h = df["high"].resample(freq).max()
    l = df["low"].resample(freq).min()
    c = df["close"].resample(freq).last()
    out = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna(subset=["close"])
    return out


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr


def atr(df: pd.DataFrame, lookback: int = C.LOOKBACK, method: str = C.ATR_METHOD) -> pd.Series:
    tr = true_range(df)
    if method == "lw":
        w = np.arange(1, lookback + 1, dtype=float)
        w /= w.sum()
        return tr.rolling(lookback).apply(lambda x: float(np.dot(x, w)), raw=True)
    if method == "ema":
        return tr.ewm(span=lookback, adjust=False).mean()
    if method == "rma":
        return tr.ewm(alpha=1.0 / lookback, adjust=False).mean()
    raise ValueError(method)


def supertrend(df: pd.DataFrame, lookback: int = C.LOOKBACK, mult: float = C.MULTIPLIER,
               method: str = C.ATR_METHOD) -> pd.DataFrame:
    """Returns DataFrame[state, upper, lower, stop] on the bar index. state in {+1, -1}."""
    a = atr(df, lookback, method)
    hl2 = (df["high"] + df["low"]) / 2.0
    up_basic = (hl2 + mult * a).values
    dn_basic = (hl2 - mult * a).values
    close = df["close"].values
    n = len(df)
    up = np.full(n, np.nan)
    dn = np.full(n, np.nan)
    state = np.zeros(n, dtype=int)
    first = int(np.argmax(~np.isnan(up_basic))) if (~np.isnan(up_basic)).any() else n
    for i in range(first, n):
        if i == first:
            up[i], dn[i] = up_basic[i], dn_basic[i]
            state[i] = 1 if close[i] > up[i] else -1
            continue
        # ratchet the bands
        up[i] = up_basic[i] if (up_basic[i] < up[i - 1] or close[i - 1] > up[i - 1]) else up[i - 1]
        dn[i] = dn_basic[i] if (dn_basic[i] > dn[i - 1] or close[i - 1] < dn[i - 1]) else dn[i - 1]
        if state[i - 1] == -1 and close[i] > up[i]:
            state[i] = 1
        elif state[i - 1] == 1 and close[i] < dn[i]:
            state[i] = -1
        else:
            state[i] = state[i - 1]
    out = pd.DataFrame({"state": state, "upper": up, "lower": dn}, index=df.index)
    out["stop"] = np.where(out["state"] == 1, out["lower"], out["upper"])
    out.loc[out.index[:first], "state"] = np.nan
    return out


def states(panel: dict[str, pd.DataFrame], labels: list[str], freq: str = C.SIGNAL_BAR) -> pd.DataFrame:
    """Weekly supertrend state for each label, aligned on the weekly grid."""
    out = {}
    for lab in labels:
        if lab in panel:
            bars = resample_ohlc(panel[lab], freq)
            out[lab] = supertrend(bars)["state"]
    return pd.DataFrame(out)
