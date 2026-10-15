"""
2. The HISTORICAL PATTERN between correlation and returns.

  by_year      For each calendar year: the average 13-week correlation among the four
               indices, its low and high, and each asset's return that year. A table a
               person can read top to bottom.

  same_quarter The trailing-quarter correlation cut into quintiles -> the return each
               asset made over THAT SAME quarter (what was happening while correlation
               was high / low) and over the NEXT quarter (what followed).

  timeline     Week by week: the 13-week correlation, N100's trailing 13-week return,
               and the VIX level if the file is present - for the chart.

  episodes     Stretches where the 13-week correlation stayed above 0.90 for 8+ weeks:
               start, end, length, and what N100 and SC100 did during and after.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
from sigcorr import pair_corrs, rolling_mean_corr


def by_year_pairs(r: pd.DataFrame) -> pd.DataFrame:
    """Every pair's average 13-week correlation in each calendar year (rows = year, columns = pair)."""
    PC = pair_corrs(r, C.HIST_WINDOW)
    out = PC.groupby(PC.index.year).mean()
    out.index.name = "year"
    return out


def same_quarter_pairs(r: pd.DataFrame) -> pd.DataFrame:
    """The quintile study pair by pair: the weeks sorted by THAT pair's 13-week correlation into five groups, and
    each leg's return over the same quarter and the next. Long: pair, leg, bucket, corr_lo, corr_hi, weeks, same/next mean and hit."""
    w, f = C.HIST_WINDOW, C.FWD_WINDOW
    PC = pair_corrs(r, w)
    labels = [f"Q{i + 1}" for i in range(C.N_BUCKETS)]
    rows = []
    for p in PC.columns:
        c = PC[p].dropna()
        if len(c) < 5 * C.N_BUCKETS:
            continue
        bk = pd.qcut(c, C.N_BUCKETS, labels=labels)
        for leg in p.split("|"):
            same = r[leg].rolling(w).sum()
            nxt = r[leg].shift(-1).rolling(f).sum().shift(-(f - 1))
            df = pd.DataFrame({"bucket": bk, "same": same.reindex(c.index), "next": nxt.reindex(c.index), "c": c})
            for b, sub in df.groupby("bucket", observed=True):
                rows.append({"pair": p, "leg": leg, "bucket": b, "corr_lo": sub["c"].min(), "corr_hi": sub["c"].max(), "weeks": len(sub),
                             "same_quarter_mean": sub["same"].mean(), "same_quarter_hit": float((sub["same"] > 0).mean()),
                             "next_quarter_mean": sub["next"].mean(), "next_quarter_hit": float((sub["next"] > 0).mean()), "n_next": int(sub["next"].notna().sum())})
    return pd.DataFrame(rows)


def by_year(r: pd.DataFrame) -> pd.DataFrame:
    mc = rolling_mean_corr(r, C.EQUITY, C.HIST_WINDOW)
    rows = []
    for y, g in r.groupby(r.index.year):
        m = mc.loc[g.index].dropna()
        row = {"year": int(y), "weeks": len(g), "mean_corr": m.mean(), "min_corr": m.min(), "max_corr": m.max(),
               "weeks_corr_above_0.9": int((m > 0.9).sum())}
        for a in C.ASSETS:
            if a in g.columns:
                row[f"ret_{a}"] = float(np.exp(g[a].sum()) - 1) if g[a].notna().sum() > 10 else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("year")


def same_quarter(r: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    w, f = C.HIST_WINDOW, C.FWD_WINDOW
    mc = rolling_mean_corr(r, C.EQUITY, w)
    labels = [f"Q{i + 1}" for i in range(C.N_BUCKETS)]
    bk = pd.qcut(mc, C.N_BUCKETS, labels=labels)
    rows = []
    for a in C.ASSETS:
        if a not in r.columns:
            continue
        same = r[a].rolling(w).sum()                            # return over the same trailing quarter
        nxt = r[a].shift(-1).rolling(f).sum().shift(-(f - 1))   # return over the following quarter (t+1..t+f)
        df = pd.DataFrame({"bucket": bk, "same": same, "next": nxt, "c": mc}).dropna(subset=["bucket", "c"])
        g = df.groupby("bucket", observed=True)
        for b in labels:
            if b not in g.groups:
                continue
            sub = g.get_group(b)
            rows.append({"asset": a, "bucket": b, "corr_lo": sub["c"].min(), "corr_hi": sub["c"].max(), "weeks": len(sub),
                         "same_quarter_mean": sub["same"].mean(), "same_quarter_hit": float((sub["same"] > 0).mean()),
                         "next_quarter_mean": sub["next"].mean(), "next_quarter_hit": float((sub["next"] > 0).mean()),
                         "n_next": int(sub["next"].notna().sum())})
    long = pd.DataFrame(rows)
    # a compact pivot for the report: rows = asset, columns = bucket, two blocks
    same_p = long.pivot(index="asset", columns="bucket", values="same_quarter_mean")
    next_p = long.pivot(index="asset", columns="bucket", values="next_quarter_mean")
    same_p.columns = [f"same_{c}" for c in same_p.columns]
    next_p.columns = [f"next_{c}" for c in next_p.columns]
    piv = pd.concat([same_p, next_p], axis=1).reindex([a for a in C.ASSETS if a in r.columns])
    return long, piv


def timeline(r: pd.DataFrame, lvl: pd.Series | None) -> pd.DataFrame:
    mc = rolling_mean_corr(r, C.EQUITY, C.HIST_WINDOW)
    out = pd.DataFrame({"mean_corr_13w": mc, "N100_ret_13w": r[C.BENCHMARK].rolling(C.HIST_WINDOW).sum(),
                        "SC100_ret_13w": r["SC100"].rolling(C.HIST_WINDOW).sum() if "SC100" in r.columns else np.nan})
    if lvl is not None and len(lvl):
        out["VIX"] = lvl.reindex(out.index)
    return pd.concat([out, pair_corrs(r, C.HIST_WINDOW).reindex(out.index)], axis=1)


def _stretches(mc: pd.Series, thresh: float, min_weeks: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of every stretch where `mc` stayed above `thresh` for at least `min_weeks` weeks (an open stretch at the end counts)."""
    above = (mc > thresh).fillna(False)
    out, start = [], None
    for d, v in above.items():
        if v and start is None:
            start = d
        if (not v) and start is not None:
            end = mc.index[mc.index.get_loc(d) - 1]
            if mc.index.get_loc(end) - mc.index.get_loc(start) + 1 >= min_weeks:
                out.append((start, end))
            start = None
    if start is not None and mc.index.get_loc(mc.index[-1]) - mc.index.get_loc(start) + 1 >= min_weeks:
        out.append((start, mc.index[-1]))
    return out


def episodes(r: pd.DataFrame, thresh: float = 0.90, min_weeks: int = 8) -> pd.DataFrame:
    mc = rolling_mean_corr(r, C.EQUITY, C.HIST_WINDOW)
    rows = []
    for start, end in _stretches(mc, thresh, min_weeks):
        i0, i1 = r.index.get_loc(start), r.index.get_loc(end)
        row = {"start": start.date(), "end": end.date(), "weeks": i1 - i0 + 1, "peak_corr": mc.loc[start:end].max()}
        for a in ("N100", "SC100", "GoldMCX"):
            if a in r.columns:
                row[f"{a}_during"] = float(np.exp(r[a].iloc[i0:i1 + 1].sum()) - 1)
                row[f"{a}_next_13w"] = float(np.exp(r[a].iloc[i1 + 1:i1 + 14].sum()) - 1) if i1 + 14 <= len(r) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def episodes_pairs(r: pd.DataFrame, thresh: float = 0.90, min_weeks: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The same stretches pair by pair: every stretch of 8+ weeks in which THAT pair's 13-week correlation stayed above 0.90,
    with each leg's return during it and over the following quarter; and a one-row-per-pair summary."""
    PC = pair_corrs(r, C.HIST_WINDOW)
    rows = []
    for p in PC.columns:
        a, b = p.split("|")
        for start, end in _stretches(PC[p], thresh, min_weeks):
            i0, i1 = r.index.get_loc(start), r.index.get_loc(end)
            row = {"pair": p, "start": start.date(), "end": end.date(), "weeks": i1 - i0 + 1, "peak_corr": PC[p].loc[start:end].max()}
            for leg, lab in ((a, "a"), (b, "b")):
                row[f"{lab}_during"] = float(np.exp(r[leg].iloc[i0:i1 + 1].sum()) - 1)
                row[f"{lab}_next_13w"] = float(np.exp(r[leg].iloc[i1 + 1:i1 + 14].sum()) - 1) if i1 + 14 <= len(r) else np.nan
            rows.append(row)
    long = pd.DataFrame(rows)
    srows = []
    for p in PC.columns:
        g = long[long["pair"] == p] if len(long) else long
        c = PC[p].dropna()
        srows.append({"pair": p, "weeks_above_0.90": int((c > thresh).sum()), "share_of_weeks_above_0.90": float((c > thresh).mean()) if len(c) else np.nan,
                      "stretches": int(len(g)), "median_weeks": float(g["weeks"].median()) if len(g) else np.nan,
                      "a_next_13w_mean": float(g["a_next_13w"].mean()) if len(g) else np.nan, "a_next_13w_positive": float((g["a_next_13w"] > 0).mean()) if g["a_next_13w"].notna().any() else np.nan,
                      "b_next_13w_mean": float(g["b_next_13w"].mean()) if len(g) else np.nan, "b_next_13w_positive": float((g["b_next_13w"] > 0).mean()) if g["b_next_13w"].notna().any() else np.nan})
    return long, pd.DataFrame(srows).set_index("pair")
