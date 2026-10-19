"""
1. Correlation and covariance WITH RESPECT TO THE SIGNAL.

Three simple things, all weekly:

  by_state     For every pair of assets, the correlation, the covariance and the two
               volatilities, computed only in weeks where BOTH signals were BUY, only
               where both were SELL, and where they disagreed. Shows whether the pair
               is more coupled in SELL, and whether the covariance rise is correlation
               or just volatility (cov = corr x vol_a x vol_b).

  around_flip  Event study, for EVERY asset as the reference. For every week the asset's
               signal flips to SELL, the average correlation among the four indices (G) and
               the asset's own correlation with the others in each of the 8 weeks before and
               after (13-week rolling corr). Same for flips to BUY. Shows whether correlation
               rises INTO a sell signal (a warning) or only after it.

  by_breadth   The 13-week rolling correlation among the indices against how many of the
               four signals are in SELL that week (0..4). Shows whether the market is
               more coupled when more signals are off.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import config as C
import supertrend as ST


def states(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    st = ST.states(panel, C.ASSETS, C.SIGNAL_BAR)
    return st[st.index >= pd.Timestamp(C.START)]


def pairs(cols) -> list[tuple[str, str]]:
    return list(itertools.combinations([c for c in C.ASSETS if c in cols], 2))


def by_state(st: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ann = C.ANNUALISE
    for a, b in pairs(r.columns):
        sa, sb = st[a].reindex(r.index), st[b].reindex(r.index)
        cells = {"both BUY": (sa == 1) & (sb == 1), "both SELL": (sa == -1) & (sb == -1), "disagree": (sa * sb == -1), "all weeks": sa.notna() & sb.notna()}
        for name, m in cells.items():
            xy = r.loc[m, [a, b]].dropna()
            if len(xy) < C.MIN_CELL:
                rows.append({"pair": f"{a}|{b}", "a": a, "b": b, "state": name, "weeks": len(xy)})
                continue
            cov = xy[a].cov(xy[b]) * ann
            va, vb = xy[a].std() * np.sqrt(ann), xy[b].std() * np.sqrt(ann)
            rows.append({"pair": f"{a}|{b}", "a": a, "b": b, "state": name, "weeks": len(xy), "corr": xy[a].corr(xy[b]),
                         "cov_ann": cov, "vol_a_ann": va, "vol_b_ann": vb, "mean_ret_a_ann": xy[a].mean() * ann, "mean_ret_b_ann": xy[b].mean() * ann})
    df = pd.DataFrame(rows)
    # decomposition: how much of the SELL-vs-BUY covariance change is correlation vs volatility
    out = []
    for pair, g in df.groupby("pair"):
        g = g.set_index("state")
        if "both BUY" in g.index and "both SELL" in g.index and pd.notna(g.loc["both BUY", "corr"]) and pd.notna(g.loc["both SELL", "corr"]):
            cb, cs = g.loc["both BUY", "corr"], g.loc["both SELL", "corr"]
            vab, vbb = g.loc["both BUY", "vol_a_ann"], g.loc["both BUY", "vol_b_ann"]
            vas, vbs = g.loc["both SELL", "vol_a_ann"], g.loc["both SELL", "vol_b_ann"]
            cov_b, cov_s = g.loc["both BUY", "cov_ann"], g.loc["both SELL", "cov_ann"]
            # covariance if only correlation had changed (vols held at BUY levels)
            cov_corr_only = cs * vab * vbb
            out.append({"pair": pair, "corr_buy": cb, "corr_sell": cs, "corr_change": cs - cb, "cov_buy": cov_b, "cov_sell": cov_s,
                        "cov_ratio_sell_over_buy": cov_s / cov_b if cov_b else np.nan,
                        "share_of_cov_change_from_corr": (cov_corr_only - cov_b) / (cov_s - cov_b) if (cov_s - cov_b) else np.nan,
                        "share_of_cov_change_from_vol": 1 - (cov_corr_only - cov_b) / (cov_s - cov_b) if (cov_s - cov_b) else np.nan,
                        "vol_a_buy": vab, "vol_a_sell": vas, "vol_b_buy": vbb, "vol_b_sell": vbs,
                        "weeks_buy": int(g.loc["both BUY", "weeks"]), "weeks_sell": int(g.loc["both SELL", "weeks"])})
    return df, pd.DataFrame(out)


def rolling_mean_corr(r: pd.DataFrame, labels: list[str], w: int = C.CORR_WINDOW) -> pd.Series:
    labels = [l for l in labels if l in r.columns]
    vals = [r[a].rolling(w, min_periods=int(0.8 * w)).corr(r[b]) for a, b in itertools.combinations(labels, 2)]
    return pd.concat(vals, axis=1).mean(axis=1, skipna=True).rename("mean_corr")


def pair_corrs(r: pd.DataFrame, w: int = C.CORR_WINDOW) -> pd.DataFrame:
    """Every pair's rolling correlation, week by week — all ten pairs, gold included; columns 'a|b'.
    G (rolling_mean_corr over the indices) is the row-mean of the six index-pair columns here."""
    return pd.DataFrame({f"{a}|{b}": r[a].rolling(w, min_periods=int(0.8 * w)).corr(r[b]) for a, b in pairs(r.columns)})


def index_pairs(cols) -> list[str]:
    return [p for p in cols if "|" in p and C.GOLD not in p]


def around_flip(st: pd.DataFrame, r: pd.DataFrame, refs: list[str] | None = None) -> pd.DataFrame:
    """For EVERY asset as the reference: the average 13-week correlation among the four indices (G), and the
    reference's own 13-week correlation with the other indices (c_ref), at k = -8..+8 weeks around each of
    the reference's flips, averaged over flips. Also the reference's own trailing 13-week return at each offset."""
    mc = rolling_mean_corr(r, C.EQUITY)
    refs = refs or [a for a in C.ASSETS if a in st.columns and a in r.columns]
    rows = []
    for ref in refs:
        others = [a for a in C.EQUITY if a in r.columns and a != ref]
        own = pd.concat([r[ref].rolling(C.CORR_WINDOW, min_periods=int(0.8 * C.CORR_WINDOW)).corr(r[b]) for b in others], axis=1).mean(axis=1, skipna=True)
        ret13 = r[ref].rolling(13).sum()
        s = st[ref].reindex(mc.index)
        for direction, val in (("flip to SELL", -1), ("flip to BUY", 1)):
            flips = mc.index[((s == val) & (s.shift(1) == -val)).fillna(False)]
            for k in range(-C.FLIP_WINDOW, C.FLIP_WINDOW + 1):
                g, gx, o, rt = [], [], [], []
                for d in flips:
                    i = mc.index.get_loc(d) + k
                    if 0 <= i < len(mc):
                        if pd.notna(mc.iloc[i]):
                            g.append(mc.iloc[i])
                            if d.year != C.FLIP_EXCLUDE_YEAR:
                                gx.append(mc.iloc[i])
                        if pd.notna(own.iloc[i]):
                            o.append(own.iloc[i])
                        if pd.notna(ret13.iloc[i]):
                            rt.append(ret13.iloc[i])
                rows.append({"ref": ref, "event": direction, "weeks_from_flip": k, "mean_corr": float(np.mean(g)) if g else np.nan,
                             "median_corr": float(np.median(g)) if g else np.nan, f"mean_corr_ex{C.FLIP_EXCLUDE_YEAR}": float(np.mean(gx)) if gx else np.nan, f"n_events_ex{C.FLIP_EXCLUDE_YEAR}": len(gx),
                             "ref_own_corr": float(np.mean(o)) if o else np.nan, "n_events": len(g), "ref_trailing_13w_return": float(np.mean(rt)) if rt else np.nan})
    return pd.DataFrame(rows)


def around_flip_pairs(st: pd.DataFrame, r: pd.DataFrame, refs: list[str] | None = None) -> pd.DataFrame:
    """The same event study pair by pair: every pair's 13-week correlation at k = -8..+8 weeks around each of the
    reference asset's flips, averaged over that asset's flips. Long: ref, event, pair, weeks_from_flip, mean_corr, n_events."""
    PC = pair_corrs(r)
    refs = refs or [a for a in C.ASSETS if a in st.columns and a in r.columns]
    rows = []
    for ref in refs:
        s = st[ref].reindex(PC.index)
        for direction, val in (("flip to SELL", -1), ("flip to BUY", 1)):
            fd = PC.index[((s == val) & (s.shift(1) == -val)).fillna(False)]
            flips = [PC.index.get_loc(d) for d in fd]
            keep = [PC.index.get_loc(d) for d in fd if d.year != C.FLIP_EXCLUDE_YEAR]
            for k in range(-C.FLIP_WINDOW, C.FLIP_WINDOW + 1):
                idx = [i + k for i in flips if 0 <= i + k < len(PC)]
                idx_x = [i + k for i in keep if 0 <= i + k < len(PC)]
                sub = PC.iloc[idx]; subx = PC.iloc[idx_x]
                for p in PC.columns:
                    v = sub[p].dropna(); vx = subx[p].dropna()
                    rows.append({"ref": ref, "event": direction, "pair": p, "weeks_from_flip": k, "mean_corr": float(v.mean()) if len(v) else np.nan, "median_corr": float(v.median()) if len(v) else np.nan,
                                 f"mean_corr_ex{C.FLIP_EXCLUDE_YEAR}": float(vx.mean()) if len(vx) else np.nan, "n_events": int(len(v)), f"n_events_ex{C.FLIP_EXCLUDE_YEAR}": int(len(vx))})
    return pd.DataFrame(rows)


BREADTH_LABELS = ["none in SELL", "1 of 3-4 in SELL", "about half", "most in SELL", "all in SELL"]


def _breadth_bucket(st: pd.DataFrame, idx: pd.Index) -> tuple[pd.Series, pd.Series]:
    eq = [c for c in C.EQUITY if c in st.columns]
    live = st[eq].notna().sum(axis=1)
    n_sell = (st[eq] == -1).sum(axis=1)
    share = (n_sell / live.replace(0, np.nan)).reindex(idx)
    return pd.cut(share, bins=[-0.01, 0.0, 0.34, 0.67, 0.99, 1.0], labels=BREADTH_LABELS), n_sell.reindex(idx)


def by_breadth(st: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    mc = rolling_mean_corr(r, C.EQUITY)
    bucket, n_sell = _breadth_bucket(st, mc.index)
    df = pd.DataFrame({"mean_corr": mc, "bucket": bucket, "n_sell": n_sell})
    g = df.dropna().groupby("bucket", observed=True)["mean_corr"]
    out = g.agg(["mean", "median", "count"]).rename(columns={"count": "weeks"})
    return out


def by_breadth_pairs(st: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    """Every pair's average 13-week correlation by how many of the four index signals were in SELL that week.
    Wide: one row per breadth bucket, one column per pair (plus 'weeks')."""
    PC = pair_corrs(r)
    bucket, _ = _breadth_bucket(st, PC.index)
    out = PC.groupby(bucket, observed=True).mean().reindex(BREADTH_LABELS)
    live = PC[index_pairs(PC.columns)].notna().any(axis=1)          # weeks with a correlation to average (the first weeks have none)
    out["weeks"] = bucket[live].value_counts().reindex(BREADTH_LABELS).fillna(0).astype(int)
    out.index.name = "bucket"
    return out


def corr_timeseries(st: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    """The 13-week mean correlation, every pair's 13-week correlation, the number of signals in SELL, and N100's state, week by week (for the chart)."""
    mc = rolling_mean_corr(r, C.EQUITY)
    eq = [c for c in C.EQUITY if c in st.columns]
    out = pd.DataFrame({"mean_corr_13w": mc, "n_sell": (st[eq] == -1).sum(axis=1).reindex(mc.index),
                        "n_live": st[eq].notna().sum(axis=1).reindex(mc.index), "N100_state": st[C.BENCHMARK].reindex(mc.index)})
    return pd.concat([out, pair_corrs(r).reindex(mc.index)], axis=1)
