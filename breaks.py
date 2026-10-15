"""
3. WHICH ASSET BREAKS FIRST.

A sell-off ("episode") = N100 falls DD_THRESHOLD (10%) from its running high, on
weekly closes. The episode's trough is N100's lowest close before it regains the
high (or the last bar). For every episode and every asset, four dates are found:

  peak        the asset's own highest close in the window from 8 weeks before N100's peak to the trough
  signal      the first week after that peak when the asset's supertrend flipped to SELL
              (searched up to 4 weeks past the trough: a "break" has to happen in the fall)
  cross       the first week the asset closed DD_CROSS (10%) below its own running high
              (same window)
  recover     the first week after the trough when the asset's supertrend flipped to BUY

Each date is turned into a LEAD in weeks relative to N100's own date (negative =
earlier than N100). Then, across episodes:

  order          who came first on each measure, episode by episode
  summary        per asset: median lead vs N100, share of episodes it was first,
                 average rank; a one-word verdict
  depth          each asset's own fall from its peak to the episode trough, so "breaks
                 first" and "falls most" are kept apart

Gold is included: it tells you whether gold's trend turns with equities or not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C


def episodes(px_w: pd.DataFrame, bench: str = C.BENCHMARK) -> pd.DataFrame:
    p = px_w[bench].dropna()
    hi = p.cummax()
    dd = p / hi - 1
    rows = []
    in_ep, start_i = False, None
    for i in range(len(p)):
        if not in_ep and dd.iloc[i] <= -C.DD_THRESHOLD:
            in_ep = True
            # the running high this drawdown is measured from: the LAST bar at that high before the fall
            start_i = int(np.where(p.values[:i + 1] == hi.iloc[i])[0][-1])
            trough_i = i
        elif in_ep:
            if dd.iloc[i] < dd.iloc[trough_i]:
                trough_i = i
            if dd.iloc[i] >= 0 or i == len(p) - 1:
                rows.append({"peak_date": p.index[start_i], "trough_date": p.index[trough_i], "end_date": p.index[i],
                             "depth": float(dd.iloc[trough_i]), "weeks_peak_to_trough": trough_i - start_i, "weeks_to_recover": i - trough_i})
                in_ep = False
    df = pd.DataFrame(rows)
    # merge episodes whose peaks are within DD_MIN_GAP weeks (same sell-off)
    keep = []
    for _, row in df.iterrows():
        if keep and (row["peak_date"] - keep[-1]["end_date"]).days / 7 < C.DD_MIN_GAP:
            if row["depth"] < keep[-1]["depth"]:
                keep[-1]["trough_date"], keep[-1]["depth"] = row["trough_date"], row["depth"]
            keep[-1]["end_date"] = row["end_date"]
            continue
        keep.append(row.to_dict())
    out = pd.DataFrame(keep)
    out.insert(0, "episode", [f"E{i + 1} {d.strftime('%Y-%m')}" for i, d in enumerate(out["peak_date"])])
    return out


def _first_after(series_bool: pd.Series, start, end=None):
    s = series_bool.loc[start:end] if end is not None else series_bool.loc[start:]
    hits = s[s]
    return hits.index[0] if len(hits) else pd.NaT


def dates_per_asset(ep: pd.DataFrame, px_w: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, e in ep.iterrows():
        t = e["trough_date"]
        i_peak = px_w.index.get_loc(e["peak_date"])
        look_from = px_w.index[max(0, i_peak - C.DD_PEAK_LOOKBACK)]
        fall_end = px_w.index[min(len(px_w) - 1, px_w.index.get_loc(t) + C.DD_FALL_GRACE)]     # the break must happen in the fall
        for a in [x for x in C.ASSETS if x in px_w.columns]:
            p = px_w[a].loc[look_from:t].dropna()
            if len(p) < 4:
                continue
            peak = p.idxmax()
            full = px_w[a].loc[look_from:fall_end].dropna()
            own_dd = full / full.cummax() - 1
            cross = _first_after(own_dd <= -C.DD_CROSS, peak, fall_end)
            s = st[a] if a in st.columns else pd.Series(dtype=float)
            flip_sell = _first_after(((s == -1) & (s.shift(1) == 1)), peak, fall_end) if len(s) else pd.NaT
            rec_end = px_w.index[min(len(px_w) - 1, px_w.index.get_loc(t) + C.RECOVERY_WINDOW)]
            flip_buy = _first_after(((s == 1) & (s.shift(1) == -1)), t, rec_end) if len(s) else pd.NaT
            fall = float(px_w[a].loc[t] / p.loc[peak] - 1) if pd.notna(px_w[a].loc[t]) else np.nan
            rows.append({"episode": e["episode"], "asset": a, "peak": peak, "signal_sell": flip_sell, "cross_10pct": cross, "recover_buy": flip_buy,
                         "own_fall_to_trough": fall})
    return pd.DataFrame(rows)


def leads(dp: pd.DataFrame, bench: str = C.BENCHMARK) -> pd.DataFrame:
    """Lead in weeks vs the benchmark's own date on each measure (negative = earlier)."""
    out = dp.copy()
    for col in ("peak", "signal_sell", "cross_10pct", "recover_buy"):
        ref = dp[dp["asset"] == bench].set_index("episode")[col]
        out[f"lead_{col}"] = [((row[col] - ref.get(row["episode"], pd.NaT)).days / 7) if pd.notna(row[col]) and pd.notna(ref.get(row["episode"], pd.NaT)) else np.nan
                              for _, row in dp.iterrows()]
    return out


def order_table(ld: pd.DataFrame, col: str) -> pd.DataFrame:
    """Episode x asset: rank on `col` (1 = first). Ties share the rank."""
    piv = ld.pivot(index="episode", columns="asset", values=col)
    return piv.rank(axis=1, method="min")


def summary(ld: pd.DataFrame) -> pd.DataFrame:
    rows = []
    assets = [a for a in C.ASSETS if a in set(ld["asset"])]
    for a in assets:
        sub = ld[ld["asset"] == a]
        row = {"asset": a, "episodes": int(sub["episode"].nunique())}
        for col in ("peak", "signal_sell", "cross_10pct", "recover_buy"):
            rk = order_table(ld, col)
            row[f"median_lead_{col}_weeks"] = float(sub[f"lead_{col}"].median())
            row[f"first_share_{col}"] = float((rk[a] == 1).mean()) if a in rk.columns else np.nan
            row[f"avg_rank_{col}"] = float(rk[a].mean()) if a in rk.columns else np.nan
            row[f"n_{col}"] = int(sub[col].notna().sum())
        row["median_own_fall"] = float(sub["own_fall_to_trough"].median())
        rows.append(row)
    df = pd.DataFrame(rows).set_index("asset")
    # verdict on the signal: the asset whose SELL flip has the lowest average rank
    eq = [a for a in C.EQUITY if a in df.index]
    first_sig = df.loc[eq, "avg_rank_signal_sell"].idxmin()
    first_cross = df.loc[eq, "avg_rank_cross_10pct"].idxmin()
    first_rec = df.loc[eq, "avg_rank_recover_buy"].idxmin()
    df["verdict"] = ""
    df.loc[first_sig, "verdict"] += "signal flips to SELL first; "
    df.loc[first_cross, "verdict"] += "crosses -10% first; "
    df.loc[first_rec, "verdict"] += "turns back to BUY first; "
    return df
