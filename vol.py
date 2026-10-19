"""
5. VOLATILITY with respect to the signal, and how it moves correlation.

All weekly. "Vol" = realised volatility: the standard deviation of the last
VOL_WINDOW (13) weekly log returns, x sqrt(52) so it reads as a yearly figure.

  rolling_vol        each asset's 13-week vol, week by week (the chart)
  by_state           each asset's vol in its OWN BUY weeks vs its own SELL weeks, and the ratio;
                     also its mean weekly return in each state (so "SELL = volatile" and
                     "SELL = falling" can be told apart)
  around_flip        event study: each index's vol (and the VIX level) at k = -8..+8 weeks
                     around EVERY asset's flips, one block per reference asset
  corr_by_regime     every pair's 13-week correlation grouped by the quintile of a volatility
                     regime in the same week, three ways: N100's 13-week vol, the pair's own
                     average 13-week vol, and the VIX level - the "does correlation rise when
                     vol rises" table, pair by pair; with a Spearman of the two series
  vol_corr_lead      does vol move before correlation or after? correlation between the
                     13-week CHANGE in N100's vol (and in the VIX) at t and the 13-week change
                     in the group correlation G at t+k, k = -4..+4
  by_state           ... also the mean VIX level in each asset's BUY and SELL weeks
  timeseries         the series behind the chart
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy import stats

import config as C
import corrbreak


def _pairs(cols) -> list[tuple[str, str]]:
    return list(itertools.combinations([c for c in C.ASSETS if c in cols], 2))


def rolling_vol(r: pd.DataFrame, w: int = C.VOL_WINDOW) -> pd.DataFrame:
    cols = [c for c in C.ASSETS if c in r.columns]
    return r[cols].rolling(w, min_periods=int(0.8 * w)).std() * np.sqrt(C.ANNUALISE)


def by_state(st: pd.DataFrame, r: pd.DataFrame, lvl: pd.Series | None = None) -> pd.DataFrame:
    rows = []
    for a in [c for c in C.ASSETS if c in r.columns and c in st.columns]:
        s = st[a].reindex(r.index)
        x = r[a]
        out = {"asset": a}
        for name, m in (("BUY", s == 1), ("SELL", s == -1)):
            xy = x[m].dropna()
            out[f"weeks_{name}"] = len(xy)
            out[f"vol_{name}"] = xy.std() * np.sqrt(C.ANNUALISE) if len(xy) >= C.MIN_CELL else np.nan
            out[f"mean_weekly_ret_{name}"] = xy.mean() if len(xy) >= C.MIN_CELL else np.nan
            out[f"share_weeks_down_{name}"] = float((xy < 0).mean()) if len(xy) >= C.MIN_CELL else np.nan
            out[f"worst_week_{name}"] = xy.min() if len(xy) else np.nan
            if lvl is not None and len(lvl):
                vv = lvl.reindex(r.index)[m].dropna()
                out[f"vix_mean_{name}"] = vv.mean() if len(vv) >= C.MIN_CELL else np.nan
                out[f"vix_median_{name}"] = vv.median() if len(vv) >= C.MIN_CELL else np.nan
        out["vol_all"] = x.dropna().std() * np.sqrt(C.ANNUALISE)
        out["vol_ratio_SELL_over_BUY"] = out["vol_SELL"] / out["vol_BUY"] if out.get("vol_BUY") else np.nan
        if lvl is not None and len(lvl):
            out["vix_ratio_SELL_over_BUY"] = out["vix_mean_SELL"] / out["vix_mean_BUY"] if out.get("vix_mean_BUY") else np.nan
        rows.append(out)
    return pd.DataFrame(rows).set_index("asset")


def state_points(st: pd.DataFrame, r: pd.DataFrame, lvl: pd.Series | None = None) -> pd.DataFrame:
    """The individual data points behind B7, one row per asset and week: the asset's own signal state, |weekly return| (the
    components of the realised vol), its 13-week realised vol, and the VIX level, its weekly change and |change|."""
    v = rolling_vol(r)
    rows = []
    for a in [c for c in C.ASSETS if c in r.columns and c in st.columns]:
        s = st[a].reindex(r.index)
        df = pd.DataFrame({"asset": a, "state": s.map({1: "BUY", -1: "SELL"}), "weekly_return": r[a], "abs_weekly_return": r[a].abs(), "vol_13w": v[a]})
        if lvl is not None and len(lvl):
            x = lvl.reindex(r.index)
            df["vix"] = x; df["vix_weekly_change"] = x.diff(); df["abs_vix_weekly_change"] = x.diff().abs()
        rows.append(df.dropna(subset=["state", "weekly_return"]))
    out = pd.concat(rows)
    out.index.name = "date"
    return out.reset_index()


def state_dist(pts: pd.DataFrame) -> pd.DataFrame:
    """B7 (review of 2026-10-19): the distribution, not just the average — per asset and own-signal state, for each measure: n, min, p25,
    median, mean, p75, max, sd. Measures: |weekly return|, the 13-week realised vol, the VIX level, its weekly change and |change|."""
    measures = [m for m in ("abs_weekly_return", "vol_13w", "vix", "vix_weekly_change", "abs_vix_weekly_change") if m in pts.columns]
    rows = []
    for (a, s), g in pts.groupby(["asset", "state"]):
        for m in measures:
            x = g[m].dropna()
            if len(x) < C.MIN_CELL:
                continue
            rows.append({"asset": a, "state": s, "measure": m, "n": int(len(x)), "min": float(x.min()), "p25": float(x.quantile(0.25)), "median": float(x.median()), "mean": float(x.mean()),
                         "p75": float(x.quantile(0.75)), "max": float(x.max()), "sd": float(x.std())})
    out = pd.DataFrame(rows)
    order = {a: i for i, a in enumerate(C.ASSETS)}
    return out.sort_values(["measure", "asset", "state"], key=lambda c: c.map(order) if c.name == "asset" else c).reset_index(drop=True)


def vix_surges(lvl: pd.Series, idx: pd.Index, pct: float = C.VIX_SURGE, weeks: int = C.VIX_SURGE_WEEKS) -> pd.DataFrame:
    """A VIX SURGE = the first week the VIX level is up at least `pct` over the last `weeks` weeks (counted once per run:
    the next surge needs the 4-week change to have dropped below the line first). Columns: date, vix, vix_4w_ago, change."""
    x = lvl.reindex(idx)
    chg = x / x.shift(weeks) - 1
    hit = (chg >= pct).fillna(False)
    starts = idx[hit & ~hit.shift(1, fill_value=False)]
    return pd.DataFrame({"date": starts, "vix": x.loc[starts].values, f"vix_{weeks}w_ago": x.shift(weeks).loc[starts].values, f"change_{weeks}w": chg.loc[starts].values})


def vix_vs_flips(st: pd.DataFrame, r: pd.DataFrame, lvl: pd.Series, window: int = C.VIX_FLIP_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """B7 (review of 2026-10-19): is the VIX a leading or a lagging indicator of the supertrend's SELL flip? Both directions:
    (a) per SELL flip of each asset: the nearest VIX surge before it (weeks) and after it (weeks), within `window`;
        the VIX's change over the 4 weeks before the flip and over the 4 after;
    (b) per VIX surge: for each index, whether its signal was already in SELL at the surge, the weeks since its last SELL flip
        and until its next one (within `window`);
    (c) the summary per asset: n flips / surges, share of flips with a surge in the `window` weeks BEFORE (the VIX led), share with
        one AFTER (the VIX lagged), median lead; share of surges with the asset already in SELL, share followed by its SELL flip;
    (d) the paths: the VIX level as a ratio to the flip-week level at k = −8 … +8 around SELL flips (mean and median) and the
        share of index signals flipping to SELL at k = −8 … +8 around surges."""
    idx = r.index
    x = lvl.reindex(idx)
    sg = vix_surges(lvl, idx)
    sdates = list(sg["date"])
    spos = {d: idx.get_loc(d) for d in sdates}
    assets = [a for a in C.EQUITY if a in st.columns]
    flips_rows, surge_rows, summ, path_rows = [], [], [], []
    ratio_paths = {k: [] for k in range(-window, window + 1)}
    for a in assets:
        s = st[a].reindex(idx)
        fl = idx[((s == -1) & (s.shift(1) == 1)).fillna(False)]
        fpos = [idx.get_loc(d) for d in fl]
        for d, i in zip(fl, fpos):
            before = [i - spos[e] for e in sdates if 0 < i - spos[e] <= window]; at = any(spos[e] == i for e in sdates)
            after = [spos[e] - i for e in sdates if 0 < spos[e] - i <= window]
            vx0 = x.iloc[i]
            row = {"asset": a, "flip_date": d.date(), "vix_at_flip": vx0, "vix_change_4w_before": (vx0 / x.iloc[i - 4] - 1) if i >= 4 and pd.notna(x.iloc[i - 4]) and pd.notna(vx0) else np.nan,
                   "vix_change_4w_after": (x.iloc[i + 4] / vx0 - 1) if i + 4 < len(idx) and pd.notna(x.iloc[i + 4]) and pd.notna(vx0) else np.nan,
                   "surge_in_flip_week": bool(at), "weeks_since_last_surge": min(before) if before else np.nan, "weeks_to_next_surge": min(after) if after else np.nan, "vix_data": bool(pd.notna(vx0))}
            flips_rows.append(row)
            for k in range(-window, window + 1):
                if 0 <= i + k < len(idx) and pd.notna(x.iloc[i + k]) and pd.notna(vx0) and vx0 > 0:
                    ratio_paths[k].append(x.iloc[i + k] / vx0)
        for d in sdates:
            i = spos[d]
            in_sell = bool(s.iloc[i] == -1) if pd.notna(s.iloc[i]) else np.nan
            since = [i - j for j in fpos if 0 <= i - j <= window]; until = [j - i for j in fpos if 0 < j - i <= window]
            surge_rows.append({"surge_date": d.date(), "asset": a, "vix": x.iloc[i], "already_in_SELL": in_sell, "weeks_since_last_SELL_flip": min(since) if since else np.nan, "weeks_to_next_SELL_flip": min(until) if until else np.nan})
        fr = pd.DataFrame([q for q in flips_rows if q["asset"] == a and q["vix_data"]]); sr = pd.DataFrame([q for q in surge_rows if q["asset"] == a])
        summ.append({"asset": a, "sell_flips_with_vix_data": int(len(fr)), "surges": int(len(sr)),
                     "share_flips_with_surge_before": float(fr["weeks_since_last_surge"].notna().mean()) if len(fr) else np.nan,
                     "share_flips_with_surge_in_week": float(fr["surge_in_flip_week"].mean()) if len(fr) else np.nan,
                     "share_flips_with_surge_after": float(fr["weeks_to_next_surge"].notna().mean()) if len(fr) else np.nan,
                     "median_weeks_surge_before_flip": float(fr["weeks_since_last_surge"].median()) if len(fr) and fr["weeks_since_last_surge"].notna().any() else np.nan,
                     "median_weeks_surge_after_flip": float(fr["weeks_to_next_surge"].median()) if len(fr) and fr["weeks_to_next_surge"].notna().any() else np.nan,
                     "mean_vix_change_4w_before_flip": float(fr["vix_change_4w_before"].mean()) if len(fr) else np.nan, "mean_vix_change_4w_after_flip": float(fr["vix_change_4w_after"].mean()) if len(fr) else np.nan,
                     "median_vix_change_4w_before_flip": float(fr["vix_change_4w_before"].median()) if len(fr) else np.nan,
                     "share_surges_already_in_SELL": float(sr["already_in_SELL"].mean()) if len(sr) else np.nan,
                     "share_surges_with_flip_before": float(sr["weeks_since_last_SELL_flip"].notna().mean()) if len(sr) else np.nan,
                     "share_surges_followed_by_flip": float(sr["weeks_to_next_SELL_flip"].notna().mean()) if len(sr) else np.nan,
                     "median_weeks_surge_to_flip": float(sr["weeks_to_next_SELL_flip"].median()) if len(sr) and sr["weeks_to_next_SELL_flip"].notna().any() else np.nan})
    for k in range(-window, window + 1):
        row = {"k": k, "vix_ratio_to_flip_week_mean": float(np.mean(ratio_paths[k])) if ratio_paths[k] else np.nan, "vix_ratio_to_flip_week_median": float(np.median(ratio_paths[k])) if ratio_paths[k] else np.nan, "n_flips": len(ratio_paths[k])}
        # around the surges: the share of index signals that flipped to SELL at k
        n, hits = 0, 0
        for d in sdates:
            i = spos[d] + k
            if 0 <= i < len(idx):
                for a in assets:
                    s = st[a].reindex(idx)
                    if pd.notna(s.iloc[i]) and i > 0 and pd.notna(s.iloc[i - 1]):
                        n += 1; hits += int(s.iloc[i] == -1 and s.iloc[i - 1] == 1)
        row["share_signals_flipping_to_SELL"] = hits / n if n else np.nan; row["n_signal_weeks"] = n
        path_rows.append(row)
    return pd.DataFrame(flips_rows), pd.DataFrame(surge_rows), pd.DataFrame(summ).set_index("asset"), pd.DataFrame(path_rows)


def around_flip(st: pd.DataFrame, r: pd.DataFrame, lvl: pd.Series | None = None, refs: list[str] | None = None) -> pd.DataFrame:
    """Each asset's 13-week vol (and the VIX level) at k = -8..+8 around each flip of EVERY reference asset, averaged over flips."""
    v = rolling_vol(r)
    if lvl is not None and len(lvl):
        v["VIX"] = lvl.reindex(v.index)
    refs = refs or [a for a in C.ASSETS if a in st.columns and a in r.columns]
    rows = []
    for ref in refs:
        s = st[ref].reindex(v.index)
        for direction, val in (("flip to SELL", -1), ("flip to BUY", 1)):
            flips = v.index[((s == val) & (s.shift(1) == -val)).fillna(False)]
            for k in range(-C.FLIP_WINDOW, C.FLIP_WINDOW + 1):
                row = {"ref": ref, "event": direction, "weeks_from_flip": k}
                for a in v.columns:
                    vals = [v[a].iloc[v.index.get_loc(d) + k] for d in flips if 0 <= v.index.get_loc(d) + k < len(v)]
                    vals = [x for x in vals if pd.notna(x)]
                    row[f"vol_{a}" if a != "VIX" else "vix_level"] = float(np.mean(vals)) if vals else np.nan
                    if a == ref:
                        row["n_events"] = len(vals)
                rows.append(row)
    return pd.DataFrame(rows)


def _pair_corrs(r: pd.DataFrame) -> dict:
    return {(a, b): r[a].rolling(C.CORR_WINDOW, min_periods=int(0.8 * C.CORR_WINDOW)).corr(r[b]) for a, b in _pairs(r.columns)}   # all ten pairs, gold included


def _by_regime(pcs: dict, regime: pd.Series, name: str, own: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Each pair's 13-week correlation grouped by the quintile of `regime` (or, with `own`, of the pair's own series)."""
    labels = [f"Q{i + 1}" for i in range(C.VOL_BUCKETS)]
    rows, long = [], []
    for (a, b), c in pcs.items():
        reg = own[(a, b)] if own is not None else regime
        ok = reg.dropna()
        if len(ok) < 5 * C.VOL_BUCKETS:
            continue
        q = pd.qcut(ok, C.VOL_BUCKETS, labels=labels)
        edges = ok.groupby(q, observed=True).agg(["min", "max"])
        df = pd.DataFrame({"corr": c, "bucket": q}).dropna()
        g = df.groupby("bucket", observed=True)["corr"]
        row = {"pair": f"{a}|{b}"}
        for bkt in labels:
            row[bkt] = g.mean().get(bkt, np.nan)
            long.append({"regime": name, "pair": f"{a}|{b}", "bucket": bkt, "lo": edges.loc[bkt, "min"] if bkt in edges.index else np.nan,
                         "hi": edges.loc[bkt, "max"] if bkt in edges.index else np.nan, "mean_corr": g.mean().get(bkt, np.nan), "weeks": int(g.count().get(bkt, 0))})
        row["Q5_minus_Q1"] = row[labels[-1]] - row[labels[0]]
        both = pd.DataFrame({"corr": c, "v": reg}).dropna()
        row["spearman_corr_vs_regime"] = stats.spearmanr(both["corr"], both["v"]).statistic if len(both) > 30 else np.nan
        row["weeks"] = len(both)
        rows.append(row)
    wide = pd.DataFrame(rows).set_index("pair") if rows else pd.DataFrame()
    return wide, pd.DataFrame(long)


def corr_by_vol_regime(r: pd.DataFrame, ref: str = C.BENCHMARK) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every pair's 13-week correlation by the quintile of `ref`'s 13-week vol that week (kept for the N100 table)."""
    v = rolling_vol(r)[ref]
    wide, lg = _by_regime(_pair_corrs(r), v, f"{ref} vol")
    ok = v.dropna()
    q = pd.qcut(ok, C.VOL_BUCKETS, labels=[f"Q{i + 1}" for i in range(C.VOL_BUCKETS)])
    wide.attrs["edges"] = ok.groupby(q, observed=True).agg(["min", "max", "count"])
    return wide, lg


def atr_pct(panel: dict, labels: list[str] | None = None, freq: str = C.SIGNAL_BAR) -> pd.DataFrame:
    """The desk's ATR (5-bar linear-weighted true range on weekly OHLC — the same ATR the supertrend uses) as a share of the
    close, per asset, on the weekly grid. A second volatility measure beside the stdev of weekly returns: it uses the week's
    high-low range, not only the close-to-close move."""
    import supertrend as ST
    labels = labels or [a for a in C.ASSETS if a in panel]
    out = {}
    for lab in labels:
        if lab in panel and {"high", "low", "close"} <= set(panel[lab].columns):
            bars = ST.resample_ohlc(panel[lab], freq)
            out[lab] = (ST.atr(bars) / bars["close"]).rename(lab)
    df = pd.DataFrame(out)
    return df[df.index >= pd.Timestamp(C.START)]


def corr_by_regime(r: pd.DataFrame, lvl: pd.Series | None = None, ref: str = C.BENCHMARK, atrp: pd.DataFrame | None = None) -> tuple[dict, pd.DataFrame]:
    """The regime definitions: the benchmark's vol, the pair's own average vol, the VIX level (if present); then each leg's own vol
    separately (the two components of the pair's vol), and — if `atrp` is given — the benchmark's ATR% and the pair's own ATR%."""
    pcs = _pair_corrs(r)
    v = rolling_vol(r)
    out, longs = {}, []
    w, lg = _by_regime(pcs, v[ref], f"{ref} vol"); out[f"{ref} vol"] = w; longs.append(lg)
    own = {(a, b): (v[a] + v[b]) / 2 for (a, b) in pcs}
    w, lg = _by_regime(pcs, None, "pair's own vol", own=own); out["pair's own vol"] = w; longs.append(lg)
    if lvl is not None and len(lvl.dropna()):
        w, lg = _by_regime(pcs, lvl.reindex(r.index), "VIX level"); out["VIX level"] = w; longs.append(lg)
    w, lg = _by_regime(pcs, None, "first leg's vol", own={(a, b): v[a] for (a, b) in pcs}); out["first leg's vol"] = w; longs.append(lg)
    w, lg = _by_regime(pcs, None, "second leg's vol", own={(a, b): v[b] for (a, b) in pcs}); out["second leg's vol"] = w; longs.append(lg)
    if atrp is not None and ref in atrp.columns:
        ap = atrp.reindex(r.index)
        w, lg = _by_regime(pcs, ap[ref], f"{ref} ATR%"); out[f"{ref} ATR%"] = w; longs.append(lg)
        own_atr = {(a, b): (ap[a] + ap[b]) / 2 for (a, b) in pcs if a in ap.columns and b in ap.columns}
        w, lg = _by_regime({k: v_ for k, v_ in pcs.items() if k in own_atr}, None, "pair's own ATR%", own=own_atr); out["pair's own ATR%"] = w; longs.append(lg)
    return out, pd.concat(longs, ignore_index=True)


def vol_corr_lead(r: pd.DataFrame, lvl: pd.Series | None = None, ref: str = C.BENCHMARK, lags: int = C.VOL_LEAD_LAGS) -> pd.DataFrame:
    """corr( change in vol at t , change in G at t+k ). k > 0: vol moves first; k < 0: correlation moves first.
    Two vol measures: realised (the benchmark's 13-week vol) and, if present, the VIX level (its 13-week log change)."""
    v = rolling_vol(r)[ref]
    G = corrbreak.series(r)["G"]
    dv = v.diff(C.VOL_WINDOW); dG = G.diff(C.CORR_WINDOW)
    dx = np.log(lvl.reindex(r.index)).diff(C.VOL_WINDOW) if lvl is not None and len(lvl.dropna()) else None
    rows = []
    for k in range(-lags, lags + 1):
        both = pd.DataFrame({"dv": dv, "dG": dG.shift(-k)}).dropna()
        row = {"k_weeks": k, "corr_dvol_t_dcorr_t_plus_k": both["dv"].corr(both["dG"]), "weeks": len(both)}
        if dx is not None:
            bx = pd.DataFrame({"dx": dx, "dG": dG.shift(-k)}).dropna()
            row["corr_dvix_t_dcorr_t_plus_k"] = bx["dx"].corr(bx["dG"]); row["weeks_vix"] = len(bx)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("k_weeks")
    lv = pd.DataFrame({"vol": v, "G": G}).dropna()
    out.attrs["level_corr"] = lv["vol"].corr(lv["G"])
    out.attrs["level_spearman"] = stats.spearmanr(lv["vol"], lv["G"]).statistic
    if dx is not None:
        lx = pd.DataFrame({"vix": lvl.reindex(r.index), "G": G}).dropna()
        out.attrs["vix_level_corr"] = lx["vix"].corr(lx["G"])
        out.attrs["vix_level_spearman"] = stats.spearmanr(lx["vix"], lx["G"]).statistic
        vx = pd.DataFrame({"vix": lvl.reindex(r.index), "vol": v}).dropna()
        out.attrs["vix_vs_realised_corr"] = vx["vix"].corr(vx["vol"])
    return out


def vol_corr_lead_pairs(r: pd.DataFrame, lvl: pd.Series | None = None, lags: int = C.VOL_LEAD_LAGS) -> pd.DataFrame:
    """The lead/lag study pair by pair: corr( change in the pair's own vol at t , change in the pair's correlation at t+k ),
    the pair's own vol = the average of its two legs' 13-week vol; and the same against the VIX. Long: pair, k_weeks, ..."""
    v = rolling_vol(r)
    pcs = _pair_corrs(r)
    dx = np.log(lvl.reindex(r.index)).diff(C.VOL_WINDOW) if lvl is not None and len(lvl.dropna()) else None
    rows = []
    for (a, b), c in pcs.items():
        dv = ((v[a] + v[b]) / 2).diff(C.VOL_WINDOW); dc = c.diff(C.CORR_WINDOW)
        for k in range(-lags, lags + 1):
            both = pd.DataFrame({"dv": dv, "dc": dc.shift(-k)}).dropna()
            row = {"pair": f"{a}|{b}", "k_weeks": k, "corr_dvol_t_dcorr_t_plus_k": both["dv"].corr(both["dc"]) if len(both) > 30 else np.nan, "weeks": len(both)}
            if dx is not None:
                bx = pd.DataFrame({"dx": dx, "dc": dc.shift(-k)}).dropna()
                row["corr_dvix_t_dcorr_t_plus_k"] = bx["dx"].corr(bx["dc"]) if len(bx) > 30 else np.nan; row["weeks_vix"] = len(bx)
            rows.append(row)
    return pd.DataFrame(rows)


def _block_stats(x: pd.Series, y: pd.Series, lag: int, min_n: int = C.LL_MIN_N) -> dict:
    """Pearson, Spearman, a Newey-West t on the slope of y on x (the weekly-stepped blocks overlap, so HAC with `lag` lags),
    and the plain Pearson on the NON-overlapping sub-sample (every `lag`-th week)."""
    import statsmodels.api as sm
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    out = {"n": int(len(d)), "pearson": np.nan, "spearman": np.nan, "nw_t": np.nan, "pearson_nonoverlap": np.nan, "n_nonoverlap": 0}
    if len(d) < min_n:
        return out
    out["pearson"] = float(d["x"].corr(d["y"])); out["spearman"] = float(stats.spearmanr(d["x"], d["y"]).statistic)
    res = sm.OLS(d["y"].values, sm.add_constant(d["x"].values)).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    out["nw_t"] = float(res.tvalues[1])
    no = d.iloc[::lag]
    out["n_nonoverlap"] = int(len(no)); out["pearson_nonoverlap"] = float(no["x"].corr(no["y"])) if len(no) >= min_n else np.nan
    return out


def lead_lag_blocks(r: pd.DataFrame, lvl: pd.Series | None = None, blocks: tuple = C.LL_BLOCKS) -> pd.DataFrame:
    """B9 redone (review of 2026-10-19) with NO overlapping weeks. For a block length L, the block B0 = weeks t−L+1 … t and the
    block B1 = weeks t+1 … t+L share no week. On each block: the pair's correlation (over the block's L weeks), the pair's own
    realised vol (the average of the two legs' stdev over the block, annualised), and the VIX change (log VIX at the block's
    end minus at its start). Then, for every pair, both directions:
        correlation → volatility : X = the correlation in B0,  Y = the own vol in B1 / the VIX change over B1
        volatility → correlation : X = the own vol in B0 / the VIX change over B0,  Y = the correlation in B1
    as LEVELS (the block values) and as CHANGES (B0 minus the block before it, B1 minus B0). Each with Pearson, Spearman, a
    Newey-West t (the blocks step weekly, so they overlap with their neighbours — HAC with L lags), and the Pearson on the
    strictly non-overlapping sub-sample (every L-th week). Long: pair, block_weeks, direction, measure, kind, …"""
    rows = []
    has_vix = lvl is not None and len(lvl.dropna())
    lx = np.log(lvl.reindex(r.index)) if has_vix else None
    for L in blocks:
        for a, b in _pairs(r.columns):
            c0 = r[a].rolling(L, min_periods=L).corr(r[b])                         # correlation over t−L+1 … t  (B0)
            v0 = (r[a].rolling(L, min_periods=L).std() + r[b].rolling(L, min_periods=L).std()) / 2 * np.sqrt(C.ANNUALISE)
            c1, v1 = c0.shift(-L), v0.shift(-L)                                    # the same over t+1 … t+L  (B1)
            cm1, vm1 = c0.shift(L), v0.shift(L)                                    # the block before B0
            x0 = (lx - lx.shift(L)) if has_vix else None                           # VIX change over B0 (end minus start)
            x1 = x0.shift(-L) if has_vix else None; xm1 = x0.shift(L) if has_vix else None
            specs = [("correlation → own vol", "own vol", "level", c0, v1), ("own vol → correlation", "own vol", "level", v0, c1),
                     ("correlation → own vol", "own vol", "change", c0 - cm1, v1 - v0), ("own vol → correlation", "own vol", "change", v0 - vm1, c1 - c0)]
            if has_vix:
                specs += [("correlation → VIX", "VIX", "level", c0, x1), ("VIX → correlation", "VIX", "level", x0, c1),
                          ("correlation → VIX", "VIX", "change", c0 - cm1, x1), ("VIX → correlation", "VIX", "change", x0, c1 - c0)]
            for direction, measure, kind, X, Y in specs:
                row = {"pair": f"{a}|{b}", "block_weeks": L, "direction": direction, "measure": measure, "kind": kind}
                row.update(_block_stats(X, Y, L))
                rows.append(row)
    return pd.DataFrame(rows)


def lead_lag_verdicts(ll: pd.DataFrame) -> pd.DataFrame:
    """Per pair and block length, from the CHANGE rows: the two directions side by side for each measure, which is the larger,
    and whether either is reliable (|NW t| ≥ 2 and the same sign on the non-overlapping sub-sample)."""
    rows = []
    ch = ll[ll["kind"] == "change"]
    for (p, L), g in ch.groupby(["pair", "block_weeks"]):
        row = {"pair": p, "block_weeks": L}
        for m, fwd, back in (("own vol", "correlation → own vol", "own vol → correlation"), ("VIX", "correlation → VIX", "VIX → correlation")):
            f = g[g["direction"] == fwd]; bk = g[g["direction"] == back]
            if not len(f) or not len(bk):
                continue
            f, bk = f.iloc[0], bk.iloc[0]
            key = m.replace(" ", "_")
            row[f"corr_leads_{key}"] = f["pearson"]; row[f"corr_leads_{key}_t"] = f["nw_t"]; row[f"{key}_leads_corr"] = bk["pearson"]; row[f"{key}_leads_corr_t"] = bk["nw_t"]
            row[f"{key}_n"] = f["n"]
            rf = abs(f["nw_t"]) >= 2 and np.sign(f["pearson"]) == np.sign(f["pearson_nonoverlap"]) if pd.notna(f["pearson_nonoverlap"]) else False
            rb = abs(bk["nw_t"]) >= 2 and np.sign(bk["pearson"]) == np.sign(bk["pearson_nonoverlap"]) if pd.notna(bk["pearson_nonoverlap"]) else False
            if rf and not rb:
                row[f"{key}_verdict"] = "correlation moves first"
            elif rb and not rf:
                row[f"{key}_verdict"] = f"{m} moves first"
            elif rf and rb:
                row[f"{key}_verdict"] = "both directions (feedback)"
            else:
                row[f"{key}_verdict"] = "neither is reliable"
        rows.append(row)
    return pd.DataFrame(rows)


def regime_weekly(r: pd.DataFrame) -> pd.DataFrame:
    """B8 (review of 2026-10-19): the components behind the regime averages — week by week, for every pair: its 13-week correlation, its own
    vol (the average of the two legs' 13-week vol), and which fifth of its own vol that week falls in (Q1 calm … Q5 wild).
    Wide: 'a|b' (corr), 'v_a|b' (own vol), 'q_a|b' (the fifth)."""
    pcs = _pair_corrs(r); v = rolling_vol(r)
    labels = [f"Q{i + 1}" for i in range(C.VOL_BUCKETS)]
    out = pd.DataFrame(index=r.index)
    for (a, b), c in pcs.items():
        own = (v[a] + v[b]) / 2
        out[f"{a}|{b}"] = c; out[f"v_{a}|{b}"] = own
        ok = own.dropna()
        if len(ok) >= 5 * C.VOL_BUCKETS:
            out[f"q_{a}|{b}"] = pd.qcut(ok, C.VOL_BUCKETS, labels=labels).reindex(r.index).astype(object)
    return out


def regime_stretches(rw: pd.DataFrame, min_weeks: int = C.REGIME_STRETCH_MIN, before: int = 8) -> pd.DataFrame:
    """Every stretch of at least `min_weeks` consecutive weeks in which the pair's own vol sat in its wildest fifth (Q5):
    when, how long, the pair's correlation in the `before` weeks before it, inside it, the change, and the correlation's peak.
    The 'components' of B8's Q5 average, one row per historical instance."""
    rows = []
    for p in [c for c in rw.columns if "|" in c and not c.startswith(("v_", "q_"))]:
        q = rw.get(f"q_{p}")
        if q is None:
            continue
        hot = (q == "Q5").fillna(False)
        on, start = False, None
        segs = []
        for d, h in hot.items():
            if h and not on:
                on, start = True, d
            elif not h and on:
                segs.append((start, hot.index[hot.index.get_loc(d) - 1])); on = False
        if on:
            segs.append((start, hot.index[-1]))
        for s0, s1 in segs:
            i0, i1 = rw.index.get_loc(s0), rw.index.get_loc(s1)
            if i1 - i0 + 1 < min_weeks:
                continue
            c = rw[p]
            inside = c.iloc[i0:i1 + 1]; pre = c.iloc[max(0, i0 - before):i0]
            rows.append({"pair": p, "start": s0.date(), "end": s1.date(), "weeks": int(i1 - i0 + 1), "own_vol_peak": float(rw[f"v_{p}"].iloc[i0:i1 + 1].max()),
                         "corr_before": float(pre.mean()) if pre.notna().any() else np.nan, "corr_inside": float(inside.mean()) if inside.notna().any() else np.nan,
                         "corr_peak_inside": float(inside.max()) if inside.notna().any() else np.nan, "change": float(inside.mean() - pre.mean()) if pre.notna().any() and inside.notna().any() else np.nan})
    return pd.DataFrame(rows)


def timeseries(st: pd.DataFrame, r: pd.DataFrame, lvl: pd.Series | None = None) -> pd.DataFrame:
    v = rolling_vol(r)
    v.columns = [f"vol_{c}" for c in v.columns]
    v["G"] = corrbreak.series(r)["G"]
    v["N100_state"] = st[C.BENCHMARK].reindex(v.index)
    if lvl is not None and len(lvl.dropna()):
        v["VIX"] = lvl.reindex(v.index)
    return v


# ----------------------------------------- the simple view: do they move together? --
def _z(s: pd.Series, n: int = C.CB_NORM) -> pd.Series:
    return (s - s.rolling(n, min_periods=int(n / 2)).mean()) / s.rolling(n, min_periods=int(n / 2)).std()


def together(r: pd.DataFrame, lvl: pd.Series | None = None, ref: str = C.BENCHMARK) -> pd.DataFrame:
    """The three series on one scale: the group correlation G, the benchmark's 13-week realised vol, and the
    VIX (log level), each as a z-score against its own 104-week history. For the overlay chart."""
    G = corrbreak.series(r)["G"]
    out = pd.DataFrame({"z_corr": _z(G), "z_vol": _z(rolling_vol(r)[ref]), "corr": G, "vol": rolling_vol(r)[ref]})
    if lvl is not None and len(lvl.dropna()):
        x = lvl.reindex(r.index)
        out["z_vix"] = _z(np.log(x)); out["vix"] = x
    return out


def corr_rise_events(tg: pd.DataFrame, thr: float = 1.0, reset: float = 0.5, min_gap: int = 8) -> list:
    """Correlation JUMPS: the weeks the group correlation's z first reaches +1 (having been below +0.5), at least min_gap weeks apart."""
    ev, armed, last = [], True, None
    for d, val in tg["z_corr"].items():
        if pd.isna(val):
            continue
        if armed and val >= thr:
            if last is None or (d - last).days > min_gap * 7:
                ev.append(d)
            armed, last = False, d
        elif val < reset:
            armed = True
    return ev


def corr_rise_paths(tg: pd.DataFrame, weeks: int = C.FLIP_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Around each correlation jump: the three z-scores from `weeks` before to `weeks` after, averaged; and per event."""
    ev = corr_rise_events(tg)
    cols = [c for c in ("z_corr", "z_vol", "z_vix") if c in tg.columns]
    rows, per = [], []
    for d in ev:
        i = tg.index.get_loc(d)
        row = {"event": d.date(), "corr_at_event": tg["corr"].iloc[i]}
        for c in cols:
            for k in (-8, -4, -2, 0, 2, 4, 8):
                row[f"{c}_{k:+d}"] = tg[c].iloc[i + k] if 0 <= i + k < len(tg) else np.nan
            # the first week in [-8, +4] the series reached its own +1
            seg = tg[c].iloc[max(0, i - weeks):i + 5]
            hit = seg[seg >= 1.0]
            row[f"{c}_first_week_above_1"] = (tg.index.get_loc(hit.index[0]) - i) if len(hit) else np.nan
        per.append(row)
        for k in range(-weeks, weeks + 1):
            if 0 <= i + k < len(tg):
                for c in cols:
                    rows.append({"event": d.date(), "k": k, "series": c, "z": tg[c].iloc[i + k]})
    long = pd.DataFrame(rows)
    paths = long.groupby(["series", "k"])["z"].agg(mean_z="mean", median_z="median", n="count").reset_index() if len(long) else pd.DataFrame(columns=["series", "k", "mean_z", "median_z", "n"])
    return paths, pd.DataFrame(per)


def together_pairs(r: pd.DataFrame, lvl: pd.Series | None = None) -> pd.DataFrame:
    """Every pair's 13-week correlation and its z-score against its own 104-week history, week by week — the pair-level
    companion of `together` (same norm as G's z there) — plus the pair's OWN realised volatility (the average of its two legs'
    13-week vol) and its z, and the VIX (log level) z. Columns: 'a|b' (corr), 'z_a|b' (its z), 'v_a|b' (vol), 'zv_a|b' (its z), 'vix', 'z_vix'."""
    S = corrbreak.series(r)
    ps = [c for c in S.columns if "|" in c and not c.startswith("z_")]
    out = S[ps + [f"z_{p}" for p in ps]].copy()
    v = rolling_vol(r)
    for p in ps:
        a, b = p.split("|")
        out[f"v_{p}"] = (v[a] + v[b]) / 2
        out[f"zv_{p}"] = _z(out[f"v_{p}"])
    if lvl is not None and len(lvl.dropna()):
        x = lvl.reindex(r.index)
        out["vix"] = x; out["z_vix"] = _z(np.log(x))
    return out


def corr_rise_paths_own(tp: pd.DataFrame, weeks: int = C.FLIP_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The 'which moves first' event study done PER PAIR on the pair's own series: the jumps are the weeks the PAIR's correlation z
    first reaches +1 (having been below +0.5, at least 8 weeks apart), and around them the pair's correlation z, the pair's own
    realised-vol z and the VIX z are averaged from `weeks` before to `weeks` after. Long: pair, series, k, mean_z, median_z, n.
    Per pair: the number of jumps, each series' z four weeks before and at the jump, and the share of jumps in which vol / the VIX
    was already above +1 before week 0 (a warning)."""
    ps = [c for c in tp.columns if "|" in c and not c.startswith(("z_", "v_", "zv_"))]
    has_vix = "z_vix" in tp.columns
    rows, per = [], []
    for p in ps:
        zc, zv = tp[f"z_{p}"], tp[f"zv_{p}"]
        ev = corr_rise_events(pd.DataFrame({"z_corr": zc}))
        series = {"z_corr": zc, "z_vol": zv} | ({"z_vix": tp["z_vix"]} if has_vix else {})
        row = {"pair": p, "jumps": len(ev)}
        warn = {s_: [] for s_ in ("z_vol", "z_vix")}
        at = {(s_, k): [] for s_ in series for k in (-4, 0)}
        for d in ev:
            i = zc.index.get_loc(d)
            for s_, z in series.items():
                for k in range(-weeks, weeks + 1):
                    if 0 <= i + k < len(z):
                        rows.append({"pair": p, "event": d.date(), "k": k, "series": s_, "z": z.iloc[i + k]})
                for k in (-4, 0):
                    if 0 <= i + k < len(z) and pd.notna(z.iloc[i + k]):
                        at[(s_, k)].append(z.iloc[i + k])
                if s_ in warn:
                    seg = z.iloc[max(0, i - weeks):i]            # strictly before the jump week
                    if seg.notna().any():
                        warn[s_].append(bool((seg >= 1.0).any()))
        for s_ in series:
            row[f"{s_}_-4"] = float(np.mean(at[(s_, -4)])) if at[(s_, -4)] else np.nan
            row[f"{s_}_+0"] = float(np.mean(at[(s_, 0)])) if at[(s_, 0)] else np.nan
        for s_ in warn:
            if s_ in series:
                row[f"share_{s_}_above1_before"] = float(np.mean(warn[s_])) if warn[s_] else np.nan
        per.append(row)
    long = pd.DataFrame(rows)
    paths = long.groupby(["pair", "series", "k"])["z"].agg(mean_z="mean", median_z="median", n="count").reset_index() if len(long) else pd.DataFrame(columns=["pair", "series", "k", "mean_z", "median_z", "n"])
    return paths, pd.DataFrame(per).set_index("pair")


def corr_rise_events_own(tp: pd.DataFrame) -> pd.DataFrame:
    """The weeks each PAIR's own correlation z first reached +1 (for the charts' markers): pair, event."""
    ps = [c for c in tp.columns if "|" in c and not c.startswith(("z_", "v_", "zv_"))]
    rows = [{"pair": p, "event": d.date()} for p in ps for d in corr_rise_events(pd.DataFrame({"z_corr": tp[f"z_{p}"]}))]
    return pd.DataFrame(rows, columns=["pair", "event"])


def corr_rise_paths_pairs(tg: pd.DataFrame, tp: pd.DataFrame, weeks: int = C.FLIP_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Around the same correlation jumps (the group's z reaching +1): each pair's z from `weeks` before to `weeks` after,
    averaged over the jumps; and per pair, the share of jumps in which the pair's own z also reached +1 within [-8, +4]."""
    ev = corr_rise_events(tg)
    ps = [c for c in tp.columns if "|" in c and not c.startswith(("z_", "v_", "zv_"))]
    rows, per = [], []
    for p in ps:
        z = tp[f"z_{p}"]
        hits, at0 = [], []
        for d in ev:
            if d not in z.index:
                continue
            i = z.index.get_loc(d)
            seg = z.iloc[max(0, i - weeks):i + 5]
            hits.append(bool((seg >= 1.0).any()) if seg.notna().any() else np.nan)
            at0.append(z.iloc[i])
            for k in range(-weeks, weeks + 1):
                if 0 <= i + k < len(z):
                    rows.append({"event": d.date(), "k": k, "series": p, "z": z.iloc[i + k]})
        h = pd.Series(hits, dtype=float)
        per.append({"pair": p, "jumps": int(h.notna().sum()), "share_pair_also_reached_plus1": float(h.mean()) if h.notna().any() else np.nan,
                    "mean_z_at_jump": float(pd.Series(at0, dtype=float).mean()) if at0 else np.nan})
    long = pd.DataFrame(rows)
    paths = long.groupby(["series", "k"])["z"].agg(mean_z="mean", median_z="median", n="count").reset_index() if len(long) else pd.DataFrame(columns=["series", "k", "mean_z", "median_z", "n"])
    return paths, pd.DataFrame(per).set_index("pair") if per else pd.DataFrame()
