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


def corr_by_regime(r: pd.DataFrame, lvl: pd.Series | None = None, ref: str = C.BENCHMARK) -> tuple[dict, pd.DataFrame]:
    """Three regime definitions: the benchmark's vol, the pair's own average vol, and the VIX level (if present)."""
    pcs = _pair_corrs(r)
    v = rolling_vol(r)
    out, longs = {}, []
    w, lg = _by_regime(pcs, v[ref], f"{ref} vol"); out[f"{ref} vol"] = w; longs.append(lg)
    own = {(a, b): (v[a] + v[b]) / 2 for (a, b) in pcs}
    w, lg = _by_regime(pcs, None, "pair's own vol", own=own); out["pair's own vol"] = w; longs.append(lg)
    if lvl is not None and len(lvl.dropna()):
        w, lg = _by_regime(pcs, lvl.reindex(r.index), "VIX level"); out["VIX level"] = w; longs.append(lg)
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
