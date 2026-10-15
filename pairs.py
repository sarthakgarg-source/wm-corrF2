"""
6. EACH PAIR SEPARATELY: the deviation (ratio z-score), what it did next, and whether
   the pair pulls back at all.

For every pair (a, b) of the five assets, a before b in config.ASSETS order:

  log ratio      lr_t = ln(P_a / P_b)   - how a has done against b, cumulatively
  deviation z    z_t = (lr_t - mean of lr over the last PAIR_WINDOW weeks) / sd over those weeks
                 z > 0 : a has run AHEAD of b relative to the last year ("a rich, b cheap")
                 z < 0 : a has LAGGED b                                 ("a cheap, b rich")
                 z_of(Z, x, y) gives the z of ln(P_x / P_y) for any order of the two.

  ratio_series   lr, its rolling mean and sd (for the band chart) and z, per pair
  stats_table    per pair: full-sample / 13w / 52w correlation, each leg's 13w vol, the vol
                 ratio, beta of a on b (52w), how often |z| is beyond 1 and beyond 2, and
                 the latest z with a plain reading
  coint_table    per pair: ADF test on the log ratio (does the ratio have a long-run level?),
                 Engle-Granger both ways, the half-life of the ratio (AR(1)), and a verdict
  z_forward      per pair and z bucket: the ratio's change over the next 4 and 13 weeks
                 (r_a - r_b summed over t+1..t+h), the share of weeks it moved back toward
                 0 (reverted), n, Newey-West t. The empirical "does a deviation pull back"
                 table - the one the pair-trading rule rests on. Also pooled over all pairs.
  path_after     after |z| first crosses 2, the average z over the next PAIR_PATH_WEEKS weeks,
                 sign-normalised so every event starts near +2: the shape of the pull-back
  current        the latest z per pair and who lagged whom
  returns_table  each leg's returns side by side: total / annualised / last 13w / last 52w, and
                 the average weekly return in both-BUY, both-SELL and disagree weeks, with the
                 spread a − b for each

Forward windows use t+1.. only (checks.py pins it).
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint

import config as C


def pairs(cols) -> list[tuple[str, str]]:
    return list(itertools.combinations([c for c in C.ASSETS if c in cols], 2))


def label(a: str, b: str) -> str:
    return f"{a}|{b}"


def forward(r: pd.Series, h: int) -> pd.Series:
    """Sum of the next h bars' log returns, aligned to the decision bar t (uses t+1..t+h only)."""
    return r.shift(-1).rolling(h).sum().shift(-(h - 1))


def hac_t(y: pd.Series, lags: int = C.HAC_LAGS, min_n: int = 20) -> float:
    y = y.dropna()
    if len(y) < min_n:
        return np.nan
    res = sm.OLS(y.values, np.ones((len(y), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


# ------------------------------------------------------------ the z-score --
def ratio_series(px_w: pd.DataFrame, w: int = C.PAIR_WINDOW, min_obs: int = C.PAIR_MIN_OBS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Z (wide, one column per pair) and BANDS (wide: '<pair>__lr', '__mean', '__sd')."""
    Z, B = {}, {}
    for a, b in pairs(px_w.columns):
        lr = (np.log(px_w[a]) - np.log(px_w[b])).rename(label(a, b))
        m = lr.rolling(w, min_periods=min_obs).mean()
        s = lr.rolling(w, min_periods=min_obs).std()
        Z[label(a, b)] = (lr - m) / s
        B[f"{label(a, b)}__lr"] = lr; B[f"{label(a, b)}__mean"] = m; B[f"{label(a, b)}__sd"] = s
    return pd.DataFrame(Z), pd.DataFrame(B)


def z_of(Z: pd.DataFrame, x: str, y: str) -> pd.Series:
    """z of ln(P_x / P_y): positive = x has run ahead of y over the trailing year."""
    if label(x, y) in Z.columns:
        return Z[label(x, y)]
    return -Z[label(y, x)]


def bucket(z: pd.Series, edges=C.PAIR_Z_EDGES) -> pd.Series:
    f = lambda v: f"{v:+.0f}".replace("-", "−")
    labels = [f"z < {f(edges[0])}", f"{f(edges[0])} ≤ z < {f(edges[1])}", f"{f(edges[1])} ≤ z < {f(edges[2])}", f"{f(edges[2])} ≤ z < {f(edges[3])}", f"z ≥ {f(edges[3])}"]
    return pd.cut(z, bins=[-np.inf, edges[0], edges[1], edges[2], edges[3], np.inf], labels=labels, right=False).rename("bucket")


BUCKET_ORDER = list(bucket(pd.Series([-3.0, -1.5, 0.0, 1.5, 3.0])).astype(str))


# ------------------------------------------------------------ per pair --
def _reading(z: float, a: str, b: str) -> str:
    if pd.isna(z):
        return "no reading yet"
    if abs(z) < 1:
        return "inside its normal range"
    who = f"{a} ahead of {b}" if z > 0 else f"{b} ahead of {a}"
    return f"{who} by {abs(z):.1f} sd" + (" — a large deviation" if abs(z) >= 2 else "")


def stats_table(r: pd.DataFrame, px_w: pd.DataFrame, st: pd.DataFrame, Z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    ann = C.ANNUALISE
    for a, b in pairs(r.columns):
        xy = r[[a, b]].dropna()
        z = Z[label(a, b)]
        c13 = xy[a].rolling(C.CORR_WINDOW, min_periods=10).corr(xy[b])
        c52 = xy[a].rolling(C.RP_CORR_WINDOW, min_periods=C.PAIR_MIN_OBS).corr(xy[b])
        v13 = xy.rolling(C.VOL_WINDOW, min_periods=10).std() * np.sqrt(ann)
        beta = xy[a].rolling(C.RP_CORR_WINDOW, min_periods=C.PAIR_MIN_OBS).cov(xy[b]) / xy[b].rolling(C.RP_CORR_WINDOW, min_periods=C.PAIR_MIN_OBS).var()
        zz = z.dropna()
        cross2 = ((zz.abs() >= 2) & (zz.abs().shift(1) < 2)).sum()
        rows.append({"pair": label(a, b), "a": a, "b": b, "weeks": len(xy), "first": xy.index[0].date(), "last": xy.index[-1].date(),
                     "corr_full": xy[a].corr(xy[b]), "corr_13w_now": c13.iloc[-1], "corr_52w_now": c52.iloc[-1],
                     "corr_13w_median": c13.median(), "corr_13w_min": c13.min(), "corr_13w_max": c13.max(),
                     "vol_a_full": xy[a].std() * np.sqrt(ann), "vol_b_full": xy[b].std() * np.sqrt(ann),
                     "vol_a_13w_now": v13[a].iloc[-1], "vol_b_13w_now": v13[b].iloc[-1], "vol_ratio_a_over_b_now": v13[a].iloc[-1] / v13[b].iloc[-1],
                     "beta_a_on_b_52w_now": beta.iloc[-1], "beta_a_on_b_full": xy[a].cov(xy[b]) / xy[b].var(),
                     "z_now": z.iloc[-1], "z_date": z.index[-1].date(), "reading_now": _reading(z.iloc[-1], a, b),
                     "share_weeks_abs_z_gt_1": float((zz.abs() > 1).mean()) if len(zz) else np.nan,
                     "share_weeks_abs_z_gt_2": float((zz.abs() > 2).mean()) if len(zz) else np.nan,
                     "n_2sd_deviations": int(cross2), "z_min": zz.min(), "z_max": zz.max(), "weeks_with_z": len(zz),
                     "ratio_total_change": float(np.exp(np.log(px_w[a] / px_w[b]).dropna().iloc[-1] - np.log(px_w[a] / px_w[b]).dropna().iloc[0]) - 1)})
    return pd.DataFrame(rows).set_index("pair")


def _halflife(y: pd.Series) -> float:
    y = y.dropna()
    dy = y.diff().dropna()
    x = y.shift(1).dropna().reindex(dy.index)
    X = sm.add_constant(x.values)
    beta = sm.OLS(dy.values, X).fit().params[1]
    return float(-np.log(2) / np.log(1 + beta)) if -1 < beta < 0 else np.inf


def coint_table(px_w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for a, b in pairs(px_w.columns):
        p = np.log(px_w[[a, b]]).dropna()
        lr = p[a] - p[b]
        try:
            adf_p = adfuller(lr.values, autolag="AIC", result_object=False)[1]
        except TypeError:
            adf_p = adfuller(lr.values, autolag="AIC")[1]
        eg_ab = coint(p[a].values, p[b].values, trend="c")[1]
        eg_ba = coint(p[b].values, p[a].values, trend="c")[1]
        hl = _halflife(lr)
        hl_w = _halflife(lr - lr.rolling(C.PAIR_WINDOW, min_periods=C.PAIR_MIN_OBS).mean())
        if adf_p < 0.05 and hl <= C.COINT_MAX_HALFLIFE:
            verdict = f"pulls back to a long-run level (half-life {hl:.0f} wk)"
        elif min(eg_ab, eg_ba) < 0.05 and hl <= C.COINT_MAX_HALFLIFE:
            verdict = f"pulls back with a fitted hedge ratio, not 1:1 (half-life {hl:.0f} wk)"
        elif adf_p < 0.10 and np.isfinite(hl) and hl <= C.COINT_MAX_HALFLIFE:
            verdict = f"some pull-back but not statistically established (half-life {hl:.0f} wk, ADF p {adf_p:.2f})"
        else:
            verdict = "no long-run level: the ratio drifts (a trend, not a spread)"
        rows.append({"pair": label(a, b), "weeks": len(lr), "adf_p_log_ratio": adf_p, "eg_p_a_on_b": eg_ab, "eg_p_b_on_a": eg_ba,
                     "halflife_weeks_full": hl, "halflife_weeks_vs_52w_mean": hl_w, "ratio_drift_per_year": float(lr.diff().mean() * C.ANNUALISE), "verdict": verdict})
    return pd.DataFrame(rows).set_index("pair")


# -------------------------------------------------- what the ratio did next --
def z_forward(Z: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pooled = {h: [] for h in C.PAIR_FWD}
    for a, b in pairs(r.columns):
        z = Z[label(a, b)]
        spread = (r[a] - r[b])
        for h in C.PAIR_FWD:
            F = forward(spread, h)
            Fa, Fb = forward(r[a], h), forward(r[b], h)
            df = pd.DataFrame({"z": z, "F": F, "Fa": Fa, "Fb": Fb}).dropna()
            df["bucket"] = bucket(df["z"]).astype(str)
            df["reverted"] = np.sign(df["F"]) == -np.sign(df["z"])
            df["toward"] = -np.sign(df["z"]) * df["F"]                  # positive = the ratio moved back toward its mean
            pooled[h].append(df.assign(pair=label(a, b)))
            for bkt, g in df.groupby("bucket"):
                rows.append({"pair": label(a, b), "a": a, "b": b, "horizon_weeks": h, "bucket": bkt, "weeks": len(g),
                             "mean_ratio_change": g["F"].mean(), "median_ratio_change": g["F"].median(),
                             "share_reverted": g["reverted"].mean() if bkt != BUCKET_ORDER[2] else np.nan,
                             "mean_move_toward_mean": g["toward"].mean() if bkt != BUCKET_ORDER[2] else np.nan,
                             "hac_t_ratio_change": hac_t(g["F"]), "mean_ret_a": g["Fa"].mean(), "mean_ret_b": g["Fb"].mean()})
    for h, parts in pooled.items():
        df = pd.concat(parts)
        for bkt, g in df.groupby("bucket"):
            rows.append({"pair": "ALL PAIRS", "a": "", "b": "", "horizon_weeks": h, "bucket": bkt, "weeks": len(g),
                         "mean_ratio_change": g["F"].mean(), "median_ratio_change": g["F"].median(),
                         "share_reverted": g["reverted"].mean() if bkt != BUCKET_ORDER[2] else np.nan,
                         "mean_move_toward_mean": g["toward"].mean() if bkt != BUCKET_ORDER[2] else np.nan,
                         "hac_t_ratio_change": hac_t(g.groupby(level=0)["F"].mean()),     # one value per week (the pairs averaged), then Newey-West
                         "mean_ret_a": g["Fa"].mean(), "mean_ret_b": g["Fb"].mean()})
    out = pd.DataFrame(rows)
    out["bucket"] = pd.Categorical(out["bucket"], BUCKET_ORDER, ordered=True)
    return out.sort_values(["pair", "horizon_weeks", "bucket"]).reset_index(drop=True)


def path_after(Z: pd.DataFrame, thr: float = C.PT_ENTRY, weeks: int = C.PAIR_PATH_WEEKS) -> pd.DataFrame:
    rows = []
    for col in Z.columns:
        z = Z[col]
        ev = z.index[((z.abs() >= thr) & (z.abs().shift(1) < thr)).fillna(False)]
        paths = []
        for d in ev:
            i = z.index.get_loc(d)
            seg = z.iloc[i:i + weeks + 1] * np.sign(z.iloc[i])
            if len(seg) == weeks + 1:
                paths.append(seg.values)
        if not paths:
            continue
        P = np.array(paths)
        for k in range(weeks + 1):
            rows.append({"pair": col, "weeks_after": k, "mean_z": float(np.nanmean(P[:, k])), "median_z": float(np.nanmedian(P[:, k])),
                         "share_back_inside_1sd": float(np.nanmean(P[:, k] <= 1.0)), "share_crossed_0": float(np.nanmean(np.nanmin(P[:, :k + 1], axis=1) <= 0)), "n_events": len(paths)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- returns --
def returns_table(r: pd.DataFrame, px_w: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    """Each leg's returns, side by side: total and annualised over the pair's common sample, the last 13 and 52
    weeks, and the average weekly return (annualised) in the weeks where both signals were BUY, both SELL, or
    they disagreed - plus the spread (a minus b) for each."""
    ann = C.ANNUALISE
    rows = []
    for a, b in pairs(r.columns):
        xy = r[[a, b]].dropna()
        sa, sb = st[a].reindex(xy.index), st[b].reindex(xy.index)
        cells = {"both_BUY": (sa == 1) & (sb == 1), "both_SELL": (sa == -1) & (sb == -1), "disagree": (sa * sb == -1)}
        yrs = (xy.index[-1] - xy.index[0]).days / 365.25
        row = {"pair": label(a, b), "a": a, "b": b, "weeks": len(xy), "years": yrs}
        for leg, x in ((a, "a"), (b, "b")):
            tot = float(np.exp(xy[leg].sum()) - 1)
            row[f"total_{x}"] = tot
            row[f"annualised_{x}"] = float((1 + tot) ** (1 / yrs) - 1) if yrs > 0 else np.nan
            row[f"last_13w_{x}"] = float(np.exp(xy[leg].iloc[-13:].sum()) - 1)
            row[f"last_52w_{x}"] = float(np.exp(xy[leg].iloc[-52:].sum()) - 1)
            row[f"vol_{x}"] = float(xy[leg].std() * np.sqrt(ann))
            row[f"share_weeks_up_{x}"] = float((xy[leg] > 0).mean())
            row[f"worst_week_{x}"] = float(xy[leg].min()); row[f"best_week_{x}"] = float(xy[leg].max())
            for name, m in cells.items():
                seg = xy.loc[m, leg]
                row[f"{name}_ann_{x}"] = float(seg.mean() * ann) if len(seg) >= C.MIN_CELL else np.nan
        for name, m in cells.items():
            row[f"{name}_weeks"] = int(m.sum())
        for k in ("total", "annualised", "last_13w", "last_52w", "both_BUY_ann", "both_SELL_ann", "disagree_ann"):
            row[f"spread_{k}"] = row[f"{k}_a"] - row[f"{k}_b"]
        row["share_weeks_a_beat_b"] = float((xy[a] > xy[b]).mean())
        row["corr_full"] = float(xy[a].corr(xy[b]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("pair")


def pair_corr_series(r: pd.DataFrame, w: int = C.CORR_WINDOW) -> pd.DataFrame:
    """Each pair's 13-week rolling correlation, week by week (for the pair charts)."""
    return pd.DataFrame({label(a, b): r[a].rolling(w, min_periods=int(0.8 * w)).corr(r[b]) for a, b in pairs(r.columns)})


def current(Z: pd.DataFrame, cw: pd.DataFrame | None = None) -> pd.DataFrame:
    rows = []
    for col in Z.columns:
        a, b = col.split("|")
        z = Z[col].dropna()
        rows.append({"pair": col, "date": z.index[-1].date() if len(z) else None, "z_now": z.iloc[-1] if len(z) else np.nan,
                     "z_4w_ago": z.iloc[-5] if len(z) > 5 else np.nan, "z_13w_ago": z.iloc[-14] if len(z) > 14 else np.nan,
                     "reading": _reading(z.iloc[-1] if len(z) else np.nan, a, b)})
    return pd.DataFrame(rows).set_index("pair")
