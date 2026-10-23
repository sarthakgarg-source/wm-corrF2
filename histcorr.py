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
from scipy import stats

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
            nx4 = _fwd(r[leg], C.QUINT_FWD)
            df = pd.DataFrame({"bucket": bk, "same": same.reindex(c.index), "next": nxt.reindex(c.index), "nx4": nx4.reindex(c.index), "c": c})
            for b, sub in df.groupby("bucket", observed=True):
                rows.append({"pair": p, "leg": leg, "bucket": b, "corr_lo": sub["c"].min(), "corr_hi": sub["c"].max(), "weeks": len(sub),
                             "same_quarter_mean": sub["same"].mean(), "same_quarter_hit": float((sub["same"].dropna() > 0).mean()),
                             "next_quarter_mean": sub["next"].mean(), "next_quarter_hit": float((sub["next"].dropna() > 0).mean()), "n_next": int(sub["next"].notna().sum()),
                             f"next_{C.QUINT_FWD}w_mean": sub["nx4"].mean(), f"next_{C.QUINT_FWD}w_median": sub["nx4"].median(), f"next_{C.QUINT_FWD}w_hit": float((sub["nx4"].dropna() > 0).mean()), f"n_next_{C.QUINT_FWD}w": int(sub["nx4"].notna().sum())})
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
        nx4 = _fwd(r[a], C.QUINT_FWD)                            # return over the next QUINT_FWD weeks (t+1..t+4) — what the chart shows
        df = pd.DataFrame({"bucket": bk, "same": same, "next": nxt, "nx4": nx4, "c": mc}).dropna(subset=["bucket", "c"])
        g = df.groupby("bucket", observed=True)
        for b in labels:
            if b not in g.groups:
                continue
            sub = g.get_group(b)
            rows.append({"asset": a, "bucket": b, "corr_lo": sub["c"].min(), "corr_hi": sub["c"].max(), "weeks": len(sub),
                         "same_quarter_mean": sub["same"].mean(), "same_quarter_hit": float((sub["same"].dropna() > 0).mean()),
                         "next_quarter_mean": sub["next"].mean(), "next_quarter_hit": float((sub["next"].dropna() > 0).mean()),
                         "n_next": int(sub["next"].notna().sum()),
                         f"next_{C.QUINT_FWD}w_mean": sub["nx4"].mean(), f"next_{C.QUINT_FWD}w_median": sub["nx4"].median(), f"next_{C.QUINT_FWD}w_hit": float((sub["nx4"].dropna() > 0).mean()), f"n_next_{C.QUINT_FWD}w": int(sub["nx4"].notna().sum())})
    long = pd.DataFrame(rows)
    # a compact pivot for the report: rows = asset, columns = bucket, three blocks
    same_p = long.pivot(index="asset", columns="bucket", values="same_quarter_mean")
    next_p = long.pivot(index="asset", columns="bucket", values="next_quarter_mean")
    nx4_p = long.pivot(index="asset", columns="bucket", values=f"next_{C.QUINT_FWD}w_mean")
    same_p.columns = [f"same_{c}" for c in same_p.columns]
    next_p.columns = [f"next_{c}" for c in next_p.columns]
    nx4_p.columns = [f"next{C.QUINT_FWD}w_{c}" for c in nx4_p.columns]
    piv = pd.concat([same_p, nx4_p, next_p], axis=1).reindex([a for a in C.ASSETS if a in r.columns])
    return long, piv


# ------------------------------------------------- the ratio study (review of 2026-10-19) --
def ratio_study(r: pd.DataFrame, st: pd.DataFrame, w: int = C.HIST_WINDOW, back: int = C.RATIO_BACK, fwd: int = C.RATIO_FWD) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One number per pair instead of two returns: the RATIO first leg ÷ second leg (its log change = the first leg's return
    minus the second's). For every pair and week t: the ratio's change over the last `back` weeks (t−back+1 … t, 'actual') and
    over the next `fwd` weeks (t+1 … t+fwd, 'forward'), cut three ways — by the pair's own 13-week correlation quintile at t,
    by the joint signal state at t (both BUY / disagree / both SELL), and by the sign of the last `back` weeks' change (does
    the ratio continue or reverse?) — with the forward change's mean, median, share positive and n. Also the correlation
    between the forward ratio change and the pair's correlation level, and the weekly rows behind it."""
    PC = pair_corrs(r, w)
    labels = [f"Q{i + 1}" for i in range(C.N_BUCKETS)]
    rows, weekly = [], []
    for p in PC.columns:
        a, b = p.split("|")
        rel = (r[a] - r[b])
        c = PC[p]
        past = rel.rolling(back).sum(); nxt = _fwd(rel, fwd)
        sa, sb = st[a].reindex(r.index), st[b].reindex(r.index)
        state = pd.Series(np.where((sa == 1) & (sb == 1), "both BUY", np.where((sa == -1) & (sb == -1), "both SELL", np.where(sa.notna() & sb.notna(), "disagree", None))), index=r.index)
        df = pd.DataFrame({"corr": c, "past": past, "next": nxt, "state": state}).dropna(subset=["corr"])
        if len(df) < 5 * C.N_BUCKETS:
            continue
        df["bucket"] = pd.qcut(df["corr"], C.N_BUCKETS, labels=labels)
        df["past_sign"] = np.where(df["past"] > 0, f"{a} ahead over the last {back} wk", np.where(df["past"] < 0, f"{b} ahead over the last {back} wk", None))
        wk = df.copy(); wk.insert(0, "pair", p); weekly.append(wk.reset_index().rename(columns={"index": "date"}))
        def _agg(sub, cut, grp):
            ok = sub["next"].dropna()
            return {"pair": p, "cut": cut, "group": grp, "weeks": int(len(sub)), "n_forward": int(len(ok)), "corr_lo": float(sub["corr"].min()), "corr_hi": float(sub["corr"].max()),
                    f"past_{back}w_mean": float(sub["past"].mean()), f"fwd_{fwd}w_mean": float(ok.mean()) if len(ok) else np.nan, f"fwd_{fwd}w_median": float(ok.median()) if len(ok) else np.nan,
                    f"fwd_{fwd}w_hit": float((ok > 0).mean()) if len(ok) else np.nan, f"fwd_{fwd}w_sd": float(ok.std()) if len(ok) > 2 else np.nan}
        for bk, sub in df.groupby("bucket", observed=True):
            rows.append(_agg(sub, "correlation quintile", bk))
        for stt in ("both BUY", "disagree", "both SELL"):
            sub = df[df["state"] == stt]
            if len(sub) >= C.MIN_CELL:
                rows.append(_agg(sub, "joint signal state", stt))
        for sg in [x for x in df["past_sign"].dropna().unique()]:
            rows.append(_agg(df[df["past_sign"] == sg], f"sign of the last {back} weeks' change", sg))
        both = df[["corr", "next", "past"]].dropna()
        rows.append({"pair": p, "cut": "all weeks", "group": "all", "weeks": int(len(df)), "n_forward": int(both.shape[0]), "corr_lo": float(df["corr"].min()), "corr_hi": float(df["corr"].max()),
                     f"past_{back}w_mean": float(df["past"].mean()), f"fwd_{fwd}w_mean": float(both["next"].mean()), f"fwd_{fwd}w_median": float(both["next"].median()), f"fwd_{fwd}w_hit": float((both["next"] > 0).mean()), f"fwd_{fwd}w_sd": float(both["next"].std()),
                     "spearman_fwd_vs_corr": float(stats.spearmanr(both["corr"], both["next"]).statistic) if len(both) > 30 else np.nan,
                     "spearman_fwd_vs_past": float(stats.spearmanr(both["past"], both["next"]).statistic) if len(both) > 30 else np.nan})
    return pd.DataFrame(rows), (pd.concat(weekly, ignore_index=True) if weekly else pd.DataFrame())


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
                      "a_next_13w_mean": float(g["a_next_13w"].mean()) if len(g) else np.nan, "a_next_13w_positive": float((g["a_next_13w"].dropna() > 0).mean()) if g["a_next_13w"].notna().any() else np.nan,
                      "b_next_13w_mean": float(g["b_next_13w"].mean()) if len(g) else np.nan, "b_next_13w_positive": float((g["b_next_13w"].dropna() > 0).mean()) if g["b_next_13w"].notna().any() else np.nan})
    return long, pd.DataFrame(srows).set_index("pair")


# ------------------------------------------------ gold against the indices --
def gold_corr_rank(r: pd.DataFrame, w: int = C.HIST_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Week by week, each index's rolling correlation with gold, and which index gold is most correlated with that week.
    Summary per index: the share of weeks it is gold's most-correlated index, and against N100: the share of weeks its gold
    correlation is above N100's and the average gap. (For the B2 question: how often is SC100×gold above N100×gold?)"""
    PC = pair_corrs(r, w)
    gcols = {p: p.split("|")[0] for p in PC.columns if "Gold" in p}
    G = PC[list(gcols)].rename(columns=gcols)
    idx = [a for a in C.EQUITY if a in G.columns]
    G = G[idx]
    live = G.notna().any(axis=1)
    top = pd.Series(np.nan, index=G.index, dtype=object); top_abs = top.copy()
    top[live] = G[live].idxmax(axis=1); top_abs[live] = G[live].abs().idxmax(axis=1)
    wk = pd.DataFrame({f"gold_corr_{a}": G[a] for a in idx}); wk["most_correlated"] = top; wk["largest_abs"] = top_abs
    rows = []
    for a in idx:
        both = G[[a, C.BENCHMARK]].dropna() if a != C.BENCHMARK else G[[a]].dropna()
        row = {"index": a, "weeks": int(G[a].notna().sum()), "mean_gold_corr": float(G[a].mean()), "share_weeks_most_correlated_with_gold": float((top == a).mean()),
               "share_weeks_largest_abs_corr_with_gold": float((top_abs == a).mean())}
        if a != C.BENCHMARK:
            row[f"weeks_above_{C.BENCHMARK}"] = int((both[a] > both[C.BENCHMARK]).sum()); row[f"share_above_{C.BENCHMARK}"] = float((both[a] > both[C.BENCHMARK]).mean())
            row[f"mean_gap_vs_{C.BENCHMARK}"] = float((both[a] - both[C.BENCHMARK]).mean()); row["weeks_compared"] = int(len(both))
        rows.append(row)
    return wk, pd.DataFrame(rows).set_index("index")


def gold_when_decoupled(r: pd.DataFrame, cep: pd.DataFrame, w: int = C.HIST_WINDOW, before: int = 8, after: int = 13) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """What the gold–index correlations do when the index correlation dies. (a) Every gold pair's 13-week correlation, and its
    absolute value, averaged inside quintiles of the group correlation G (Q1 = the indices least coupled … Q5 most). (b) Around the
    start of each decorrelation episode: each gold pair's correlation from `before` weeks before to `after` weeks after, averaged;
    and per episode, the pair's mean correlation in the 8 weeks before vs inside the episode, and its largest |corr| inside."""
    PC = pair_corrs(r, w)
    gp = [p for p in PC.columns if "Gold" in p]; ip = [p for p in PC.columns if "Gold" not in p]
    G = PC[ip].mean(axis=1, skipna=True)
    labels = [f"Q{i + 1}" for i in range(C.N_BUCKETS)]
    bk = pd.qcut(G.dropna(), C.N_BUCKETS, labels=labels)
    rows = []
    for p in gp:
        df = pd.DataFrame({"c": PC[p], "bk": bk}).dropna()
        for b, g in df.groupby("bk", observed=True):
            rows.append({"pair": p, "bucket": b, "G_lo": float(G.loc[g.index].min()), "G_hi": float(G.loc[g.index].max()), "weeks": int(len(g)), "mean_corr": float(g["c"].mean()), "mean_abs_corr": float(g["c"].abs().mean()), "share_abs_above_0.3": float((g["c"].abs() > 0.3).mean())})
    byq = pd.DataFrame(rows)
    paths, per = [], []
    for _, e in cep.iterrows():
        s0, s1 = pd.Timestamp(e["start"]), pd.Timestamp(e["end"])
        if s0 not in PC.index:
            continue
        i = PC.index.get_loc(s0)
        for p in gp:
            c = PC[p]
            pre = c.iloc[max(0, i - before):i]; dur = c.loc[s0:s1]
            if pre.notna().any() and dur.notna().any():
                per.append({"episode": e["episode"], "pair": p, "corr_before": float(pre.mean()), "corr_during": float(dur.mean()), "abs_before": float(pre.abs().mean()), "abs_during": float(dur.abs().mean()), "max_abs_during": float(dur.abs().max()), "change_abs": float(dur.abs().mean() - pre.abs().mean())})
            for k in range(-before, after + 1):
                if 0 <= i + k < len(c) and pd.notna(c.iloc[i + k]):
                    paths.append({"episode": e["episode"], "pair": p, "k": k, "corr": float(c.iloc[i + k]), "abs_corr": float(abs(c.iloc[i + k]))})
    pl = pd.DataFrame(paths)
    path = pl.groupby(["pair", "k"]).agg(mean_corr=("corr", "mean"), mean_abs_corr=("abs_corr", "mean"), n=("corr", "count")).reset_index() if len(pl) else pd.DataFrame(columns=["pair", "k", "mean_corr", "mean_abs_corr", "n"])
    return byq, path, pd.DataFrame(per)


def _fwd(s: pd.Series, h: int) -> pd.Series:
    """Sum over t+1 … t+h, aligned to t (the same rule as pairs.forward)."""
    return s.shift(-1).rolling(h).sum().shift(-(h - 1))


def _fit(x: pd.Series, y: pd.Series, lags: int = C.HAC_LAGS) -> dict:
    """A straight line y = a + b·x with a Newey-West t on the slope (the weeks overlap), plus Pearson and Spearman."""
    import statsmodels.api as sm
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(d) < 20:
        return {"n": int(len(d)), "pearson": np.nan, "spearman": np.nan, "slope": np.nan, "intercept": np.nan, "r2": np.nan, "nw_t_slope": np.nan}
    res = sm.OLS(d["y"].values, sm.add_constant(d["x"].values)).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"n": int(len(d)), "pearson": float(d["x"].corr(d["y"])), "spearman": float(stats.spearmanr(d["x"], d["y"]).statistic), "slope": float(res.params[1]), "intercept": float(res.params[0]),
            "r2": float(res.rsquared), "nw_t_slope": float(res.tvalues[1])}


def gold_vs_group(r: pd.DataFrame, w: int = C.HIST_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """B2 (review of 2026-10-19): is there a relationship between the index-pair chart and the gold-vs-index chart? Week by week, the group
    correlation G against each gold pair's 13-week correlation and against the SC100×Gold − N100×Gold gap. Returns the weekly
    points (for the scatter), a fit table (a straight line per y, with Pearson / Spearman / R² / Newey-West t on the slope,
    and a quadratic's R² as the 'is there a better model' check) and the binned means (deciles of G → the average y)."""
    PC = pair_corrs(r, w)
    gp = [p for p in PC.columns if "Gold" in p]; ip = [p for p in PC.columns if "Gold" not in p]
    G = PC[ip].mean(axis=1, skipna=True)
    wk = pd.DataFrame({"G": G})
    for p in gp:
        wk[p] = PC[p]
    if "SC100|GoldMCX" in PC.columns and "N100|GoldMCX" in PC.columns:
        wk["gap_SC100_minus_N100"] = PC["SC100|GoldMCX"] - PC["N100|GoldMCX"]
    ys = [c for c in wk.columns if c != "G"]
    fits, bins = [], []
    dec = pd.qcut(G.dropna(), 10, labels=[f"D{i + 1}" for i in range(10)])
    for y in ys:
        f = _fit(wk["G"], wk[y]); f["y"] = y
        d = wk[["G", y]].dropna()
        if len(d) >= 20:
            q = np.polyfit(d["G"], d[y], 2); pred = np.polyval(q, d["G"])
            f["r2_quadratic"] = float(1 - ((d[y] - pred) ** 2).sum() / ((d[y] - d[y].mean()) ** 2).sum())
            lo, hi = d[d["G"] <= d["G"].quantile(0.2)], d[d["G"] >= d["G"].quantile(0.8)]
            f["mean_y_G_lowest_fifth"] = float(lo[y].mean()); f["mean_y_G_highest_fifth"] = float(hi[y].mean())
        fits.append(f)
        for b, g in pd.DataFrame({"y": wk[y], "d": dec}).dropna().groupby("d", observed=True):
            bins.append({"y": y, "decile_of_G": b, "G_lo": float(G.loc[g.index].min()), "G_hi": float(G.loc[g.index].max()), "weeks": int(len(g)), "mean_y": float(g["y"].mean()), "median_y": float(g["y"].median())})
    cols = ["y", "n", "pearson", "spearman", "slope", "intercept", "r2", "nw_t_slope", "r2_quadratic", "mean_y_G_lowest_fifth", "mean_y_G_highest_fifth"]
    return wk, pd.DataFrame(fits).reindex(columns=cols).set_index("y"), pd.DataFrame(bins)


def gap_reversion(r: pd.DataFrame, w: int = C.HIST_WINDOW, sd_k: float = C.GAP_SD, fwd: tuple = C.GAP_FWD) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """B2 (review of 2026-10-19): the SC100×Gold − N100×Gold gap looks mean-reverting — is it, and can it be used? (a) The gap's statistics:
    mean, sd, autocorrelation at 1/4/13 weeks, AR(1) coefficient and half-life, ADF p-value — with the caveat that a 13-week
    rolling correlation reverts by construction as the window rolls off. (b) What followed an extreme gap (beyond ±sd_k sd of
    its own full-sample sd, counted once per run): the gap's change over the next 4 / 13 weeks and how often it narrowed, and
    the returns that would make it usable — SC100 minus N100, and gold, over the same forward weeks (t+1 …). (c) The weekly series."""
    from statsmodels.tsa.stattools import adfuller
    PC = pair_corrs(r, w)
    if not {"SC100|GoldMCX", "N100|GoldMCX"} <= set(PC.columns):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    gap = (PC["SC100|GoldMCX"] - PC["N100|GoldMCX"]).rename("gap")
    s = gap.dropna()
    x, y = s.shift(1).dropna(), s.diff().dropna()
    phi = 1.0 + float(np.polyfit(x.values, y.values, 1)[0])
    st = {"weeks": int(len(s)), "mean": float(s.mean()), "sd": float(s.std()), "min": float(s.min()), "max": float(s.max()),
          "autocorr_1w": float(s.autocorr(1)), "autocorr_4w": float(s.autocorr(4)), "autocorr_13w": float(s.autocorr(13)), "autocorr_26w": float(s.autocorr(26)),
          "ar1_phi": phi, "half_life_weeks": float(-np.log(2) / np.log(phi)) if 0 < phi < 1 else np.nan,
          "adf_p": float(adfuller(s.values, autolag="AIC", result_object=False)[1]) if len(s) > 30 else np.nan,
          "share_weeks_beyond_1sd": float((s.abs() > sd_k * s.std()).mean()), "weeks_above_zero_share": float((s > 0).mean())}
    rel = (r["SC100"] - r[C.BENCHMARK]) if "SC100" in r.columns else None
    wk = pd.DataFrame({"gap": gap, "upper": s.mean() + sd_k * s.std(), "lower": s.mean() - sd_k * s.std()})
    rows = []
    for sign, name in ((1, f"gap above +{sd_k:g} sd (Smallcap's gold corr the higher)"), (-1, f"gap below −{sd_k:g} sd (Nifty's gold corr the higher)")):
        thr = s.mean() + sign * sd_k * s.std()
        hit = (s * sign) > (thr * sign)
        starts = s.index[hit & ~hit.shift(1, fill_value=False)]          # the first week of each run beyond the line
        for h in fwd:
            chg = gap.shift(-h) - gap
            toward = (chg * sign) < 0
            fr = _fwd(rel, h) if rel is not None else None; fg = _fwd(r["GoldMCX"], h) if "GoldMCX" in r.columns else None
            ok = [d for d in starts if pd.notna(chg.get(d))]
            row = {"extreme": name, "threshold": float(thr), "horizon_weeks": h, "events": int(len(ok)),
                   "mean_gap_change": float(chg.loc[ok].mean()) if ok else np.nan, "share_gap_narrowed": float(toward.loc[ok].mean()) if ok else np.nan,
                   "mean_gap_at_event": float(s.loc[ok].mean()) if ok else np.nan}
            if fr is not None:
                row["SC100_minus_N100_mean"] = float(fr.loc[ok].mean()) if ok else np.nan; row["SC100_minus_N100_hit"] = float((fr.loc[ok] > 0).mean()) if ok else np.nan
                row["SC100_minus_N100_all_weeks_mean"] = float(fr.mean())
            if fg is not None:
                row["gold_mean"] = float(fg.loc[ok].mean()) if ok else np.nan; row["gold_hit"] = float((fg.loc[ok] > 0).mean()) if ok else np.nan; row["gold_all_weeks_mean"] = float(fg.mean())
            rows.append(row)
    return pd.DataFrame([st]), pd.DataFrame(rows), wk


def index_pair_vs_gold_pair(r: pd.DataFrame, w: int = C.HIST_WINDOW) -> tuple[pd.DataFrame, pd.DataFrame]:
    """B2 (review of 2026-10-22): do an index pair's correlation and a gold pair's correlation move in opposite directions?
    For every index pair × gold pair: the correlation between the two weekly series (levels: Pearson with a Newey-West t,
    Spearman), between their 4-week and 13-week changes (weekly-stepped, and on the strictly non-overlapping sub-sample),
    and the gold pair's average correlation in the weeks the index pair was in its loosest fifth vs its tightest fifth.
    Plus the weekly difference series (index pair − gold pair) for each combination, with its mean and sd."""
    PC = pair_corrs(r, w)
    ip = [p for p in PC.columns if "Gold" not in p]; gp = [p for p in PC.columns if "Gold" in p]
    rows, wk = [], pd.DataFrame(index=PC.index)
    for a in ip:
        for b in gp:
            d = pd.DataFrame({"a": PC[a], "b": PC[b]}).dropna()
            f = _fit(d["a"], d["b"], lags=w)
            row = {"index_pair": a, "gold_pair": b, "weeks": int(len(d)), "pearson_levels": f["pearson"], "nw_t_levels": f["nw_t_slope"], "spearman_levels": f["spearman"]}
            for L in (4, 13):
                dd = pd.DataFrame({"a": PC[a].diff(L), "b": PC[b].diff(L)}).dropna(); no = dd.iloc[::L]
                row[f"pearson_{L}w_changes"] = float(dd["a"].corr(dd["b"])) if len(dd) >= 20 else np.nan
                row[f"pearson_{L}w_changes_nonoverlap"] = float(no["a"].corr(no["b"])) if len(no) >= 20 else np.nan
                row[f"n_{L}w_nonoverlap"] = int(len(no))
            if len(d) >= 5 * C.N_BUCKETS:
                q = pd.qcut(d["a"], C.N_BUCKETS, labels=[f"Q{i + 1}" for i in range(C.N_BUCKETS)])
                g = d.groupby(q, observed=True)["b"].mean()
                row["gold_pair_mean_when_index_pair_loosest"] = float(g.get("Q1", np.nan)); row["gold_pair_mean_when_index_pair_tightest"] = float(g.get(f"Q{C.N_BUCKETS}", np.nan))
            diff = PC[a] - PC[b]
            row["diff_mean"] = float(diff.mean()); row["diff_sd"] = float(diff.std())
            wk[f"{a} - {b}"] = diff
            rows.append(row)
    return pd.DataFrame(rows), wk


# ------------------------------------- what follows a difference series crossing ±1 sd (review of 2026-10-23) --
def _difference_series(PC: pd.DataFrame) -> dict:
    """The difference series the B2 charts draw, each with its two components and the assets involved:
    the gap (SC100×Gold − N100×Gold) and every index pair minus every gold pair."""
    out = {}
    if {"SC100|GoldMCX", "N100|GoldMCX"} <= set(PC.columns):
        out["gap: SC100|GoldMCX - N100|GoldMCX"] = ("SC100|GoldMCX", "N100|GoldMCX")
    for a in [p for p in PC.columns if "Gold" not in p]:
        for b in [p for p in PC.columns if "Gold" in p]:
            out[f"{a} - {b}"] = (a, b)
    return out


def _fwd_stats(r: pd.Series, i: int, h: int) -> tuple[float, float, float]:
    """Over t+1 … t+h from row i: the cumulative log return, the largest peak-to-trough fall inside the window (max drawdown,
    the peak starting at t), and the lowest point reached from t (max adverse excursion). NaN when the window is short."""
    seg = r.iloc[i + 1:i + 1 + h]
    if len(seg) < h or seg.isna().any():
        return np.nan, np.nan, np.nan
    c = np.concatenate([[0.0], np.cumsum(seg.values)])
    peak = np.maximum.accumulate(c)
    return float(c[-1]), float((peak - c).max()), float(c.min())


def extreme_followups(r: pd.DataFrame, w: int = C.HIST_WINDOW, sd_k: float = C.GAP_SD, horizons: tuple = C.EXTREME_FWD, path: int = C.EXTREME_PATH) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """What happens after a difference series (the gap, or an index pair's correlation minus a gold pair's) first crosses
    +sd_k or −sd_k of its own whole-history sd. For every such crossing (the first week of each run beyond the line) and
    horizon h: (1) the series itself — its change over h weeks and whether it moved back toward its mean; (2) its two
    components — each correlation's change over h; (3) returns — each asset involved over t+1 … t+h, and the two relative
    returns (the index pair's first leg minus its second; the gold pair's index leg minus gold); (4) drawdowns — each asset's
    largest peak-to-trough fall inside the window and its lowest point from t. Every number comes with the all-weeks
    baseline (the same statistic started from every week). Returns: per event (wide), the summary (long: series, side,
    horizon, metric, mean, median, hit, n, all_weeks_mean, all_weeks_hit), and the average paths k = −8 … +path."""
    PC = pair_corrs(r, w)
    series = _difference_series(PC)
    ev_rows, sum_rows, path_rows = [], [], []
    for name, (a, b) in series.items():
        d = (PC[a] - PC[b]).rename("d")
        s = d.dropna()
        if len(s) < 60:
            continue
        m, sd = float(s.mean()), float(s.std())
        legs = list(dict.fromkeys(a.split("|") + b.split("|")))
        rel = {f"{a.split('|')[0]} - {a.split('|')[1]}": r[a.split("|")[0]] - r[a.split("|")[1]]}
        gi = [x for x in b.split("|") if x != C.GOLD]
        if gi and gi[0] != a.split("|")[0] and gi[0] != a.split("|")[1]:
            rel[f"{gi[0]} - {C.GOLD}"] = r[gi[0]] - r[C.GOLD]
        elif gi:
            rel[f"{gi[0]} - {C.GOLD}"] = r[gi[0]] - r[C.GOLD]
        idx = d.index
        # all-weeks baselines, per horizon: the same forward statistics started from every week with the series present
        base = {}
        for h in horizons:
            rows = []
            for i in range(len(idx)):
                if pd.isna(d.iloc[i]) or i + h >= len(idx):
                    continue
                row = {"series_change": d.iloc[i + h] - d.iloc[i], "comp_a_change": PC[a].iloc[i + h] - PC[a].iloc[i], "comp_b_change": PC[b].iloc[i + h] - PC[b].iloc[i]}
                for x in legs:
                    ret, dd, mae = _fwd_stats(r[x], i, h); row[f"ret_{x}"] = ret; row[f"maxdd_{x}"] = dd; row[f"worst_{x}"] = mae
                for k_, v in rel.items():
                    row[f"rel_{k_}"] = v.iloc[i + 1:i + 1 + h].sum() if v.iloc[i + 1:i + 1 + h].notna().all() and len(v.iloc[i + 1:i + 1 + h]) == h else np.nan
                rows.append(row)
            base[h] = pd.DataFrame(rows)
        for sign, side in ((1, f"above +{sd_k:g} sd"), (-1, f"below -{sd_k:g} sd")):
            thr = m + sign * sd_k * sd
            hit = ((d * sign) > (thr * sign)).fillna(False)
            starts = [idx.get_loc(x) for x in idx[hit & ~hit.shift(1, fill_value=False)]]
            per = []
            for i in starts:
                row = {"series": name, "side": side, "date": idx[i].date(), "value": float(d.iloc[i]), "threshold": thr, "comp_a": a, "comp_b": b}
                # weeks until the series is back at its mean (the crossing's side), within `path`
                back = next((k for k in range(1, path + 1) if i + k < len(idx) and pd.notna(d.iloc[i + k]) and (d.iloc[i + k] - m) * sign <= 0), np.nan)
                row["weeks_to_mean"] = back
                for h in horizons:
                    if i + h < len(idx):
                        row[f"series_change_{h}"] = d.iloc[i + h] - d.iloc[i]; row[f"toward_mean_{h}"] = bool((d.iloc[i + h] - d.iloc[i]) * sign < 0)
                        row[f"comp_a_change_{h}"] = PC[a].iloc[i + h] - PC[a].iloc[i]; row[f"comp_b_change_{h}"] = PC[b].iloc[i + h] - PC[b].iloc[i]
                    for x in legs:
                        ret, dd, mae = _fwd_stats(r[x], i, h); row[f"ret_{x}_{h}"] = ret; row[f"maxdd_{x}_{h}"] = dd; row[f"worst_{x}_{h}"] = mae
                    for k_, v in rel.items():
                        seg = v.iloc[i + 1:i + 1 + h]; row[f"rel_{k_}_{h}"] = seg.sum() if len(seg) == h and seg.notna().all() else np.nan
                per.append(row)
                for k in range(-8, path + 1):
                    if 0 <= i + k < len(idx):
                        prow = {"series": name, "side": side, "date": idx[i].date(), "k": k, "d": d.iloc[i + k], "comp_a": PC[a].iloc[i + k], "comp_b": PC[b].iloc[i + k]}
                        for x in legs:
                            prow[f"cum_{x}"] = r[x].iloc[i + 1:i + 1 + k].sum() if k > 0 and r[x].iloc[i + 1:i + 1 + k].notna().all() else (0.0 if k == 0 else np.nan)
                        path_rows.append(prow)
            ev_rows += per
            pe = pd.DataFrame(per)
            if not len(pe):
                continue
            for h in horizons:
                bh = base[h]
                def _add(metric, col, bcol, hit_fn=None):
                    x = pe[col].dropna() if col in pe.columns else pd.Series(dtype=float)
                    sum_rows.append({"series": name, "side": side, "horizon_weeks": h, "metric": metric, "events": int(len(x)), "mean": float(x.mean()) if len(x) else np.nan, "median": float(x.median()) if len(x) else np.nan,
                                     "hit": float(hit_fn(x).mean()) if len(x) and hit_fn is not None else np.nan, "all_weeks_mean": float(bh[bcol].mean()) if bcol in bh.columns else np.nan, "all_weeks_median": float(bh[bcol].median()) if bcol in bh.columns else np.nan,
                                     "all_weeks_hit": float(hit_fn(bh[bcol].dropna()).mean()) if bcol in bh.columns and hit_fn is not None else np.nan})
                _add("the series: change", f"series_change_{h}", "series_change", lambda x: (x * sign) < 0)
                _add(f"component {a}: change", f"comp_a_change_{h}", "comp_a_change", lambda x: x > 0)
                _add(f"component {b}: change", f"comp_b_change_{h}", "comp_b_change", lambda x: x > 0)
                for x_ in legs:
                    _add(f"return {x_}", f"ret_{x_}_{h}", f"ret_{x_}", lambda x: x > 0)
                    _add(f"max drawdown {x_}", f"maxdd_{x_}_{h}", f"maxdd_{x_}", lambda x: x > 0.05)
                    _add(f"worst point {x_}", f"worst_{x_}_{h}", f"worst_{x_}", lambda x: x < -0.05)
                for k_ in rel:
                    _add(f"relative return {k_}", f"rel_{k_}_{h}", f"rel_{k_}", lambda x: x > 0)
    paths = pd.DataFrame(path_rows)
    pth = paths.groupby(["series", "side", "k"]).mean(numeric_only=True).reset_index() if len(paths) else pd.DataFrame()
    cnt = paths.groupby(["series", "side", "k"])["d"].count().rename("n").reset_index() if len(paths) else pd.DataFrame()
    pth = pth.merge(cnt, on=["series", "side", "k"]) if len(pth) else pth
    return pd.DataFrame(ev_rows), pd.DataFrame(sum_rows), pth


def gold_after_break(r: pd.DataFrame, cep: pd.DataFrame, w: int = C.HIST_WINDOW, after: int = C.GOLD_AFTER_BREAK) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """B2 (review of 2026-10-19): after the index correlation structure breaks, how long does a gold pair's correlation stay positive?
    For every decorrelation episode and gold pair: the pair's 13-week correlation from the episode's first week (k = 0) to
    k = `after`; per episode the correlation at k = 0, the first week it was ≤ 0, the number of positive weeks in the first
    `after`, the longest positive run and the peak. Returns the per-episode paths (long), the averaged path per pair, and the
    per-episode summary. Micro250 exists only from 2022, so it is in the later episodes only — `episodes` says how many."""
    PC = pair_corrs(r, w)
    gp = [p for p in PC.columns if "Gold" in p]
    paths, per = [], []
    for _, e in cep.iterrows():
        s0 = pd.Timestamp(e["start"])
        if s0 not in PC.index:
            continue
        i = PC.index.get_loc(s0)
        for p in gp:
            c = PC[p]
            seg = c.iloc[i:i + after + 1]
            if seg.isna().all() or pd.isna(c.iloc[i]):
                continue
            vals = seg.values
            pos = vals > 0
            first_np = next((k for k, v in enumerate(vals) if pd.notna(v) and v <= 0), np.nan)
            run, best = 0, 0
            for v in vals:
                run = run + 1 if (pd.notna(v) and v > 0) else 0; best = max(best, run)
            per.append({"episode": e["episode"], "start": s0.date(), "pair": p, "corr_at_break": float(vals[0]), "positive_at_break": bool(vals[0] > 0),
                        "first_week_at_or_below_zero": first_np, "weeks_positive_of_first_" + str(after): int(pos.sum()), "longest_positive_run": int(best),
                        "peak_corr": float(np.nanmax(vals)), "peak_week": int(np.nanargmax(vals)), "mean_corr_0_to_" + str(after): float(np.nanmean(vals)), "weeks_available": int(pd.notna(vals).sum())})
            for k, v in enumerate(vals):
                if pd.notna(v):
                    paths.append({"episode": e["episode"], "pair": p, "k": k, "corr": float(v)})
    pl = pd.DataFrame(paths)
    avg = pl.groupby(["pair", "k"]).agg(mean_corr=("corr", "mean"), median_corr=("corr", "median"), share_positive=("corr", lambda v: float((v > 0).mean())), n=("corr", "count")).reset_index() if len(pl) else pd.DataFrame()
    return pl, avg, pd.DataFrame(per)
