"""
3. WHICH ASSET BREAKS THE CORRELATION FIRST.

The idea in one line: each index normally moves with the others; when the market
"decorrelates", one of them usually lets go first. This module finds those moments
and ranks the assets by who let go first.

Definitions (all weekly, all in config.py):

  c_i,t      asset i's CURRENT correlation with the other indices: the average of its
             13-week rolling correlations with each of the other three size indices.
             (Gold is measured against the four indices but is not ranked - it is
             ~uncorrelated with them all the time, so "breaking" means nothing for it.)
  G_t        the group's current correlation: the average of all six pairwise 13-week
             correlations among the four indices.
  z_i,t      how unusual c_i,t is for THAT asset: (c_i,t - mean over the last 104 weeks)
             / sd over the last 104 weeks. A z of -1 means "a full standard deviation
             below its own two-year norm".
  break      the first week z_i,t <= CB_Z (-1) after having been above CB_RESET (-0.5).
             The same rule applied to G_t gives a GROUP break: the market decorrelating.
  episode    one group break, from the week z_G crosses -1 until it is back above -0.5.
             Episodes closer than CB_MIN_GAP weeks are merged.

For every episode, each asset's own break date is searched from CB_LOOKBACK weeks
before the group's break to the episode's end; lead = asset's date - group's date in
weeks (negative = the asset broke BEFORE the market did). Ranked 1..n per episode.

Outputs:
  series            c_i,t, z_i,t, G_t, z_G,t week by week (for the chart)
  episodes          the group-break episodes with dates, depth (lowest G), length
  asset_breaks      per episode x asset: break date, lead, rank, the asset's correlation
                    drop (pre-episode level -> minimum), and its 13-week relative return at
                    the break (up-break = it decoupled by outperforming, down-break by
                    underperforming)
  summary           per asset across episodes: median lead, share first, average rank,
                    share of breaks that were up-breaks, average drop
  pair_breaks       the same question at pair level: which PAIR's correlation breaks first
  before_selloff    for each PRICE sell-off (breaks.episodes): did a correlation break occur
                    in the 13 weeks before N100's peak, and whose - the early-warning test
  after_break       what the breaker, the rest of the group and gold returned over the 4
                    and 13 weeks after an asset's break; split by up-break / down-break
  sensitivity       the asset ranking re-run at window 13/26 and z -1/-1.5
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

import config as C


# -------------------------------------------------------------- series --
def pair_corrs(r: pd.DataFrame, w: int) -> dict[tuple[str, str], pd.Series]:
    eq = [a for a in C.EQUITY if a in r.columns]
    return {(a, b): r[a].rolling(w, min_periods=int(0.8 * w)).corr(r[b]) for a, b in itertools.combinations(eq, 2)}


def series(r: pd.DataFrame, w: int = C.CB_WINDOW, norm: int = C.CB_NORM) -> pd.DataFrame:
    eq = [a for a in C.EQUITY if a in r.columns]
    pc = pair_corrs(r, w)
    out = pd.DataFrame(index=r.index)
    for a in eq:
        cols = [pc[k] for k in pc if a in k]
        out[f"c_{a}"] = pd.concat(cols, axis=1).mean(axis=1, skipna=True)
    out["G"] = pd.concat(list(pc.values()), axis=1).mean(axis=1, skipna=True)
    if C.GOLD in r.columns:
        out[f"c_{C.GOLD}"] = pd.concat([r[C.GOLD].rolling(w, min_periods=int(0.8 * w)).corr(r[a]) for a in eq], axis=1).mean(axis=1, skipna=True)
    for a, b in itertools.combinations([x for x in C.ASSETS if x in r.columns], 2):      # every pair on its own, gold included
        out[f"{a}|{b}"] = pc[(a, b)] if (a, b) in pc else r[a].rolling(w, min_periods=int(0.8 * w)).corr(r[b])
    for col in [c for c in out.columns]:
        m = out[col].rolling(norm, min_periods=int(0.5 * norm)).mean()
        s = out[col].rolling(norm, min_periods=int(0.5 * norm)).std()
        out[f"z_{col}"] = (out[col] - m) / s
    return out


# ------------------------------------------------------------ episodes --
def _breaks(z: pd.Series, thr: float = C.CB_Z, reset: float = C.CB_RESET) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of each stretch: start = first week z <= thr after being > reset; end = first week back above reset."""
    out, armed, start = [], True, None
    for d, v in z.items():
        if pd.isna(v):
            continue
        if start is None:
            if armed and v <= thr:
                start = d
            elif v > reset:
                armed = True
        else:
            if v > reset:
                out.append((start, d))
                start, armed = None, True
    if start is not None:
        out.append((start, z.index[-1]))
    return out


def episodes(S: pd.DataFrame) -> pd.DataFrame:
    eps = _breaks(S["z_G"])
    rows = []
    for s, e in eps:
        seg = S.loc[s:e]
        pre = S["G"].loc[:s].iloc[-C.CB_LOOKBACK - 1:-1]
        rows.append({"start": s, "end": e, "weeks": len(seg), "G_before": float(pre.mean()) if len(pre) else np.nan,
                     "G_min": float(seg["G"].min()), "G_min_date": seg["G"].idxmin(), "z_min": float(seg["z_G"].min())})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    keep = []
    for _, row in df.iterrows():
        if keep and (row["start"] - keep[-1]["end"]).days / 7 < C.CB_MIN_GAP:
            if row["G_min"] < keep[-1]["G_min"]:
                keep[-1]["G_min"], keep[-1]["G_min_date"], keep[-1]["z_min"] = row["G_min"], row["G_min_date"], row["z_min"]
            keep[-1]["end"] = row["end"]
            keep[-1]["weeks"] = int((keep[-1]["end"] - keep[-1]["start"]).days / 7) + 1
            continue
        keep.append(row.to_dict())
    out = pd.DataFrame(keep)
    out.insert(0, "episode", [f"C{i + 1} {d.strftime('%Y-%m')}" for i, d in enumerate(out["start"])])
    out["drop"] = out["G_min"] - out["G_before"]
    return out


def asset_breaks(S: pd.DataFrame, ep: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    eq = [a for a in C.EQUITY if f"c_{a}" in S.columns]
    rows = []
    for _, e in ep.iterrows():
        i_start = S.index.get_loc(e["start"])
        look_from = S.index[max(0, i_start - C.CB_LOOKBACK)]
        for a in eq:
            z = S[f"z_c_{a}"].loc[look_from:e["end"]]
            c = S[f"c_{a}"].loc[look_from:e["end"]]
            if z.notna().sum() < 4:
                continue
            # the asset's own break: first week z <= CB_Z within the window (armed if it entered the window above reset or at the window start)
            hit = z[z <= C.CB_Z]
            bdate = hit.index[0] if len(hit) else pd.NaT
            pre = S[f"c_{a}"].loc[:e["start"]].iloc[-C.CB_LOOKBACK - 1:-1]
            rel = np.nan
            if pd.notna(bdate):
                others = [o for o in eq if o != a]
                i = r.index.get_loc(bdate)
                own = r[a].iloc[max(0, i - 12):i + 1].sum()
                grp = r[others].iloc[max(0, i - 12):i + 1].sum(axis=0).mean()
                rel = float(own - grp)
            z_at = S[f"z_c_{a}"].get(e["start"], np.nan)
            rows.append({"episode": e["episode"], "asset": a, "break_date": bdate,
                         "lead_vs_group_weeks": ((bdate - e["start"]).days / 7) if pd.notna(bdate) else np.nan,
                         "z_at_group_break": float(z_at) if pd.notna(z_at) else np.nan,
                         "corr_before": float(pre.mean()) if len(pre) else np.nan, "corr_min": float(c.min()), "corr_drop": float(c.min() - pre.mean()) if len(pre) else np.nan,
                         "z_min": float(z.min()), "rel_13w_return_at_break": rel,
                         "direction": ("up-break (outperforming)" if rel > 0 else "down-break (underperforming)") if pd.notna(rel) else ""})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["rank"] = df.groupby("episode")["break_date"].rank(method="min")            # 1 = earliest date (ties share)
    df["rank_deepest"] = df.groupby("episode")["z_at_group_break"].rank(method="min")  # 1 = furthest below its norm when the group broke
    tot = df.groupby("episode")["corr_drop"].transform(lambda s: s.sum())
    df["drop_share"] = df["corr_drop"] / tot                                         # share of the episode's total correlation loss
    df["rank_drop"] = df.groupby("episode")["corr_drop"].rank(method="min")           # 1 = biggest fall in its own correlation
    df["broke"] = df["break_date"].notna()
    return df


def summary(ab: pd.DataFrame) -> pd.DataFrame:
    ab = ab.copy()
    ab["n_first_in_episode"] = ab.groupby("episode")["rank"].transform(lambda s: int((s == 1).sum()))
    rows = []
    for a, g in ab.groupby("asset"):
        b = g[g["broke"]]
        rows.append({"asset": a, "episodes_present": int(len(g)), "episodes_broke": int(len(b)), "share_broke": float(len(b) / len(g)) if len(g) else np.nan,
                     "share_first": float((g["rank"] == 1).mean()), "avg_rank": float(g["rank"].mean()),
                     "share_first_alone": float(((g["rank"] == 1) & (g["n_first_in_episode"] == 1)).mean()),
                     "avg_rank_deepest": float(g["rank_deepest"].mean()), "share_deepest": float((g["rank_deepest"] == 1).mean()),
                     "avg_rank_drop": float(g["rank_drop"].mean()), "share_biggest_drop": float((g["rank_drop"] == 1).mean()),
                     "avg_drop_share": float(g["drop_share"].mean()),
                     "median_lead_weeks": float(b["lead_vs_group_weeks"].median()) if len(b) else np.nan,
                     "share_before_group": float((b["lead_vs_group_weeks"] < 0).mean()) if len(b) else np.nan,
                     "avg_corr_drop": float(b["corr_drop"].mean()) if len(b) else np.nan,
                     "share_up_breaks": float((b["rel_13w_return_at_break"] > 0).mean()) if len(b) else np.nan,
                     "median_rel_13w_return_at_break": float(b["rel_13w_return_at_break"].median()) if len(b) else np.nan})
    df = pd.DataFrame(rows).set_index("asset").reindex([a for a in C.EQUITY if a in set(ab["asset"])])
    # a combined score: average of the three ranks (date, deepest, biggest drop); lower = more of a first-breaker
    df["combined_rank"] = df[["avg_rank", "avg_rank_deepest", "avg_rank_drop"]].mean(axis=1)
    df["verdict"] = ""
    df.loc[df["avg_rank"].idxmin(), "verdict"] += "earliest by date; "
    df.loc[df["avg_rank_deepest"].idxmin(), "verdict"] += "deepest below its norm when the market broke; "
    df.loc[df["avg_rank_drop"].idxmin(), "verdict"] += "biggest fall in its own correlation; "
    df.loc[df["combined_rank"].idxmin(), "verdict"] += "=> FIRST-BREAKER on the combined score"
    return df


# -------------------------------------------------------- pair level --
def pair_breaks(r: pd.DataFrame, ep: pd.DataFrame, w: int = C.CB_WINDOW, norm: int = C.CB_NORM) -> tuple[pd.DataFrame, pd.DataFrame]:
    pc = pair_corrs(r, w)
    rows = []
    for _, e in ep.iterrows():
        i_start = r.index.get_loc(e["start"])
        look_from = r.index[max(0, i_start - C.CB_LOOKBACK)]
        for (a, b), s in pc.items():
            m = s.rolling(norm, min_periods=int(0.5 * norm)).mean()
            sd = s.rolling(norm, min_periods=int(0.5 * norm)).std()
            z = ((s - m) / sd).loc[look_from:e["end"]]
            hit = z[z <= C.CB_Z]
            bdate = hit.index[0] if len(hit) else pd.NaT
            rows.append({"episode": e["episode"], "pair": f"{a}|{b}", "break_date": bdate,
                         "lead_vs_group_weeks": ((bdate - e["start"]).days / 7) if pd.notna(bdate) else np.nan,
                         "corr_min": float(s.loc[look_from:e["end"]].min()), "z_min": float(z.min()) if z.notna().any() else np.nan})
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby("episode")["break_date"].rank(method="min")
    sm = df.groupby("pair").agg(episodes=("episode", "nunique"), broke=("break_date", lambda s: int(s.notna().sum())),
                                share_first=("rank", lambda s: float((s == 1).mean())), avg_rank=("rank", "mean"),
                                median_lead_weeks=("lead_vs_group_weeks", "median")).sort_values("avg_rank")
    return df, sm


# ------------------------------------------------ before a price sell-off --
def before_selloff(S: pd.DataFrame, price_ep: pd.DataFrame, lookback: int = 13) -> pd.DataFrame:
    eq = [a for a in C.EQUITY if f"c_{a}" in S.columns]
    rows = []
    for _, e in price_ep.iterrows():
        pk = e["peak_date"]
        if pk not in S.index:
            continue
        i = S.index.get_loc(pk)
        win = S.iloc[max(0, i - lookback):i + 1]
        row = {"selloff": e["episode"], "N100_peak": pk.date(), "depth": e["depth"]}
        row["group_min_z_before_peak"] = float(win["z_G"].min()) if win["z_G"].notna().any() else np.nan
        row["group_broke_before_peak"] = bool((win["z_G"] <= C.CB_Z).any())
        firsts = {}
        for a in eq:
            z = win[f"z_c_{a}"]
            hit = z[z <= C.CB_Z]
            firsts[a] = hit.index[0] if len(hit) else pd.NaT
            row[f"{a}_broke_weeks_before_peak"] = ((pk - hit.index[0]).days / 7) if len(hit) else np.nan
        valid = {a: d for a, d in firsts.items() if pd.notna(d)}
        row["first_to_break"] = min(valid, key=valid.get) if valid else "none"
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------- after a break --
def after_break(ab: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    eq = [a for a in C.EQUITY if a in r.columns]
    rows = []
    for _, b in ab[ab["broke"]].iterrows():
        a = b["asset"]
        i = r.index.get_loc(b["break_date"])
        others = [o for o in eq if o != a]
        for h in C.CB_FWD:
            seg = r.iloc[i + 1:i + 1 + h]
            if len(seg) < h:
                continue
            rows.append({"episode": b["episode"], "asset": a, "direction": b["direction"], "weeks_ahead": h,
                         "breaker": float(seg[a].sum()), "rest_of_group": float(seg[others].sum(axis=0).mean()),
                         "gold": float(seg[C.GOLD].sum()) if C.GOLD in seg.columns else np.nan})
    df = pd.DataFrame(rows)
    if df.empty:
        return df, df
    df["breaker_minus_rest"] = df["breaker"] - df["rest_of_group"]
    sm = df.groupby(["direction", "weeks_ahead"]).agg(n=("episode", "count"), breaker=("breaker", "mean"), rest_of_group=("rest_of_group", "mean"),
                                                     breaker_minus_rest=("breaker_minus_rest", "mean"),
                                                     share_breaker_ahead=("breaker_minus_rest", lambda s: float((s > 0).mean())), gold=("gold", "mean")).reset_index()
    return df, sm


# ---------------------------------------------------------- sensitivity --
def sensitivity(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for w in C.CB_SENSITIVITY["window"]:
        for zt in C.CB_SENSITIVITY["z"]:
            S = series(r, w=w)
            eps = _breaks(S["z_G"], thr=zt)
            # build a minimal episode frame
            ep_rows = []
            for s, e in eps:
                ep_rows.append({"episode": f"{s.strftime('%Y-%m')}", "start": s, "end": e})
            ep = pd.DataFrame(ep_rows)
            if ep.empty:
                continue
            # merge close ones
            keep = []
            for _, row in ep.iterrows():
                if keep and (row["start"] - keep[-1]["end"]).days / 7 < C.CB_MIN_GAP:
                    keep[-1]["end"] = row["end"]; continue
                keep.append(row.to_dict())
            ep = pd.DataFrame(keep)
            eq = [a for a in C.EQUITY if f"c_{a}" in S.columns]
            recs = []
            for _, e in ep.iterrows():
                i_start = S.index.get_loc(e["start"])
                look_from = S.index[max(0, i_start - C.CB_LOOKBACK)]
                for a in eq:
                    z = S[f"z_c_{a}"].loc[look_from:e["end"]]
                    hit = z[z <= zt]
                    recs.append({"episode": e["episode"], "asset": a, "break_date": hit.index[0] if len(hit) else pd.NaT})
            d = pd.DataFrame(recs)
            d["rank"] = d.groupby("episode")["break_date"].rank(method="min")
            for a in eq:
                g = d[d["asset"] == a]
                rows.append({"window_weeks": w, "z_threshold": zt, "episodes": int(len(ep)), "asset": a,
                             "share_first": float((g["rank"] == 1).mean()), "avg_rank": float(g["rank"].mean()),
                             "share_broke": float(g["break_date"].notna().mean())})
    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="asset", columns=["window_weeks", "z_threshold"], values="avg_rank")
    piv.columns = [f"avg_rank w{w} z{z}" for w, z in piv.columns]
    piv2 = df.pivot_table(index="asset", columns=["window_weeks", "z_threshold"], values="share_first")
    piv2.columns = [f"first w{w} z{z}" for w, z in piv2.columns]
    n = df.groupby(["window_weeks", "z_threshold"])["episodes"].first()
    out = pd.concat([piv, piv2], axis=1).reindex([a for a in C.EQUITY if a in set(df["asset"])])
    out.attrs["episodes"] = n.to_dict()
    return out, df
