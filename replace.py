"""
8. WHO REPLACES A SELL: when an asset's supertrend flips to SELL, which asset should take
   its place in the book - and how did that choice perform?

Event = a week t in which asset A's state goes from BUY to SELL ("the seller"). At the close
of that week the book has to put A's money somewhere. The candidates are the other four
assets, plus cash (the liquid ETF) as the plain alternative.

For every event and candidate B, measured AT t (nothing after t is used to describe it):
  corr          the 52-week correlation of B with A            -> "correlated" if >= RP_CORR_MIN
  z             z_of(Z, B, A): the deviation of B against A over the trailing year
                z <= -RP_Z_DEV  "lagging"  (B has fallen behind A)
                z >= +RP_Z_DEV  "leading"  (B has run ahead of A)
                otherwise       "neutral"
  state         B's own supertrend state at t (BUY / SELL)
  rel_13w       B's 13-week return minus A's (a shorter deviation measure, for reference)

and AFTER t:
  fwd_<x>_h     the return of B, of A (staying put), of cash and of gold over t+1..t+h,
                h in RP_FWD (4 and 13 weeks)
  hold_until_own_sell   B's return from t+1 until the first week B's own signal is SELL
                (capped at RP_MAX_HOLD weeks), with the weeks held and how it ended; A's and
                cash's return over the same weeks

  events          the SELL flips
  candidate_rows  one row per event x candidate (the long table everything else is built from)
  cells           by (deviation, correlated, B's state): B's return vs staying in A vs cash
                  - answers "if it is correlated and has deviated, and we enter it, how did it go"
  by_pair         seller x candidate: how each replacement did on that seller's flips
  rules           twelve replacement rules applied to the SAME events, one pick per event,
                  falling back to cash when a rule finds nobody: the actionable comparison
  z_path          the average z of each candidate against the seller from RP_PATH_WEEKS before
                  the flip to RP_PATH_WEEKS after - what the deviation chart looked like
  event_paths     the same per event (for the event-grid charts)
  current         the latest week: every asset in SELL and what each rule would pick now

Forward windows use t+1.. only (checks.py pins it). Events of one seller do not overlap much,
but events of different sellers often fall in the same weeks and one event feeds several
candidate rows; t-stats are therefore reported only on the per-event rule picks, with
Newey-West, and only where n >= 20.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import pairs as PR
from pairs import forward, hac_t, z_of


def events(st: pd.DataFrame, idx: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Every BUY -> SELL flip of every asset. `idx` (the weekly return index) drops flips with no return week to follow."""
    rows = []
    for a in [c for c in C.ASSETS if c in st.columns]:
        s = st[a]
        flips = s.index[((s == -1) & (s.shift(1) == 1)).fillna(False)]
        rows += [{"seller": a, "date": d} for d in flips if idx is None or d in idx]
    return pd.DataFrame(rows).sort_values(["date", "seller"]).reset_index(drop=True)


def _hold_until_sell(r_b: pd.Series, s_b: pd.Series, i: int, cap: int) -> tuple[float, int, str]:
    j, cum = i + 1, 0.0
    n = len(r_b)
    while j < n:
        cum += r_b.iloc[j] if pd.notna(r_b.iloc[j]) else 0.0
        if s_b.iloc[j] == -1:
            return cum, j - i, "own SELL"
        if j - i >= cap:
            return cum, j - i, "cap"
        j += 1
    return cum, min(j, n - 1) - i, "end of data"


def candidate_rows(ev: pd.DataFrame, st: pd.DataFrame, r: pd.DataFrame, Z: pd.DataFrame) -> pd.DataFrame:
    idx = r.index
    assets = [c for c in C.ASSETS if c in r.columns]
    cw = {}
    for a in assets:
        for b in assets:
            if a < b:
                cw[(a, b)] = cw[(b, a)] = r[a].rolling(C.RP_CORR_WINDOW, min_periods=C.PAIR_MIN_OBS).corr(r[b])
    F = {(x, h): forward(r[x], h) for x in assets + ([C.CASH] if C.CASH in r.columns else []) for h in C.RP_FWD}
    ret13 = r[assets].rolling(13).sum()
    stx = st.reindex(idx)
    rows = []
    for _, e in ev.iterrows():
        a, d = e["seller"], e["date"]
        if d not in idx:
            continue
        i = idx.get_loc(d)
        for b in assets:
            if b == a or pd.isna(r[b].iloc[i]):            # a candidate that does not exist yet (Micro250 before 2022) is no candidate
                continue
            z = z_of(Z, b, a).reindex(idx).iloc[i] if PR.label(a, b) in Z.columns or PR.label(b, a) in Z.columns else np.nan
            corr = cw[(a, b)].iloc[i]
            sb = stx[b].iloc[i] if b in stx.columns else np.nan
            dev = "lagging" if (pd.notna(z) and z <= -C.RP_Z_DEV) else ("leading" if (pd.notna(z) and z >= C.RP_Z_DEV) else ("neutral" if pd.notna(z) else "no z"))
            row = {"seller": a, "date": d, "candidate": b, "corr_52w": corr, "correlated": bool(pd.notna(corr) and corr >= C.RP_CORR_MIN),
                   "z_cand_vs_seller": z, "deviation": dev, "cand_state": "BUY" if sb == 1 else ("SELL" if sb == -1 else "none"),
                   "cand_rel_13w": ret13[b].iloc[i] - ret13[a].iloc[i], "cand_ret_13w": ret13[b].iloc[i], "seller_ret_13w": ret13[a].iloc[i]}
            for h in C.RP_FWD:
                row[f"fwd_cand_{h}"] = F[(b, h)].iloc[i]
                row[f"fwd_seller_{h}"] = F[(a, h)].iloc[i]
                row[f"fwd_cash_{h}"] = F[(C.CASH, h)].iloc[i] if (C.CASH, h) in F else np.nan
                row[f"fwd_gold_{h}"] = F[(C.GOLD, h)].iloc[i] if C.GOLD in assets else np.nan
            if b in stx.columns and pd.notna(stx[b].iloc[i]):
                cum, wk, how = _hold_until_sell(r[b], stx[b], i, C.RP_MAX_HOLD)
                row.update({"hold_ret": cum, "hold_weeks": wk, "hold_end": how,
                            "hold_seller_ret": float(r[a].iloc[i + 1:i + 1 + wk].sum()) if wk else 0.0,
                            "hold_cash_ret": float(r[C.CASH].iloc[i + 1:i + 1 + wk].sum()) if C.CASH in r.columns and wk else np.nan})
            else:
                row.update({"hold_ret": np.nan, "hold_weeks": np.nan, "hold_end": "no signal", "hold_seller_ret": np.nan, "hold_cash_ret": np.nan})
            rows.append(row)
    return pd.DataFrame(rows)


def _agg(g: pd.DataFrame, h: int) -> dict:
    c, s, k = g[f"fwd_cand_{h}"], g[f"fwd_seller_{h}"], g[f"fwd_cash_{h}"]
    return {"rows": len(g), "events": int(g[["seller", "date"]].drop_duplicates().shape[0]),
            "cand_mean": c.mean(), "cand_median": c.median(), "cand_hit_positive": float((c > 0).mean()),
            "stay_mean": s.mean(), "cash_mean": k.mean(), "cand_minus_stay": (c - s).mean(), "cand_minus_cash": (c - k).mean(),
            "share_cand_beat_stay": float((c > s).mean()), "share_cand_beat_cash": float((c > k).mean()) if k.notna().any() else np.nan}


def cells(cr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in C.RP_FWD:
        for (dev, corr, stt), g in cr.groupby(["deviation", "correlated", "cand_state"]):
            rows.append({"horizon_weeks": h, "deviation": dev, "correlated": corr, "cand_state": stt, **_agg(g, h)})
        for (dev, corr), g in cr.groupby(["deviation", "correlated"]):
            rows.append({"horizon_weeks": h, "deviation": dev, "correlated": corr, "cand_state": "any", **_agg(g, h)})
        for (dev, stt), g in cr.groupby(["deviation", "cand_state"]):
            rows.append({"horizon_weeks": h, "deviation": dev, "correlated": "any", "cand_state": stt, **_agg(g, h)})
        for dev, g in cr.groupby("deviation"):
            rows.append({"horizon_weeks": h, "deviation": dev, "correlated": "any", "cand_state": "any", **_agg(g, h)})
        for stt, g in cr.groupby("cand_state"):
            rows.append({"horizon_weeks": h, "deviation": "any", "correlated": "any", "cand_state": stt, **_agg(g, h)})
        for corr, g in cr.groupby("correlated"):
            rows.append({"horizon_weeks": h, "deviation": "any", "correlated": corr, "cand_state": "any", **_agg(g, h)})
        for (corr, stt), g in cr.groupby(["correlated", "cand_state"]):
            rows.append({"horizon_weeks": h, "deviation": "any", "correlated": corr, "cand_state": stt, **_agg(g, h)})
        rows.append({"horizon_weeks": h, "deviation": "any", "correlated": "any", "cand_state": "any", **_agg(cr, h)})
    out = pd.DataFrame(rows)
    out["deviation"] = pd.Categorical(out["deviation"], ["lagging", "neutral", "leading", "no z", "any"], ordered=True)
    out["correlated"] = out["correlated"].astype(str)
    return out.sort_values(["horizon_weeks", "deviation", "correlated", "cand_state"]).reset_index(drop=True)


def by_pair(cr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (a, b), g in cr.groupby(["seller", "candidate"]):
        row = {"seller": a, "candidate": b, "events": len(g), "avg_corr_52w": g["corr_52w"].mean(), "avg_z_cand_vs_seller": g["z_cand_vs_seller"].mean(),
               "share_lagging": float((g["deviation"] == "lagging").mean()), "share_leading": float((g["deviation"] == "leading").mean()),
               "share_cand_in_BUY": float((g["cand_state"] == "BUY").mean())}
        for h in C.RP_FWD:
            row[f"cand_mean_{h}"] = g[f"fwd_cand_{h}"].mean(); row[f"stay_mean_{h}"] = g[f"fwd_seller_{h}"].mean(); row[f"cash_mean_{h}"] = g[f"fwd_cash_{h}"].mean()
            row[f"share_cand_beat_cash_{h}"] = float((g[f"fwd_cand_{h}"] > g[f"fwd_cash_{h}"]).mean())
        if "hold_ret" in g:
            row["hold_mean"] = g["hold_ret"].mean(); row["hold_weeks_mean"] = g["hold_weeks"].mean(); row["hold_seller_mean"] = g["hold_seller_ret"].mean(); row["hold_cash_mean"] = g["hold_cash_ret"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rules --
def _pick(g: pd.DataFrame, mask: pd.Series, by: str, largest: bool = True):
    sub = g[mask & g[by].notna()]
    if not len(sub):
        return None
    return sub.loc[sub[by].idxmax() if largest else sub[by].idxmin()]


RULES = [
    ("stay in the seller", "the seller's own return (no replacement)"),
    ("cash", "the liquid ETF"),
    ("gold", "gold (cash when gold itself is the seller)"),
    ("most correlated", "the candidate with the highest 52w correlation to the seller"),
    ("most correlated, in BUY", "the same, among candidates whose own signal is BUY; cash if none"),
    ("correlated & lagging", "corr >= 0.5 and z <= -1: the one most correlated; cash if none"),
    ("correlated & lagging, in BUY", "as above, and its own signal is BUY; cash if none"),
    ("correlated & leading", "corr >= 0.5 and z >= +1: the one most correlated; cash if none"),
    ("correlated & leading, in BUY", "as above, and its own signal is BUY; cash if none"),
    ("most lagging (any corr)", "the candidate with the lowest z against the seller"),
    ("most leading (any corr)", "the candidate with the highest z against the seller"),
    ("best 13w momentum, in BUY", "the BUY-state candidate with the highest own 13-week return; cash if none"),
]


def rules(cr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    picks = []
    for (a, d), g in cr.groupby(["seller", "date"]):
        g = g.copy()
        buy = g["cand_state"] == "BUY"
        corr = g["correlated"]
        lag = g["deviation"] == "lagging"; lead = g["deviation"] == "leading"
        base = g.iloc[0]
        cash = {f"ret_{h}": base[f"fwd_cash_{h}"] for h in C.RP_FWD}
        cash.update({"hold_ret": base["hold_cash_ret"] if "hold_cash_ret" in base else np.nan, "hold_weeks": np.nan})
        gold_row = g[g["candidate"] == C.GOLD]
        choices = {
            "stay in the seller": ({f"ret_{h}": base[f"fwd_seller_{h}"] for h in C.RP_FWD} | {"hold_ret": np.nan, "hold_weeks": np.nan}, a),
            "cash": (cash, C.CASH),
            "gold": (({f"ret_{h}": gold_row.iloc[0][f"fwd_cand_{h}"] for h in C.RP_FWD} | {"hold_ret": gold_row.iloc[0].get("hold_ret", np.nan), "hold_weeks": gold_row.iloc[0].get("hold_weeks", np.nan)}, C.GOLD) if len(gold_row) else (cash, C.CASH)),
            "most correlated": _pick(g, pd.Series(True, index=g.index), "corr_52w"),
            "most correlated, in BUY": _pick(g, buy, "corr_52w"),
            "correlated & lagging": _pick(g, corr & lag, "corr_52w"),
            "correlated & lagging, in BUY": _pick(g, corr & lag & buy, "corr_52w"),
            "correlated & leading": _pick(g, corr & lead, "corr_52w"),
            "correlated & leading, in BUY": _pick(g, corr & lead & buy, "corr_52w"),
            "most lagging (any corr)": _pick(g, pd.Series(True, index=g.index), "z_cand_vs_seller", largest=False),
            "most leading (any corr)": _pick(g, pd.Series(True, index=g.index), "z_cand_vs_seller", largest=True),
            "best 13w momentum, in BUY": _pick(g, buy, "cand_ret_13w"),
        }
        for rule, _ in RULES:
            ch = choices[rule]
            if isinstance(ch, tuple):
                vals, who = ch
                had_pick = who not in (C.CASH,) or rule == "cash"
            elif ch is None:
                vals, who, had_pick = cash, C.CASH, False
            else:
                vals = {f"ret_{h}": ch[f"fwd_cand_{h}"] for h in C.RP_FWD} | {"hold_ret": ch.get("hold_ret", np.nan), "hold_weeks": ch.get("hold_weeks", np.nan)}
                who, had_pick = ch["candidate"], True
            picks.append({"rule": rule, "seller": a, "date": d, "pick": who, "had_pick": had_pick, **vals,
                          **{f"cash_{h}": base[f"fwd_cash_{h}"] for h in C.RP_FWD}, **{f"stay_{h}": base[f"fwd_seller_{h}"] for h in C.RP_FWD}})
    P = pd.DataFrame(picks)
    rows = []
    for rule, desc in RULES:
        g = P[P["rule"] == rule].sort_values("date")
        row = {"rule": rule, "what it does": desc, "events": len(g), "share_with_a_pick": float(g["had_pick"].mean())}
        gp = g[g["had_pick"]]
        for h in C.RP_FWD:
            y = g[f"ret_{h}"]
            row[f"mean_{h}"] = y.mean(); row[f"median_{h}"] = y.median(); row[f"hit_positive_{h}"] = float((y > 0).mean())
            row[f"beat_cash_{h}"] = float((y > g[f"cash_{h}"]).mean()); row[f"beat_stay_{h}"] = float((y > g[f"stay_{h}"]).mean())
            row[f"minus_cash_{h}"] = (y - g[f"cash_{h}"]).mean()
            row[f"hac_t_minus_cash_{h}"] = hac_t((y - g[f"cash_{h}"]).reset_index(drop=True)) if len(g) >= 20 else np.nan
            row[f"worst_{h}"] = y.min()
            # the same, counting only the events where the rule actually found somebody (the fallback-to-cash events removed)
            yp = gp[f"ret_{h}"]
            row[f"when_pick_mean_{h}"] = yp.mean() if len(gp) else np.nan
            row[f"when_pick_minus_cash_{h}"] = (yp - gp[f"cash_{h}"]).mean() if len(gp) else np.nan
            row[f"when_pick_beat_cash_{h}"] = float((yp > gp[f"cash_{h}"]).mean()) if len(gp) else np.nan
            row[f"when_pick_beat_stay_{h}"] = float((yp > gp[f"stay_{h}"]).mean()) if len(gp) else np.nan
        row["events_with_a_pick"] = int(len(gp))
        row["hold_mean"] = gp["hold_ret"].mean() if len(gp) else np.nan; row["hold_weeks_mean"] = gp["hold_weeks"].mean() if len(gp) else np.nan
        rows.append(row)
    R = pd.DataFrame(rows).set_index("rule")
    return R, P


def rules_by_seller(P: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (rule, a), g in P.groupby(["rule", "seller"]):
        row = {"rule": rule, "seller": a, "events": len(g), "share_with_a_pick": float(g["had_pick"].mean())}
        for h in C.RP_FWD:
            row[f"mean_{h}"] = g[f"ret_{h}"].mean(); row[f"beat_cash_{h}"] = float((g[f"ret_{h}"] > g[f"cash_{h}"]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- z path --
def event_paths(ev: pd.DataFrame, Z: pd.DataFrame, weeks: int = C.RP_PATH_WEEKS) -> pd.DataFrame:
    rows = []
    idx = Z.index
    assets = [c for c in C.ASSETS if any(c in col.split("|") for col in Z.columns)]
    for _, e in ev.iterrows():
        a, d = e["seller"], e["date"]
        if d not in idx:
            continue
        i = idx.get_loc(d)
        for b in assets:
            if b == a:
                continue
            z = z_of(Z, b, a)
            for k in range(-weeks, weeks + 1):
                if 0 <= i + k < len(idx):
                    rows.append({"seller": a, "date": d, "candidate": b, "k": k, "z": z.iloc[i + k]})
    return pd.DataFrame(rows)


def z_path(ep: pd.DataFrame) -> pd.DataFrame:
    g = ep.dropna(subset=["z"]).groupby(["seller", "candidate", "k"])["z"]
    out = g.agg(mean_z="mean", median_z="median", n="count").reset_index()
    pooled = ep.dropna(subset=["z"]).groupby(["seller", "k"])["z"].agg(mean_z="mean", median_z="median", n="count").reset_index().assign(candidate="all candidates")
    return pd.concat([out, pooled[out.columns]], ignore_index=True)


# ---------------------------------------------------------------- current --
def current(st: pd.DataFrame, r: pd.DataFrame, Z: pd.DataFrame) -> pd.DataFrame:
    assets = [c for c in C.ASSETS if c in r.columns and c in st.columns]
    d = r.index[-1]                                   # the last week every index has a close (gold's own calendar runs a little later)
    stx = st.reindex(r.index)
    Zx = Z.reindex(r.index)

    def _s(x):
        v = stx[x].iloc[-1]
        return "BUY" if v == 1 else ("SELL" if v == -1 else "none")

    rows = []
    for a in assets:
        for b in assets:
            if a == b:
                continue
            corr = r[a].rolling(C.RP_CORR_WINDOW, min_periods=C.PAIR_MIN_OBS).corr(r[b]).iloc[-1]
            z = z_of(Zx, b, a).iloc[-1]
            rows.append({"date": d.date(), "seller_if_in_SELL": a, "seller_state": _s(a), "candidate": b,
                         "corr_52w": corr, "correlated": bool(corr >= C.RP_CORR_MIN), "z_cand_vs_seller": z,
                         "deviation": "lagging" if z <= -C.RP_Z_DEV else ("leading" if z >= C.RP_Z_DEV else "neutral"),
                         "cand_state": _s(b)})
    return pd.DataFrame(rows)
