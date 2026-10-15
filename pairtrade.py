"""
7. THE PAIR-TRADING RULE, built from the deviation z-score and evaluated pair by pair.

The rule (every number in config.py):
  * at the close of week t, if |z_t| >= PT_ENTRY (2 sd) and there is no open trade in this pair:
      z > 0  (a rich, b cheap)  ->  short a, long b
      z < 0  (a cheap, b rich)  ->  long a, short b
    i.e. always long the CHEAP leg and short the RICH leg, one unit each (dollar-neutral)
  * the trade earns from week t+1: each week  side x (r_a - r_b)
  * close at the first week u where |z_u| <= PT_EXIT, or after PT_MAX_HOLD weeks, or at the
    last bar of data
  * cost: PT_COST_BPS one-way per leg, so 4 x that per round trip

Two readings of the same trade:
  long/short  gross = sum of side x (r_a - r_b) over t+1..u;  net = gross - cost
  long-only   the cheap leg's own return over the same weeks vs the rich leg's: "had the book
              simply held the cheap leg instead of the rich one". The difference is the gross
              spread, so the long-only switch and the long/short trade are the same bet seen
              from a long-only book.

  trades       every trade of every pair, with its dates, legs, z at entry / exit, exit reason,
               gross / cost / net, the two legs' returns, and the worst point during the trade
  summary      per pair and pooled: n, win rate, mean / median / total net, a plain t on the
               trade P&Ls (trades do not overlap within a pair), average weeks held, how trades
               ended, worst and best trade, net per year, drawdown of the trade-by-trade
               equity, a verdict
  equity       the weekly cumulative net P&L per pair (for the chart)
  sensitivity  the pooled result at entry {1.5, 2, 2.5} x exit {0, 0.5} x require the cheap
               leg's own signal to be BUY {no, yes}; robustness, not a menu

This is an evaluation of a rule on history, not a backtest of a strategy: one unit per leg,
no sizing, no margin, no slippage beyond the cost assumption, no overlap between pairs
considered.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config as C
import pairs as PR


def trades(Z: pd.DataFrame, r: pd.DataFrame, a: str, b: str, entry: float = C.PT_ENTRY, exit_: float = C.PT_EXIT,
           max_hold: int = C.PT_MAX_HOLD, cost_bps: float = C.PT_COST_BPS, require_buy: bool = False, st: pd.DataFrame | None = None) -> pd.DataFrame:
    z = Z[PR.label(a, b)]
    spread = (r[a] - r[b]).reindex(z.index)
    ra, rb = r[a].reindex(z.index), r[b].reindex(z.index)
    sa = st[a].reindex(z.index) if st is not None else None
    sb = st[b].reindex(z.index) if st is not None else None
    cost = 4 * cost_bps / 1e4
    out, i, n = [], 0, len(z)
    while i < n - 1:
        zi = z.iloc[i]
        if pd.isna(zi) or abs(zi) < entry:
            i += 1
            continue
        side = -1 if zi > 0 else 1                      # +1 = long a short b (a cheap); -1 = long b short a (b cheap)
        cheap, rich = (a, b) if side == 1 else (b, a)
        if require_buy:
            s_cheap = (sa if cheap == a else sb).iloc[i]
            if s_cheap != 1:
                i += 1
                continue
        j, cum, worst, reason = i + 1, 0.0, 0.0, "end of data"
        while j < n:
            cum += side * spread.iloc[j] if pd.notna(spread.iloc[j]) else 0.0
            worst = min(worst, cum)
            zj = z.iloc[j]
            if pd.notna(zj) and abs(zj) <= exit_:
                reason = "z back inside"
                break
            if j - i >= max_hold:
                reason = "max hold"
                break
            j += 1
        j = min(j, n - 1)
        seg = slice(i + 1, j + 1)
        gross = float(side * spread.iloc[seg].sum())
        out.append({"pair": PR.label(a, b), "a": a, "b": b, "entry_date": z.index[i], "exit_date": z.index[j], "weeks": j - i, "side": side,
                    "cheap_leg": cheap, "rich_leg": rich, "z_entry": float(zi), "z_exit": float(z.iloc[j]) if pd.notna(z.iloc[j]) else np.nan,
                    "exit_reason": reason, "gross": gross, "cost": cost, "net": gross - cost, "worst_point": worst,
                    "ret_cheap_leg": float((ra if cheap == a else rb).iloc[seg].sum()), "ret_rich_leg": float((rb if cheap == a else ra).iloc[seg].sum()),
                    "cheap_leg_state_at_entry": (sa if cheap == a else sb).iloc[i] if st is not None else np.nan})
        i = j + 1
    return pd.DataFrame(out)


def all_trades(Z: pd.DataFrame, r: pd.DataFrame, st: pd.DataFrame | None = None, **kw) -> pd.DataFrame:
    parts = [trades(Z, r, a, b, st=st, **kw) for a, b in PR.pairs(r.columns)]
    parts = [p for p in parts if len(p)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _verdict(n: int, mean_net: float, win: float, t: float) -> str:
    if n < C.PT_MIN_TRADES:
        return f"too few trades ({n}) to say"
    if mean_net > 0 and win >= 0.6 and t >= 2:
        return "worked on this history"
    if mean_net > 0:
        return "positive on average but not reliable"
    return "did not work"


def _dd(eq: pd.Series) -> float:
    if not len(eq):
        return np.nan
    e = eq.cumsum()
    return float((e - e.cummax()).min())


def summary(tr: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    groups = list(tr.groupby("pair")) + [("ALL PAIRS", tr)] if len(tr) else []
    for name, g in groups:
        n = len(g)
        t = g["net"].mean() / (g["net"].std(ddof=1) / np.sqrt(n)) if n >= C.PT_MIN_TRADES and g["net"].std(ddof=1) > 0 else np.nan
        years = (r.index[-1] - r.index[0]).days / 365.25 if name == "ALL PAIRS" else (r[[name.split("|")[0], name.split("|")[1]]].dropna().index[-1] - r[[name.split("|")[0], name.split("|")[1]]].dropna().index[0]).days / 365.25
        rows.append({"pair": name, "trades": n, "win_rate_net": float((g["net"] > 0).mean()), "mean_net": g["net"].mean(), "median_net": g["net"].median(),
                     "total_net": g["net"].sum(), "total_gross": g["gross"].sum(), "total_cost": g["cost"].sum(), "t_mean_net": t,
                     "avg_weeks": g["weeks"].mean(), "share_exit_z": float((g["exit_reason"] == "z back inside").mean()), "share_exit_max_hold": float((g["exit_reason"] == "max hold").mean()),
                     "mean_ret_cheap_leg": g["ret_cheap_leg"].mean(), "mean_ret_rich_leg": g["ret_rich_leg"].mean(), "share_cheap_beat_rich": float((g["ret_cheap_leg"] > g["ret_rich_leg"]).mean()),
                     "worst_trade": g["net"].min(), "best_trade": g["net"].max(), "mean_worst_point": g["worst_point"].mean(),
                     "net_per_year": g["net"].sum() / years if years else np.nan, "equity_drawdown": _dd(g.sort_values("exit_date")["net"]),
                     "weeks_in_market_share": g["weeks"].sum() / len(r) if name != "ALL PAIRS" else np.nan,
                     "verdict": _verdict(n, g["net"].mean(), float((g["net"] > 0).mean()), t)})
    return pd.DataFrame(rows).set_index("pair")


def equity(tr: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    """Weekly cumulative net P&L per pair (spread earned while in a trade; the cost charged on the exit bar)."""
    out = pd.DataFrame(index=r.index)
    for pair, g in tr.groupby("pair"):
        a, b = pair.split("|")
        spread = r[a] - r[b]
        pnl = pd.Series(0.0, index=r.index)
        for _, t in g.iterrows():
            i0, i1 = r.index.get_loc(t["entry_date"]) + 1, r.index.get_loc(t["exit_date"])
            pnl.iloc[i0:i1 + 1] += t["side"] * spread.iloc[i0:i1 + 1].fillna(0.0)
            pnl.iloc[i1] -= t["cost"]
        out[pair] = pnl.cumsum()
    out["ALL PAIRS"] = out.sum(axis=1)
    return out


def sensitivity(Z: pd.DataFrame, r: pd.DataFrame, st: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for entry in C.PT_SENSITIVITY["entry"]:
        for exit_ in C.PT_SENSITIVITY["exit"]:
            for req in C.PT_SENSITIVITY["require_buy"]:
                tr = all_trades(Z, r, st=st, entry=entry, exit_=exit_, require_buy=req)
                if not len(tr):
                    rows.append({"entry": entry, "exit": exit_, "require_cheap_leg_BUY": req, "trades": 0})
                    continue
                n = len(tr)
                rows.append({"entry": entry, "exit": exit_, "require_cheap_leg_BUY": req, "trades": n, "win_rate_net": float((tr["net"] > 0).mean()),
                             "mean_net": tr["net"].mean(), "total_net": tr["net"].sum(), "avg_weeks": tr["weeks"].mean(),
                             "t_mean_net": tr["net"].mean() / (tr["net"].std(ddof=1) / np.sqrt(n)) if n >= C.PT_MIN_TRADES else np.nan,
                             "pairs_positive": int((tr.groupby("pair")["net"].sum() > 0).sum()), "pairs_traded": int(tr["pair"].nunique())})
    return pd.DataFrame(rows)
