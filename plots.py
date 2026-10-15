"""Charts, read from outputs/tables/. Same design system as the sibling repos."""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd

import config as C

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COL = {"N100": "#2a78d6", "MC100": "#eb6834", "SC100": "#1baf7a", "Micro250": "#4a3aa7", "GoldMCX": "#eda100", "VIX": "#e34948"}
BUY, SELL = "#2a78d6", "#e34948"
plt.rcParams.update({
    "font.family": ["Inter", "Helvetica", "Arial", "DejaVu Sans", "sans-serif"], "font.size": 9.5,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "savefig.dpi": C.CHART_DPI,
    "legend.frameon": False, "legend.fontsize": 8.5, "text.color": INK, "lines.linewidth": 2.0,
})
T = C.TABLES
PCT = mt.PercentFormatter(1.0, decimals=0)


def _csv(name, **kw):
    return pd.read_csv(T / f"{name}.csv", **kw)


def _save(fig, name, note=None):
    C.CHARTS.mkdir(parents=True, exist_ok=True)
    if json.loads((T / "meta.json").read_text()).get("synthetic"):
        fig.text(0.5, 0.5, "SYNTHETIC DEMO DATA", ha="center", va="center", fontsize=40, color="#e34948", alpha=0.18, rotation=25, zorder=100)
    if note:
        fig.text(0.01, 0.005, note, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    fig.savefig(C.CHARTS / f"{name}.png", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _end(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(4, dy), textcoords="offset points", va="center", ha="left", fontsize=8.5, color=INK2)


# ---------------------------------------------------------------- 1 --------
def corr_by_state():
    bs = _csv("1_corr_by_signal_state")
    bs = bs[bs["state"].isin(["both BUY", "both SELL"])]
    pairs = [p for p in dict.fromkeys(bs["pair"])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, what, title in ((axes[0], "corr", "Correlation of the pair"), (axes[1], "cov_ann", "Covariance of the pair (annualised)")):
        y = np.arange(len(pairs))
        b = bs[bs["state"] == "both BUY"].set_index("pair").reindex(pairs)[what]
        s = bs[bs["state"] == "both SELL"].set_index("pair").reindex(pairs)[what]
        for i in range(len(pairs)):
            ax.plot([b.iloc[i], s.iloc[i]], [i, i], color=GRID, lw=3, zorder=1)
        ax.scatter(b, y, color=BUY, s=40, zorder=2, label="weeks when BOTH signals were BUY", edgecolor=SURFACE)
        ax.scatter(s, y, color=SELL, s=40, zorder=3, label="weeks when BOTH signals were SELL", edgecolor=SURFACE)
        ax.set_yticks(y); ax.set_yticklabels([p.replace("|", " × ") for p in pairs])
        ax.axvline(0, color=AXIS, lw=0.8); ax.set_title(title); ax.grid(axis="y", visible=False)
    axes[0].set_xlim(-0.4, 1.02); axes[0].legend(loc="lower right")
    _save(fig, "1_corr_by_state", "Covariance = correlation × vol(a) × vol(b). The right panel moves far more than the left: in SELL the pairs are a little more correlated and a lot more volatile.")


def corr_around_flip():
    af = _csv("1_corr_around_flip"); af = af[af["ref"] == C.BENCHMARK]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for ax, ev, col in ((axes[0], "flip to SELL", SELL), (axes[1], "flip to BUY", BUY)):
        sub = af[af["event"] == ev]
        ax.plot(sub["weeks_from_flip"], sub["mean_corr"], color=col, marker="o", ms=4)
        ax.axvline(0, color=AXIS, lw=1, ls="--")
        ax.set_xlabel(f"weeks from N100's {ev}   (n = {int(sub['n_events'].max())} flips)")
        ax.set_title(f"Average correlation among the four indices around N100's {ev}")
        ax.set_ylim(0.7, 0.95)
    _save(fig, "1_corr_around_flip", "13-week rolling correlation. If the line rises BEFORE week 0, the market couples up ahead of the signal; if after, the signal fires first.")


def corr_timeseries():
    ts = _csv("1_corr_timeseries", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(ts.index, ts["mean_corr_13w"], color=CAT[0], lw=1.6)
    sell = ts["N100_state"] == -1
    ax.fill_between(ts.index, 0, 1, where=sell, color=SELL, alpha=0.08, lw=0, transform=ax.get_xaxis_transform())
    ax.set_ylim(0.3, 1.0); ax.set_title("13-week average correlation among the four indices — shaded where N100's signal was in SELL")
    ax2 = axes[1]
    ax2.fill_between(ts.index, 0, ts["n_sell"], step="mid", color=SELL, alpha=0.35, lw=0)
    ax2.plot(ts.index, ts["n_live"], color=MUTED, lw=1, ls="--", label="signals that exist")
    ax2.set_title("How many of the four index signals were in SELL"); ax2.set_ylim(0, 4.3); ax2.legend(loc="upper left")
    fig.tight_layout()
    _save(fig, "1_corr_timeseries")


# ---------------------------------------------------------------- 2 --------
def by_year():
    y = _csv("2_by_year", index_col=0)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    ax = axes[0]
    x = np.arange(len(y))
    ax.bar(x, y["mean_corr"], color=CAT[0], width=0.6)
    ax.vlines(x, y["min_corr"], y["max_corr"], color=INK2, lw=1.2)
    for i, v in enumerate(y["mean_corr"]):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_ylim(0.3, 1.05); ax.set_title("Average correlation among the four indices by year (bar) and its low–high range within the year (line)")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    cols = [c for c in ("ret_N100", "ret_MC100", "ret_SC100", "ret_Micro250", "ret_GoldMCX") if c in y.columns]
    w = 0.8 / len(cols)
    for k, c in enumerate(cols):
        lab = c.replace("ret_", "")
        ax.bar(x + (k - len(cols) / 2 + 0.5) * w, y[c], w, color=COL[lab], label=lab)
    ax.set_xticks(x); ax.set_xticklabels(y.index.astype(str))
    ax.yaxis.set_major_formatter(PCT); ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title("Return of each asset in that year"); ax.legend(loc="upper left", ncol=5); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "2_by_year", "The last year is partial. Returns are price-index returns (no dividends).")


def quintiles():
    lg = _csv("2_corr_quintile_returns_long")
    buckets = [f"Q{i + 1}" for i in range(C.N_BUCKETS)]
    assets = [a for a in C.ASSETS if a in set(lg["asset"])]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, what, title in ((axes[0], "same_quarter_mean", "Return over the SAME quarter the correlation was measured on"),
                            (axes[1], "next_quarter_mean", "Return over the NEXT quarter")):
        x = np.arange(len(buckets)); w = 0.8 / len(assets)
        for k, a in enumerate(assets):
            s = lg[lg["asset"] == a].set_index("bucket")[what].reindex(buckets)
            ax.bar(x + (k - len(assets) / 2 + 0.5) * w, s.values, w, color=COL[a], label=a)
        rng = lg[lg["asset"] == assets[0]].set_index("bucket")
        ax.set_xticks(x); ax.set_xticklabels([f"{b}\ncorr {rng.loc[b, 'corr_lo']:.2f}–{rng.loc[b, 'corr_hi']:.2f}" for b in buckets], fontsize=8)
        ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=1)); ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(title); ax.grid(axis="x", visible=False)
    axes[0].legend(loc="upper right", ncol=5)
    fig.suptitle("Weeks sorted by the trailing 13-week correlation among the indices, calm (Q1) → tightly coupled (Q5)", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "2_corr_quintiles", "Left answers 'what was happening while correlation was high'; right answers 'what came after'.")


def timeline():
    tl = _csv("2_timeline", index_col=0, parse_dates=True)
    n = 3 if "VIX" in tl.columns else 2
    fig, axes = plt.subplots(n, 1, figsize=(13, 2.6 * n + 1), sharex=True)
    ax = axes[0]
    ax.plot(tl.index, tl["mean_corr_13w"], color=CAT[0], lw=1.6)
    ax.axhline(0.9, color=SELL, lw=0.8, ls="--"); ax.set_ylim(0.3, 1.0)
    ax.set_title("13-week average correlation among the four indices (dashed = 0.90)")
    ax = axes[1]
    ax.plot(tl.index, tl["N100_ret_13w"], color=COL["N100"], lw=1.4, label="N100")
    ax.plot(tl.index, tl["SC100_ret_13w"], color=COL["SC100"], lw=1.4, label="SC100")
    ax.axhline(0, color=AXIS, lw=0.8); ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Trailing 13-week return"); ax.legend(loc="upper left", ncol=2)
    if n == 3:
        ax = axes[2]
        ax.fill_between(tl.index, 0, tl["VIX"], color=COL["VIX"], alpha=0.2, lw=0); ax.plot(tl.index, tl["VIX"], color=COL["VIX"], lw=1.2)
        ax.set_title("India VIX level")
    fig.tight_layout()
    _save(fig, "2_timeline")


# ---------------------------------------------------------------- 3 --------
def breaks_leads():
    sm = _csv("4_summary", index_col=0)
    assets = [a for a in C.ASSETS if a in sm.index]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, col, title in ((axes[0], "signal_sell", "Supertrend flips to SELL"), (axes[1], "cross_10pct", "Price crosses −10% from its own high"),
                           (axes[2], "recover_buy", "Supertrend flips back to BUY after the trough")):
        y = np.arange(len(assets))
        v = sm.loc[assets, f"first_share_{col}"]
        ax.barh(y, v, color=[COL[a] for a in assets], height=0.6)
        for i, a in enumerate(assets):
            ax.text(v.iloc[i] + 0.02, i, f"avg rank {sm.loc[a, f'avg_rank_{col}']:.1f} · median lead {sm.loc[a, f'median_lead_{col}_weeks']:+.0f} wk · n {int(sm.loc[a, f'n_{col}'])}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels(assets); ax.invert_yaxis()
        ax.xaxis.set_major_formatter(PCT); ax.set_xlim(0, 1.45); ax.set_title(f"{title}: share of sell-offs it was FIRST")
        ax.grid(axis="y", visible=False)
    _save(fig, "4_who_breaks_first", "Lead = weeks relative to N100's own date (negative = earlier). Ties share first place. Micro250 exists only from 2022.")


def breaks_episodes():
    ep = _csv("4_price_episodes", parse_dates=["peak_date", "trough_date", "end_date"])
    ld = _csv("4_dates_and_leads", parse_dates=["peak", "signal_sell", "cross_10pct", "recover_buy"])
    px = _csv("px_weekly", index_col=0, parse_dates=True)
    n = len(ep)
    fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(15, 3.6 * int(np.ceil(n / 3))))
    axes = axes.ravel()
    for ax, (_, e) in zip(axes, ep.iterrows()):
        i0 = max(0, px.index.get_loc(e["peak_date"]) - 8)
        i1 = min(len(px) - 1, px.index.get_loc(e["trough_date"]) + 16)
        seg = px.iloc[i0:i1 + 1]
        for a in [x for x in C.ASSETS if x in seg.columns and x != "GoldMCX"]:
            s = seg[a].dropna()
            if len(s) < 4:
                continue
            base = s.loc[:e["peak_date"]].max() if (s.index <= e["peak_date"]).any() else s.iloc[0]
            ax.plot(s.index, s / base - 1, color=COL[a], lw=1.5, label=a)
            d = ld[(ld["episode"] == e["episode"]) & (ld["asset"] == a)]
            if len(d) and pd.notna(d["signal_sell"].iloc[0]) and d["signal_sell"].iloc[0] in s.index:
                dt = d["signal_sell"].iloc[0]
                ax.scatter([dt], [s.loc[dt] / base - 1], color=COL[a], marker="v", s=55, zorder=5, edgecolor=SURFACE)
        ax.axhline(-C.DD_CROSS, color=SELL, lw=0.8, ls="--")
        ax.axvline(e["trough_date"], color=AXIS, lw=0.8, ls=":")
        ax.yaxis.set_major_formatter(PCT)
        ax.set_title(f"{e['episode']}: N100 {e['depth']:.0%} over {int(e['weeks_peak_to_trough'])} wk")
        ax.tick_params(axis="x", labelrotation=30, labelsize=7.5)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(loc="lower left", ncol=2, fontsize=8)
    fig.suptitle("Each sell-off: every index from N100's peak, ▼ = the week its supertrend flipped to SELL, dashed = −10%, dotted = N100's trough", x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "4_price_episodes")


# ------------------------------------------------ 3. correlation breaks --
def cb_series():
    S = _csv("3_corr_series", index_col=0, parse_dates=True)
    ep = _csv("3_corr_episodes", parse_dates=["start", "end", "G_min_date"])
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.3, 1]})
    ax = axes[0]
    for a in [x for x in C.EQUITY if f"c_{x}" in S.columns]:
        ax.plot(S.index, S[f"c_{a}"], color=COL[a], lw=1.3, label=a)
    ax.plot(S.index, S["G"], color=INK, lw=2.0, label="group (all pairs)")
    for _, e in ep.iterrows():
        ax.axvspan(e["start"], e["end"], color=SELL, alpha=0.10, lw=0)
    ax.set_ylim(0, 1.02); ax.legend(loc="lower left", ncol=5)
    ax.set_title("Each index's 13-week correlation with the other indices, and the group average — shaded: decorrelation episodes")
    ax = axes[1]
    for a in [x for x in C.EQUITY if f"z_c_{x}" in S.columns]:
        ax.plot(S.index, S[f"z_c_{a}"], color=COL[a], lw=1.1)
    ax.plot(S.index, S["z_G"], color=INK, lw=1.8)
    ax.axhline(C.CB_Z, color=SELL, lw=1, ls="--"); ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_ylim(-4, 3); ax.set_title(f"The same, as z-scores against each series' own {C.CB_NORM}-week norm — dashed: the break line (z = {C.CB_Z:.0f})")
    ax = axes[2]
    if f"c_{C.GOLD}" in S.columns:
        ax.plot(S.index, S[f"c_{C.GOLD}"], color=COL["GoldMCX"], lw=1.4)
        ax.axhline(0, color=AXIS, lw=0.8); ax.set_ylim(-0.6, 0.6)
        ax.set_title("Gold's 13-week correlation with the four indices (for reference; not ranked)")
    fig.tight_layout()
    _save(fig, "3_corr_series")


def cb_episodes():
    S = _csv("3_corr_series", index_col=0, parse_dates=True)
    ep = _csv("3_corr_episodes", parse_dates=["start", "end", "G_min_date"])
    ab = _csv("3_asset_breaks", parse_dates=["break_date"])
    n = len(ep)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows))
    axes = axes.ravel()
    for ax, (_, e) in zip(axes, ep.iterrows()):
        i = S.index.get_loc(e["start"])
        seg = S.iloc[max(0, i - 12):min(len(S), i + 13)]
        k = np.arange(len(seg)) - (i - max(0, i - 12))
        for a in [x for x in C.EQUITY if f"z_c_{x}" in S.columns]:
            ax.plot(k, seg[f"z_c_{a}"], color=COL[a], lw=1.3, label=a)
            b = ab[(ab["episode"] == e["episode"]) & (ab["asset"] == a)]
            if len(b) and pd.notna(b["break_date"].iloc[0]) and b["break_date"].iloc[0] in seg.index:
                pos = seg.index.get_loc(b["break_date"].iloc[0])
                ax.scatter([pos - (i - max(0, i - 12))], [seg[f"z_c_{a}"].iloc[pos]], color=COL[a], s=50, marker="v", zorder=5, edgecolor=SURFACE)
        ax.plot(k, seg["z_G"], color=INK, lw=1.8, label="group")
        ax.axhline(C.CB_Z, color=SELL, lw=0.9, ls="--"); ax.axvline(0, color=AXIS, lw=0.8, ls=":")
        ax.set_ylim(-4, 2.5); ax.set_title(f"{e['episode']}: group {e['G_before']:.2f} → {e['G_min']:.2f}", fontsize=9.5)
        ax.set_xlabel("weeks from the group break", fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    axes[0].legend(loc="lower left", ncol=3, fontsize=7.5)
    fig.suptitle("Every decorrelation episode: each index's z-score in the 12 weeks before and after the group break (▼ = its own break week)", x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "3_corr_episodes")


def cb_summary():
    sm = _csv("3_summary", index_col=0)
    assets = list(sm.index)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.4))
    specs = (("share_first", "Earliest by date (ties count)", "share of episodes first"), ("share_deepest", "Deepest below its norm when the market broke", "share of episodes"),
             ("share_biggest_drop", "Biggest fall in its own correlation", "share of episodes"), ("share_up_breaks", "Broke by OUTperforming (up-break)", "share of its breaks"))
    for ax, (col, title, xl) in zip(axes, specs):
        y = np.arange(len(assets))
        ax.barh(y, sm[col], color=[COL[a] for a in assets], height=0.6)
        for i, a in enumerate(assets):
            extra = f"avg rank {sm.loc[a, 'avg_rank']:.1f}" if col == "share_first" else (f"avg rank {sm.loc[a, 'avg_rank_deepest']:.1f}" if col == "share_deepest" else (f"avg rank {sm.loc[a, 'avg_rank_drop']:.1f}" if col == "share_biggest_drop" else f"n {int(sm.loc[a, 'episodes_broke'])}"))
            ax.text(sm.loc[a, col] + 0.02, i, extra, va="center", fontsize=8, color=INK2)
        ax.set_yticks(y); ax.set_yticklabels(assets); ax.invert_yaxis(); ax.set_xlim(0, 1.35)
        ax.xaxis.set_major_formatter(PCT); ax.set_title(title, fontsize=10); ax.set_xlabel(xl); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "3_who_breaks_correlation", "Micro250 exists only from 2022 and is present in fewer episodes.")


def cb_after():
    afs = _csv("3_after_break_summary")
    fig, ax = plt.subplots(figsize=(11, 4.6))
    cats = [(d, h) for d in afs["direction"].unique() for h in sorted(afs["weeks_ahead"].unique())]
    x = np.arange(len(cats)); w = 0.26
    for k, (col, lab, c) in enumerate((("breaker", "the asset that broke", CAT[0]), ("rest_of_group", "the other indices", MUTED), ("gold", "gold", COL["GoldMCX"]))):
        vals = [afs[(afs["direction"] == d) & (afs["weeks_ahead"] == h)][col].iloc[0] for d, h in cats]
        ax.bar(x + (k - 1) * w, vals, w, color=c, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"{d.split(' (')[0]}\nnext {h} wk (n {int(afs[(afs['direction'] == d) & (afs['weeks_ahead'] == h)]['n'].iloc[0])})" for d, h in cats], fontsize=8.5)
    ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=1)); ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title("After an index breaks away from the group: what it, the others and gold returned next"); ax.legend(loc="upper left", ncol=3); ax.grid(axis="x", visible=False)
    _save(fig, "3_after_break", "Up-break = it decoupled by outperforming over the prior 13 weeks; down-break = by underperforming.")


# ---------------------------------------------------------- 5. volatility --
def _label_lines(ax, ys: dict, x, dy_pts: float = 9):
    """Direct labels at the right end of several lines, pushed apart so they do not overlap."""
    items = sorted(((v, k) for k, v in ys.items() if pd.notna(v)), key=lambda t: t[0])
    ymin, ymax = ax.get_ylim()
    px_per_unit = ax.get_window_extent().height / (ymax - ymin) if ymax > ymin else 1
    min_gap = dy_pts / 72 * ax.figure.dpi / px_per_unit
    placed = []
    for v, k in items:
        y = v if not placed else max(v, placed[-1] + min_gap)
        placed.append(y)
        ax.annotate(k, (x, v), xytext=(5, (y - v) * px_per_unit * 72 / ax.figure.dpi), textcoords="offset points", va="center", ha="left", fontsize=8.5, color=INK2)


def vol_timeseries():
    ts = _csv("5_vol_timeseries", index_col=0, parse_dates=True)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
    ax = axes[0]
    for a in [x for x in C.ASSETS if f"vol_{x}" in ts.columns]:
        ax.plot(ts.index, ts[f"vol_{a}"], color=COL[a], lw=1.3 if a != "GoldMCX" else 1.1, ls="-" if a != "GoldMCX" else "--", label=a)
    sell = ts["N100_state"] == -1
    ax.fill_between(ts.index, 0, 1, where=sell, color=SELL, alpha=0.08, lw=0, transform=ax.get_xaxis_transform())
    ax.yaxis.set_major_formatter(PCT); ax.set_ylim(0, min(0.9, float(np.nanmax(ts.filter(like="vol_").values)) * 1.05))
    ax.set_title(f"Each asset's {C.VOL_WINDOW}-week realised volatility (annualised) — shaded where N100's signal was in SELL"); ax.legend(loc="upper right", ncol=5)
    ax = axes[1]
    ax.plot(ts.index, ts["G"], color=INK, lw=1.6); ax.set_ylim(0.3, 1.0)
    ax.set_title("The group correlation G (the four indices' average 13-week pairwise correlation), for comparison")
    fig.tight_layout()
    _save(fig, "5_vol_timeseries", "Volatility spikes and correlation spikes line up: both are features of the same sell-off weeks.")


def vol_by_state():
    vs = _csv("5_vol_by_state", index_col=0)
    assets = [a for a in C.ASSETS if a in vs.index]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    y = np.arange(len(assets))
    for i, a in enumerate(assets):
        ax.plot([vs.loc[a, "vol_BUY"], vs.loc[a, "vol_SELL"]], [i, i], color=GRID, lw=3, zorder=1)
    ax.scatter(vs.loc[assets, "vol_BUY"], y, color=BUY, s=48, zorder=2, label="in its own BUY weeks", edgecolor=SURFACE)
    ax.scatter(vs.loc[assets, "vol_SELL"], y, color=SELL, s=48, zorder=3, label="in its own SELL weeks", edgecolor=SURFACE)
    for i, a in enumerate(assets):
        ax.text(max(vs.loc[a, "vol_BUY"], vs.loc[a, "vol_SELL"]) + 0.012, i, f"×{vs.loc[a, 'vol_ratio_SELL_over_BUY']:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(assets); ax.invert_yaxis(); ax.xaxis.set_major_formatter(PCT); ax.set_xlim(0, 0.36)
    ax.set_title("Realised volatility by the asset's own signal state"); ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    ax = axes[1]
    w = 0.38
    ax.bar(y - w / 2, vs.loc[assets, "mean_weekly_ret_BUY"], w, color=BUY, label="BUY weeks")
    ax.bar(y + w / 2, vs.loc[assets, "mean_weekly_ret_SELL"], w, color=SELL, label="SELL weeks")
    ax.set_xticks(y); ax.set_xticklabels(assets); ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=1)); ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_title("Average weekly return in each state"); ax.legend(loc="upper right"); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "5_vol_by_state", "Left: SELL weeks are not just falling weeks, they are bigger-swing weeks. Gold's swings do not change with its signal.")


def vol_around_flip():
    af = _csv("5_vol_around_flip"); af = af[af["ref"] == C.BENCHMARK]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, ev in ((axes[0], "flip to SELL"), (axes[1], "flip to BUY")):
        sub = af[af["event"] == ev]
        for a in [x for x in C.EQUITY if f"vol_{x}" in sub.columns]:
            ax.plot(sub["weeks_from_flip"], sub[f"vol_{a}"], color=COL[a], marker="o", ms=3.5, lw=1.6, label=a)
        ax.axvline(0, color=AXIS, lw=1, ls="--"); ax.yaxis.set_major_formatter(PCT)
        ax.set_xlabel(f"weeks from N100's {ev}   (n = {int(sub['n_events'].max())} flips)")
        ax.set_title(f"Each index's {C.VOL_WINDOW}-week volatility around N100's {ev}")
    axes[0].legend(loc="upper left", ncol=4)
    fig.tight_layout()
    _save(fig, "5_vol_around_flip", "Volatility is already rising before the SELL flip (the flip is caused by a big down move) and keeps rising for weeks after it.")


def corr_by_vol_regime():
    cv = _csv("5_corr_by_vol_regime", index_col=0)
    edges = _csv("5_vol_regime_edges", index_col=0)
    qs = [c for c in cv.columns if c.startswith("Q") and "_" not in c]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(qs))
    ends = {}
    for pair in cv.index:
        a, b = pair.split("|")
        gold = "Gold" in pair
        col = COL[a] if not gold else COL[b if a == "GoldMCX" else a]
        ax.plot(x, cv.loc[pair, qs], color=col, lw=1.1 if gold else 1.8, ls="--" if gold else "-", marker="o", ms=3.5, alpha=0.9)
        ends[pair.replace("|", " × ")] = cv.loc[pair, qs[-1]]
    ax.set_xticks(x); ax.set_xticklabels([f"{q}\nN100 vol {edges.loc[q, 'min']:.0%}–{edges.loc[q, 'max']:.0%}" for q in qs], fontsize=8.5)
    ax.set_ylim(-0.2, 1.02); ax.axhline(0, color=AXIS, lw=0.8); ax.set_xlim(-0.3, len(qs) + 1.6)
    ax.set_title("Each pair's 13-week correlation by the volatility regime of that week (quintiles of N100's 13-week vol)")
    _label_lines(ax, ends, x[-1])
    _save(fig, "5_corr_by_vol_regime", "Solid: index pairs. Dashed: gold pairs. Index pairs climb from left to right; gold stays near zero in every regime.")


def vol_corr_lead():
    vl = _csv("5_vol_corr_lead", index_col=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    cols = [SELL if k == 0 else CAT[0] for k in vl.index]
    ax.bar(vl.index, vl["corr_dvol_t_dcorr_t_plus_k"], color=cols, width=0.7)
    for k, v in vl["corr_dvol_t_dcorr_t_plus_k"].items():
        ax.text(k, v + 0.01, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
    ax.set_xlabel("k (weeks): correlation between the change in N100's vol at week t and the change in the group correlation at week t + k")
    ax.set_ylim(0, max(0.65, vl["corr_dvol_t_dcorr_t_plus_k"].max() * 1.2)); ax.set_title("Does volatility move before correlation, or with it?"); ax.grid(axis="x", visible=False)
    _save(fig, "5_vol_corr_lead", "A peak at k = 0 means they move together; a peak at k > 0 would mean vol leads; at k < 0, correlation leads.")


# ------------------------------------------------------------ 6. pairs --
def pair_cards():
    Z = _csv("6_ratio_z", index_col=0, parse_dates=True)
    B = _csv("6_ratio_bands", index_col=0, parse_dates=True)
    PC = _csv("6_pair_corr13", index_col=0, parse_dates=True)
    V = _csv("5_vol_rolling", index_col=0, parse_dates=True)
    st = _csv("sig_states_weekly", index_col=0, parse_dates=True)
    tr = _csv("7_trades", parse_dates=["entry_date", "exit_date"]) if (T / "7_trades.csv").exists() and (T / "7_trades.csv").stat().st_size > 10 else pd.DataFrame()
    for pair in Z.columns:
        a, b = pair.split("|")
        fig, axes = plt.subplots(4, 1, figsize=(13, 12.5), sharex=True, gridspec_kw={"height_ratios": [1.4, 1.6, 1, 1]})
        lr, m, s = B[f"{pair}__lr"], B[f"{pair}__mean"], B[f"{pair}__sd"]
        ax = axes[0]
        ax.fill_between(lr.index, m - 2 * s, m + 2 * s, color=GRID, alpha=0.5, lw=0, label="±2 sd")
        ax.fill_between(lr.index, m - s, m + s, color=AXIS, alpha=0.45, lw=0, label="±1 sd")
        ax.plot(lr.index, m, color=MUTED, lw=1, ls="--", label=f"{C.PAIR_WINDOW}-week mean")
        ax.plot(lr.index, lr, color=INK, lw=1.5, label=f"ln({a} / {b})")
        ax.set_title(f"{a} × {b} — 1. the price ratio ln({a}/{b}) with its trailing {C.PAIR_WINDOW}-week mean and bands (up = {a} doing better than {b})")
        ax.legend(loc="upper left", ncol=4)
        ax = axes[1]
        z = Z[pair]
        ax.plot(z.index, z, color=CAT[0], lw=1.3)
        for lvl in C.PAIR_Z_LINES:
            ax.axhline(lvl, color=SELL if lvl >= 2 else AXIS, lw=0.9, ls="--"); ax.axhline(-lvl, color=SELL if lvl >= 2 else AXIS, lw=0.9, ls="--")
        ax.axhline(0, color=AXIS, lw=0.8)
        for leg, mark_y in ((a, 3.4), (b, -3.4)):
            if leg in st.columns:
                sl = st[leg].reindex(z.index)
                flips = z.index[((sl == -1) & (sl.shift(1) == 1)).fillna(False)]
                ax.scatter(flips, [mark_y] * len(flips), marker="v" if mark_y > 0 else "^", color=COL[leg], s=42, zorder=5, edgecolor=SURFACE, label=f"{leg} flips to SELL")
        if len(tr):
            g = tr[tr["pair"] == pair]
            ax.scatter(g["entry_date"], g["z_entry"], marker="o", color=INK, s=34, zorder=6, label="pair-trade entry")
            ax.scatter(g["exit_date"], g["z_exit"], marker="o", facecolor=SURFACE, edgecolor=INK, s=34, zorder=6, label="exit")
        ax.set_ylim(-4.7, 3.9); ax.set_yticks([-3, -2, -1, 0, 1, 2, 3])
        ax.set_title(f"2. the deviation z-score of that ratio (dashed: ±1 and ±2 sd) · ▼ {a} SELL flip (top row) · ▲ {b} SELL flip (bottom row) · ● pair-trade entries, ○ exits")
        ax.legend(loc="lower center", ncol=4, fontsize=7.5)
        ax = axes[2]
        ax.plot(PC.index, PC[pair], color=CAT[0], lw=1.3); ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_ylim(-0.6 if "Gold" in pair else 0.0, 1.02); ax.set_title(f"3. the pair's {C.CORR_WINDOW}-week correlation")
        ax = axes[3]
        for leg in (a, b):
            if f"{leg}" in V.columns:
                ax.plot(V.index, V[leg], color=COL[leg], lw=1.3, label=leg)
        ax.yaxis.set_major_formatter(PCT); ax.set_title(f"4. each leg's {C.VOL_WINDOW}-week volatility (annualised)"); ax.legend(loc="upper right", ncol=2)
        fig.tight_layout()
        _save(fig, f"6_pair_{a}_{b}")


def z_forward_chart():
    zf = _csv("6_z_forward")
    buckets = list(dict.fromkeys(zf["bucket"]))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, h in zip(axes, C.PAIR_FWD):
        pooled = zf[(zf["pair"] == "ALL PAIRS") & (zf["horizon_weeks"] == h)].set_index("bucket").reindex(buckets)
        x = np.arange(len(buckets))
        ax.bar(x, pooled["mean_ratio_change"], color=[CAT[1] if i < 2 else (MUTED if i == 2 else CAT[0]) for i in range(len(buckets))], width=0.6, zorder=2)
        per = zf[(zf["pair"] != "ALL PAIRS") & (zf["horizon_weeks"] == h)]
        for i, bkt in enumerate(buckets):
            v = per[per["bucket"] == bkt]["mean_ratio_change"]
            ax.scatter(np.full(len(v), i) + np.linspace(-0.18, 0.18, len(v)), v, color=INK2, s=12, zorder=3, alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels([f"{b}\nt {pooled.loc[b, 'hac_t_ratio_change']:+.1f} · n {int(pooled.loc[b, 'weeks'])}" for b in buckets], fontsize=8.5); ax.axhline(0, color=AXIS, lw=0.8)
        ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=1)); ax.set_title(f"Change in ln(a/b) over the next {h} weeks, by this week's deviation z"); ax.grid(axis="x", visible=False)
    fig.suptitle("If a deviation pulled back, the left bars would be positive and the right bars negative. Bars = all pairs pooled; dots = each pair.", x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "6_z_forward", "z < 0 means a has lagged b. A pull-back would then need the ratio to RISE (positive bar).")


def path_after_chart():
    pa = _csv("6_path_after_2sd")
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ends = {}
    for pair, g in pa.groupby("pair"):
        a, b = pair.split("|")
        gold = "Gold" in pair
        col = COL[a] if not gold else COL[b if a == "GoldMCX" else a]
        ax.plot(g["weeks_after"], g["mean_z"], color=col, lw=1.1 if gold else 1.7, ls="--" if gold else "-")
        ends[f"{pair.replace('|', ' × ')} (n {int(g['n_events'].iloc[0])})"] = g["mean_z"].iloc[-1]
    ax.axhline(1, color=AXIS, lw=0.9, ls="--"); ax.axhline(0, color=AXIS, lw=0.8); ax.axhline(2, color=SELL, lw=0.9, ls="--")
    ax.set_xlabel("weeks after |z| first reached 2"); ax.set_ylim(-0.5, 2.8); ax.set_xlim(0, C.PAIR_PATH_WEEKS + 5.5)
    ax.set_title("The average z-score after a 2-sd deviation, sign-flipped so every event starts near +2")
    _label_lines(ax, ends, C.PAIR_PATH_WEEKS)
    _save(fig, "6_path_after_2sd", "The z-score always drifts back toward 0 as the 52-week window absorbs the move - this chart shows the SHAPE, the 6_z_forward chart shows whether the RATIO itself came back.")


# ------------------------------------------------------- 7. pair trade --
def pt_equity():
    eq = _csv("7_equity", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(13, 6))
    ends = {}
    for pair in [c for c in eq.columns if c != "ALL PAIRS"]:
        a, b = pair.split("|")
        gold = "Gold" in pair
        col = COL[a] if not gold else COL[b if a == "GoldMCX" else a]
        ax.plot(eq.index, eq[pair], color=col, lw=1.1 if gold else 1.5, ls="--" if gold else "-")
        ends[pair.replace("|", " × ")] = eq[pair].iloc[-1]
    ax.plot(eq.index, eq["ALL PAIRS"], color=INK, lw=2.2); ends["ALL PAIRS"] = eq["ALL PAIRS"].iloc[-1]
    ax.axhline(0, color=AXIS, lw=0.8); ax.yaxis.set_major_formatter(PCT)
    ax.set_title(f"Cumulative net P&L of the pair-trading rule, one unit per leg (enter |z| ≥ {C.PT_ENTRY:.0f}, exit |z| ≤ {C.PT_EXIT:.1f} or {C.PT_MAX_HOLD} weeks, cost {C.PT_COST_BPS} bp per leg per side)")
    _label_lines(ax, ends, eq.index[-1])
    _save(fig, "7_equity", "Log-return units: +0.10 = a unit long/short position gained 10 points of spread. Solid: index pairs. Dashed: gold pairs.")


def pt_summary():
    sm = _csv("7_summary", index_col=0)
    pairs_ = [p for p in sm.index if p != "ALL PAIRS"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    y = np.arange(len(pairs_))
    ax = axes[0]
    ax.barh(y, sm.loc[pairs_, "mean_net"], color=[BUY if v > 0 else SELL for v in sm.loc[pairs_, "mean_net"]], height=0.6)
    for i, p in enumerate(pairs_):
        ax.text(max(sm.loc[p, "mean_net"], 0) + 0.003, i, f"n {int(sm.loc[p, 'trades'])} · wins {sm.loc[p, 'win_rate_net']:.0%} · {sm.loc[p, 'avg_weeks']:.0f} wk", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([p.replace("|", " × ") for p in pairs_]); ax.invert_yaxis(); ax.axvline(0, color=AXIS, lw=0.8)
    ax.xaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=1)); ax.set_xlim(min(sm.loc[pairs_, "mean_net"].min() * 1.3, -0.02), 0.09); ax.set_title("Average net P&L per trade"); ax.grid(axis="y", visible=False)
    ax = axes[1]
    ax.barh(y - 0.2, sm.loc[pairs_, "mean_ret_cheap_leg"], 0.4, color=BUY, label="the cheap leg (the one the rule buys)")
    ax.barh(y + 0.2, sm.loc[pairs_, "mean_ret_rich_leg"], 0.4, color=MUTED, label="the rich leg (the one it sells)")
    ax.set_yticks(y); ax.set_yticklabels([""] * len(pairs_)); ax.invert_yaxis(); ax.axvline(0, color=AXIS, lw=0.8)
    ax.xaxis.set_major_formatter(PCT); ax.set_title("Average return of each leg over the trade (the long-only view)"); ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    _save(fig, "7_summary", "If the cheap leg's bar is shorter than the rich leg's, the lagging index kept lagging: holding the laggard instead of the leader lost money.")


# ------------------------------------------------------- 8. replacement --
def rp_rules():
    R = _csv("8_rules", index_col=0)
    rules_ = list(R.index)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    y = np.arange(len(rules_))
    for ax, h in zip(axes, C.RP_FWD):
        v = R[f"mean_{h}"]
        ax.barh(y, v, color=[COL["GoldMCX"] if r == "gold" else (MUTED if r in ("cash", "stay in the seller") else (BUY if x > R.loc["cash", f"mean_{h}"] else SELL)) for r, x in zip(rules_, v)], height=0.62)
        for i, r in enumerate(rules_):
            ax.text(max(v.iloc[i], 0) + 0.003, i, f"beat cash {R.loc[r, f'beat_cash_{h}']:.0%} · had a pick {R.loc[r, 'share_with_a_pick']:.0%}" + (f" · t {R.loc[r, f'hac_t_minus_cash_{h}']:+.1f}" if pd.notna(R.loc[r, f"hac_t_minus_cash_{h}"]) else ""), va="center", fontsize=7.8, color=INK2)
        ax.axvline(R.loc["cash", f"mean_{h}"], color=INK, lw=0.9, ls="--")
        ax.set_yticks(y); ax.set_yticklabels(rules_, fontsize=9); ax.invert_yaxis(); ax.axvline(0, color=AXIS, lw=0.8)
        ax.xaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=1)); ax.set_xlim(min(v.min() * 1.2, -0.01), max(v.max() * 1.9, 0.09))
        ax.set_title(f"Average return over the next {h} weeks"); ax.grid(axis="y", visible=False)
    fig.suptitle(f"What each replacement rule earned after a SELL flip (n = {int(R['events'].iloc[0])} flips of the five assets)", x=0.01, ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "8_rules", "Dashed line = cash. Every rule is applied to the same SELL flips; a rule with no candidate that week falls back to cash, so 'had a pick' says how often it actually did something.")


def rp_cells():
    ce = _csv("8_cells")
    h = C.RP_FWD[-1]
    sub = ce[(ce["horizon_weeks"] == h) & (ce["correlated"] == "any") & (ce["cand_state"].isin(["BUY", "SELL"])) & (ce["deviation"].isin(["lagging", "neutral", "leading"]))]
    sub2 = ce[(ce["horizon_weeks"] == h) & (ce["correlated"].isin(["True", "False"])) & (ce["cand_state"] == "any") & (ce["deviation"].isin(["lagging", "neutral", "leading"]))]
    devs = ["lagging", "neutral", "leading"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, (frame, key, vals, cols, title) in zip(axes, ((sub, "cand_state", ["BUY", "SELL"], [BUY, SELL], "by the candidate's OWN signal"), (sub2, "correlated", ["True", "False"], [CAT[0], MUTED], "by whether it was correlated with the seller (52w corr ≥ 0.5)"))):
        x = np.arange(len(devs)); w = 0.36
        for k, (val, col) in enumerate(zip(vals, cols)):
            s = frame[frame[key] == val].set_index("deviation").reindex(devs)
            lab = {"True": "correlated", "False": "not correlated"}.get(val, f"candidate in {val}")
            ax.bar(x + (k - 0.5) * w, s["cand_mean"], w, color=col, label=lab)
            for i, d in enumerate(devs):
                if pd.notna(s.loc[d, "cand_mean"]):
                    ax.text(i + (k - 0.5) * w, max(s.loc[d, "cand_mean"], 0) + 0.002, f"n {int(s.loc[d, 'rows'])}", ha="center", fontsize=8, color=INK2)
        allrow = ce[(ce["horizon_weeks"] == h) & (ce["deviation"] == "any") & (ce["correlated"] == "any") & (ce["cand_state"] == "any")].iloc[0]
        ax.axhline(allrow["cash_mean"], color=INK, lw=0.9, ls="--"); ax.text(len(devs) - 0.55, allrow["cash_mean"] + 0.002, "cash", fontsize=8, color=INK2)
        ax.axhline(allrow["stay_mean"], color=SELL, lw=0.9, ls=":"); ax.text(len(devs) - 0.55, allrow["stay_mean"] - 0.007, "stay in seller", fontsize=8, color=SELL)
        ax.set_xticks(x); ax.set_xticklabels([f"candidate {d}\n(z vs seller {'≤ −1' if d == 'lagging' else ('≥ +1' if d == 'leading' else 'between')})" for d in devs], fontsize=8.5)
        ax.yaxis.set_major_formatter(mt.PercentFormatter(1.0, decimals=1)); ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_title(f"Candidate's return over the next {h} weeks, {title}"); ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, "8_cells", "One bar = all (SELL flip, candidate) pairs in that cell. n counts candidate rows, not distinct flips.")


def rp_z_path():
    zp = _csv("8_z_path")
    sellers = [a for a in C.ASSETS if a in set(zp["seller"])]
    fig, axes = plt.subplots(1, len(sellers), figsize=(3.4 * len(sellers) + 1, 4.4), sharey=True)
    for ax, a in zip(np.atleast_1d(axes), sellers):
        sub = zp[zp["seller"] == a]
        for b, g in sub.groupby("candidate"):
            if b == "all candidates":
                ax.plot(g["k"], g["mean_z"], color=INK, lw=2.2, label="all candidates")
            else:
                ax.plot(g["k"], g["mean_z"], color=COL[b], lw=1.2, label=b)
        ax.axvline(0, color=AXIS, lw=1, ls="--"); ax.axhline(0, color=AXIS, lw=0.8)
        n = int(sub[sub["candidate"] == "all candidates"]["n"].max()) if (sub["candidate"] == "all candidates").any() else 0
        ax.set_title(f"{a} flips to SELL (n {n // max(1, len(set(sub['candidate'])) - 1)})", fontsize=10); ax.set_xlabel("weeks from the flip", fontsize=8.5)
        ax.set_ylim(-1.6, 1.6); ax.legend(fontsize=7, loc="lower left")
    np.atleast_1d(axes)[0].set_ylabel("z of the candidate vs the seller\n(+ = candidate ahead)")
    fig.suptitle("The deviation of every other asset against the seller, before and after the seller's SELL flip (average over flips)", x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    fig.tight_layout()
    _save(fig, "8_z_path", "A line that keeps rising after week 0 means the candidate kept pulling ahead of the seller after the seller's signal fired.")


def rp_event_grids():
    ep = _csv("8_event_paths", parse_dates=["date"])
    P = _csv("8_rule_picks", parse_dates=["date"])
    h = C.RP_FWD[-1]
    for a in [x for x in C.ASSETS if x in set(ep["seller"])]:
        sub = ep[ep["seller"] == a]
        dates = sorted(sub["date"].unique())
        n = len(dates)
        if n == 0:
            continue
        cols = 4; rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 3.1 * rows), sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, d in zip(axes, dates):
            g = sub[sub["date"] == d]
            for b, gg in g.groupby("candidate"):
                ax.plot(gg["k"], gg["z"], color=COL[b], lw=1.3, label=b)
            ax.axvline(0, color=AXIS, lw=1, ls="--"); ax.axhline(0, color=AXIS, lw=0.8)
            for lvl in (1, -1):
                ax.axhline(lvl, color=GRID, lw=0.8)
            pk = P[(P["seller"] == a) & (P["date"] == d)].set_index("rule")
            stay = pk.loc["stay in the seller", f"ret_{h}"] if "stay in the seller" in pk.index else np.nan
            cash = pk.loc["cash", f"ret_{h}"] if "cash" in pk.index else np.nan
            lead = pk.loc["most leading (any corr)"] if "most leading (any corr)" in pk.index else None
            ttl = f"{pd.Timestamp(d).date()}\nnext {h} wk: stay {stay:+.0%} · cash {cash:+.0%}" + (f" · most-leading pick {lead['pick']} {lead[f'ret_{h}']:+.0%}" if lead is not None and pd.notna(lead[f"ret_{h}"]) else "")
            ax.set_title(ttl, fontsize=7.8)
            ax.set_ylim(-3.5, 3.5); ax.tick_params(labelsize=7.5)
        for ax in axes[n:]:
            ax.axis("off")
        axes[0].legend(fontsize=7, loc="lower left", ncol=2)
        fig.suptitle(f"Every SELL flip of {a}: the z-score of each other asset against {a} from {C.RP_PATH_WEEKS} weeks before to {C.RP_PATH_WEEKS} after (+ = that asset ahead of {a})", x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
        fig.tight_layout()
        _save(fig, f"8_events_{a}")


def all_charts():
    corr_by_state(); corr_around_flip(); corr_timeseries(); by_year(); quintiles(); timeline()
    cb_series(); cb_episodes(); cb_summary(); cb_after(); breaks_leads(); breaks_episodes()
    vol_timeseries(); vol_by_state(); vol_around_flip(); corr_by_vol_regime(); vol_corr_lead()
    pair_cards(); z_forward_chart(); path_after_chart()
    pt_equity(); pt_summary()
    rp_rules(); rp_cells(); rp_z_path(); rp_event_grids()


if __name__ == "__main__":
    all_charts()
