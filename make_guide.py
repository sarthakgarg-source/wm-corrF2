"""
Build outputs/report_guide.html - the READING GUIDE to the report.

Part A  every question the study asks, with its answer and the chart that shows it
Part B  chart by chart, in the report's order: the chart itself, what every element on it is (axes, lines,
        markers, colours, panels), how it answers the question, and a reading FOR EACH COMBINATION
        (every pair, or every asset), with the numbers
Part C  a dictionary of every column name used in the report's tables

Every number is computed from outputs/tables/; the charts are the same figs.py figures as the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import figs

T = C.TABLES
PLOTLY_JS = Path(__import__("plotly").__file__).parent / "package_data" / "plotly.min.js"
H4, H13 = C.PAIR_FWD


def _csv(name, **kw):
    return pd.read_csv(T / f"{name}.csv", **kw)


def _fig(name, **kw):
    try:
        return f"<div class='fig'>{figs.html(name, **kw)}</div>"
    except Exception as e:  # noqa: BLE001
        return f"<p class='missing'>chart {name} failed: {e}</p>"


def _tbl(df, fmt="{:.2f}", index=True, pct_cols=(), int_cols=(), max_rows=80):
    d = df.head(max_rows).copy()
    for c in d.columns:
        if c in pct_cols:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v * 100:+.1f}%")
        elif c in int_cols:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    return d.to_html(index=index, border=0, classes="t", escape=False, na_rep="")


def _pct(v, d=1):
    return "n/a" if pd.isna(v) else f"{v * 100:+.{d}f}%"


def _p(v, d=0):
    return "n/a" if pd.isna(v) else f"{v * 100:.{d}f}%"


def _n(v, d=2):
    return "n/a" if pd.isna(v) else f"{v:.{d}f}"


def _t(v):
    return "n/a" if pd.isna(v) else f"{v:+.1f}"


def _x(p):
    return p.replace("|", " × ")


def _level(c):
    if pd.isna(c):
        return "no reading"
    if c >= 0.9:
        return "effectively one asset"
    if c >= 0.7:
        return "move together"
    if c >= 0.4:
        return "related, with a life of their own"
    if c > -0.2:
        return "unrelated"
    return "move opposite"


def box(kind, title, body):
    return f"<div class='{kind}'><b>{title}</b> {body}</div>"


def elements(items):
    """What is on the chart: a list of (element, meaning)."""
    return "<div class='els'><b>What is on the chart</b><ul>" + "".join(f"<li><b>{k}</b> — {v}</li>" for k, v in items) + "</ul></div>"


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def columns(items):
    return "<div class='cols'><b>The table's columns</b><ul>" + "".join(f"<li><code>{_esc(k)}</code> — {v}</li>" for k, v in items) + "</ul></div>"


def combos(title, rows):
    """Reading for each combination: rows = list of (label, text)."""
    return f"<div class='combo'><b>{title}</b><table class='t'>" + "".join(f"<tr><td><b>{a}</b></td><td>{b}</td></tr>" for a, b in rows) + "</table></div>"


# ======================================================================== build ==
def build() -> str:
    meta = json.loads((T / "meta.json").read_text())
    M = _csv("0_corr_matrix_full", index_col=0); M52 = _csv("0_corr_matrix_52w", index_col=0); M13 = _csv("0_corr_matrix_13w", index_col=0)
    pst = _csv("6_pair_stats", index_col=0); rt = _csv("6_pair_returns", index_col=0)
    bs = _csv("1_corr_by_signal_state"); dec = _csv("1_cov_decomposition").set_index("pair"); bb = _csv("1_corr_by_breadth", index_col=0)
    afa = _csv("1_corr_around_flip")
    vs = _csv("5_vol_by_state", index_col=0); vafa = _csv("5_vol_around_flip"); regl = _csv("5_corr_by_regime_long"); vl = _csv("5_vol_corr_lead", index_col=0)
    lg = _csv("2_corr_quintile_returns_long"); yr = _csv("2_by_year", index_col=0); hc = _csv("2_high_corr_episodes")
    cep = _csv("3_corr_episodes", parse_dates=["start", "end"]); sm = _csv("3_summary", index_col=0); afs = _csv("3_after_break_summary"); ab = _csv("3_asset_breaks")
    pep = _csv("4_price_episodes"); sm4 = _csv("4_summary", index_col=0)
    pco = _csv("6_pair_coint", index_col=0); zf = _csv("6_z_forward"); cur6 = _csv("6_current", index_col=0)
    pts = _csv("7_summary", index_col=0); tr = _csv("7_trades")
    ev = _csv("8_events"); cells = _csv("8_cells"); bp = _csv("8_by_pair"); R = _csv("8_rules", index_col=0); zp = _csv("8_z_path"); cur8 = _csv("8_current")
    S = _csv("3_corr_series", index_col=0, parse_dates=True)
    afp = _csv("1_corr_around_flip_pairs"); bpd = _csv("1_corr_by_breadth_pairs", index_col=0); qp = _csv("2_corr_quintile_returns_pairs")
    hps = _csv("2_high_corr_pairs_summary", index_col=0); pps = _csv("5_corr_rise_pairs_summary", index_col=0); vlp = _csv("5_vol_corr_lead_pairs")
    psm = _csv("3_pair_summary", index_col=0); pbk = _csv("3_pair_breaks"); own = _csv("5_corr_rise_own_summary", index_col=0)
    has_vix = "vix_mean_BUY" in vs.columns and vs["vix_mean_BUY"].notna().any()
    pairs = list(pst.index); idx_pairs = [p for p in pairs if "Gold" not in p]; gold_pairs = [p for p in pairs if "Gold" in p]
    assets = [a for a in C.ASSETS if a in vs.index]
    eq = [a for a in C.EQUITY if a in vs.index]
    bs3 = bs[bs["state"].isin(["both BUY", "disagree", "both SELL", "all weeks"])].pivot(index="pair", columns="state", values="corr").reindex(pairs)
    bsw = bs[bs["state"].isin(["both BUY", "disagree", "both SELL"])].pivot(index="pair", columns="state", values="weeks").reindex(pairs)
    eqp = dec.loc[idx_pairs]
    def af(ref, ev_, k, col="mean_corr"):
        g = afa[(afa["ref"] == ref) & (afa["event"] == ev_) & (afa["weeks_from_flip"] == k)]
        return g[col].iloc[0] if len(g) else np.nan
    def vaf(ref, ev_, k, col):
        g = vafa[(vafa["ref"] == ref) & (vafa["event"] == ev_) & (vafa["weeks_from_flip"] == k)]
        return g[col].iloc[0] if len(g) and col in g.columns else np.nan
    regimes = list(dict.fromkeys(regl["regime"]))
    regw = {rg: regl[regl["regime"] == rg].pivot(index="pair", columns="bucket", values="mean_corr") for rg in regimes}
    qn = {a: lg[lg["asset"] == a].set_index("bucket") for a in assets if a in set(lg["asset"])}
    pz = zf[(zf["pair"] == "ALL PAIRS") & (zf["horizon_weeks"] == H13)].set_index("bucket"); B = list(pz.index)
    zfp = zf[(zf["pair"] != "ALL PAIRS") & (zf["horizon_weeks"] == H13)]
    c13 = cells[cells["horizon_weeks"] == H13]
    def cell(dev, corr, stt):
        m = c13[(c13["deviation"] == dev) & (c13["correlated"] == corr) & (c13["cand_state"] == stt)]
        return m.iloc[0] if len(m) else None
    ptall = pts.loc["ALL PAIRS"]
    last = S.dropna(subset=["z_G"]).iloc[-1]
    date_first = sm["avg_rank"].idxmin(); drop_first = sm["avg_rank_drop"].idxmin()
    eq4 = [a for a in C.EQUITY if a in sm4.index]
    first_cross = sm4.loc[eq4, "avg_rank_cross_10pct"].idxmin(); first_sig = sm4.loc[eq4, "avg_rank_signal_sell"].idxmin()
    up = afs[afs["direction"].str.startswith("up")].set_index("weeks_ahead")
    lead_rule = R.loc["most leading (any corr)"]; corrbuy = R.loc["most correlated, in BUY"]; gold_rule = R.loc["gold"]; lag_rule = R.loc["correlated & lagging"]; lagbuy = R.loc["correlated & lagging, in BUY"]
    cash13 = R.loc["cash", f"mean_{H13}"]; stay13 = R.loc["stay in the seller", f"mean_{H13}"]
    coint_ok = [p for p in pco.index if pco.loc[p, "verdict"].startswith("pulls back")]
    idx_names = [a for a in C.EQUITY if a in M.index]
    off = M.loc[idx_names, idx_names].where(~np.eye(len(idx_names), dtype=bool)).stack()

    css = """
    body{font-family:Inter,Helvetica,Arial,sans-serif;max-width:1320px;margin:0 auto;padding:24px 20px;color:#0b0b0b;background:#fcfcfb;line-height:1.55}
    h1{font-size:24px;margin:0 0 4px} h2{font-size:20px;margin:46px 0 10px;border-bottom:2px solid #e1e0d9;padding-bottom:4px} h3{font-size:16px;margin:34px 0 6px;padding:6px 10px;background:#f3f3ef;border-left:4px solid #0b0b0b} h4{font-size:13.5px;margin:14px 0 4px;color:#52514e}
    .sub{color:#52514e;margin-bottom:18px} .fig{margin:12px 0;border:1px solid #e1e0d9;border-radius:4px;padding:4px 4px 2px;background:#fcfcfb} .fignote{font-size:12px;color:#52514e;margin:2px 6px 8px}
    .t{border-collapse:collapse;font-size:12.6px;margin:8px 0;display:block;overflow-x:auto;max-width:100%} .t th,.t td{padding:4px 9px;text-align:right;border-bottom:1px solid #eeede8;vertical-align:top} .t th{color:#52514e;font-weight:600} .t td:first-child,.t th:first-child{text-align:left}
    .combo .t td{text-align:left} .combo .t td:first-child{white-space:nowrap;color:#52514e}
    .box{background:#f3f3ef;border-left:4px solid #2a78d6;padding:10px 14px;margin:14px 0}
    .q{background:#eef5fd;border-left:4px solid #2a78d6;padding:9px 14px;margin:10px 0;font-size:13.8px} .shows{background:#f0faf5;border-left:4px solid #1baf7a;padding:9px 14px;margin:10px 0;font-size:13.6px}
    .els,.cols{font-size:13px;margin:10px 0;padding:8px 14px;border:1px solid #e1e0d9;border-radius:4px;background:#fbfbf7} .els ul,.cols ul{margin:4px 0 0;padding-left:18px} .els li,.cols li{margin:2px 0}
    .combo{margin:10px 0} .combo > b{font-size:13.5px}
    .toc{columns:2;font-size:13px;list-style:none;padding-left:4px} .toc li{margin:1px 0} .dict{columns:2;font-size:12.8px;column-gap:28px} .dict p{margin:3px 0;break-inside:avoid} .dict code{background:#f0efec;padding:1px 4px;border-radius:3px}
    .missing{color:#e34948} .small{font-size:12.5px;color:#52514e} code{background:#f0efec;padding:1px 4px;border-radius:3px;font-size:12.5px} ul{margin:6px 0} li{margin:3px 0}
    """
    H = [f"<!doctype html><html><head><meta charset='utf-8'><title>wm-BREAK · reading guide</title><style>{css}</style><script>{PLOTLY_JS.read_text(encoding='utf-8')}</script></head><body>"]
    H.append("<h1>Study I — the reading guide: every question, every chart, every combination, every column</h1>")
    H.append(f"<div class='sub'>Companion to <code>outputs/report.html</code> · weekly, {C.START} → {meta['last_session']} · built {meta['run_at']}</div>")
    H.append("<div class='box'><b>How to use this.</b> <b>Part A</b> lists every question the study asks, with its answer and the chart that shows it. <b>Part B</b> goes chart by chart in the report's order: the chart itself; <i>what is on the chart</i> (every axis, line, marker, colour and panel); <i>how it shows the answer</i>; a reading <i>for each combination</i> — every pair of assets, or every asset — with the numbers; and the meaning of every column in the table that sits under it in the report. <b>Part C</b> is a dictionary of every column name used anywhere in the report. Charts are the same as in the report: hover for values, click a legend entry to hide or show it.</div>")
    H.append("<ul class='toc'><li>Part A — the questions and their answers</li><li>Part B — chart by chart (B1 … B20)</li><li>Part C — the column dictionary</li></ul>")

    # ======================================================== PART A ==
    H.append("<h2>Part A — the questions, and their answers</h2>")
    Q = [
        ("How correlated are the asset classes with each other?", f"The four size indices are one block: {off.min():.2f}–{off.max():.2f} over the whole sample, highest between neighbours in size. Gold is unrelated to all of them ({M.loc['GoldMCX', idx_names].min():+.2f} to {M.loc['GoldMCX', idx_names].max():+.2f})" + (f"; the VIX moves against them ({M.loc[C.VIXCHG, idx_names].min():+.2f} to {M.loc[C.VIXCHG, idx_names].max():+.2f})." if C.VIXCHG in M.index else "."), "B1 (the correlation matrix), B2 (rolling correlation)"),
        ("Does correlation depend on what the signal says — both BUY, disagree, both SELL?", f"Yes, a little: index pairs average {bs3.loc[idx_pairs, 'both BUY'].mean():.2f} in both-BUY weeks, {bs3.loc[idx_pairs, 'disagree'].mean():.2f} when the signals disagree, {bs3.loc[idx_pairs, 'both SELL'].mean():.2f} in both-SELL weeks. Gold's stays near zero in all three.", "B3"),
        ("Does correlation rise before a signal flips (a warning) or after (a consequence)?", f"After. Around every asset's SELL flip the group correlation is flat before and climbs ~0.1 in the two months after (N100: {af('N100', 'flip to SELL', -4):.2f} → {af('N100', 'flip to SELL', 0):.2f} → {af('N100', 'flip to SELL', 8):.2f}). It is a consequence of the sell-off that fired the signal. Pair by pair, the climb is in the N100 pairs; the pairs among the smaller indices were already coupled — the rise is Nifty re-joining the block (B4b).", "B4"),
        ("Is the market more coupled when more signals are in SELL?", f"Yes, monotonically: {bb.loc['none in SELL', 'mean']:.2f} with none in SELL → {bb.loc['all in SELL', 'mean']:.2f} with all four. Pair by pair: every index pair rises with breadth, the N100 pairs most (N100×SC100 {bpd.loc['none in SELL', 'N100|SC100']:.2f} → {bpd.loc['all in SELL', 'N100|SC100']:.2f}); gold pairs stay near zero.", "B5 (table) and B6 (timeline)"),
        ("Is a SELL state a high-correlation state or a high-volatility state?", f"A high-volatility state. Swings are {vs.loc[eq, 'vol_ratio_SELL_over_BUY'].mean():.1f}× bigger in an index's own SELL weeks while correlation rises only {bs3.loc[idx_pairs, 'both SELL'].mean() - bs3.loc[idx_pairs, 'both BUY'].mean():.2f}; {eqp['share_of_cov_change_from_vol'].mean() * 100:.0f}% of the extra covariance (risk) in SELL is volatility.", "B7, and B3's right panel"),
        ("Does correlation rise with volatility — pair by pair, whichever vol we use?", f"Yes for every index pair under all three regime definitions (N100×SC100: {regw[regimes[0]].loc['N100|SC100', 'Q1']:.2f} in the calmest fifth of weeks → {regw[regimes[0]].loc['N100|SC100', 'Q5']:.2f} in the wildest). Gold's correlation does not rise.", "B8"),
        ("Which moves first — volatility or correlation?", f"Neither. In the correlation jumps since 2016 all three — correlation, realised volatility and the VIX — were at their usual level four weeks before and unusually high together in the jump week. A finer test gives the VIX a small statistical head start of a few weeks, too small to act on. Pair by pair, with the pair's own volatility: the same — in every index pair the own-vol change peaks at k = 0 and at the pair's own jumps its vol was above +1 beforehand in only {own.loc[[p for p in own.index if 'Gold' not in p], 'share_z_vol_above1_before'].mean():.0%} of them; the gold pairs' correlation does not move with volatility at all. The pairs whose correlation makes the group jump are the N100 pairs ({pps.loc[[p for p in pps.index if p.startswith('N100|') and 'Gold' not in p], 'share_pair_also_reached_plus1'].mean():.0%} of group jumps).", "B9"),
        ("When correlation is high, what is the market doing — and what does it do next?", f"Falling, then recovering. The most-correlated fifth of weeks coincided with N100 {_pct(qn['N100'].loc['Q5', 'same_quarter_mean'])} over that quarter and was followed by {_pct(qn['N100'].loc['Q5', 'next_quarter_mean'])} the next (positive {qn['N100'].loc['Q5', 'next_quarter_hit'] * 100:.0f}% of the time); the least-correlated fifth: {_pct(qn['N100'].loc['Q1', 'same_quarter_mean'])} then {_pct(qn['N100'].loc['Q1', 'next_quarter_mean'])}. Pair by pair the same shape holds in each index pair with a full history; gold's own next quarter is flat whichever index it happened to move with.", "B10, B11, B12"),
        ("Which index breaks the correlation first, and which way?", f"The smaller index, by running ahead ({date_first} most often by date); Nifty is the one left behind (by size of collapse, {drop_first}). {sm.loc['MC100', 'share_up_breaks'] * 100:.0f}% of Midcap's and {sm.loc['SC100', 'share_up_breaks'] * 100:.0f}% of Smallcap's breaks are upward, {100 - sm.loc['N100', 'share_up_breaks'] * 100:.0f}% of Nifty's downward. An index that decoupled by outperforming returned {_pct(up.loc[13, 'breaker'])} over the next quarter — the run was ending.", "B13, B14"),
        ("In a price sell-off, who falls first and whose signal fires first?", f"{first_cross} crosses −10% first ({sm4.loc[first_cross, 'first_share_cross_10pct'] * 100:.0f}% of sell-offs) and falls most; {first_sig}'s supertrend fires first ({sm4.loc[first_sig, 'first_share_signal_sell'] * 100:.0f}%).", "B15"),
        ("Do deviations between two assets pull back? Is there a pair-trading strategy?", f"No. {10 - len(coint_ok)} of 10 price ratios have no long-run level; a 1–2 sd deviation continued ({_pct(pz.loc[B[1], 'mean_ratio_change'])} further over 13 weeks after a lag, t {_t(pz.loc[B[1], 'hac_t_ratio_change'])}); the classic pair trade lost in 10 of 10 pairs ({int(ptall['trades'])} trades, {_pct(ptall['mean_net'])} each).", "B16, B17, B18"),
        ("When an asset flips to SELL, which asset should replace it?", f"One whose own signal is BUY. The most-leading asset returned {_pct(lead_rule[f'mean_{H13}'])} over 13 weeks, the most correlated asset still in BUY {_pct(corrbuy[f'mean_{H13}'])}, gold {_pct(gold_rule[f'mean_{H13}'])} — vs staying {_pct(stay13)} and cash {_pct(cash13)}. The correlated laggard never had a BUY-state candidate in {int(ev.shape[0])} flips.", "B19, B20"),
    ]
    H.append("<ol>")
    for q, a, where in Q:
        H.append(f"<li><div class='q'><b>{q}</b><br>{a} <span class='small'>→ {where}</span></div></li>")
    H.append("</ol>")

    # ======================================================== PART B ==
    H.append("<h2>Part B — chart by chart</h2>")
    H.append("<p class='small'>Every study that uses the group correlation G is also shown for each of the ten pairs (gold included): the pair lines are on the same chart — index pairs solid, gold pairs dashed and off until clicked, one colour per pair throughout — and each section reads the result pair by pair.</p>")

    # ---- B1 heatmap
    H.append("<h3>B1 · Correlation between the asset classes — the matrix (report §2)</h3>")
    H.append(_fig("0_corr_heatmap"))
    H.append(elements([("Three panels", "the same matrix on three windows: the whole sample (2016 →), the last 52 weeks, the last 13 weeks."),
                       ("Rows and columns", "the five assets, plus two reference rows: VIX (change) = the weekly change in India VIX, Cash (liquid) = the liquid ETF."),
                       ("A cell", "the correlation of the two series' weekly returns, −1 … +1, printed in the cell."),
                       ("Colour", "blue = move together (+1 dark blue), grey = unrelated (0), red = move opposite (−1)."),
                       ("Blank cell", "too few weeks in that window — the VIX file ends 2025-09, so it is blank in the recent windows."),
                       ("The diagonal", "always 1.00 (an asset with itself).")]))
    H.append(box("shows", "How it shows the answer.", "The index block is solid blue in every panel (the four move together); the gold row and column are grey (unrelated); the VIX row is red (moves against). Compare the three panels to see whether the block is tighter now than on average."))
    rows = []
    for p in pairs:
        a, b = p.split("|")
        rows.append((_x(p), f"whole sample {M.loc[a, b]:.2f} ({_level(M.loc[a, b])}); last 52 weeks {M52.loc[a, b]:.2f}; last 13 weeks {M13.loc[a, b]:.2f}" + (f" — {'tighter' if M13.loc[a, b] > M.loc[a, b] + 0.05 else ('looser' if M13.loc[a, b] < M.loc[a, b] - 0.05 else 'about usual')} now" if pd.notna(M13.loc[a, b]) else "")))
    if C.VIXCHG in M.index:
        rows.append(("VIX change vs the indices", f"whole sample " + ", ".join(f"{a} {M.loc[C.VIXCHG, a]:+.2f}" for a in idx_names) + f"; vs gold {M.loc[C.VIXCHG, 'GoldMCX']:+.2f} — a rising VIX is a falling equity market, by construction; gold is untouched"))
    H.append(combos("Reading it for each combination", rows))
    H.append(columns([("N100 … GoldMCX, VIX (change), Cash (liquid)", "the series; a cell is the correlation of the row series with the column series.")]))

    # ---- B2 rolling
    H.append("<h3>B2 · How each pair's correlation has moved (report §2.1)</h3>")
    H.append(_fig("0_corr_rolling"))
    H.append(elements([("Top panel", "the six index pairs' 13-week rolling correlation, week by week. Solid lines = pairs with N100; dotted = the other index pairs."),
                       ("Bottom panel", "gold against each of the four indices, on its own scale (−0.7 … +0.7) because it sits near zero."),
                       ("13-week rolling", "each point is the correlation computed on that week and the 12 before it — the 'current' correlation, recomputed weekly."),
                       ("Line colours", "the colour of the first-named asset of the pair (N100 blue, MC100 orange, SC100 green, Micro250 purple); gold pairs take the index's colour.")]))
    H.append(box("shows", "How it shows the answer.", "The index lines live between about 0.6 and 1.0 and dip together in the decorrelation episodes (the sharp Vs); the gold lines wander around zero without trend. The dips are where Part II §6 looks."))
    rows = [(_x(p), f"typical (median) {pst.loc[p, 'corr_13w_median']:.2f}, ranged {pst.loc[p, 'corr_13w_min']:.2f} to {pst.loc[p, 'corr_13w_max']:.2f}, now {pst.loc[p, 'corr_13w_now']:.2f} (13w) / {pst.loc[p, 'corr_52w_now']:.2f} (52w)") for p in pairs]
    H.append(combos("Reading it for each combination", rows))
    H.append(columns([("whole sample", "the correlation over all weeks since 2016."), ("13-week: typical (median)", "the middle value of the 13-week rolling correlation over the sample."), ("13-week: lowest / highest", "how far the 13-week correlation has ranged."), ("last 52 weeks / last 13 weeks", "the correlation over the most recent year / quarter.")]))

    # ---- B3 by state
    H.append("<h3>B3 · Correlation by what the two signals said (report §3.1)</h3>")
    H.append(_fig("1_corr_by_state"))
    H.append(elements([("Left panel", "the pair's correlation; right panel: the pair's covariance (annualised) — the risk the pair contributes to a book."),
                       ("One row per pair", "the ten pairs, listed top to bottom."),
                       ("Blue dot", "the value in the weeks both signals were BUY; red dot: both SELL; grey diamond: the signals disagreed (one BUY, one SELL)."),
                       ("Grey bar", "joins the BUY dot to the SELL dot — its length is how much the state changes the number."),
                       ("Covariance", "= correlation × vol(a) × vol(b); the numbers are small because they are products of fractions — read the ratio between red and blue, not the level.")]))
    H.append(box("shows", "How it shows the answer.", "On the left the red dots sit a little to the right of the blue ones (more correlated in SELL) with the diamonds between; on the right the red dots sit far to the right (several times the risk). The gap on the right is mostly volatility, which the table under the chart splits out."))
    rows = []
    for p in pairs:
        b_, d_, s_ = bs3.loc[p, "both BUY"], bs3.loc[p, "disagree"], bs3.loc[p, "both SELL"]
        txt = f"{b_:.2f} in both-BUY ({int(bsw.loc[p, 'both BUY'])} wk) → {'—' if pd.isna(d_) else f'{d_:.2f}'} disagree ({int(bsw.loc[p, 'disagree'])} wk) → {s_:.2f} in both-SELL ({int(bsw.loc[p, 'both SELL'])} wk)"
        if "Gold" not in p:
            txt += f"; risk {dec.loc[p, 'cov_ratio_sell_over_buy']:.1f}× larger in SELL, {dec.loc[p, 'share_of_cov_change_from_vol'] * 100:.0f}% of that from volatility"
            txt += f" — {'coupling rises in SELL' if s_ - b_ > 0.1 else 'coupling barely changes'}"
        else:
            txt += " — near zero in every state; disagree is its most common state because gold's signal runs on its own cycle"
        rows.append((_x(p), txt))
    H.append(combos("Reading it for each combination", rows))
    H.append(columns([("both BUY / disagree / both SELL", "the pair's correlation in the weeks the two signals were both BUY / one BUY one SELL / both SELL."), ("weeks", "how many weeks fall in that state (a state needs 20 to be reported)."), ("all weeks", "the correlation over every week."), ("SELL minus BUY", "the change in correlation between the two clean states."), ("risk (covariance) SELL ÷ BUY", "how many times larger the pair's covariance is in both-SELL weeks than in both-BUY weeks."), ("share of the extra risk that is volatility", "the fraction of that rise that comes from the two volatilities rising rather than from the correlation rising (the two shares add to 1).")]))

    # ---- B4 around flip
    H.append("<h3>B4 · Correlation around every asset's flips (report §3.2)</h3>")
    H.append(_fig("1_corr_around_flip"))
    H.append(elements([("x-axis", "weeks from the flip: 0 is the week the asset's supertrend flipped; −8 … −1 before, +1 … +8 after."),
                       ("y-axis", "the group correlation G — the average of the six 13-week correlations among the four indices."),
                       ("Left panel", "around flips to SELL; right panel: around flips back to BUY."),
                       ("One solid line per asset", "that asset's flips, averaged (n = the number of flips, in the legend). Click a legend entry to hide or show it."),
                       ("Dotted lines (off by default)", "the flipping asset's own correlation with the other indices instead of G."),
                       ("Dashed vertical line", "the flip week.")]))
    H.append(box("shows", "How it shows the answer.", "If correlation were a warning, the lines would be rising before week 0. They are flat (or falling) before 0 and climb after it, for every index; gold's line (yellow) does nothing at all. On the right, the lines peak at the BUY flip and fall away — a BUY flip comes out of the bottom of a sell-off."))
    rows = []
    for a in assets:
        s4, s0, s8 = af(a, "flip to SELL", -4), af(a, "flip to SELL", 0), af(a, "flip to SELL", 8)
        b0, b8 = af(a, "flip to BUY", 0), af(a, "flip to BUY", 8)
        n = int(afa[(afa["ref"] == a) & (afa["event"] == "flip to SELL")]["n_events"].max())
        rows.append((f"{a}'s flips (n {n})", f"SELL flip: {s4:.2f} four weeks before → {s0:.2f} at the flip → {s8:.2f} eight weeks after ({'rises after, not before' if s8 - s0 > 0.02 and abs(s0 - s4) < 0.12 and a != 'GoldMCX' else ('nothing happens' if abs(s8 - s4) < 0.05 else 'mixed')}); BUY flip: {b0:.2f} at the flip → {b8:.2f} after"))
    H.append(combos("Reading it for each asset", rows))
    H.append(columns([("8 wk before … 8 wk after", "the group correlation at that offset from the flip, averaged over the asset's flips."), ("n_flips", "how many flips were averaged.")]))
    H.append("<p><b>The same, pair by pair.</b></p>")
    H.append(_fig("1_corr_around_flip_pairs"))
    H.append(elements([("The menu (top right)", "whose flips the lines are drawn around — N100's by default; pick any asset."),
                       ("One line per pair", "that pair's 13-week correlation at each week from −8 to +8 around the chosen asset's flips, averaged over its flips (n in the legend). Each pair keeps the same colour in every chart of the report; gold pairs are dashed and off until clicked."),
                       ("Left / right panel", "around flips to SELL / back to BUY; the grey band is the 8 weeks before; the dashed vertical line is the flip week.")]))
    afn = afp[(afp["ref"] == "N100") & (afp["event"] == "flip to SELL")].pivot(index="pair", columns="weeks_from_flip", values="mean_corr")
    n100p = [p for p in pairs if p.startswith("N100|") and "Gold" not in p and p in afn.index]; othp = [p for p in pairs if not p.startswith("N100|") and "Gold" not in p and p in afn.index]
    H.append(box("shows", "How it shows the answer.", f"With N100's flips selected, the lines that step up at week 0 and keep climbing are the three N100 pairs ({afn.loc[n100p, -4].mean():.2f} four weeks before → {afn.loc[n100p, 8].mean():.2f} eight weeks after); the pairs among the smaller indices run flat along the top ({afn.loc[othp, -4].mean():.2f} → {afn.loc[othp, 8].mean():.2f}) — they were already coupled. So the rise in G after a SELL flip is Nifty re-joining the block. The gold pairs (click them on) dip below zero in the flip week: gold went up while the indices went down."))
    rows = []
    for p in [p for p in pairs if p in afn.index]:
        rows.append((_x(p), f"around N100's SELL flips: {afn.loc[p, -4]:.2f} four weeks before → {afn.loc[p, 0]:.2f} at the flip → {afn.loc[p, 8]:.2f} eight weeks after ({afn.loc[p, 8] - afn.loc[p, -4]:+.2f})" + (" — climbs: this pair contains the flipping asset" if p.startswith("N100|") and "Gold" not in p else (" — dips below zero at the flip, then drifts back" if "Gold" in p else " — already coupled, barely moves"))))
    H.append(combos("Reading it for each pair (N100's flips)", rows))
    H.append(columns([("ref", "the asset whose flips the window is centred on."), ("event", "flip to SELL / flip to BUY."), ("pair", "the pair whose correlation is averaged."), ("weeks_from_flip, mean_corr, n_events", "the offset, the pair's average 13-week correlation there, and how many flips it was averaged over.")]))

    # ---- B5 breadth + B6 timeseries
    H.append("<h3>B5 · Correlation by how many signals are in SELL (report §3.3) and B6 · the week-by-week chart</h3>")
    H.append(_fig("1_corr_by_breadth"))
    H.append(_tbl(bb, fmt="{:.2f}", int_cols=("weeks",)))
    H.append(_fig("1_corr_timeseries"))
    H.append(elements([("B5 x-axis", "the weeks grouped by how many of the four index signals were in SELL that week: none, one, about half, most, all — with the number of weeks in each group under the label."),
                       ("B5 lines", "one per pair (its average 13-week correlation in those weeks); the black line is the group correlation G; gold pairs dashed, off until clicked."),
                       ("B6 top panel", "the group correlation G (black) and every pair's 13-week correlation (thin lines) week by week; the red shading marks the weeks N100's signal was in SELL."),
                       ("B6 bottom panel", "how many of the four signals were in SELL (red area) and how many signals existed (dashed; Micro250 only from 2022).")]))
    H.append(box("shows", "How it shows the answer.", f"On B5 every solid line rises from left to right and the black line with them ({bb.loc['none in SELL', 'mean']:.2f} → {bb.loc['all in SELL', 'mean']:.2f}); the steepest are the N100 pairs, the flattest the pairs among the smaller indices, which start high. On B6 the spikes of G — and of every pair line — sit inside the red-shaded stretches (SELL weeks). The same fact as B3 seen from the breadth side."))
    H.append(combos("Reading each band", [(str(i), f"group {bb.loc[i, 'mean']:.2f} (median {bb.loc[i, 'median']:.2f}) over {int(bb.loc[i, 'weeks'])} weeks; " + ", ".join(f"{_x(p)} {bpd.loc[i, p]:.2f}" for p in pairs if p in bpd.columns)) for i in bb.index]))
    H.append(combos("Reading it for each pair (none in SELL → all in SELL)", [(_x(p), f"{bpd.loc['none in SELL', p]:.2f} → {bpd.loc['all in SELL', p]:.2f} ({bpd.loc['all in SELL', p] - bpd.loc['none in SELL', p]:+.2f})" + (" — rises with breadth" if "Gold" not in p else " — stays near zero")) for p in pairs if p in bpd.columns]))
    H.append(columns([("mean / median", "the average and the middle value of G in the weeks of that band."), ("weeks", "how many weeks are in the band."), ("<pair> columns (1_corr_by_breadth_pairs)", "that pair's average 13-week correlation in the band's weeks.")]))

    # ---- B7 vol by state
    H.append("<h3>B7 · Volatility by the asset's own signal state (report §4.1)</h3>")
    H.append(_fig("5_vol_by_state"))
    H.append(elements([("Left panel", "each asset's realised volatility (sd of weekly returns × √52) in its own BUY weeks (blue) and its own SELL weeks (red); the number beside the red dot is SELL ÷ BUY."),
                       ("Middle panel", "the average weekly return in each state (blue BUY, red SELL)."),
                       ("Right panel (if the VIX file is present)", "the average India VIX level in the asset's BUY and SELL weeks, same colours, with its ratio."),
                       ("One row / group per asset", "the five assets.")]))
    H.append(box("shows", "How it shows the answer.", "The red dots sit far to the right of the blue ones for every index (1.6–1.8× the swings) and on top of each other for gold (its signal does not sort its weeks by volatility). The middle panel shows SELL weeks are only mildly negative on average — the state is about the size of moves, not their direction. The VIX panel shows the same ordering with a much smaller gap: the option market prices part of it in advance."))
    rows = []
    for a in assets:
        txt = f"vol {_p(vs.loc[a, 'vol_BUY'])} in BUY → {_p(vs.loc[a, 'vol_SELL'])} in SELL (×{vs.loc[a, 'vol_ratio_SELL_over_BUY']:.2f}); average week {_pct(vs.loc[a, 'mean_weekly_ret_BUY'], 2)} vs {_pct(vs.loc[a, 'mean_weekly_ret_SELL'], 2)}; {_p(vs.loc[a, 'share_weeks_down_SELL'])} of SELL weeks were down; worst week {_pct(vs.loc[a, 'worst_week_SELL'])}"
        if has_vix:
            txt += f"; VIX {vs.loc[a, 'vix_mean_BUY']:.1f} → {vs.loc[a, 'vix_mean_SELL']:.1f} (×{vs.loc[a, 'vix_ratio_SELL_over_BUY']:.2f})"
        rows.append((a, txt))
    H.append(combos("Reading it for each asset", rows))
    H.append(columns([("weeks_BUY / weeks_SELL", "how many weeks the asset's own signal was BUY / SELL."), ("vol_BUY / vol_SELL", "its realised volatility, annualised, in those weeks."), ("vol_ratio", "vol_SELL ÷ vol_BUY."), ("mean_weekly_ret_BUY / _SELL", "its average weekly return in each state."), ("share_weeks_down_SELL", "the share of its SELL weeks that were down weeks."), ("worst_week_SELL", "its worst single week while in SELL."), ("vix_mean_BUY / _SELL, vix_ratio", "the average India VIX level in its BUY / SELL weeks, and the ratio.")]))

    # ---- B8 regimes
    H.append("<h3>B8 · Correlation by volatility regime — three definitions (report §4.2)</h3>")
    H.append(_fig("5_corr_by_regime"))
    H.append(elements([("Three panels", "three ways of defining the volatility regime of a week: N100's 13-week realised vol; the pair's own average 13-week vol; the India VIX level."),
                       ("x-axis", "Q1 … Q5: the weeks sorted into five equal groups by that measure — Q1 the calmest fifth, Q5 the wildest. Hover shows the vol range of each group."),
                       ("y-axis", "the pair's 13-week correlation averaged inside the group."),
                       ("One line per pair", "solid = index pairs, dashed = gold pairs; each pair keeps its own colour in every chart of the report. Click the legend to isolate a pair."),
                       ("The zero line", "no relationship.")]))
    H.append(box("shows", "How it shows the answer.", "Every solid line climbs from left to right in every panel — index pairs are more correlated in wild weeks whichever volatility is used; the climb is flattest in the VIX panel (a forecast sorts weeks less sharply). The dashed lines stay flat around zero: gold does not couple to the indices even in the wildest weeks."))
    rows = []
    for p in pairs:
        parts = []
        for rg in regimes:
            w = regw[rg]
            if p in w.index:
                parts.append(f"{rg}: {w.loc[p, 'Q1']:.2f} → {w.loc[p, 'Q5']:.2f} ({w.loc[p, 'Q5'] - w.loc[p, 'Q1']:+.2f})")
        rows.append((_x(p), "; ".join(parts) + (" — rises with volatility under every definition" if "Gold" not in p else " — no rise: stays near zero")))
    H.append(combos("Reading it for each combination (calm Q1 → wild Q5, and the difference)", rows))
    H.append(columns([("<regime>: calm (Q1) / wild (Q5)", "the pair's average 13-week correlation in the calmest / wildest fifth of weeks under that regime definition."), ("<regime>: wild − calm", "how much more correlated the pair is in wild weeks than calm ones."), ("spearman_corr_vs_regime (in the full tables)", "the rank correlation between the pair's correlation series and the regime series: 1 = they rise and fall together perfectly.")]))

    # ---- B9 together / who comes first
    per = _csv("5_corr_rise_events"); pa = _csv("5_corr_rise_paths"); pav = pa.pivot(index="k", columns="series", values="mean_z")
    H.append("<h3>B9 · Which moves first — volatility or correlation? (report §4.3)</h3>")
    H.append(_fig("5_vol_corr_lead"))
    H.append(elements([("x-axis", "k, in weeks. Each bar is the correlation between the change in volatility at week t and the change in the group correlation at week t + k."),
                       ("Blue bars", "realised volatility (N100's 13-week vol); red bars: the India VIX (13-week change in its log level)."),
                       ("k = 0", "the two change in the same week; k > 0: the correlation change comes k weeks after the volatility change (volatility leads); k < 0: correlation leads."),
                       ("The number on each bar", "the correlation of the two changes.")]))
    vlpw = vlp.pivot(index="k_weeks", columns="pair", values="corr_dvol_t_dcorr_t_plus_k")
    vlxw = vlp.pivot(index="k_weeks", columns="pair", values="corr_dvix_t_dcorr_t_plus_k") if "corr_dvix_t_dcorr_t_plus_k" in vlp.columns else None
    H.append(box("shows", "How it shows the answer.", f"The blue bars peak at k = 0 ({vl.loc[0, 'corr_dvol_t_dcorr_t_plus_k']:.2f}) and fall away symmetrically — realised vol and correlation are the same event seen through two windows." + (f" The red bars keep rising to the right (k = 0: {vl.loc[0, 'corr_dvix_t_dcorr_t_plus_k']:.2f}, k = {int(vl['corr_dvix_t_dcorr_t_plus_k'].idxmax()):+d}: {vl['corr_dvix_t_dcorr_t_plus_k'].max():.2f}) — a VIX change is a little more related to the correlation change a few weeks later than to the same week's: implied vol leads, modestly." if "corr_dvix_t_dcorr_t_plus_k" in vl.columns else "")))
    H.append(combos("Reading each lag", [(f"k = {int(k):+d}", f"realised {vl.loc[k, 'corr_dvol_t_dcorr_t_plus_k']:.2f}" + (f", VIX {vl.loc[k, 'corr_dvix_t_dcorr_t_plus_k']:.2f}" if "corr_dvix_t_dcorr_t_plus_k" in vl.columns else "")) for k in vl.index]))
    H.append(columns([("k_weeks", "the lead / lag in weeks."), ("corr_dvol_t_dcorr_t_plus_k", "the correlation between the 13-week change in realised vol at t and the 13-week change in G at t + k."), ("corr_dvix_t_dcorr_t_plus_k", "the same with the VIX."), ("weeks / weeks_vix", "how many weeks the correlation is computed on.")]))
    H.append("<p><b>The same test for each pair on its own</b> — one chart per pair: the change in the pair's own volatility (the average of its two legs' 13-week vol), and the change in the VIX, against the change in that pair's correlation.</p>")
    H.append(_fig("5_vol_corr_lead_pairs"))
    H.append(elements([("One panel per pair", "index pairs first, gold pairs last; the panel title says where the blue and the red bars peak."),
                       ("Blue bars", "the pair's own realised vol (13-week, the two legs averaged) at week t against the pair's correlation at week t + k."),
                       ("Red bars", "the India VIX (13-week change in its log level) at week t against the pair's correlation at week t + k."),
                       ("Reading", "a peak at k = 0 means the pair's own swings and its coupling change in the same week; a peak at k > 0 means its volatility moved first; bars near zero mean the pair's correlation has nothing to do with volatility.")]))
    oe = [p for p in pairs if "Gold" not in p and p in vlpw.columns]; og = [p for p in pairs if "Gold" in p and p in vlpw.columns]
    H.append(box("shows", "How it shows the answer.", f"In every index pair the blue bars peak at k = 0 (" + ", ".join(f"{_x(p)} {vlpw.loc[0, p]:.2f}" for p in oe) + ") and fall away on both sides — the pair's own swings and its coupling are the same event, as for the group." + (f" The red bars peak later (k = {', '.join(f'{int(vlxw[p].idxmax()):+d}' for p in oe[:3])}) in the N100 pairs and not in the small-cap pairs: the VIX's small head start is a large-cap thing." if vlxw is not None else "") + f" In the gold pairs both sets of bars sit near zero ({', '.join(f'{_x(p)} {vlpw.loc[0, p]:.2f}' for p in og)}): a gold pair's correlation does not move with volatility at all."))
    H.append(combos("Reading it for each pair", [(_x(p), f"own vol: best match at k = {int(vlpw[p].idxmax()):+d} ({vlpw[p].max():.2f}; at k = 0 {vlpw.loc[0, p]:.2f})" + (f"; VIX: best at k = {int(vlxw[p].idxmax()):+d} ({vlxw[p].max():.2f}; at k = 0 {vlxw.loc[0, p]:.2f})" if vlxw is not None else "") + (" — the same week" if int(vlpw[p].idxmax()) == 0 else (" — its own vol moved first" if int(vlpw[p].idxmax()) > 0 else " — no relation to volatility" if "Gold" in p else " — correlation moved first"))) for p in pairs if p in vlpw.columns]))
    H.append(columns([("pair, k_weeks", "the pair and the lead / lag in weeks."), ("corr_dvol_t_dcorr_t_plus_k", "the correlation between the 13-week change in the pair's own vol at t and the 13-week change in the pair's correlation at t + k."), ("corr_dvix_t_dcorr_t_plus_k", "the same with the 13-week change in the log VIX."), ("weeks / weeks_vix", "how many weeks each is computed on.")]))
    H.append("<p><b>The simpler view: the three series on one scale, and the weeks the correlation jumped</b> — first the group, then each pair on its own.</p>")
    H.append(_fig("5_together"))
    H.append(_fig("5_corr_rise_paths"))
    H.append(elements([("The three lines", "the correlation among the four indices (black), N100's realised volatility (blue), and the India VIX (red)."),
                       ("Why they can share one axis", "each is shown as a z-score against its own trailing two years: 0 = its usual level, +1 (the dashed red line) = unusually high, +3 = extreme. So '+1' means the same thing for all three — 'unusually high for this series'."),
                       ("First chart", "the whole history. The dotted vertical lines are the correlation jumps — the weeks the correlation's z first reached +1."),
                       ("Second chart", f"those {int(len(per))} jumps lined up at week 0 and the three lines averaged around them, from 8 weeks before to 8 weeks after."),
                       ("How a warning would look", "if volatility (or the VIX) moved first, its line would already be above the dashed +1 line to the LEFT of week 0 while the black line was still low.")]))
    H.append(box("shows", "How it shows the answer.", f"In the second chart all three lines sit near 0 on the left and rise through +1 in the same week (week 0: correlation {pav.loc[0, 'z_corr']:+.1f}, realised vol {pav.loc[0, 'z_vol']:+.1f}" + (f", VIX {pav.loc[0, 'z_vix']:+.1f}" if "z_vix" in pav.columns else "") + "). Nothing is high before the correlation is. In the first chart the spikes of the three lines coincide at every sell-off. They move together — no warning from either."))
    rows = []
    for _, e in per.iterrows():
        parts = [f"correlation {e['corr_at_event']:.2f}"]
        for c, nm in (("z_vol", "realised vol"), ("z_vix", "VIX")):
            if f"{c}_-4" in e.index and pd.notna(e.get(f"{c}_+0")):
                parts.append(f"{nm} z {e[f'{c}_-4']:+.1f} four weeks before → {e[f'{c}_+0']:+.1f} in the jump week")
        rows.append((str(e["event"]), "; ".join(parts)))
    H.append(combos("Reading each correlation jump", rows))
    pn = [p for p in pairs if p.startswith("N100|") and "Gold" not in p and p in pps.index]; po = [p for p in pairs if not p.startswith("N100|") and "Gold" not in p and p in pps.index]
    H.append(box("shows", "Which pairs make the group jump.", f"A group jump is the N100 pairs' jump: their own z reached +1 in {pps.loc[pn, 'share_pair_also_reached_plus1'].mean():.0%} of the group's jumps; the pairs among the smaller indices got there in only {pps.loc[po, 'share_pair_also_reached_plus1'].mean():.0%}, because they were already near their usual high level. A group-correlation jump is Nifty re-joining a block that was already coupled — the same thing B4 shows pair by pair around the SELL flips."))
    H.append(combos("Reading it for each pair (around the group's correlation jumps)", [(_x(p), f"its own z reached +1 in {pps.loc[p, 'share_pair_also_reached_plus1']:.0%} of the {int(pps.loc[p, 'jumps'])} jumps it existed in; z in the jump week {pps.loc[p, 'mean_z_at_jump']:+.2f}") for p in pairs if p in pps.index]))
    H.append("<p><b>Each pair on its own</b> — the same two charts repeated per pair, with the pair's own correlation, the pair's own volatility and the VIX, and the jumps defined on the pair's own correlation.</p>")
    H.append(_fig("5_together_pairs"))
    H.append(_fig("5_corr_rise_paths_own"))
    H.append(elements([("One panel per pair (both charts)", "index pairs first, gold pairs last."),
                       ("Black", "the pair's 13-week correlation as a z-score against its own two years."),
                       ("Blue", "the pair's own realised volatility — the average of its two legs' 13-week vol — as a z-score."),
                       ("Red", "the India VIX (log level) as a z-score."),
                       ("Dotted verticals / week 0", "the weeks THAT pair's correlation z first reached +1 (its own jumps; n in the panel title); the second chart lines the pair's series up at those weeks."),
                       ("How a warning would look", "blue or red above the dashed +1 line to the LEFT of week 0 while black is still low.")]))
    oe2 = [p for p in pairs if "Gold" not in p and p in own.index]; og2 = [p for p in pairs if "Gold" in p and p in own.index]
    H.append(box("shows", "How it shows the answer.", f"In the index-pair panels the three lines spike in the same weeks, and at the pair's own jumps the blue line goes from its usual level four weeks before (z {own.loc[oe2, 'z_vol_-4'].mean():+.1f}) to up in the jump week ({own.loc[oe2, 'z_vol_+0'].mean():+.1f}); it was already above +1 before the jump in only {own.loc[oe2, 'share_z_vol_above1_before'].mean():.0%} of jumps" + (f", the VIX in {own.loc[oe2, 'share_z_vix_above1_before'].mean():.0%}" if "share_z_vix_above1_before" in own.columns else "") + f" — together, not before, pair by pair. In the gold-pair panels the black line spikes on its own: at a gold pair's jumps its own vol is {own.loc[og2, 'z_vol_+0'].mean():+.1f}" + (f" and the VIX {own.loc[og2, 'z_vix_+0'].mean():+.1f}" if "z_vix_+0" in own.columns else "") + " — a gold pair coupling or decoupling is not a volatility event."))
    H.append(combos("Reading it for each pair (its own jumps)", [(_x(p), f"{int(own.loc[p, 'jumps'])} jumps; its correlation z {own.loc[p, 'z_corr_-4']:+.1f} four weeks before → {own.loc[p, 'z_corr_+0']:+.1f} at the jump; its own vol z {own.loc[p, 'z_vol_-4']:+.1f} → {own.loc[p, 'z_vol_+0']:+.1f}" + (f"; VIX z {own.loc[p, 'z_vix_-4']:+.1f} → {own.loc[p, 'z_vix_+0']:+.1f}" if "z_vix_-4" in own.columns else "") + f"; vol already above +1 before the jump in {own.loc[p, 'share_z_vol_above1_before']:.0%} of them" + (f", the VIX in {own.loc[p, 'share_z_vix_above1_before']:.0%}" if "share_z_vix_above1_before" in own.columns else "")) for p in pairs if p in own.index]))
    H.append(columns([("event / correlation at the jump", "the week the correlation's z first reached +1, and the correlation level then."), ("correlation z / realised vol z / VIX z at −4, +0, +4", "each series' z-score four weeks before, in, and four weeks after the jump week."), ("jumps, z_corr_-4 / z_corr_+0, z_vol_-4 / z_vol_+0, z_vix_-4 / z_vix_+0, share_z_vol_above1_before, share_z_vix_above1_before (5_corr_rise_own_summary)", "per pair, on its own jumps: how many; each series' z four weeks before and in the jump week; the share of jumps in which the pair's own vol / the VIX was already above +1 in the 8 weeks before."), ("pair, series, k, mean_z (5_corr_rise_paths_own)", "the averaged paths behind the per-pair chart."), ("v_a|b, zv_a|b, vix, z_vix (5_together_pairs)", "the pair's own vol and its z; the VIX and its z."), ("jumps, share_pair_also_reached_plus1, mean_z_at_jump (5_corr_rise_pairs_summary)", "how many of the group's jumps the pair existed in; the share in which the pair's own z reached +1 within −8 … +4 weeks; the pair's z in the jump week."), ("z_<pair> (5_together_pairs)", "the pair's 13-week correlation as a z-score against its own 104-week history.")]))

    # ---- B10 quintiles
    H.append("<h3>B10 · Correlation quintiles vs returns (report §5.1)</h3>")
    H.append(_fig("2_corr_quintiles"))
    H.append(elements([("x-axis", "Q1 … Q5: all weeks sorted by the 13-week correlation among the indices into five equal groups; the label gives each group's correlation range."),
                       ("Left panel", "each asset's return over the SAME 13 weeks the correlation was measured on — what was happening while correlation sat at that level."),
                       ("Right panel", "each asset's return over the NEXT 13 weeks (t+1 … t+13) — the forward-looking one."),
                       ("One bar per asset", "N100 blue, MC100 orange, SC100 green, Micro250 purple, gold yellow.")]))
    H.append(box("shows", "How it shows the answer.", "On the left the bars step down from Q1 to Q5 — the most-correlated weeks are the ones where the market had been falling (a sell-off in progress). On the right Q5 is the tallest for the indices — the quarter after the most-correlated weeks has been the best, because sell-offs end. Gold's bars do not follow the pattern."))
    rows = []
    for a in assets:
        if a in qn:
            q = qn[a]
            rows.append((a, f"same quarter: Q1 {_pct(q.loc['Q1', 'same_quarter_mean'])} → Q5 {_pct(q.loc['Q5', 'same_quarter_mean'])}; next quarter: Q1 {_pct(q.loc['Q1', 'next_quarter_mean'])} → Q5 {_pct(q.loc['Q5', 'next_quarter_mean'])} (positive {q.loc['Q5', 'next_quarter_hit'] * 100:.0f}% of the time after Q5)"))
    H.append(combos("Reading it for each asset", rows))
    H.append(columns([("correlation range", "the lowest and highest group correlation inside the bucket."), ("weeks", "how many weeks are in the bucket."), ("same quarter", "the asset's average return over the 13 weeks ending in those weeks."), ("next quarter", "its average return over the following 13 weeks."), ("next quarter positive", "the share of those following quarters that were positive.")]))
    H.append("<p><b>The same, pair by pair.</b></p>")
    H.append(_fig("2_corr_quintiles_pairs"))
    H.append(elements([("One panel per pair", "the weeks sorted by THAT pair's own 13-week correlation into five equal groups (the correlation range under each label); index pairs first, gold pairs last."),
                       ("Bars", "each leg's average return over the NEXT quarter (t+1 … t+13), in the asset's colour."),
                       ("Dotted lines with diamonds", "each leg's average return over the SAME quarter the correlation was measured on."),
                       ("Shared y-axis", "all panels on one scale so the heights compare across pairs.")]))
    qe = qp[~qp["pair"].str.contains("Gold")]; qg = qp[qp["pair"].str.contains("Gold") & (qp["leg"] == "GoldMCX")]
    H.append(box("shows", "How it shows the answer.", f"In the index panels the dotted lines fall from Q1 to Q5 (the most-correlated weeks end a fall: legs {_pct(qe[qe['bucket'] == 'Q5']['same_quarter_mean'].mean())} that quarter against {_pct(qe[qe['bucket'] == 'Q1']['same_quarter_mean'].mean())} for the least-correlated) and the Q5 bars are taller than the Q1 bars in most pairs (next quarter {_pct(qe[qe['bucket'] == 'Q5']['next_quarter_mean'].mean())} vs {_pct(qe[qe['bucket'] == 'Q1']['next_quarter_mean'].mean())}) — the B10 pattern is in each pair, not an artefact of averaging six into G. In the gold panels gold's own bars are flat ({_pct(qg[qg['bucket'] == 'Q1']['next_quarter_mean'].mean())} after Q1, {_pct(qg[qg['bucket'] == 'Q5']['next_quarter_mean'].mean())} after Q5): how much gold moved with an index says nothing about gold's next quarter."))
    rows = []
    for p in [p for p in pairs if p in set(qp["pair"])]:
        a_, b_ = p.split("|"); ga = qp[(qp["pair"] == p) & (qp["leg"] == a_)].set_index("bucket"); gb = qp[(qp["pair"] == p) & (qp["leg"] == b_)].set_index("bucket")
        rows.append((_x(p), f"Q5 = correlation {ga.loc['Q5', 'corr_lo']:.2f}–{ga.loc['Q5', 'corr_hi']:.2f}. {a_}: same quarter Q1 {_pct(ga.loc['Q1', 'same_quarter_mean'])} → Q5 {_pct(ga.loc['Q5', 'same_quarter_mean'])}, next quarter {_pct(ga.loc['Q1', 'next_quarter_mean'])} → {_pct(ga.loc['Q5', 'next_quarter_mean'])} (positive {ga.loc['Q5', 'next_quarter_hit']:.0%} after Q5). {b_}: same {_pct(gb.loc['Q1', 'same_quarter_mean'])} → {_pct(gb.loc['Q5', 'same_quarter_mean'])}, next {_pct(gb.loc['Q1', 'next_quarter_mean'])} → {_pct(gb.loc['Q5', 'next_quarter_mean'])} ({gb.loc['Q5', 'next_quarter_hit']:.0%})"))
    H.append(combos("Reading it for each pair", rows))
    H.append(columns([("pair / leg", "the pair whose correlation sorts the weeks, and which of its two assets the returns belong to."), ("bucket, corr_lo, corr_hi, weeks", "the quintile of the pair's own 13-week correlation, its range and its week count."), ("same_quarter_mean / _hit, next_quarter_mean / _hit, n_next", "the leg's average return over the same / the following quarter, the share positive, and the number of following quarters available.")]))

    # ---- B11 by year, B12 timeline
    H.append("<h3>B11 · By calendar year, and B12 · the timeline and the stretches above 0.90 (report §5.2–5.3)</h3>")
    H.append(_fig("2_by_year"))
    H.append(_fig("2_timeline"))
    H.append(elements([("B11 top", "the average 13-week correlation in each year — the group G as bars with its lowest–highest range within the year (the thin line), and every pair as a line with markers (gold pairs off until clicked)."),
                       ("B11 bottom", "each asset's return in that year (price only, no dividends); the last year is partial."),
                       ("B12 top", "the group correlation (black) and every pair's correlation (thin lines) week by week; dashed = 0.90."),
                       ("B12 middle", "N100's and SC100's trailing 13-week return."),
                       ("B12 bottom", "the India VIX level (until its file ends)."),
                       ("The stretches table", "every run of at least 8 weeks with correlation above 0.90: when it ran, how long, its peak, what N100 / SC100 / gold returned during it and over the 13 weeks after it ended.")]))
    H.append(box("shows", "How it shows the answer.", "The yearly bars drift up over the decade; the most-correlated years are not the worst years because a year averages the sell-off and the recovery. On the timeline the correlation spikes line up with the troughs of the 13-week return and the VIX spikes — three views of the same sell-offs."))
    rows = [(str(y), f"average {yr.loc[y, 'mean_corr']:.2f} (range {yr.loc[y, 'min_corr']:.2f}–{yr.loc[y, 'max_corr']:.2f}, {int(yr.loc[y, 'weeks_corr_above_0.9'])} weeks above 0.90); N100 {_pct(yr.loc[y, 'ret_N100'])}, SC100 {_pct(yr.loc[y, 'ret_SC100'])}, gold {_pct(yr.loc[y, 'ret_GoldMCX'])}") for y in yr.index]
    H.append(combos("Reading each year", rows))
    rows = [(f"{r['start']} → {r['end']}", f"{int(r['weeks'])} weeks, peak {r['peak_corr']:.2f}; during: N100 {_pct(r['N100_during'])}, SC100 {_pct(r['SC100_during'])}; the 13 weeks after: N100 {_pct(r['N100_next_13w'])}, SC100 {_pct(r['SC100_next_13w'])}, gold {_pct(r['GoldMCX_next_13w'])}") for _, r in hc.iterrows()]
    H.append(combos("Reading each stretch above 0.90", rows))
    H.append(combos("Above 0.90, pair by pair", [(_x(p), f"above 0.90 in {hps.loc[p, 'share_of_weeks_above_0.90']:.0%} of weeks; {int(hps.loc[p, 'stretches'])} stretches of 8+ weeks" + (f" (median {hps.loc[p, 'median_weeks']:.0f} weeks); the quarter after a stretch: {p.split('|')[0]} {_pct(hps.loc[p, 'a_next_13w_mean'])} (positive {hps.loc[p, 'a_next_13w_positive']:.0%}), {p.split('|')[1]} {_pct(hps.loc[p, 'b_next_13w_mean'])} ({hps.loc[p, 'b_next_13w_positive']:.0%})" if hps.loc[p, 'stretches'] > 0 else " — never a stretch")) for p in pairs if p in hps.index]))
    H.append(columns([("average correlation / lowest / highest", "of the 13-week group correlation across the year's weeks."), ("<pair> columns (2_by_year_pairs)", "that pair's average 13-week correlation in the year."), ("weeks_above_0.90, share_of_weeks_above_0.90, stretches, median_weeks, a_/b_next_13w_mean, a_/b_next_13w_positive (2_high_corr_pairs_summary)", "for each pair: weeks and share of weeks above 0.90, the number and median length of its 8+-week stretches, and each leg's average return (and share positive) over the quarter after a stretch ended; 'a' = the first leg, 'b' = the second."), ("weeks above 0.90", "how many weeks the indices were effectively one block."), ("return <asset>", "the asset's return over the year."), ("start / end / weeks / peak correlation", "the stretch's dates, length and highest correlation."), ("<asset>_during / <asset>_next_13w", "the asset's return during the stretch and over the 13 weeks after it ended.")]))

    # ---- B13 corr series + B14 who breaks
    H.append("<h3>B13 · The correlation structure over time, and B14 · who breaks the correlation first (Part II §6)</h3>")
    H.append(_fig("3_corr_series"))
    H.append(_fig("3_who_breaks_correlation"))
    H.append(elements([("B13 top", "each index's 13-week correlation with the other three (its colour) and the group average G (black). Red shading = decorrelation episodes (G fell at least 1 sd below its own 2-year norm)."),
                       ("B13 middle", "the same series as z-scores against their own 104-week history; the dashed line is the break line z = −1. A series crossing it has 'let go' of the others."),
                       ("B13 third panel", "every pair's own 13-week correlation (gold pairs off until clicked), with the same episode shading — so an episode can be traced to the pair that loosened."),
                       ("B13 bottom", "gold's correlation with the four indices — reference only, never ranked."),
                       ("B14 four panels", "the share of episodes in which the index was: earliest to cross the break line (ties count); deepest below its norm in the week the group broke; the one whose own correlation fell most; and the share of its breaks that were upward (it decoupled by outperforming). Bars in the asset's colour; the text gives its average rank or count.")]))
    H.append(box("shows", "How it shows the answer.", "In the middle panel the coloured lines cross the dashed line at about the same time as the black one in each shaded episode — ties are built in. B14 then shows who is first by three different measures and, in the last panel, the robust fact: the smaller indices' bars are tall (they leave upward), Nifty's is short (it is left behind)."))
    rows = [(a, f"present in {int(sm.loc[a, 'episodes_present'])} episodes; earliest by date {_p(sm.loc[a, 'share_first'])} (alone {_p(sm.loc[a, 'share_first_alone'])}); deepest at the break {_p(sm.loc[a, 'share_deepest'])}; biggest collapse {_p(sm.loc[a, 'share_biggest_drop'])} (its share of the group's loss {_p(sm.loc[a, 'avg_drop_share'])}); combined rank {sm.loc[a, 'combined_rank']:.2f}; broke upward {_p(sm.loc[a, 'share_up_breaks'])} of the time") for a in sm.index]
    H.append(combos("Reading it for each index", rows))
    rows = [(str(e["episode"]), f"{e['start'].date()} → {e['end'].date()}, {int(e['weeks'])} weeks; group {e['G_before']:.2f} → {e['G_min']:.2f}; order: " + ", ".join(f"{g['asset']} ({'same week' if g['lead_vs_group_weeks'] == 0 else (str(int(-g['lead_vs_group_weeks'])) + ' wk before' if g['lead_vs_group_weeks'] < 0 else str(int(g['lead_vs_group_weeks'])) + ' wk after')}, {str(g['direction']).split(' (')[0]})" for _, g in ab[(ab["episode"] == e["episode"]) & ab["break_date"].notna()].sort_values("break_date").iterrows())) for _, e in cep.iterrows()]
    H.append(combos("Reading each episode", rows))
    prow = []
    for p in [p for p in pairs if p in psm.index]:
        g = pbk[(pbk["pair"] == p) & pbk["break_date"].notna()]
        prow.append((_x(p), f"broke (its own z ≤ −1) in {int(psm.loc[p, 'broke'])} of {int(psm.loc[p, 'episodes'])} episodes; first among the pairs {_p(psm.loc[p, 'share_first'])} of the time, average rank {psm.loc[p, 'avg_rank']:.2f}, median lead {psm.loc[p, 'median_lead_weeks']:+.0f} wk vs the group" + (f"; lowest correlation inside an episode {g['corr_min'].min():.2f}" if len(g) else "")))
    H.append(combos("Reading it for each pair — which relationship lets go first (the third panel; tables 3_pair_breaks, 3_pair_summary)", prow))
    H.append(columns([("episodes", "how many decorrelation episodes the index existed in."), ("broke / share_first / avg_rank / median_lead_weeks (3_pair_summary)", "for a pair: how many episodes its own correlation crossed its break line; how often it was the earliest pair to do so; its average rank among the pairs; its median lead in weeks against the group's break (negative = earlier)."), ("break_date, lead_vs_group_weeks, corr_min, z_min, rank (3_pair_breaks)", "per episode and pair: the week the pair's z crossed −1, its lead vs the group, the pair's lowest correlation and z inside the episode, and its rank by date."), ("first by date", "share of episodes it crossed the break line earliest (ties counted)."), ("deepest at the break", "share of episodes it was furthest below its own norm in the week the group broke."), ("biggest collapse", "share of episodes its own correlation with the others fell most."), ("its share of the group's loss", "on average, how much of the group's total correlation loss was its own."), ("combined rank", "the average of its three average ranks; lower = more of a first-breaker."), ("broke upward", "share of its breaks that were up-breaks — it decoupled by outperforming the others over the prior 13 weeks."), ("lead_vs_group_weeks", "its own break date minus the group's, in weeks; negative = earlier."), ("direction", "up-break / down-break.")]))

    # ---- B15 price / signal
    H.append("<h3>B15 · Who breaks in price and signal (Part II §7)</h3>")
    H.append(_fig("4_who_breaks_first"))
    H.append(elements([("Three panels", "in the sell-offs (10%+ falls in N100), the share of sell-offs in which the asset was FIRST to: flip its supertrend to SELL; close 10% below its own high; flip back to BUY after the trough."),
                       ("Bars", "in the asset's colour; the text gives its average rank, its median lead in weeks relative to N100 (negative = earlier) and the number of sell-offs it had the event in."),
                       ("Ties", "two assets doing it in the same week are both first.")]))
    H.append(box("shows", "How it shows the answer.", "The middle panel's tallest bar is SC100 (it falls 10% first and most); the left panel's is N100 (its signal fires first) — the band of the supertrend scales with each asset's own volatility, so a 10% fall trips the band of a low-vol large cap before a high-vol small cap's."))
    rows = [(a, f"first to fall 10% in {_p(sm4.loc[a, 'first_share_cross_10pct'])} of sell-offs (median lead {sm4.loc[a, 'median_lead_cross_10pct_weeks']:+.0f} wk vs N100); signal fired first in {_p(sm4.loc[a, 'first_share_signal_sell'])} (lead {sm4.loc[a, 'median_lead_signal_sell_weeks']:+.0f} wk); first back to BUY in {_p(sm4.loc[a, 'first_share_recover_buy'])}; median own fall {_pct(sm4.loc[a, 'median_own_fall'])}") for a in sm4.index]
    H.append(combos("Reading it for each asset", rows))
    H.append(columns([("episodes", "sell-offs the asset was present in."), ("first to fall 10% / signal fires first / first back to BUY", "the share of sell-offs it was first on that measure (ties counted)."), ("lead (wk)", "its median lead in weeks relative to N100's own date; negative = earlier."), ("median own fall", "its median fall from its own peak to N100's trough week.")]))

    # ---- B16 z_forward, B17 pair cards
    H.append("<h3>B16 · Do deviations pull back? (Part II §8.1) and B17 · the pair cards (§8.2)</h3>")
    H.append(_fig("6_z_forward"))
    H.append(elements([("x-axis", "the pair's deviation z this week, in five buckets: z < −2 (the first asset has lagged the second by more than 2 sd of the last year's ratio), −2 ≤ z < −1, −1 ≤ z < +1 (normal), +1 ≤ z < +2, z ≥ +2 (the first asset has run ahead). Under each label: the Newey-West t and the number of weeks."),
                       ("y-axis", "the change in ln(a/b) over the next 4 weeks (left) or 13 weeks (right): positive = the first asset gained on the second."),
                       ("Bars", "all ten pairs pooled; orange = the 'a lagged' buckets, grey = normal, blue = the 'a led' buckets."),
                       ("Dots", "each pair on its own (click the legend to isolate one); diamonds are gold pairs."),
                       ("What a pull-back would look like", "orange bars above zero (the laggard catching up) and blue bars below zero (the leader giving it back).")]))
    H.append(box("shows", "How it shows the answer.", f"The orange 1–2 sd bar is below zero and the blue 1–2 sd bar is above zero — the opposite of a pull-back: moderate deviations continued ({_pct(pz.loc[B[1], 'mean_ratio_change'])} and {_pct(pz.loc[B[3], 'mean_ratio_change'])} further over 13 weeks). The outer bars sit near zero: beyond 2 sd the run stalls but does not come back."))
    rows = []
    for p in pairs:
        g = zfp[zfp["pair"] == p].set_index("bucket")
        lagb = g.loc[B[1]] if B[1] in g.index else None; leadb = g.loc[B[3]] if B[3] in g.index else None
        a, b = p.split("|")
        txt = f"pull-back test: {pco.loc[p, 'verdict']}"
        if lagb is not None:
            txt += f"; after {a} lagged by 1–2 sd the ratio moved {_pct(lagb['mean_ratio_change'])} over 13 weeks ({'a pull-back' if lagb['mean_ratio_change'] > 0 else 'kept going'}, t {_t(lagb['hac_t_ratio_change'])}, n {int(lagb['weeks'])})"
        if leadb is not None:
            txt += f"; after {a} led by 1–2 sd, {_pct(leadb['mean_ratio_change'])} ({'a pull-back' if leadb['mean_ratio_change'] < 0 else 'kept going'}, t {_t(leadb['hac_t_ratio_change'])}, n {int(leadb['weeks'])})"
        rows.append((_x(p), txt))
    H.append(combos("Reading it for each combination", rows))
    H.append(_fig("6_pair", pair="N100|SC100"))
    H.append(elements([("Panel 1", "each leg's cumulative return, both starting at 100 on the pair's first common week, log scale — the returns the ratio is made of."),
                       ("Panel 2", "the price ratio ln(a/b) (black) with its trailing 52-week mean (dashed) and ±1 sd (darker) / ±2 sd (lighter) bands. Up = the first asset doing better."),
                       ("Panel 3", "the deviation z of that ratio (blue line); dashed lines at ±1 (grey) and ±2 (red). ▼ in the top row = the first leg's flips to SELL, ▲ in the bottom row = the second leg's; ● = a pair-trade entry, ○ = its exit (hover for the trade's result)."),
                       ("Panel 4", "the pair's 13-week correlation."),
                       ("Panel 5", "each leg's 13-week volatility (left axis) and the India VIX (dotted, right axis)."),
                       ("The facts grid and the returns table under each card", "the pair's correlation (whole sample / now), each leg's annualised return and last-52-week return, volatility and beta, the deviation now in words, the pull-back verdict; then total / annualised / last 13w / last 52w returns and each leg's annualised return in both-BUY, both-SELL and disagree weeks with the spread.")]))
    H.append(box("shows", "How it shows the answer.", "One card is shown here (N100 × SC100); the report has all ten. Read panel 3 against panel 2: every time z reaches ±2 the ratio in panel 2 is at the edge of its band — and then, more often than not, keeps going rather than returning to the dashed mean. The ▼/▲ markers show that SELL flips cluster where z is moving fast, not at its extremes."))
    rows = []
    for p in pairs:
        a, b = p.split("|")
        rr = rt.loc[p]; s = pst.loc[p]
        ptr = pts.loc[p] if p in pts.index else None
        rows.append((_x(p), f"correlation {s['corr_full']:.2f} (now {s['corr_13w_now']:.2f}); {a} {_pct(rr['annualised_a'])} a year vs {b} {_pct(rr['annualised_b'])} ({b if rr['spread_annualised'] < 0 else a} led by {abs(rr['spread_annualised']) * 100:.1f} points); in both-SELL weeks {_pct(rr['both_SELL_ann_a'], 0)} vs {_pct(rr['both_SELL_ann_b'], 0)} annualised; deviation now z {s['z_now']:+.1f} ({s['reading_now']}); |z| &gt; 2 in {_p(s['share_weeks_abs_z_gt_2'])} of weeks" + (f"; pair trade: {int(ptr['trades'])} trades, {_pct(ptr['mean_net'])} each — {ptr['verdict']}" if ptr is not None else "")))
    H.append(combos("Reading the card of each pair", rows))
    H.append(columns([("half-life (wk)", "how many weeks a deviation from the ratio's long-run level takes to halve (∞ = it does not shrink)."), ("drift per year", "the ratio's average yearly change — a steady negative number means the second asset has out-run the first year after year."), ("ADF p", "the test that the ratio has a long-run level; below 0.05 = established."), ("verdict", "pulls back / pulls back with a hedge ratio / some pull-back / no long-run level (drifts)."), ("bucket / weeks / mean_ratio_change / share_reverted / hac_t_ratio_change", "the z bucket; weeks in it; the ratio's average change over the next 13 weeks; the share of weeks the ratio moved back toward its mean; the Newey-West t of the change."), ("mean_ret_a / mean_ret_b", "each leg's own average return over the next 13 weeks from those weeks."), ("total return / annualised / last 13 weeks / last 52 weeks", "each leg's return over the common sample, per year, and recently."), ("both BUY weeks / both SELL weeks / disagree weeks, annualised", "each leg's average weekly return × 52 in the weeks the two signals were in that state (the count in brackets)."), ("spread a − b", "the first leg minus the second, for each of those rows.")]))

    # ---- B18 pair trade
    H.append("<h3>B18 · The pair-trading rule (Part II §9)</h3>")
    H.append(_fig("7_summary"))
    H.append(elements([("Left panel", "the average net P&L per trade for each pair (blue positive, red negative), in log-return units (+0.05 = a one-unit spread position made 5 points); the text gives the number of trades, the share of winners and the average weeks held."),
                       ("Right panel", "the long-only view: over the same trades, the return of the leg the rule bought (the cheap leg, blue) and the leg it sold (the rich leg, grey)."),
                       ("The rule", f"enter at |z| ≥ {C.PT_ENTRY:.0f} long the cheap leg / short the rich leg; exit at |z| ≤ {C.PT_EXIT:.1f} or after {C.PT_MAX_HOLD} weeks; {C.PT_COST_BPS} bp per leg per side.")]))
    H.append(box("shows", "How it shows the answer.", "Every bar on the left is red (every pair lost); on the right the grey bar is longer than the blue one for most pairs — the leg the rule sold kept beating the leg it bought: the laggard kept lagging. The equity chart in the report's 'More…' box shows the cumulative loss."))
    rows = [(_x(p) if p != "ALL PAIRS" else "ALL PAIRS", f"{int(pts.loc[p, 'trades'])} trades, {_p(pts.loc[p, 'win_rate_net'])} winners, {_pct(pts.loc[p, 'mean_net'])} net per trade, {_pct(pts.loc[p, 'total_net'])} in total, {pts.loc[p, 'avg_weeks']:.0f} weeks held, {_p(pts.loc[p, 'share_exit_z'])} closed by z; cheap leg {_pct(pts.loc[p, 'mean_ret_cheap_leg'])} vs rich leg {_pct(pts.loc[p, 'mean_ret_rich_leg'])} — {pts.loc[p, 'verdict']}") for p in list(pts.index)]
    H.append(combos("Reading it for each pair", rows))
    H.append(columns([("trades", "how many trades the rule made in the pair."), ("winners", "the share with a positive net P&L."), ("net per trade / total net", "the average and the sum of net P&L (gross spread minus costs)."), ("weeks held", "average holding time."), ("closed by z", "the share of trades that ended because the deviation came back inside the exit line (the rest hit the 26-week limit or the end of the data)."), ("cheap leg's return / rich leg's return", "each leg's own return over the trade."), ("verdict", "'worked' needs ≥ 5 trades, a positive mean, ≥ 60% winners and t ≥ 2.")]))

    # ---- B19 rules, B20 cells / z-path
    H.append("<h3>B19 · Who replaces a SELL — the rules (Part II §10.1), and B20 · why: the candidate's own signal (§10.2) and the z-path (§10.4)</h3>")
    H.append(_fig("8_rules"))
    H.append(elements([("One row per rule", "twelve replacement rules applied to the same SELL flips, one pick each; a rule with no qualifying candidate sits in cash that time."),
                       ("Bars", "the average return of the rule's pick over the next 4 weeks (left) and 13 weeks (right). Blue = beats cash, red = does not; grey = the two references (stay in the seller, cash); yellow = gold."),
                       ("Dashed vertical line", "cash."),
                       ("The text on each bar", "how often the pick beat cash, how often the rule actually had a pick, and the Newey-West t of its edge over cash.")]))
    H.append(box("shows", "How it shows the answer.", "The long blue bars are the rules that pick something in BUY or something that pulled ahead of the seller (and gold); the red bars are the rules that pick the most correlated or the most lagging asset regardless of its signal — those are usually in SELL too. The proposed rule, 'correlated & lagging, in BUY', has no bar of its own: it never found a candidate."))
    rows = [(r, f"had a pick {_p(R.loc[r, 'share_with_a_pick'])} of the time; return over 13 weeks {_pct(R.loc[r, f'mean_{H13}'])} (when it had a pick {_pct(R.loc[r, f'when_pick_mean_{H13}'])}); beat cash {_p(R.loc[r, f'beat_cash_{H13}'])}, beat staying {_p(R.loc[r, f'beat_stay_{H13}'])}; t vs cash {_t(R.loc[r, f'hac_t_minus_cash_{H13}'])}; held to its own SELL {_pct(R.loc[r, 'hold_mean'])} over {R.loc[r, 'hold_weeks_mean']:.0f} weeks" if pd.notna(R.loc[r, 'hold_weeks_mean']) else f"had a pick {_p(R.loc[r, 'share_with_a_pick'])} of the time; return over 13 weeks {_pct(R.loc[r, f'mean_{H13}'])}; beat cash {_p(R.loc[r, f'beat_cash_{H13}'])}, beat staying {_p(R.loc[r, f'beat_stay_{H13}'])}") for r in R.index]
    H.append(combos("Reading each rule", rows))
    H.append(_fig("8_cells"))
    H.append(_fig("8_z_path"))
    H.append(elements([("B20 cells, x-axis", "the candidate's deviation against the seller at the flip: lagging (z ≤ −1, it has fallen behind the seller over the last year), neutral, leading (z ≥ +1, it has run ahead)."),
                       ("B20 cells, bars", "left panel: by the candidate's OWN signal at the flip (blue BUY, red SELL); right panel: by whether it was correlated with the seller (52-week correlation ≥ 0.5). Height = the candidate's average return over the next 13 weeks; n = candidate rows."),
                       ("Dashed and dotted lines", "cash, and staying in the seller."),
                       ("B20 z-path, panels", "one per seller; x = weeks from the seller's SELL flip; y = the deviation z of each other asset against the seller (+ = that asset ahead), averaged over the seller's flips; black = all candidates."),
                       ("B20 z-path, reading", "a line still rising after week 0 means the seller kept falling behind the others after its signal fired.")]))
    H.append(box("shows", "How it shows the answer.", "In the cells chart the blue (BUY) bars are tall in every deviation bucket and the red (SELL) bars are near or below zero — the candidate's own signal is what matters; 'correlated and lagging' is the lowest bar because such a candidate is almost always in SELL. In the z-path panels the lines mostly keep rising after the flip: the seller keeps falling behind."))
    rows = []
    for dev in ("lagging", "neutral", "leading"):
        for stt in ("BUY", "SELL"):
            c = cell(dev, "any", stt)
            if c is not None:
                rows.append((f"candidate {dev}, own signal {stt}", f"{int(c['rows'])} candidate rows: returned {_pct(c['cand_mean'])} over 13 weeks (beat cash {_p(c['share_cand_beat_cash'])}); the seller made {_pct(c['stay_mean'])}, cash {_pct(c['cash_mean'])}"))
    for corr_ in ("True", "False"):
        for dev in ("lagging", "neutral", "leading"):
            c = cell(dev, corr_, "any")
            if c is not None:
                rows.append((f"candidate {dev}, {'correlated' if corr_ == 'True' else 'not correlated'} with the seller", f"{int(c['rows'])} rows: {_pct(c['cand_mean'])} (beat cash {_p(c['share_cand_beat_cash'])})"))
    H.append(combos("Reading each cell", rows))
    prow = []
    for _, r in bp.sort_values(["seller", "candidate"]).iterrows():
        prow.append((f"{r['seller']} sells → {r['candidate']}", f"{int(r['events'])} flips; 52-week correlation with the seller {r['avg_corr_52w']:+.2f} on average; the candidate was lagging {_p(r['share_lagging'])} / leading {_p(r['share_leading'])} of the time and in BUY {_p(r['share_cand_in_BUY'])}; next 13 weeks: candidate {_pct(r[f'cand_mean_{H13}'])} vs staying {_pct(r[f'stay_mean_{H13}'])} vs cash {_pct(r[f'cash_mean_{H13}'])} (beat cash {_p(r[f'share_cand_beat_cash_{H13}'])}); held until its own SELL: {_pct(r['hold_mean'])} over {r['hold_weeks_mean']:.0f} weeks"))
    H.append(combos("Reading it for each pair — seller × candidate (table 8_by_pair)", prow))
    zpa = zp[zp["candidate"] == "all candidates"].pivot(index="seller", columns="k", values="mean_z")
    rows = [(f"after {a} sells", f"the other assets' deviation against {a}: {zpa.loc[a, -4]:+.2f} four weeks before → {zpa.loc[a, 0]:+.2f} at the flip → {zpa.loc[a, 13]:+.2f} thirteen weeks after ({'they kept pulling ahead' if zpa.loc[a, 13] > zpa.loc[a, 0] + 0.1 else ('they fell back' if zpa.loc[a, 13] < zpa.loc[a, 0] - 0.1 else 'little change')}); best replacement by average: " + (lambda g: f"{g.loc[g[f'cand_mean_{H13}'].idxmax(), 'candidate']} {_pct(g[f'cand_mean_{H13}'].max())} vs staying {_pct(g[f'stay_mean_{H13}'].iloc[0])}")(bp[bp["seller"] == a].reset_index(drop=True))) for a in zpa.index]
    H.append(combos("Reading each seller", rows))
    H.append(columns([("had a pick", "how often the rule found a qualifying candidate (otherwise it sat in cash)."), ("return, 13 wk", "the rule's average return over the 13 weeks after the flip, over all flips."), ("when it had a pick", "the same, counting only the flips where it chose someone."), ("beat cash / beat staying", "the share of flips the pick beat cash / beat staying in the seller."), ("t vs cash", "the Newey-West t of its average edge over cash (≥ 20 flips)."), ("held to its own SELL / weeks held", "the pick held until its own signal turned SELL (capped at 26 weeks), and the average weeks that took."), ("candidate's deviation", "lagging / neutral / leading: the candidate's z against the seller at the flip."), ("own signal", "the candidate's supertrend state at the flip."), ("rows", "candidate rows in the cell (each flip contributes up to four candidates)."), ("candidate return, 13 wk / beat cash / the seller made", "the candidate's average return over the next 13 weeks, the share of rows it beat cash, and the seller's own return over the same weeks.")]))

    # ======================================================== PART C ==
    H.append("<h2>Part C — the column dictionary</h2>")
    H.append("<p class='small'>Every column name that appears in the report's tables, in one line each. Grouped by the table it comes from; the <code>outputs/tables/</code> file is in brackets.</p>")
    D = {
        "The correlation matrices (0_corr_matrix_*)": [("N100, MC100, SC100, Micro250, GoldMCX", "the five assets; a cell is the correlation of the row series with the column series"), ("VIX (change)", "the weekly change in India VIX — a reference row, not an asset"), ("Cash (liquid)", "the liquid ETF — a reference row"), ("first / last / weeks (coverage)", "each series' first and last week with a return, and the count")],
        "By signal state (1_corr_by_signal_state, 1_cov_decomposition)": [("pair / a / b", "the two assets; a is the first-named"), ("state", "both BUY / both SELL / disagree / all weeks"), ("weeks", "weeks in that state"), ("corr", "the pair's correlation on those weeks"), ("cov_ann", "the pair's covariance on those weeks, annualised (× 52)"), ("vol_a_ann / vol_b_ann", "each leg's volatility on those weeks, annualised"), ("mean_ret_a_ann / mean_ret_b_ann", "each leg's average weekly return on those weeks × 52"), ("corr_buy / corr_sell / corr_change", "the pair's correlation in both-BUY / both-SELL weeks, and the difference"), ("cov_buy / cov_sell", "its covariance in each"), ("cov_ratio_sell_over_buy", "cov_sell ÷ cov_buy — how many times larger the risk is in SELL"), ("share_of_cov_change_from_corr / _from_vol", "the split of the covariance rise into the correlation part and the volatility part (they add to 1)"), ("vol_a_buy / vol_a_sell / vol_b_buy / vol_b_sell", "each leg's volatility in each state"), ("weeks_buy / weeks_sell", "weeks in each state")],
        "Around the flips (1_corr_around_flip, 5_vol_around_flip)": [("ref", "the asset whose flips are lined up"), ("event", "flip to SELL / flip to BUY"), ("weeks_from_flip", "the offset k: 0 is the flip week"), ("mean_corr", "the group correlation G at that offset, averaged over the asset's flips"), ("ref_own_corr", "the flipping asset's own correlation with the other indices at that offset"), ("ref_trailing_13w_return", "its trailing 13-week return at that offset"), ("n_events", "flips averaged"), ("vol_<asset>", "that asset's 13-week vol at the offset"), ("vix_level", "the India VIX at the offset")],
        "Pair by pair, where the report repeats a group study for every pair (1_corr_around_flip_pairs, 1_corr_by_breadth_pairs, 2_by_year_pairs, 2_corr_quintile_returns_pairs, 2_high_corr_episodes_pairs, 2_high_corr_pairs_summary, 5_vol_corr_lead_pairs, 5_together_pairs, 5_corr_rise_paths_pairs, 5_corr_rise_pairs_summary; the pair columns of 1_corr_timeseries, 2_timeline and 3_corr_series)": [("pair, or a column named a|b", "the two assets; a is the first-named; all ten pairs, gold included"), ("leg", "which of the pair's two assets a return belongs to"), ("a_… / b_…", "a figure for the first / the second leg"), ("z_a|b", "the pair's 13-week correlation as a z-score against its own 104-week history"), ("share_pair_also_reached_plus1", "the share of the group's correlation jumps in which the pair's own z reached +1 within −8 … +4 weeks"), ("mean_z_at_jump", "the pair's z in the group's jump week"), ("weeks_above_0.90 / share_of_weeks_above_0.90 / stretches / median_weeks", "how often the pair's 13-week correlation was above 0.90, and its stretches of 8+ weeks there"), ("series (5_corr_rise_paths_pairs)", "the pair the path belongs to")],
        "By breadth (1_corr_by_breadth)": [("none in SELL … all in SELL", "the bands of how many of the four index signals were in SELL"), ("mean / median / weeks", "G's average, middle value and the week count in the band")],
        "Volatility by state (5_vol_by_state)": [("weeks_BUY / weeks_SELL", "weeks the asset's own signal was BUY / SELL"), ("vol_BUY / vol_SELL / vol_all", "its realised vol in each, and overall"), ("vol_ratio_SELL_over_BUY", "vol_SELL ÷ vol_BUY"), ("mean_weekly_ret_BUY / _SELL", "its average weekly return in each state"), ("share_weeks_down_BUY / _SELL", "the share of those weeks that were down"), ("worst_week_BUY / _SELL", "its worst single week in each state"), ("vix_mean_BUY / vix_mean_SELL / vix_median_* / vix_ratio_SELL_over_BUY", "the average (median) India VIX in each state and the ratio")],
        "Regimes (5_corr_by_regime_long, 5_corr_by_regime_*, 5_corr_by_vol_regime)": [("regime", "N100 vol / pair's own vol / VIX level"), ("bucket (Q1 … Q5)", "the fifth of weeks the week falls in under that measure, calmest to wildest"), ("lo / hi", "the range of the measure inside the bucket"), ("mean_corr", "the pair's average 13-week correlation in the bucket"), ("Q5_minus_Q1 / wild − calm", "the difference between the wildest and calmest fifths"), ("spearman_corr_vs_regime", "the rank correlation between the pair's correlation series and the regime series")],
        "Lead / lag (5_vol_corr_lead)": [("k_weeks", "the lead / lag in weeks"), ("corr_dvol_t_dcorr_t_plus_k", "corr(change in realised vol at t, change in G at t + k)"), ("corr_dvix_t_dcorr_t_plus_k", "the same with the VIX"), ("weeks / weeks_vix", "weeks used")],
        "History (2_by_year, 2_corr_quintile_returns_long, 2_high_corr_episodes)": [("mean_corr / min_corr / max_corr", "G's average / lowest / highest over the year"), ("weeks_corr_above_0.9", "weeks the indices were effectively one block"), ("ret_<asset>", "the asset's return over the year"), ("bucket (Q1 … Q5)", "the fifth of weeks by G, least to most correlated"), ("corr_lo / corr_hi", "G's range inside the bucket"), ("same_quarter_mean / same_quarter_hit", "the asset's average return over the 13 weeks ending in those weeks, and the share positive"), ("next_quarter_mean / next_quarter_hit", "over the following 13 weeks (t+1 … t+13)"), ("start / end / weeks / peak_corr", "a stretch above 0.90: its dates, length and peak"), ("<asset>_during / <asset>_next_13w", "the asset's return during the stretch and over the 13 weeks after")],
        "Correlation breaks (3_corr_episodes, 3_asset_breaks, 3_summary, 3_after_break_summary)": [("c_<asset> / G / z_c_<asset> / z_G", "each index's correlation with the others, the group average, and their z-scores vs their own 104-week history"), ("episode / start / end / weeks", "a decorrelation episode (z_G ≤ −1 until it is back above −0.5)"), ("G_before / G_min / G_min_date / drop / z_min", "the group's correlation over the 8 weeks before, its lowest point and date, the fall, the lowest z"), ("break_date / lead_vs_group_weeks", "the index's own break date and its distance from the group's (negative = earlier)"), ("rank / rank_deepest / rank_drop", "its rank in the episode by date / by depth at the group break / by size of its own fall"), ("z_at_group_break", "its z in the week the group broke"), ("corr_before / corr_min / corr_drop / drop_share", "its correlation with the others before, its lowest, the fall, and the fall as a share of all indices' falls"), ("rel_13w_return_at_break / direction", "its 13-week return minus the others' at its break; up-break (positive) / down-break"), ("share_first / share_first_alone / avg_rank / median_lead_weeks", "earliest-by-date shares and ranks"), ("share_deepest / avg_rank_deepest / share_biggest_drop / avg_rank_drop / avg_drop_share", "the two tie-breakers"), ("combined_rank", "the average of the three average ranks"), ("share_up_breaks", "the share of its breaks that were upward"), ("breaker / rest_of_group / gold / share_breaker_ahead", "after a break: the breaker's return, the others' average, gold's, and how often the breaker was ahead")],
        "Price sell-offs (4_price_episodes, 4_summary)": [("peak_date / trough_date / end_date / depth", "a sell-off in N100: its peak, trough, recovery date and depth"), ("weeks_peak_to_trough / weeks_to_recover", "its lengths"), ("first_share_<measure> / avg_rank_<measure> / median_lead_<measure>_weeks / n_<measure>", "per asset and measure (signal_sell, cross_10pct, recover_buy): the share of sell-offs it was first, its average rank, its median lead vs N100 in weeks, and the count"), ("median_own_fall", "its median fall from its own peak")],
        "Pairs (6_pair_stats, 6_pair_returns, 6_pair_coint, 6_z_forward, 6_ratio_z)": [("corr_full / corr_13w_now / corr_52w_now / corr_13w_median / corr_13w_min / corr_13w_max", "the pair's correlation: whole sample, now (13w / 52w), and the 13-week series' median / lowest / highest"), ("vol_a_full / vol_b_full / vol_a_13w_now / vol_b_13w_now / vol_ratio_a_over_b_now", "each leg's volatility, overall and now, and the ratio"), ("beta_a_on_b_52w_now / beta_a_on_b_full", "how much a moves for a 1% move in b"), ("z_now / z_date / reading_now", "the deviation z of ln(a/b) against its trailing 52 weeks, in words"), ("share_weeks_abs_z_gt_1 / _gt_2 / n_2sd_deviations / z_min / z_max", "how often and how far the pair deviated"), ("ratio_total_change", "a's total out- or under-performance of b over the sample"), ("total_a / total_b / annualised_a / annualised_b / last_13w_* / last_52w_*", "each leg's returns"), ("both_BUY_ann_* / both_SELL_ann_* / disagree_ann_* / *_weeks", "each leg's annualised average return in the weeks the two signals were in that state, and the counts"), ("spread_<k>", "a minus b for each of those"), ("share_weeks_a_beat_b", "the share of weeks a's return beat b's"), ("adf_p_log_ratio / eg_p_a_on_b / eg_p_b_on_a", "the tests that the ratio has a long-run level (p below 0.05 = established)"), ("halflife_weeks_full / halflife_weeks_vs_52w_mean", "how many weeks a deviation takes to halve, against the long-run level / against the moving 52-week mean"), ("ratio_drift_per_year", "the ratio's average yearly change"), ("verdict", "pulls back / with a hedge ratio / some pull-back / no long-run level"), ("horizon_weeks / bucket / weeks / mean_ratio_change / median_ratio_change / share_reverted / mean_move_toward_mean / hac_t_ratio_change / mean_ret_a / mean_ret_b", "the z-bucket table: the horizon, the bucket, weeks in it, the ratio's later change (mean, median), the share of weeks it moved back toward its mean, the average move in the pull-back direction, the Newey-West t, each leg's own later return")],
        "The pair-trading rule (7_trades, 7_summary, 7_sensitivity)": [("entry_date / exit_date / weeks / side / cheap_leg / rich_leg", "a trade: its dates, length, direction (+1 long a short b), the leg bought and the leg sold"), ("z_entry / z_exit / exit_reason", "the deviation at entry and exit; why it closed (z back inside / max hold / end of data)"), ("gross / cost / net / worst_point", "the spread earned, the cost, the net, and the lowest the running P&L got"), ("ret_cheap_leg / ret_rich_leg", "each leg's own return over the trade"), ("trades / win_rate_net / mean_net / median_net / total_net / t_mean_net / avg_weeks", "per pair: count, share of winners, average / median / total net, a plain t on the trade P&Ls, average weeks held"), ("share_exit_z / share_exit_max_hold", "how trades ended"), ("mean_ret_cheap_leg / mean_ret_rich_leg / share_cheap_beat_rich", "the long-only view"), ("worst_trade / best_trade / net_per_year / equity_drawdown / verdict", "extremes, net per year, the deepest fall of the cumulative P&L, and the verdict"), ("entry / exit / require_cheap_leg_BUY / pairs_positive / pairs_traded", "the sensitivity grid's settings and results")],
        "Who replaces a SELL (8_events, 8_candidates, 8_cells, 8_by_pair, 8_rules, 8_z_path, 8_current)": [("seller / date", "the asset whose signal flipped to SELL, and the week"), ("candidate", "one of the other assets"), ("corr_52w / correlated", "the candidate's 52-week correlation with the seller; correlated if ≥ 0.5"), ("z_cand_vs_seller / deviation", "the candidate's deviation against the seller (+ = ahead); lagging (≤ −1) / neutral / leading (≥ +1)"), ("cand_state", "the candidate's own signal at the flip"), ("cand_rel_13w / cand_ret_13w / seller_ret_13w", "13-week returns at the flip"), ("fwd_cand_h / fwd_seller_h / fwd_cash_h / fwd_gold_h", "returns over the h weeks after the flip (t+1 … t+h)"), ("hold_ret / hold_weeks / hold_end / hold_seller_ret / hold_cash_ret", "the candidate held until its own SELL (capped at 26 weeks), how long, how it ended, and the seller / cash over the same weeks"), ("rows / events", "candidate rows and distinct flips in a cell"), ("cand_mean / cand_median / cand_hit_positive / stay_mean / cash_mean / cand_minus_stay / cand_minus_cash / share_cand_beat_stay / share_cand_beat_cash", "the cell's averages and hit rates"), ("rule / what it does / share_with_a_pick / events_with_a_pick", "the replacement rule, its definition, how often it found a candidate"), ("mean_h / median_h / hit_positive_h / beat_cash_h / beat_stay_h / minus_cash_h / hac_t_minus_cash_h / worst_h", "the rule's results over h weeks"), ("when_pick_*", "the same counting only the flips where the rule picked someone"), ("hold_mean / hold_weeks_mean", "the pick held until its own SELL"), ("k / mean_z / median_z / n", "the z-path: offset from the flip, the average and median deviation of the candidate against the seller, and the count"), ("seller_if_in_SELL / seller_state", "the current table: each asset as a would-be seller, and whether it is in SELL now")],
    }
    for grp, items in D.items():
        H.append(f"<h4>{grp}</h4><div class='dict'>" + "".join(f"<p><code>{_esc(k)}</code> — {v}</p>" for k, v in items) + "</div>")
    H.append(f"<p class='small'>Built from the same tables as the report ({meta.get('run_at')}). <code>GLOSSARY.md</code> has the longer, plain-language version of every entry.</p>")
    H.append("<script>document.querySelectorAll('details').forEach(d => d.addEventListener('toggle', () => { if (d.open) d.querySelectorAll('.js-plotly-plot').forEach(p => { try { Plotly.Plots.resize(p); } catch (e) {} }); }));</script></body></html>")
    return "\n".join(H)


def main() -> int:
    out = build()
    C.REPORT_GUIDE_HTML.write_text(out, encoding="utf-8")
    print(f"wrote {C.REPORT_GUIDE_HTML.relative_to(C.ROOT)} ({len(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
