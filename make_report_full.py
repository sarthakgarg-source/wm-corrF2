"""FULL version: build outputs/report_full.html - every table, every chart, the five-part scaffold in every section.
The default report (make_report.py) is the clean version; this one is for digging.

Every section follows the same shape so it can be read without the code:
  THE QUESTION   what is being asked, in one or two sentences
  HOW IT IS MEASURED   the steps, with the parameters from config.py
  HOW TO READ THE TABLE / CHART   what a row, a column, a big or small number means
  WHAT IT SAYS   the findings, every number computed from the tables
  WHAT IT DOES NOT SAY   the caveats that stop the finding being over-read
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
import figs

T, CH = C.TABLES, C.CHARTS
PLOTLY_JS = (Path(__import__("plotly").__file__).parent / "package_data" / "plotly.min.js")


# ------------------------------------------------------------- helpers --
def _csv(name, **kw):
    return pd.read_csv(T / f"{name}.csv", **kw)


def _img(name, **kw):
    """An interactive Plotly figure (figs.py), rendered as a div; the library is embedded once in <head>."""
    try:
        return f"<div class='fig'>{figs.html(name, **kw)}</div>"
    except Exception as e:  # noqa: BLE001 - a missing table must show in the report, not kill it
        return f"<p class='missing'>chart {name} failed: {e}</p>"


def _tbl(df, fmt="{:.2f}", index=True, max_rows=80, pct_cols=(), int_cols=()):
    d = df.head(max_rows).copy()
    for c in d.columns:
        if c in pct_cols:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v * 100:+.1f}%")
        elif c in int_cols:
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{int(v)}")
        elif pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else fmt.format(v))
    return d.to_html(index=index, border=0, classes="t", escape=True, na_rep="")


def _pct(v, d=1):
    return "n/a" if pd.isna(v) else f"{v * 100:+.{d}f}%"


def _p(v, d=0):
    return "n/a" if pd.isna(v) else f"{v * 100:.{d}f}%"


def _n(v, d=2):
    return "n/a" if pd.isna(v) else f"{v:.{d}f}"


def _t(v):
    return "n/a" if pd.isna(v) else f"{v:+.1f}"


def _wk(v):
    if pd.isna(v):
        return "did not break"
    v = int(v)
    return "the same week as" if v == 0 else (f"{-v} week{'s' if v != -1 else ''} before" if v < 0 else f"{v} week{'s' if v != 1 else ''} after")


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _ask(q, how, read):
    return (f"<div class='ask'><p><b>The question.</b> {q}</p><p><b>How it is measured.</b> {how}</p><p><b>How to read it.</b> {read}</p></div>")


def _says(txt):
    return f"<div class='says'><b>What it says.</b> {txt}</div>"


def _not(txt):
    return f"<div class='not'><b>What it does not say.</b> {txt}</div>"


def _verdict_word(mean, cash, hit):
    if pd.isna(mean):
        return "no reading"
    if mean > cash and hit >= 0.6:
        return "better than cash, most of the time"
    if mean > cash:
        return "better than cash on average, not reliably"
    return "no better than cash"


# ------------------------------------------------------------- build --
def build() -> str:
    meta = json.loads((T / "meta.json").read_text())
    # section 1-4 tables (existing modules)
    bs = _csv("1_corr_by_signal_state"); dec = _csv("1_cov_decomposition"); afa = _csv("1_corr_around_flip"); af = afa[afa["ref"] == C.BENCHMARK]; bb = _csv("1_corr_by_breadth", index_col=0)
    yr = _csv("2_by_year", index_col=0); lg = _csv("2_corr_quintile_returns_long"); piv = _csv("2_corr_quintile_returns", index_col=0); hc = _csv("2_high_corr_episodes")
    S = _csv("3_corr_series", index_col=0, parse_dates=True)
    cep = _csv("3_corr_episodes", parse_dates=["start", "end", "G_min_date"])
    ab = _csv("3_asset_breaks", parse_dates=["break_date"])
    sm = _csv("3_summary", index_col=0); psm = _csv("3_pair_summary", index_col=0); pb = _csv("3_pair_breaks", parse_dates=["break_date"])
    bsf = _csv("3_before_selloff"); afs = _csv("3_after_break_summary"); sens = _csv("3_sensitivity", index_col=0)
    pep = _csv("4_price_episodes"); sm4 = _csv("4_summary", index_col=0); ld4 = _csv("4_dates_and_leads")
    cov = _csv("coverage_weekly", index_col=0)
    # 5 volatility
    vs = _csv("5_vol_by_state", index_col=0); vafa = _csv("5_vol_around_flip"); vaf = vafa[vafa["ref"] == C.BENCHMARK]; regl = _csv("5_corr_by_regime_long"); rt = _csv("6_pair_returns", index_col=0); cv = _csv("5_corr_by_vol_regime", index_col=0); vedges = _csv("5_vol_regime_edges", index_col=0); vl = _csv("5_vol_corr_lead", index_col=0)
    # 6 pairs
    pst = _csv("6_pair_stats", index_col=0); pco = _csv("6_pair_coint", index_col=0); zf = _csv("6_z_forward"); pa = _csv("6_path_after_2sd"); cur6 = _csv("6_current", index_col=0)
    # 7 pair trade
    tr = _csv("7_trades", parse_dates=["entry_date", "exit_date"]); pts = _csv("7_summary", index_col=0); ptsens = _csv("7_sensitivity")
    # 8 replacement
    ev = _csv("8_events", parse_dates=["date"]); cr = _csv("8_candidates", parse_dates=["date"]); cells = _csv("8_cells"); bp = _csv("8_by_pair"); R = _csv("8_rules", index_col=0); Rs = _csv("8_rules_by_seller"); cur8 = _csv("8_current")

    # ---- derived quantities used in several places
    eqp = dec[~dec["pair"].str.contains("Gold")]; gp = dec[dec["pair"].str.contains("Gold")]
    sell_flip = af[af["event"] == "flip to SELL"].set_index("weeks_from_flip")["mean_corr"]
    n100 = lg[lg["asset"] == "N100"].set_index("bucket"); sc = lg[lg["asset"] == "SC100"].set_index("bucket")
    eq = list(sm.index)
    combined_first = sm["combined_rank"].idxmin()
    date_first = sm["avg_rank"].idxmin(); deep_first = sm["avg_rank_deepest"].idxmin(); drop_first = sm["avg_rank_drop"].idxmin()
    up = afs[afs["direction"].str.startswith("up")].set_index("weeks_ahead"); dn = afs[afs["direction"].str.startswith("down")].set_index("weeks_ahead")
    n_warn = int(bsf["group_broke_before_peak"].sum()); n_sell = len(bsf)
    last = S.dropna(subset=["z_G"]).iloc[-1]
    eq4 = [a for a in C.EQUITY if a in sm4.index]
    first_cross = sm4.loc[eq4, "avg_rank_cross_10pct"].idxmin(); first_sig = sm4.loc[eq4, "avg_rank_signal_sell"].idxmin(); first_rec = sm4.loc[eq4, "avg_rank_recover_buy"].idxmin()
    H13 = C.PAIR_FWD[-1]; H4 = C.PAIR_FWD[0]
    pooled = zf[(zf["pair"] == "ALL PAIRS")]
    pz = pooled[pooled["horizon_weeks"] == H13].set_index("bucket"); pz4 = pooled[pooled["horizon_weeks"] == H4].set_index("bucket")
    B = list(pz.index)                                                   # the five bucket labels in order
    idx_pairs = [p for p in pst.index if "Gold" not in p]; gold_pairs = [p for p in pst.index if "Gold" in p]
    vsf = vaf[vaf["event"] == "flip to SELL"].set_index("weeks_from_flip"); vbf = vaf[vaf["event"] == "flip to BUY"].set_index("weeks_from_flip")
    qcols = [c for c in cv.columns if c.startswith("Q") and "_" not in c]
    has_vix = "vix_mean_BUY" in vs.columns and vs["vix_mean_BUY"].notna().any()
    vix_last = meta.get("vix_last")
    regimes = list(dict.fromkeys(regl["regime"]))
    regw = {rg: regl[regl["regime"] == rg].pivot(index="pair", columns="bucket", values="mean_corr").reindex([p for p in pst.index if p in set(regl["pair"])]) for rg in regimes}
    vlv = meta.get("vol_corr_level", {})
    # around-flip summaries for every reference asset
    def _af_row(ref, ev, k, col="mean_corr"):
        g = afa[(afa["ref"] == ref) & (afa["event"] == ev) & (afa["weeks_from_flip"] == k)]
        return g[col].iloc[0] if len(g) else np.nan
    def _vaf_row(ref, ev, k, col):
        g = vafa[(vafa["ref"] == ref) & (vafa["event"] == ev) & (vafa["weeks_from_flip"] == k)]
        return g[col].iloc[0] if len(g) and col in g.columns else np.nan
    refs_all = [a for a in C.ASSETS if a in set(afa["ref"])]
    ptall = pts.loc["ALL PAIRS"]; ptp = pts.drop(index="ALL PAIRS")
    n_pairs_neg = int((ptp["total_net"] < 0).sum()); n_pairs_judged = int((ptp["trades"] >= C.PT_MIN_TRADES).sum())
    best_pt = ptp["mean_net"].idxmax(); worst_pt = ptp["mean_net"].idxmin()
    sens_best = ptsens.sort_values("mean_net", ascending=False).iloc[0]
    rules_rank = R.sort_values(f"mean_{H13}", ascending=False)
    top_rule = rules_rank.index[0]; cash13 = R.loc["cash", f"mean_{H13}"]; stay13 = R.loc["stay in the seller", f"mean_{H13}"]
    lag_rule = R.loc["correlated & lagging"]; lagbuy_rule = R.loc["correlated & lagging, in BUY"]; lead_rule = R.loc["most leading (any corr)"]; corrbuy = R.loc["most correlated, in BUY"]; gold_rule = R.loc["gold"]; mc_rule = R.loc["most correlated"]; mlag = R.loc["most lagging (any corr)"]; mom = R.loc["best 13w momentum, in BUY"]
    c13 = cells[cells["horizon_weeks"] == H13]
    _empty_cell = pd.Series({"rows": 0, "events": 0, "cand_mean": np.nan, "cand_median": np.nan, "cand_hit_positive": np.nan, "stay_mean": np.nan, "cash_mean": np.nan, "cand_minus_stay": np.nan, "cand_minus_cash": np.nan, "share_cand_beat_stay": np.nan, "share_cand_beat_cash": np.nan})

    def cell(dev, corr, stt):
        m = c13[(c13["deviation"] == dev) & (c13["correlated"] == corr) & (c13["cand_state"] == stt)]
        return m.iloc[0] if len(m) else _empty_cell
    c_lag_buy = cell("lagging", "any", "BUY"); c_lag_sell = cell("lagging", "any", "SELL"); c_lead_buy = cell("leading", "any", "BUY"); c_lead_sell = cell("leading", "any", "SELL"); c_neu_buy = cell("neutral", "any", "BUY"); c_neu_sell = cell("neutral", "any", "SELL")
    c_buy = cell("any", "any", "BUY"); c_sellst = cell("any", "any", "SELL"); c_all = cell("any", "any", "any")
    c_lag_corr = cell("lagging", "True", "any"); c_lead_corr = cell("leading", "True", "any"); c_lag_unc = cell("lagging", "False", "any"); c_lead_unc = cell("leading", "False", "any"); c_neu_corr = cell("neutral", "True", "any"); c_neu_unc = cell("neutral", "False", "any")
    n_ev = len(ev); ev_by = ev.groupby("seller").size()
    coint_ok = [p for p in pco.index if pco.loc[p, "verdict"].startswith("pulls back")]
    nz = (cur8[cur8["seller_state"] == "SELL"]["seller_if_in_SELL"].unique().tolist())
    cash_ok = bool(meta.get("cash_present", False))
    flips_per = {a: int(ev_by.get(a, 0)) for a in C.ASSETS}
    _st = _csv("sig_states_weekly", index_col=0)
    sell_share = {a: float((_st[a].dropna() == -1).mean()) for a in C.ASSETS if a in _st.columns}
    wk_ex = bs[(bs["pair"] == "N100|SC100") & (bs["state"] == "both SELL")].iloc[0]
    flip_all_str = "; ".join(f"{a} {_af_row(a, 'flip to SELL', -4):.2f} → {_af_row(a, 'flip to SELL', 0):.2f} → {_af_row(a, 'flip to SELL', 8):.2f}" for a in refs_all if a != C.BENCHMARK)
    vol_bs_str = ", ".join(f"{a} {_p(vs.loc[a, 'vol_BUY'])} → {_p(vs.loc[a, 'vol_SELL'])}" for a in C.EQUITY if a in vs.index)
    worst_str = ", ".join(f"{a} {_pct(vs.loc[a, 'worst_week_SELL'])}" for a in C.EQUITY if a in vs.index)
    mean_sell_str = ", ".join(f"{a} {_pct(vs.loc[a, 'mean_weekly_ret_SELL'], 2)}/wk" for a in C.EQUITY if a in vs.index)
    gold_q_str = ", ".join(f"{p.replace('|', '×')} {cv.loc[p, 'Q5_minus_Q1']:+.2f}" for p in gold_pairs)
    order4_str = ", ".join(f"{r} {_pct(R.loc[r, f'mean_{H4}'])}" for r in ["most leading (any corr)", "gold", "most correlated, in BUY", "stay in the seller", "cash"])
    now_pairs_str = "; ".join(f"{p.replace('|', '×')} z {cur6.loc[p, 'z_now']:+.1f} ({cur6.loc[p, 'reading']})" for p in idx_pairs)

    css = """
    body{font-family:Inter,Helvetica,Arial,sans-serif;max-width:1400px;margin:0 auto;padding:24px 20px;color:#0b0b0b;background:#fcfcfb;line-height:1.55}
    h1{font-size:25px;margin:0 0 4px} h2{font-size:20px;margin:46px 0 8px;border-bottom:2px solid #e1e0d9;padding-bottom:4px} h3{font-size:15.5px;margin:28px 0 6px} h4{font-size:13.5px;margin:18px 0 4px;color:#52514e}
    .sub{color:#52514e;margin-bottom:18px} .fig{margin:14px 0;border:1px solid #e1e0d9;border-radius:4px;padding:4px 4px 2px;background:#fcfcfb} .fignote{font-size:12px;color:#52514e;margin:2px 6px 8px}
    .t{border-collapse:collapse;font-size:12.3px;margin:8px 0;display:block;overflow-x:auto;max-width:100%} .t th,.t td{padding:3px 8px;text-align:right;border-bottom:1px solid #eeede8} .t th{color:#52514e;font-weight:600} .t td:first-child,.t th:first-child{text-align:left}
    .box{background:#f3f3ef;border-left:4px solid #2a78d6;padding:10px 14px;margin:14px 0} .warn{background:#fdecea;border-left:4px solid #e34948;padding:10px 14px;margin:14px 0;font-weight:600}
    .ask{background:#fbfbf7;border:1px solid #e1e0d9;padding:8px 14px;margin:10px 0;font-size:13.2px;color:#2b2b28} .ask p{margin:5px 0}
    .says{background:#eef5fd;border-left:4px solid #2a78d6;padding:9px 14px;margin:12px 0;font-size:13.6px} .not{background:#fbf6ea;border-left:4px solid #eda100;padding:8px 14px;margin:10px 0;font-size:13px;color:#2b2b28}
    .ep{border-left:3px solid #c3c2b7;padding:4px 12px;margin:8px 0;font-size:13.5px} .card{border:1px solid #e1e0d9;border-radius:6px;padding:10px 16px;margin:22px 0;background:#fdfdfb}
    .missing{color:#e34948} .small{font-size:12.5px;color:#52514e} code{background:#f0efec;padding:1px 4px;border-radius:3px;font-size:12.5px} ul{margin:6px 0} li{margin:3px 0} ol li{margin:6px 0}
    .toc{columns:2;font-size:13px;list-style:none;padding-left:4px} .toc li{margin:1px 0} .kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:6px 18px;font-size:12.8px;margin:8px 0} .kv div{border-bottom:1px dotted #e1e0d9;padding:2px 0} .kv b{color:#52514e;font-weight:600}
    """
    H = [f"<!doctype html><html><head><meta charset='utf-8'><title>wm-BREAK · Study I</title><style>{css}</style><script>{PLOTLY_JS.read_text(encoding='utf-8')}</script></head><body>"]
    H.append("<h1>Study I — full version: correlation, volatility and the signal; each pair on its own; the pair-trading rule; who replaces a SELL</h1>")
    H.append(f"<div class='sub'>wm-BREAK · weekly, {C.START} → {meta['last_session']} · assets {', '.join(C.ASSETS)}{' · cash comparator ' + C.CASH if cash_ok else ''} · built {meta['run_at']}</div>")
    if meta.get("synthetic"):
        H.append("<div class='warn'>SYNTHETIC DEMO DATA — nothing below is a market result.</div>")
    H.append("<div class='box'><b>How this report is laid out.</b> Every section has the same five parts: <i>the question</i>, <i>how it is measured</i> (with the settings, all of which live in <code>config.py</code>), <i>how to read the table or chart</i>, <i>what it says</i> (every number computed from <code>outputs/tables/</code>), and <i>what it does not say</i>. "
             "Three ideas are used over and over and are explained once, in §1: the <b>z-score</b> (how unusual a number is against its own recent history), <b>forward returns</b> (always the weeks <i>after</i> the decision week), and the <b>t-statistic</b> (whether an average could be luck). <code>GLOSSARY.md</code> has every column in plain words.</div>")
    H.append("<ul class='toc'><li>0. The answers</li><li>1. The data and the building blocks</li><li>2. Correlation and covariance vs the signal</li><li>3. Volatility vs the signal, and volatility vs correlation</li><li>4. The historical pattern: correlation and returns</li><li>5. Who breaks the correlation first</li><li>6. Who breaks in price and signal</li><li>7. Each pair separately (ten cards)</li><li>8. The pair-trading rule</li><li>9. Who replaces a SELL</li><li>10. Reading it all together</li></ul>")

    # ============================================================ 0 ==
    H.append("<h2>0. The answers</h2><div class='box'><ol>")
    H.append(f"<li><b>Correlation and covariance vs the signal (§2).</b> Index pairs are {eqp['corr_sell'].mean():.2f} correlated in weeks when both signals are SELL against {eqp['corr_buy'].mean():.2f} when both are BUY. Covariance is {eqp['cov_ratio_sell_over_buy'].mean():.1f}× larger in SELL weeks, and {eqp['share_of_cov_change_from_vol'].mean() * 100:.0f}% of that rise is volatility, not correlation. Correlation does not climb ahead of N100's SELL flip ({sell_flip.loc[-4]:.2f} four weeks before → {sell_flip.loc[0]:.2f} at the flip → {sell_flip.loc[8]:.2f} eight weeks after), and the same holds for every other asset's flips ({flip_all_str}): the signal fires first, the market couples up afterwards.</li>")
    vix_bit = (f" The VIX tells the same story more faintly: it averages {vs.loc['N100', 'vix_mean_BUY']:.1f} in N100's BUY weeks and {vs.loc['N100', 'vix_mean_SELL']:.1f} in its SELL weeks (×{vs.loc['N100', 'vix_ratio_SELL_over_BUY']:.2f}, against ×{vs.loc['N100', 'vol_ratio_SELL_over_BUY']:.2f} for realised vol)." if has_vix else "")
    lead_bit = (f" A rise in the VIX, by contrast, runs a few weeks <i>ahead</i> of the rise in correlation (its changes correlate {vl.loc[0, 'corr_dvix_t_dcorr_t_plus_k']:.2f} with the same week's correlation change and {vl.loc[C.VOL_LEAD_LAGS, 'corr_dvix_t_dcorr_t_plus_k']:.2f} with the change {C.VOL_LEAD_LAGS} weeks later)." if has_vix and "corr_dvix_t_dcorr_t_plus_k" in vl.columns else "")
    H.append(f"<li><b>Volatility (§3).</b> Each index's swings are {vs.loc[C.EQUITY, 'vol_ratio_SELL_over_BUY'].mean():.1f}× bigger in its own SELL weeks ({vol_bs_str}); gold's do not change ({_p(vs.loc['GoldMCX', 'vol_BUY'])} → {_p(vs.loc['GoldMCX', 'vol_SELL'])}).{vix_bit} Correlation rises with volatility in every index pair under all three regime definitions (N100's vol, the pair's own vol, the VIX level): N100×SC100 goes from {cv.loc['N100|SC100', qcols[0]]:.2f} in the calmest fifth of weeks to {cv.loc['N100|SC100', qcols[-1]]:.2f} in the wildest; gold's correlation stays near zero in every regime. Realised volatility and correlation move <i>together</i> (their changes correlate {vl.loc[0, 'corr_dvol_t_dcorr_t_plus_k']:.2f} in the same week, less at every lead or lag).{lead_bit}</li>")
    H.append(f"<li><b>Correlation and returns, historically (§4).</b> The most-correlated fifth of weeks coincided with N100 {_pct(n100.loc['Q5', 'same_quarter_mean'])} and SC100 {_pct(sc.loc['Q5', 'same_quarter_mean'])} over that quarter; the least-correlated fifth with {_pct(n100.loc['Q1', 'same_quarter_mean'])} and {_pct(sc.loc['Q1', 'same_quarter_mean'])}. High correlation describes a sell-off in progress. The quarter <i>after</i> the most-correlated weeks: N100 {_pct(n100.loc['Q5', 'next_quarter_mean'])}, positive {n100.loc['Q5', 'next_quarter_hit'] * 100:.0f}% of the time.</li>")
    H.append(f"<li><b>Who breaks the correlation first (§5).</b> {len(cep)} decorrelation episodes since 2016. By date {date_first} most often ({sm.loc[date_first, 'share_first'] * 100:.0f}% of episodes incl. ties); by depth at the moment the market broke {deep_first} ({sm.loc[deep_first, 'share_deepest'] * 100:.0f}%); by size of its own collapse {drop_first} ({sm.loc[drop_first, 'share_biggest_drop'] * 100:.0f}%). Combined, <b>{combined_first}</b>. The robust fact is the <i>direction</i>: N100 breaks away by underperforming ({sm.loc['N100', 'share_up_breaks'] * 100:.0f}% up-breaks), the smaller indices by outperforming ({sm.loc['MC100', 'share_up_breaks'] * 100:.0f}% / {sm.loc['SC100', 'share_up_breaks'] * 100:.0f}%). As an early warning of a sell-off it worked {n_warn} times in {n_sell}.</li>")
    H.append(f"<li><b>Who breaks in price and signal (§6).</b> {first_cross} crosses −10% first ({sm4.loc[first_cross, 'first_share_cross_10pct'] * 100:.0f}% of the {len(pep)} sell-offs) and falls most; {first_sig}'s supertrend flips to SELL first ({sm4.loc[first_sig, 'first_share_signal_sell'] * 100:.0f}%); {first_rec} turns back to BUY first.</li>")
    H.append(f"<li><b>Each pair separately (§7).</b> Of the ten pairs, {len(coint_ok)} ({', '.join(p.replace('|', '×') for p in coint_ok) if coint_ok else 'none'}) has a price ratio that pulls back to a long-run level; the other {10 - len(coint_ok)} drift. A deviation of 1–2 sd has <i>continued</i>, not reverted: when a had lagged b by 1–2 sd, the ratio moved a further {_pct(pz.loc[B[1], 'mean_ratio_change'])} over the next {H13} weeks — the lag widened (t {_t(pz.loc[B[1], 'hac_t_ratio_change'])}); when a had led by 1–2 sd it moved a further {_pct(pz.loc[B[3], 'mean_ratio_change'])} — the lead widened (t {_t(pz.loc[B[3], 'hac_t_ratio_change'])}). Beyond 2 sd the ratio went nowhere on average ({_pct(pz.loc[B[0], 'mean_ratio_change'])} and {_pct(pz.loc[B[4], 'mean_ratio_change'])}).</li>")
    H.append(f"<li><b>The pair-trading rule (§8).</b> Enter at |z| ≥ {C.PT_ENTRY:.0f}, long the cheap leg / short the rich leg, exit at |z| ≤ {C.PT_EXIT:.1f} or after {C.PT_MAX_HOLD} weeks: <b>{ptall['verdict']}</b>. {int(ptall['trades'])} trades, {_p(ptall['win_rate_net'])} winners, {_pct(ptall['mean_net'])} per trade net of costs (t {_t(ptall['t_mean_net'])}), {n_pairs_neg} of 10 pairs lost money. The long-only reading is the same: the cheap leg returned {_pct(ptall['mean_ret_cheap_leg'])} over the average trade, the rich leg {_pct(ptall['mean_ret_rich_leg'])} — the laggard kept lagging. No setting in the sensitivity grid turns it into a reliable rule (best: entry {sens_best['entry']:.1f} / exit {sens_best['exit']:.1f}{' / cheap leg in BUY' if sens_best['require_cheap_leg_BUY'] else ''}, {_pct(sens_best['mean_net'])} per trade, t {_t(sens_best['t_mean_net'])}, {int(sens_best['trades'])} trades).</li>")
    H.append(f"<li><b>Who replaces a SELL (§9).</b> {n_ev} SELL flips. The rule as first proposed — <i>enter the correlated asset that has deviated (lagged) from the seller</i> — had a candidate in only {int(lag_rule['events_with_a_pick'])} of {n_ev} flips and, when it did, returned {_pct(lag_rule[f'when_pick_mean_{H13}'])} over {H13} weeks against cash {_pct(cash13)}; with the extra condition that the candidate's own signal is BUY it <b>never had a candidate</b> in ten years (a correlated asset that has lagged a seller is itself in SELL). What worked: the candidate that has deviated <i>upward</i> — the most <b>leading</b> asset — {_pct(lead_rule[f'mean_{H13}'])} (beat cash {_p(lead_rule[f'beat_cash_{H13}'])} of the time, t {_t(lead_rule[f'hac_t_minus_cash_{H13}'])}); the most correlated asset <b>still in BUY</b> — {_pct(corrbuy[f'mean_{H13}'])} (t {_t(corrbuy[f'hac_t_minus_cash_{H13}'])}); and plain <b>gold</b> — {_pct(gold_rule[f'mean_{H13}'])} (t {_t(gold_rule[f'hac_t_minus_cash_{H13}'])}). Staying in the seller returned {_pct(stay13)}; the most correlated asset regardless of its signal {_pct(mc_rule[f'mean_{H13}'])}; the most lagging {_pct(mlag[f'mean_{H13}'])}. The candidate's own signal is what matters: candidates in BUY returned {_pct(c_buy['cand_mean'])} over {H13} weeks, candidates in SELL {_pct(c_sellst['cand_mean'])}.</li>")
    H.append(f"<li><b>Right now</b> ({cur8['date'].iloc[0]}): " + (f"in SELL: {', '.join(nz)}. " if nz else "no asset is in SELL. ") + f"The group correlation is {last['G']:.2f} (z {last['z_G']:+.1f}, {'a decorrelation episode' if last['z_G'] <= C.CB_Z else 'inside its normal range'}); " + now_pairs_str + ".</li>")
    H.append("</ol></div>")

    # ============================================================ 1 ==
    H.append("<h2>1. The data and the building blocks</h2>")
    H.append("<p>Everything in this report is built from six ideas: a weekly return, the signal, correlation, volatility (realised and implied), covariance, and the z-score. This section explains each one in plain words, shows how it is computed, and works through one example with the real numbers, so that every later table can be read without the code.</p>")
    H.append("<h3>1.1 What is in the data</h3>")
    H.append(_tbl(cov, fmt="{:.0f}"))
    H.append(f"<p><b>How to read the table:</b> one row per series; <code>first</code> / <code>last</code> are the first and last week with a close; <code>obs</code> the number of weeks. "
             f"The five assets are <b>N100</b> (Nifty 100, the large caps), <b>MC100</b> (Nifty Midcap 100), <b>SC100</b> (Nifty Smallcap 100), <b>Micro250</b> (Nifty Microcap 250 — it exists only from 2022, so every number involving it rests on {int(cov.loc['Micro250', 'obs']) if 'Micro250' in cov.index else 0} weeks) and <b>GoldMCX</b> (the near gold future on MCX, carried onto the NSE calendar). "
             f"{'<b>' + C.CASH + '</b> is a liquid ETF (growth NAV): it stands for cash in §9. ' if cash_ok else ''}<b>VIXchg</b> is the weekly change in India VIX; the VIX file ends {cov.loc['VIXchg', 'last'] if 'VIXchg' in cov.index else 'n/a'}, so everything that uses the VIX stops there and says so. N50 is loaded only to set the trading calendar.</p>")
    H.append(f"<p><b>Weekly bars.</b> One row per trading week: the close on Friday (or the last trading day of the week). Weekly because the desk's signal is weekly; daily data would add noise without adding decisions. {len(_csv('ret_weekly', index_col=0))} weeks from {C.START} to {meta['last_session']}.</p>")
    H.append("<p><b>Returns.</b> A week's return is ln(this week's close ÷ last week's close) — the <i>log return</i>. For ordinary moves it is the percentage change (a 2.0% rise is a log return of 0.0198); the advantage is that returns over several weeks simply add, and exp(sum) − 1 turns the total back into a percentage. These are price indices, so dividends are not included.</p>")
    H.append("<h3>1.2 The signal: the desk's supertrend, and what a 'flip' is</h3>")
    ex_d = ev[ev["seller"] == "N100"]["date"]
    ex_d = ex_d[ex_d.dt.year == 2020].iloc[0] if (ex_d.dt.year == 2020).any() else ex_d.iloc[0]
    pxw = _csv("px_weekly", index_col=0, parse_dates=True)
    ii = pxw.index.get_loc(ex_d)
    H.append(f"<p>Every week, for every asset, the supertrend says <b>BUY</b> (+1) or <b>SELL</b> (−1). How it decides: it takes the <b>ATR</b> — the average size of a week's price range over the last {C.LOOKBACK} weeks, newest week weighted most (weights 1/15 … 5/15) — and draws a band {C.MULTIPLIER:.0f} ATRs above and below the week's mid-price (high + low) ÷ 2. The state turns BUY when the close rises through the upper band, SELL when it falls through the lower band; in between, the bands only tighten in the direction of the trend (they 'ratchet'), so a BUY is given up only after a fall of about two average weekly ranges from the band. It is recovered from the desk's wm-STstrat and not tuned here.</p>"
             f"<p>A <b>flip</b> is the week the state changes. Because the band is two ATRs wide, a flip to SELL happens <i>after</i> a sizeable fall, never at the top — that is why, throughout this report, correlation and volatility are already moving by the week of the flip. Worked example: N100 closed at {pxw['N100'].iloc[ii - 1]:,.0f} on {pxw.index[ii - 1].date()} and {pxw['N100'].iloc[ii]:,.0f} on {ex_d.date()} ({pxw['N100'].iloc[ii] / pxw['N100'].iloc[ii - 1] - 1:+.1%} in the week) — that close fell through the lower band and the state flipped to SELL on {ex_d.date()}.</p>"
             f"<p>Share of weeks in SELL: " + ", ".join(f"{a} {_p(sell_share[a])}" for a in C.ASSETS if a in sell_share) + f". Flips to SELL since {C.START}: " + ", ".join(f"{a} {flips_per[a]}" for a in C.ASSETS) + ". Several sections sort the weeks by what the signal said — <i>both BUY</i>, <i>both SELL</i>, <i>disagree</i> for a pair; the asset's <i>own</i> state for one asset — and compare the groups.</p>")
    H.append("<h3>1.3 Correlation</h3>")
    ex_c = pst.loc["N100|SC100", "corr_full"]
    H.append(f"<p><b>What it is.</b> A number from −1 to +1 saying how much two assets move in the same direction week to week. +1: every up week of one is an up week of the other in proportion; 0: no relation; −1: always opposite. Squared, it is the share of one asset's weekly variation that moves with the other: N100 and SC100 are {ex_c:.2f} correlated over the whole sample, so {ex_c ** 2:.0%} of Smallcap's weekly movement is 'the market' and {1 - ex_c ** 2:.0%} is its own.</p>"
             f"<p><b>How it is computed.</b> The usual (Pearson) correlation of the two weekly log-return series. <b>'13-week' correlation</b> = computed on the last 13 weeks only (one quarter: enough weeks to measure, few enough to move with the market), recomputed every week — this is the 'current correlation' used almost everywhere. <b>'52-week'</b> = the last year; steadier, used where a judgement about a relationship is needed (§9's 'is the candidate correlated with the seller'). <b>'full'</b> = the whole sample.</p>"
             "<p><b>How to read the level.</b> Above 0.90: effectively one asset. 0.70–0.90: move together (the normal state of the four indices). 0.40–0.70: related but with a life of their own. Near 0: unrelated (gold vs the indices, all the time). Below 0: opposite.</p>"
             "<p><b>Why the desk cares.</b> A rotation book holds one or two of these indices at a time. When their correlation is 0.95 the choice between them is mostly cosmetic — they are one bet; when it is 0.60 the choice matters, and holding the wrong one costs real money. Correlation is also the ingredient that decides whether 'diversifying' into the other index does anything.</p>")
    H.append("<p><b>Between the asset classes, week by week, and gold against the indices.</b></p>")
    H.append(_img("0_corr_heatmap"))
    H.append(_img("0_corr_rolling"))
    grk = _csv("0_gold_corr_rank", index_col=0); gq = _csv("0_gold_by_group_corr"); gper = _csv("0_gold_in_episodes")
    H.append(_img("0_gold_when_decoupled"))
    H.append(_tbl(grk, fmt="{:+.3f}", int_cols=("weeks", "weeks_above_N100", "weeks_compared"), pct_cols=("share_weeks_most_correlated_with_gold", "share_weeks_largest_abs_corr_with_gold", "share_above_N100")))
    H.append(_says(f"<b>Gold against the indices.</b> Smallcap's 13-week correlation with gold is above Nifty's in {int(grk.loc['SC100', 'weeks_above_N100'])} of {int(grk.loc['SC100', 'weeks_compared'])} weeks ({grk.loc['SC100', 'share_above_N100']:.0%}; average gap {grk.loc['SC100', 'mean_gap_vs_N100']:+.2f}); gold's most-correlated index is Nifty {grk.loc['N100', 'share_weeks_most_correlated_with_gold']:.0%} of the time, Smallcap {grk.loc['SC100', 'share_weeks_most_correlated_with_gold']:.0%}, Midcap {grk.loc['MC100', 'share_weeks_most_correlated_with_gold']:.0%}. When the indices decouple, gold does not couple to any of them: |corr| with gold {gq[gq['bucket'] == 'Q1']['mean_abs_corr'].mean():.2f} in the least-coupled fifth of weeks vs {gq[gq['bucket'] == 'Q5']['mean_abs_corr'].mean():.2f} in the most; {gper['abs_before'].mean():.2f} in the 8 weeks before a decorrelation episode vs {gper['abs_during'].mean():.2f} inside it, with the sign drifting from slightly negative to slightly positive in the first weeks of an episode."))
    gvf = _csv("0_gold_vs_group_fit", index_col=0); ggs = _csv("0_gold_gap_stats"); gga = _csv("0_gold_gap_after_extreme"); gbe = _csv("0_gold_after_break_episodes"); gba = _csv("0_gold_after_break_avg")
    H.append(_img("0_gold_vs_group"))
    H.append(_tbl(gvf, fmt="{:+.3f}", int_cols=("n",)))
    H.append(_says(f"<b>Gold's correlation against how coupled the indices are (added 2026-10-19).</b> No relationship: Pearson {gvf.loc[[p for p in gvf.index if 'Gold' in p], 'pearson'].min():+.2f} to {gvf.loc[[p for p in gvf.index if 'Gold' in p], 'pearson'].max():+.2f}, R² at most {gvf['r2'].max():.3f}, no slope's |t| above {gvf['nw_t_slope'].abs().max():.1f}, a quadratic no better. The scatter is the justification: a relationship that existed would slope; this does not."))
    H.append(_img("0_gold_gap_reversion"))
    H.append(_tbl(ggs, fmt="{:.3f}", index=False, int_cols=("weeks",)))
    H.append(_tbl(gga, fmt="{:+.3f}", index=False, int_cols=("horizon_weeks", "events"), pct_cols=tuple(c for c in gga.columns if c.startswith(("SC100_", "gold_")) and not c.endswith("_hit"))))
    s0 = ggs.iloc[0]
    H.append(_says(f"<b>The SC100×Gold − N100×Gold gap.</b> Mean-reverting (AR(1) {s0['ar1_phi']:.2f}, half-life {s0['half_life_weeks']:.1f} weeks, ADF p {s0['adf_p']:.4f}) by construction — the difference of two 13-week rolling correlations of an unrelated series. It is not a price and cannot be traded; and what followed its extremes in returns is a thin tilt on 18–20 events, read as a tendency and no more."))
    H.append(_img("0_index_vs_gold_pair"))
    ipg_ = _csv("0_index_vs_gold_pairs")
    H.append(_tbl(ipg_.assign(index_pair=ipg_["index_pair"].str.replace("|", " × ", regex=False), gold_pair=ipg_["gold_pair"].str.replace("|", " × ", regex=False)), fmt="{:+.3f}", index=False, int_cols=("weeks", "n_4w_nonoverlap", "n_13w_nonoverlap")))
    H.append(_says("<b>An index pair's correlation against a gold pair's (added 2026-10-22).</b> For every index pair × gold pair: the correlation between the two weekly series, between their 4- and 13-week changes, and the gold pair's mean when the index pair is loosest vs tightest. N100 × SC100 vs SC100 × Gold: unrelated (−0.03). Nothing in the grid is more opposite than about −0.26, and that does not survive on non-overlapping changes."))
    H.append(_img("0_extreme_followups"))
    xsm_ = _csv("0_extreme_summary"); xsel = xsm_[xsm_["series"].str.startswith("gap:") | (xsm_["series"] == f"{C.PAIR_VS_GOLD_DEFAULT[0]} - {C.PAIR_VS_GOLD_DEFAULT[1]}")]
    H.append(_tbl(xsel, fmt="{:+.3f}", index=False, int_cols=("horizon_weeks", "events"), max_rows=400))
    H.append(_says("<b>What follows a crossing of ±1 sd (added 2026-10-23).</b> For the gap and for every index pair − gold pair difference (25 series; the two charted ones tabled here, the rest in <code>0_extreme_summary.csv</code>): at 4 / 8 / 13 weeks after the first week beyond the line, the series' change and whether it moved back, each component's change, each asset's return, largest peak-to-trough fall and lowest point, and the relative returns — each beside the same statistic from every week. The series come back (the window); the gold-side component does the moving; a −1 sd gap and a +1 sd index-minus-gold difference were followed by above-average, calmer quarters for the indices, a −1 sd index-minus-gold difference by a choppy one for small-caps. Small counts, overlapping windows, no t-stats."))
    H.append(_img("0_gold_after_break"))
    H.append(_tbl(gbe, fmt="{:+.2f}", index=False, int_cols=("first_week_at_or_below_zero", f"weeks_positive_of_first_{C.GOLD_AFTER_BREAK}", "longest_positive_run", "peak_week", "weeks_available")))
    em = gbe[gbe["pair"] == "Micro250|GoldMCX"]
    H.append(_says(f"<b>After a break.</b> Micro250 × Gold was positive in the break week in {int(em['positive_at_break'].sum())} of its {len(em)} episodes and negative in the rest; the median episode has it positive {em[f'weeks_positive_of_first_{C.GOLD_AFTER_BREAK}'].median():.0f} of the first {C.GOLD_AFTER_BREAK} weeks. No gold pair holds a positive correlation after a break for long; the averaged paths sit near zero."))
    H.append("<h3>1.4 Volatility — realised, and the VIX</h3>")
    exv = vs.loc["N100", "vol_all"]
    H.append(f"<p><b>Realised volatility</b> is how big an asset's weekly moves typically are: the standard deviation of its weekly log returns, multiplied by √52 so it reads as a yearly figure. N100's is {_p(exv)} over the sample: a typical week moves about {exv / np.sqrt(52):.1%} ({_p(exv)} ÷ √52), a typical year ±{_p(exv)}. <b>'13-week vol'</b> = the same on the last 13 weeks, recomputed weekly — the 'current' volatility.</p>"
             + (f"<p><b>The India VIX</b> is the other kind of volatility: not what the market <i>did</i>, but what option prices say it <i>expects</i> to do over the next 30 days, for Nifty 50, annualised. A VIX of 15 means the options market expects a yearly swing of about ±15%, i.e. about ±{15 / np.sqrt(12):.1f}% over the next month. Realised vol looks back, the VIX looks forward; they agree most of the time (their levels correlate {vlv.get('vix_vs_realised_corr', np.nan):.2f} here) and disagree exactly when it is interesting. The VIX series ends {vix_last}; it is used in §3 as a second volatility measure, as a regime, and in the lead/lag test, and stops there.</p>" if has_vix else "")
             + "<p><b>Why it matters.</b> Half of what 'risk' means in a rotation book is volatility: a SELL signal that cuts exposure in high-vol weeks is doing most of its work through volatility, not through correlation (§2–§3 measure exactly how much).</p>")
    H.append("<h3>1.5 Covariance — the one formula</h3>")
    H.append(f"<p><b>Covariance = correlation × vol(a) × vol(b).</b> It is correlation with the size of the moves put back in, and it is what portfolio risk actually runs on: the variance of a two-asset book is w²σ² + w²σ² + 2·w·w·cov. Two assets can be equally correlated in two periods and contribute four times the risk in one of them because both are moving twice as much.</p>"
             f"<p><b>Worked example</b> (N100 × SC100, from the §2 table): in weeks when both signals were BUY, correlation {bs.loc[(bs['pair'] == 'N100|SC100') & (bs['state'] == 'both BUY'), 'corr'].iloc[0]:.2f} × {_p(bs.loc[(bs['pair'] == 'N100|SC100') & (bs['state'] == 'both BUY'), 'vol_a_ann'].iloc[0])} × {_p(bs.loc[(bs['pair'] == 'N100|SC100') & (bs['state'] == 'both BUY'), 'vol_b_ann'].iloc[0])} = {bs.loc[(bs['pair'] == 'N100|SC100') & (bs['state'] == 'both BUY'), 'cov_ann'].iloc[0]:.4f}; in weeks when both were SELL, {wk_ex['corr']:.2f} × {_p(wk_ex['vol_a_ann'])} × {_p(wk_ex['vol_b_ann'])} = {wk_ex['cov_ann']:.4f} — {wk_ex['cov_ann'] / bs.loc[(bs['pair'] == 'N100|SC100') & (bs['state'] == 'both BUY'), 'cov_ann'].iloc[0]:.1f}× larger. The numbers look small because they are products of fractions; what matters is the ratio, and how much of the rise came from the correlation term and how much from the two volatility terms (§2 splits it).</p>")
    H.append("<h3>1.6 The z-score — used three times in this report</h3>")
    zex = S.dropna(subset=["z_G"]).iloc[-1]
    H.append(f"<p>A z-score answers <i>'how unusual is this number for this series?'</i>: (this week's value − its average over a trailing window) ÷ its standard deviation over that window. z = 0 is 'its usual self'; z = ±1 is one standard deviation away (about one week in three is at least that far); z = ±2 is rare (about one week in twenty); z = ±3 very rare. The same reading is used for three different things:</p>"
             f"<ol><li><b>§5 — correlation-to-group.</b> Each index's 13-week correlation with the others against its own {C.CB_NORM}-week history. Today the group correlation is {zex['G']:.2f}, its two-year average is {zex['G'] - zex['z_G'] * (S['G'].rolling(C.CB_NORM, min_periods=52).std().iloc[-1]):.2f} and its spread {S['G'].rolling(C.CB_NORM, min_periods=52).std().iloc[-1]:.2f}, so z = {zex['z_G']:+.1f}: unusually low.</li>"
             f"<li><b>§7–§9 — the deviation of a pair.</b> The log price ratio ln(P<sub>a</sub>/P<sub>b</sub>) against its own trailing {C.PAIR_WINDOW}-week history. <b>z &gt; 0: a has run ahead of b</b> relative to the last year (a 'rich', b 'cheap'); <b>z &lt; 0: a has lagged b</b>. z = +2 means a is two standard deviations ahead of where the last year's relationship would put it. For N100×SC100 today z = {cur6.loc['N100|SC100', 'z_now']:+.1f}: {cur6.loc['N100|SC100', 'reading']}.</li>"
             f"<li><b>§9 — a candidate against a seller.</b> The same ratio z, read from the candidate's side: z(B vs A) &gt; 0 means B has run ahead of A. A candidate is called <b>lagging</b> at z ≤ {-C.RP_Z_DEV:.0f}, <b>leading</b> at z ≥ {C.RP_Z_DEV:+.0f}, <b>neutral</b> in between.</li></ol>"
             f"<p>One trap, stated once: <b>a z-score always comes back toward 0 on its own</b>, because the trailing window gradually absorbs the move — the average catches up with the level even if the level never returns. So 'the z went back to 0' does <i>not</i> mean 'the ratio went back'. Wherever this report tests whether a deviation reverts, it looks at the change in the <i>ratio itself</i> after the deviation (§7.1), never at the z-score.</p>")
    H.append("<h3>1.7 Forward returns, and the t-statistic</h3>")
    H.append(f"<p>Whenever something is measured 'after' a week t — after a SELL flip, after a deviation — the return counted is the sum of the weeks t+1 … t+h, with h = {H4} or {H13}. The week of the decision itself is never included, because the signal is read at its close and the money moves from the next bar. <code>checks.py</code> pins this alignment. Where such windows overlap (every week can be a decision week, and a 13-week window shares 12 weeks with the next one), the <b>t-statistic</b> is computed with a Newey-West correction ({C.HAC_LAGS} lags) that allows for the overlap. <b>How to read a t:</b> it is the average divided by its own uncertainty; beyond about ±2 the average is unlikely to be luck (roughly a 1-in-20 chance), between −2 and +2 it could easily be. Where the count is small — the {len(cep)} correlation episodes, the {len(pep)} sell-offs — no t is reported at all, only ranks, medians and shares, because a t on six events means nothing.</p>")

    # ============================================================ 2 ==
    H.append("<h2>2. Correlation and covariance with respect to the signal</h2>")
    H.append(_ask("Does a SELL signal mean the assets are <i>more coupled</i>, or just <i>moving more</i>? And does correlation rise before a signal fires — which would make it a warning — or after it, which would make it a consequence? Asked for every asset's flips, not only N100's.",
                  f"<b>(a) By joint state.</b> For each pair, every week is labelled by what the two supertrend signals said: <i>both BUY</i>, <i>both SELL</i>, <i>disagree</i>. Inside each label the pair's correlation, each leg's volatility and the covariance are computed on those weeks only (a label needs at least {C.MIN_CELL} weeks). The rise in covariance from both-BUY to both-SELL is then split in two: the part that would have happened if only the correlation had changed (both volatilities held at their BUY level), and the remainder, which is the volatility. "
                  f"<b>(b) Around the flips.</b> An event study: for every asset in turn, every week its signal flipped to SELL is lined up at 'week 0', and the group correlation G (the average of the six 13-week correlations among the four indices) is read at each week from {C.FLIP_WINDOW} before to {C.FLIP_WINDOW} after, then averaged over that asset's flips; the same for flips to BUY. "
                  "<b>(c) By breadth.</b> G grouped by how many of the four index signals were in SELL that week.",
                  "In table (a) one row is a pair. <code>corr_buy</code> / <code>corr_sell</code> are its correlation in the two states, <code>corr_change</code> the difference (positive = more coupled in SELL); <code>cov_ratio_sell_over_buy</code> how many times larger the covariance is in SELL; <code>share_of_cov_change_from_vol</code> what fraction of that rise is volatility (the two shares add to 1); <code>weeks_buy</code> / <code>weeks_sell</code> how many weeks each state has. In the event-study chart, week 0 is the flip; a line that is already rising before 0 means correlation climbs ahead of the signal, a line flat before and rising after means the signal fires first. The legend lets you switch each asset's line on and off."))
    H.append("<h3>2.1 By joint state</h3>")
    H.append(_img("1_corr_by_state"))
    H.append(_tbl(dec[["pair", "corr_buy", "corr_sell", "corr_change", "cov_buy", "cov_sell", "cov_ratio_sell_over_buy", "share_of_cov_change_from_corr", "share_of_cov_change_from_vol", "vol_a_buy", "vol_a_sell", "vol_b_buy", "vol_b_sell", "weeks_buy", "weeks_sell"]], fmt="{:.2f}", index=False, int_cols=("weeks_buy", "weeks_sell"), pct_cols=("vol_a_buy", "vol_a_sell", "vol_b_buy", "vol_b_sell")))
    H.append("<p class='small'>The full table behind it (every state including 'disagree' and 'all weeks', with each leg's annualised mean return in that state) is <code>1_corr_by_signal_state.csv</code>.</p>")
    d_ex = dec[dec["pair"] == "N100|SC100"].iloc[0]
    H.append(_says(f"<b>Reading one row — N100×SC100.</b> In the {int(d_ex['weeks_buy'])} weeks when both signals were BUY the pair was {d_ex['corr_buy']:.2f} correlated, with N100 at {_p(d_ex['vol_a_buy'])} vol and SC100 at {_p(d_ex['vol_b_buy'])}. In the {int(d_ex['weeks_sell'])} both-SELL weeks the correlation was {d_ex['corr_sell']:.2f} ({d_ex['corr_change']:+.2f}) and the vols {_p(d_ex['vol_a_sell'])} and {_p(d_ex['vol_b_sell'])}. Covariance went up {d_ex['cov_ratio_sell_over_buy']:.1f}×, and {d_ex['share_of_cov_change_from_vol'] * 100:.0f}% of that came from the two volatilities rising, {d_ex['share_of_cov_change_from_corr'] * 100:.0f}% from the correlation rising. "
                   f"<b>The pattern across the index pairs:</b> correlation {eqp['corr_buy'].mean():.2f} → {eqp['corr_sell'].mean():.2f} on average (every index pair goes up: smallest change {eqp.set_index('pair')['corr_change'].idxmin().replace('|', '×')} {eqp['corr_change'].min():+.2f}, largest {eqp.set_index('pair')['corr_change'].idxmax().replace('|', '×')} {eqp['corr_change'].max():+.2f}); covariance {eqp['cov_ratio_sell_over_buy'].min():.1f}–{eqp['cov_ratio_sell_over_buy'].max():.1f}× larger; {eqp['share_of_cov_change_from_vol'].mean() * 100:.0f}% of the extra covariance is volatility. "
                   f"<b>Gold</b> is different in kind: its correlation with the indices is {gp['corr_buy'].mean():+.2f} in BUY and {gp['corr_sell'].mean():+.2f} in SELL — zero either way — so a SELL state does not make gold start moving with the market."))
    H.append("<h3>2.2 Does correlation rise before the signal, or after? — every asset's flips</h3>")
    H.append(_img("1_corr_around_flip"))
    piv_af = afa[afa["event"] == "flip to SELL"].pivot(index="ref", columns="weeks_from_flip", values="mean_corr").reindex(refs_all)
    piv_af["n_flips"] = afa[afa["event"] == "flip to SELL"].groupby("ref")["n_events"].max().reindex(refs_all)
    H.append("<p class='small'>The group correlation G around each asset's flip to SELL (rows), at each week from −8 to +8:</p>")
    H.append(_tbl(piv_af[[k for k in (-8, -6, -4, -2, 0, 2, 4, 6, 8)] + ["n_flips"]], fmt="{:.2f}", int_cols=("n_flips",)))
    piv_ab = afa[afa["event"] == "flip to BUY"].pivot(index="ref", columns="weeks_from_flip", values="mean_corr").reindex(refs_all)
    H.append("<p class='small'>The same around each asset's flip back to BUY:</p>")
    H.append(_tbl(piv_ab[[k for k in (-8, -6, -4, -2, 0, 2, 4, 6, 8)]], fmt="{:.2f}"))
    bf = af[af["event"] == "flip to BUY"].set_index("weeks_from_flip")["mean_corr"]
    eq_sell_str = "; ".join(f"{a} {_af_row(a, 'flip to SELL', -4):.2f} → {_af_row(a, 'flip to SELL', 0):.2f} → {_af_row(a, 'flip to SELL', 8):.2f}" for a in refs_all if a in C.EQUITY)
    H.append(_says(f"<b>N100:</b> around its flip to SELL the group correlation reads {sell_flip.loc[-8]:.2f} eight weeks before, {sell_flip.loc[-4]:.2f} four weeks before, {sell_flip.loc[0]:.2f} at the flip and {sell_flip.loc[8]:.2f} eight weeks after (n {int(af[af['event'] == 'flip to SELL']['n_events'].max())} flips): it rises <i>after</i> the signal, not before. "
                   f"<b>The same for every index</b> (four weeks before → at the flip → eight weeks after): {eq_sell_str}. The shape is identical: flat or falling into the flip, then a climb of about 0.1 over the following two months. "
                   f"<b>Gold's flips</b> are different: around a gold SELL flip the index correlation goes {_af_row('GoldMCX', 'flip to SELL', -4):.2f} → {_af_row('GoldMCX', 'flip to SELL', 0):.2f} → {_af_row('GoldMCX', 'flip to SELL', 8):.2f} — nothing happens, because gold's signal says nothing about the equity market. "
                   f"<b>Around the flips back to BUY</b> the market is at its most coupled ({bf.loc[-4]:.2f} four weeks before N100's BUY flip, {bf.loc[0]:.2f} at it) and loosens afterwards ({bf.loc[8]:.2f}): a BUY flip comes out of the bottom of a sell-off, when everything has been moving together. The dotted lines in the chart (off by default) show the flipping asset's <i>own</i> correlation with the others; they tell the same story."))
    H.append("<h4>Pair by pair</h4>")
    afp = _csv("1_corr_around_flip_pairs")
    PAIRS_ALL = [f"{a}|{b}" for a, b in itertools.combinations(C.ASSETS, 2)]
    afn = afp[(afp["ref"] == "N100") & (afp["event"] == "flip to SELL")].pivot(index="pair", columns="weeks_from_flip", values="mean_corr")
    afn = afn.reindex([p for p in PAIRS_ALL if p in afn.index]); afn.index.name = None; afn.columns.name = None
    n100p = [p for p in afn.index if p.startswith("N100|") and "Gold" not in p]; othp = [p for p in afn.index if not p.startswith("N100|") and "Gold" not in p]; gldp = [p for p in afn.index if "Gold" in p]
    H.append(_img("1_corr_around_flip_pairs"))
    H.append("<p class='small'>Every pair's 13-week correlation around N100's flips to SELL (the other assets' flips are in the chart's menu and in <code>1_corr_around_flip_pairs.csv</code>):</p>")
    H.append(_tbl(afn[[k for k in (-8, -6, -4, -2, 0, 2, 4, 6, 8)]].rename(index=lambda p: p.replace("|", " × ")), fmt="{:.2f}"))
    afp_all = _csv("1_corr_around_flip_pairs"); afx = afp_all[(afp_all["ref"] == "N100") & (afp_all["event"] == "flip to SELL")]
    rob = pd.DataFrame({"mean −4": afx.pivot(index="pair", columns="weeks_from_flip", values="mean_corr")[-4], "mean 0": afx.pivot(index="pair", columns="weeks_from_flip", values="mean_corr")[0], "mean +8": afx.pivot(index="pair", columns="weeks_from_flip", values="mean_corr")[8],
                        "median −4": afx.pivot(index="pair", columns="weeks_from_flip", values="median_corr")[-4], "median 0": afx.pivot(index="pair", columns="weeks_from_flip", values="median_corr")[0], "median +8": afx.pivot(index="pair", columns="weeks_from_flip", values="median_corr")[8],
                        f"ex-{C.FLIP_EXCLUDE_YEAR} −4": afx.pivot(index="pair", columns="weeks_from_flip", values=f"mean_corr_ex{C.FLIP_EXCLUDE_YEAR}")[-4], f"ex-{C.FLIP_EXCLUDE_YEAR} 0": afx.pivot(index="pair", columns="weeks_from_flip", values=f"mean_corr_ex{C.FLIP_EXCLUDE_YEAR}")[0], f"ex-{C.FLIP_EXCLUDE_YEAR} +8": afx.pivot(index="pair", columns="weeks_from_flip", values=f"mean_corr_ex{C.FLIP_EXCLUDE_YEAR}")[8]}).reindex([p for p in PAIRS_ALL if p in set(afx["pair"])])
    H.append(f"<p class='small'>The same under the median and without the {C.FLIP_EXCLUDE_YEAR} flips (added 2026-10-19 — the step at the flip is not the COVID crash):</p>")
    H.append(_tbl(rob.rename(index=lambda p: p.replace("|", " × ")), fmt="{:.2f}"))
    H.append(_says(f"<b>The rise after a SELL flip is in the N100 pairs.</b> Around N100's SELL flips the pairs that climb are {', '.join(p.replace('|', '×') for p in n100p)} — on average {afn.loc[n100p, -4].mean():.2f} four weeks before → {afn.loc[n100p, 0].mean():.2f} at the flip → {afn.loc[n100p, 8].mean():.2f} eight weeks after — while the pairs among the smaller indices were already coupled and barely move ({afn.loc[othp, -4].mean():.2f} → {afn.loc[othp, 8].mean():.2f}). The group number G averages the two, so 'correlation rises after the flip' means, concretely, that Nifty re-joins the mid/small block it had drifted away from during the run-up. The gold pairs dip below zero in the flip week ({afn.loc[gldp, 0].mean():+.2f} on average): gold rose while the indices fell, and the dip fades over the following weeks. The same pattern holds around the other indices' flips (pick them in the menu): the pairs that include the flipping asset move, the rest do not."))
    H.append("<h3>2.3 Correlation vs how many signals are in SELL</h3>")
    H.append(_tbl(bb, fmt="{:.2f}", int_cols=("weeks",)))
    H.append(_img("1_corr_by_breadth"))
    bpd = _csv("1_corr_by_breadth_pairs", index_col=0)
    bp_show = bpd[[p for p in PAIRS_ALL if p in bpd.columns] + ["weeks"]].rename(columns=lambda c: c.replace("|", " × ")); bp_show.insert(0, "group G", bb["mean"].reindex(bpd.index))
    H.append("<p class='small'>The same by pair — each pair's average 13-week correlation in the weeks grouped by how many of the four index signals were in SELL:</p>")
    H.append(_tbl(bp_show, fmt="{:.2f}", int_cols=("weeks",)))
    H.append(_says(f"<b>Pair by pair:</b> every index pair is more correlated the more signals are in SELL; the gain is largest in the N100 pairs (N100×SC100 {bpd.loc['none in SELL', 'N100|SC100']:.2f} with none in SELL → {bpd.loc['all in SELL', 'N100|SC100']:.2f} with all four; N100×MC100 {bpd.loc['none in SELL', 'N100|MC100']:.2f} → {bpd.loc['all in SELL', 'N100|MC100']:.2f}) and smallest in the pairs among the smaller indices, which start high (MC100×SC100 {bpd.loc['none in SELL', 'MC100|SC100']:.2f} → {bpd.loc['all in SELL', 'MC100|SC100']:.2f}). The gold pairs stay within {bpd.loc[:, [c for c in bpd.columns if 'Gold' in c]].min().min():+.2f} … {bpd.loc[:, [c for c in bpd.columns if 'Gold' in c]].max().max():+.2f} in every group."))
    H.append(_img("1_corr_timeseries"))
    H.append(_says(f"With no index in SELL the four move with a correlation of {bb.loc['none in SELL', 'mean']:.2f} ({int(bb.loc['none in SELL', 'weeks'])} weeks); with all four in SELL, {bb.loc['all in SELL', 'mean']:.2f} ({int(bb.loc['all in SELL', 'weeks'])} weeks). The relationship is monotone across the five bands: the more signals are off, the more the market moves as one block — which is the same fact as 2.1 seen from the breadth side."))
    H.append(_not("That SELL weeks are more correlated does not make correlation a warning: 2.2 shows it climbs after the flip, for every asset. The practical point is the covariance split — SELL weeks are dangerous mainly because everything moves more, so a defensive sleeve earns its keep by cutting exposure, not by hunting for a less-correlated index — with the one exception of gold, which is uncorrelated in both states (§3 and §9 pick this up). The 'disagree' weeks are a mixed bag (one leg already off, the other not yet) and are not read on their own."))

    # ============================================================ 3 vol ==
    H.append("<h2>3. Volatility with respect to the signal, and volatility vs correlation</h2>")
    H.append(_ask("How much bigger are an asset's swings when its own signal is SELL — in realised vol and in the VIX? Does volatility build before a flip, for every asset? Does correlation rise when volatility rises, pair by pair, and does it matter whose volatility we use? And which moves first, volatility or correlation?",
                  f"<b>(a) By own state.</b> For each asset, its realised vol, its average weekly return, the share of down weeks and the worst week, on the weeks its <i>own</i> signal was BUY and on the weeks it was SELL" + ("; and the average VIX level in those same weeks" if has_vix else "") + f". <b>(b) Around the flips.</b> The event study of §2.2 repeated on volatility, for every asset as the reference: each index's 13-week vol" + (" and the VIX level" if has_vix else "") + f" at each week from {C.FLIP_WINDOW} before to {C.FLIP_WINDOW} after that asset's flips. <b>(c) By regime.</b> Every week is sorted into {C.VOL_BUCKETS} volatility regimes — quintiles, from the calmest fifth of weeks (Q1) to the wildest (Q5) — and each pair's 13-week correlation is averaged inside each regime. The regime is defined three ways, because the answer could depend on whose volatility is used: N100's 13-week vol (the market), the pair's own average 13-week vol, " + ("and the VIX level. " if has_vix else "and (where the VIX file covers the week) the VIX level. ") + f"<b>(d) Lead / lag.</b> The correlation between the {C.VOL_WINDOW}-week change in N100's vol at week t and the 13-week change in the group correlation G at week t + k, for k from −{C.VOL_LEAD_LAGS} to +{C.VOL_LEAD_LAGS}" + ("; the same with the 13-week change in the VIX" if has_vix else "") + ".",
                  "<code>vol_ratio_SELL_over_BUY</code> of 1.8 means the asset swings 1.8× as much in its SELL weeks; <code>vix_ratio_SELL_over_BUY</code> is the same for the VIX level. In the regime tables a row is a pair and Q1 … Q5 run from calm to wild; <code>Q5_minus_Q1</code> is how much more correlated the pair is in wild weeks than calm ones, <code>spearman</code> the rank correlation between the pair's correlation series and the regime series (1 = they rise and fall together). In the lead/lag chart a peak at k = 0 means the two move together; a peak at k &gt; 0 means volatility leads (the correlation change comes k weeks later)."))
    H.append("<h3>3.0 Volatility over time</h3>")
    H.append(_img("5_vol_timeseries"))
    H.append("<h3>3.1 Volatility by the asset's own signal state</h3>")
    H.append(_img("5_vol_by_state"))
    vcols = ["weeks_BUY", "vol_BUY", "mean_weekly_ret_BUY", "share_weeks_down_BUY", "worst_week_BUY", "weeks_SELL", "vol_SELL", "mean_weekly_ret_SELL", "share_weeks_down_SELL", "worst_week_SELL", "vol_all", "vol_ratio_SELL_over_BUY"] + (["vix_mean_BUY", "vix_mean_SELL", "vix_ratio_SELL_over_BUY"] if has_vix else [])
    H.append(_tbl(vs[vcols], fmt="{:.2f}", int_cols=("weeks_BUY", "weeks_SELL"), pct_cols=("vol_BUY", "vol_SELL", "vol_all", "mean_weekly_ret_BUY", "mean_weekly_ret_SELL", "worst_week_BUY", "worst_week_SELL")))
    sdist = _csv("5_state_dist")
    H.append(_img("5_vol_state_box"))
    H.append(_tbl(sdist, fmt="{:.4f}", index=False, int_cols=("n",), max_rows=60))
    H.append(_says("<b>The distribution behind the averages (added 2026-10-19).</b> Every week as a dot; min / p25 / median / mean / p75 / max per asset, state and measure (|weekly return|, 13-week vol, the VIX level, its weekly change and |change|). The medians separate BUY from SELL as the means do; the VIX's SELL mean is pulled up by March 2020 (max 70), the median is not."))
    if (T / "5_vix_flip_summary.csv").exists():
        H.append(_img("5_vix_flip_leadlag"))
        H.append(_tbl(_csv("5_vix_flip_summary", index_col=0), fmt="{:.2f}", int_cols=("sell_flips_with_vix_data", "surges")))
        H.append(_tbl(_csv("5_vix_flip_paths"), fmt="{:.3f}", index=False, int_cols=("k", "n_flips", "n_signal_weeks")))
        H.append(_says(f"<b>Is the VIX leading or lagging the SELL flip?</b> Leading by a few weeks, as the same event: the VIX rises over the four weeks before a flip (the supertrend needs that long to cross its band) and the flip week is the commonest VIX-surge week (a surge = up {C.VIX_SURGE:.0%} in {C.VIX_SURGE_WEEKS} weeks). A surge alone is a weak warning: most surges find the index already in SELL and only about a quarter are followed by its flip within {C.VIX_FLIP_WINDOW} weeks. Every flip and every surge is in <code>5_vix_flip_events.csv</code> and <code>5_vix_surge_events.csv</code>."))
    vix_says = (f" <b>The VIX agrees but separates the states far less:</b> it averages " + ", ".join(f"{a} {vs.loc[a, 'vix_mean_BUY']:.1f} → {vs.loc[a, 'vix_mean_SELL']:.1f}" for a in C.EQUITY if a in vs.index) + f" (BUY → SELL), a ratio of {vs.loc[C.EQUITY, 'vix_ratio_SELL_over_BUY'].mean():.2f} against {vs.loc[C.EQUITY, 'vol_ratio_SELL_over_BUY'].mean():.2f} for realised vol. The option market prices some of the extra volatility of a SELL state in advance and only part of it is 'new' when it arrives; realised vol is the bigger number because it includes the moves that caused the flip. Gold's VIX ratio is {vs.loc['GoldMCX', 'vix_ratio_SELL_over_BUY']:.2f}: gold's signal does not sort the equity market's fear either." if has_vix else "")
    H.append(_says(f"<b>Reading one row — N100.</b> In its {int(vs.loc['N100', 'weeks_BUY'])} BUY weeks N100's realised vol was {_p(vs.loc['N100', 'vol_BUY'])}, it averaged {_pct(vs.loc['N100', 'mean_weekly_ret_BUY'], 2)} a week, {_p(vs.loc['N100', 'share_weeks_down_BUY'])} of weeks were down and the worst was {_pct(vs.loc['N100', 'worst_week_BUY'])}; in its {int(vs.loc['N100', 'weeks_SELL'])} SELL weeks the vol was {_p(vs.loc['N100', 'vol_SELL'])} (×{vs.loc['N100', 'vol_ratio_SELL_over_BUY']:.2f}), the average {_pct(vs.loc['N100', 'mean_weekly_ret_SELL'], 2)}, {_p(vs.loc['N100', 'share_weeks_down_SELL'])} down weeks, worst {_pct(vs.loc['N100', 'worst_week_SELL'])}. "
                   f"<b>Every index</b> is {vs.loc[C.EQUITY, 'vol_ratio_SELL_over_BUY'].min():.1f}–{vs.loc[C.EQUITY, 'vol_ratio_SELL_over_BUY'].max():.1f}× more volatile in its own SELL weeks ({vol_bs_str}), the worst single weeks all fall in SELL ({worst_str}), and SELL weeks are on average falling weeks ({mean_sell_str}) — but the share of down weeks in SELL is only {vs.loc[C.EQUITY, 'share_weeks_down_SELL'].mean() * 100:.0f}% against {vs.loc[C.EQUITY, 'share_weeks_down_BUY'].mean() * 100:.0f}% in BUY. The state is about the <i>size</i> of the moves far more than their direction: a SELL week is a coin-flip week with twice the stakes. "
                   f"<b>Gold</b> is the exception: {_p(vs.loc['GoldMCX', 'vol_BUY'])} in BUY, {_p(vs.loc['GoldMCX', 'vol_SELL'])} in SELL (ratio {vs.loc['GoldMCX', 'vol_ratio_SELL_over_BUY']:.2f}); its signal does not sort its weeks by volatility at all." + vix_says))
    H.append("<h3>3.2 Volatility around every asset's flips</h3>")
    H.append(_img("5_vol_around_flip"))
    own_rows = []
    for a in refs_all:
        row = {"asset": a}
        for k in (-8, -4, 0, 4, 8):
            row[f"own vol {k:+d}"] = _vaf_row(a, "flip to SELL", k, f"vol_{a}")
        if has_vix:
            for k in (-8, -4, 0, 4, 8):
                row[f"VIX {k:+d}"] = _vaf_row(a, "flip to SELL", k, "vix_level")
        row["n_flips"] = _vaf_row(a, "flip to SELL", 0, "n_events")
        own_rows.append(row)
    own_df = pd.DataFrame(own_rows).set_index("asset")
    H.append("<p class='small'>Around each asset's flip to SELL: its own 13-week vol" + (" and the VIX level" if has_vix else "") + " at −8, −4, 0, +4, +8 weeks:</p>")
    H.append(_tbl(own_df, fmt="{:.1f}", int_cols=("n_flips",), pct_cols=tuple(c for c in own_df.columns if c.startswith("own vol"))))
    vix_flip = (f" The VIX moves the same way but earlier and less: around N100's SELL flip it reads {_vaf_row('N100', 'flip to SELL', -4, 'vix_level'):.1f} four weeks before, {_vaf_row('N100', 'flip to SELL', 0, 'vix_level'):.1f} at the flip, {_vaf_row('N100', 'flip to SELL', 8, 'vix_level'):.1f} eight weeks after." if has_vix else "")
    H.append(_says(f"<b>N100:</b> its own vol runs {_p(vsf.loc[-8, 'vol_N100'])} eight weeks before its SELL flip, {_p(vsf.loc[-4, 'vol_N100'])} four weeks before, {_p(vsf.loc[0, 'vol_N100'])} at the flip and {_p(vsf.loc[8, 'vol_N100'])} eight weeks after. <b>Every index shows the same shape</b> (four weeks before → at the flip → eight weeks after, own vol): " + "; ".join(f"{a} {_p(_vaf_row(a, 'flip to SELL', -4, f'vol_{a}'))} → {_p(_vaf_row(a, 'flip to SELL', 0, f'vol_{a}'))} → {_p(_vaf_row(a, 'flip to SELL', 8, f'vol_{a}'))}" for a in refs_all if a in C.EQUITY) + f". The rise begins in the flip week itself — the flip <i>is</i> a large down move — and continues for about two months. Into the BUY flip, vol is at its highest ({_p(vbf.loc[-4, 'vol_N100'])} four weeks before, {_p(vbf.loc[0, 'vol_N100'])} at N100's BUY flip) and falls afterwards ({_p(vbf.loc[8, 'vol_N100'])}). <b>Gold's flips</b> move gold's own vol ({_p(_vaf_row('GoldMCX', 'flip to SELL', -4, 'vol_GoldMCX'))} → {_p(_vaf_row('GoldMCX', 'flip to SELL', 0, 'vol_GoldMCX'))} → {_p(_vaf_row('GoldMCX', 'flip to SELL', 8, 'vol_GoldMCX'))}) and leave the indices' vol alone.{vix_flip} Compare §2.2: correlation follows the same path, after the flip. The two are one phenomenon."))
    H.append("<h3>3.3 Correlation by volatility regime, pair by pair — three definitions of the regime</h3>")
    H.append(_img("5_corr_by_regime"))
    H.append(_img("5_corr_by_regime_more"))
    rgw = _csv("5_regime_weeks")
    H.append(_img("5_regime_scatter"))
    H.append(_img("5_regime_timeline"))
    rstx_ = _csv("5_regime_stretches")
    H.append(_tbl(rstx_.assign(pair=rstx_["pair"].str.replace("|", " × ", regex=False)), fmt="{:+.2f}", index=False, int_cols=("weeks",), pct_cols=("own_vol_peak",), max_rows=100))
    H.append(_says("<b>The components of the averages, and the hypothesis on a time-series basis (added 2026-10-19).</b> Every week as a dot with the median per fifth: the median climbs with the mean in every index pair, so the regime averages are not an outlier effect. The timeline shades every stretch of 2+ weeks in the pair's wildest fifth; the table lists each with the correlation before, inside and the change — the hypothesis holds in most instances, the exceptions being second legs of a sell-off and the 2018-type stretches where small-caps fell alone."))
    H.append("<p class='small'>The second regime chart: the regime defined by each leg of the pair separately (the two components of 'the pair's own vol' = their average), and by the desk's ATR (5-bar weekly average true range ÷ close — ATR divided by the index value, never the ATR in points) instead of the standard deviation. Weeks in each fifth, regime = the pair's own vol:</p>")
    H.append(_tbl(rgw[rgw["regime"] == "pair's own vol"].set_index("pair")[["Q1", "Q2", "Q3", "Q4", "Q5"]].rename(index=lambda p: p.replace("|", " × ")), fmt="{:.0f}", int_cols=("Q1", "Q2", "Q3", "Q4", "Q5")))
    for rg in regimes:
        w = regw[rg].copy()
        w["Q5_minus_Q1"] = w["Q5"] - w["Q1"]
        edges = regl[regl["regime"] == rg].groupby("bucket")[["lo", "hi"]].median()
        if rg.startswith(C.BENCHMARK):
            w = w.rename(columns={q: f"{q} (vol {edges.loc[q, 'lo']:.0%}–{edges.loc[q, 'hi']:.0%})" for q in ("Q1", "Q2", "Q3", "Q4", "Q5") if q in edges.index})
        elif rg == "VIX level":
            w = w.rename(columns={q: f"{q} (VIX {edges.loc[q, 'lo']:.0f}–{edges.loc[q, 'hi']:.0f})" for q in ("Q1", "Q2", "Q3", "Q4", "Q5") if q in edges.index})
        H.append(f"<p class='small'>Regime = <b>{rg}</b>" + (" (each pair's own quintile edges differ; the chart's hover shows them)" if rg == "pair's own vol" else "") + ":</p>")
        H.append(_tbl(w, fmt="{:.2f}"))
    cvi = cv.loc[idx_pairs]; cvg = cv.loc[gold_pairs]
    own_w = regw.get("pair's own vol"); vix_w = regw.get("VIX level")
    own_bit = (f" Using the <b>pair's own volatility</b> instead gives the same ordering (index pairs Q5 − Q1 from {(own_w.loc[idx_pairs, 'Q5'] - own_w.loc[idx_pairs, 'Q1']).min():+.2f} to {(own_w.loc[idx_pairs, 'Q5'] - own_w.loc[idx_pairs, 'Q1']).max():+.2f})." if own_w is not None else "")
    vix_bit3 = (f" Using the <b>VIX level</b> as the regime, the index pairs still climb (Q5 − Q1 from {(vix_w.loc[idx_pairs, 'Q5'] - vix_w.loc[idx_pairs, 'Q1']).min():+.2f} to {(vix_w.loc[idx_pairs, 'Q5'] - vix_w.loc[idx_pairs, 'Q1']).max():+.2f}) but less cleanly than with realised vol — the VIX is Nifty 50's implied vol and a forecast, so it sorts the weeks less sharply than what actually happened; gold's pairs sit between {vix_w.loc[gold_pairs].drop(columns=[c for c in vix_w.columns if c not in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5')]).min().min():+.2f} and {vix_w.loc[gold_pairs].drop(columns=[c for c in vix_w.columns if c not in ('Q1', 'Q2', 'Q3', 'Q4', 'Q5')]).max().max():+.2f}." if vix_w is not None else "")
    H.append(_says(f"<b>With N100's vol as the regime:</b> every index pair is more correlated when the market is more volatile — from Q1 to Q5 the six index pairs gain {cvi['Q5_minus_Q1'].min():+.2f} to {cvi['Q5_minus_Q1'].max():+.2f} (largest {cvi['Q5_minus_Q1'].idxmax().replace('|', '×')}, {cvi.loc[cvi['Q5_minus_Q1'].idxmax(), qcols[0]]:.2f} → {cvi.loc[cvi['Q5_minus_Q1'].idxmax(), qcols[-1]]:.2f}; smallest {cvi['Q5_minus_Q1'].idxmin().replace('|', '×')}, already {cvi.loc[cvi['Q5_minus_Q1'].idxmin(), qcols[0]]:.2f} in calm weeks). The rank correlations between a pair's correlation and the vol level run {cvi['spearman_corr_vs_regime'].min():.2f}–{cvi['spearman_corr_vs_regime'].max():.2f}. The four gold pairs sit between {cvg[qcols].min().min():+.2f} and {cvg[qcols].max().max():+.2f} in every regime (Q5 − Q1: {gold_q_str}): gold does not couple to the indices even when volatility is extreme.{own_bit}{vix_bit3} <b>What it means for the book:</b> the diversification between two indices is smallest exactly in the weeks the book needs it most; the only diversifier that survives a high-vol regime is gold."))
    H.append("<h3>3.4 Which moves first — volatility or correlation?</h3>")
    H.append(_img("5_together"))
    H.append(_img("5_corr_rise_paths"))
    pps = _csv("5_corr_rise_pairs_summary", index_col=0); pps = pps.reindex([p for p in PAIRS_ALL if p in pps.index])
    pn = [p for p in pps.index if p.startswith("N100|") and "Gold" not in p]; po = [p for p in pps.index if not p.startswith("N100|") and "Gold" not in p]
    pps_show = pps.rename(index=lambda p: p.replace("|", " × ")); pps_show["share_pair_also_reached_plus1"] = pps_show["share_pair_also_reached_plus1"].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    H.append("<p class='small'>Around the same correlation jumps, pair by pair — did the pair's own correlation z also reach +1 within the window, and where was it in the jump week:</p>")
    H.append(_tbl(pps_show, fmt="{:.2f}", int_cols=("jumps",)))
    H.append(_says(f"<b>Which pairs jump:</b> the N100 pairs reach their own +1 in {pps.loc[pn, 'share_pair_also_reached_plus1'].mean():.0%} of the group's correlation jumps (z at the jump {pps.loc[pn, 'mean_z_at_jump'].mean():+.2f} on average); the pairs among the smaller indices only in {pps.loc[po, 'share_pair_also_reached_plus1'].mean():.0%} (z {pps.loc[po, 'mean_z_at_jump'].mean():+.2f}) — they were already near their usual, high, level. A correlation jump is Nifty re-joining a block that was already coupled, which is the same thing §2.2 found around the SELL flips."))
    H.append("<h4>Pair by pair — each pair's own correlation, its own volatility and the VIX</h4>")
    own = _csv("5_corr_rise_own_summary", index_col=0); own = own.reindex([p for p in PAIRS_ALL if p in own.index])
    oe = [p for p in own.index if "Gold" not in p]; og = [p for p in own.index if "Gold" in p]
    H.append(_img("5_together_pairs"))
    H.append(_img("5_corr_rise_paths_own"))
    H.append(_tbl(own.rename(index=lambda p: p.replace("|", " × ")), fmt="{:+.2f}", int_cols=("jumps",), pct_cols=tuple(c for c in own.columns if c.startswith("share_"))))
    H.append(_says(f"<b>Each pair on its own</b> — the jumps defined on the pair's own correlation z, and the volatility being the pair's own (the average of its two legs' 13-week vol). Index pairs: own vol z {own.loc[oe, 'z_vol_-4'].mean():+.2f} four weeks before the pair's jump → {own.loc[oe, 'z_vol_+0'].mean():+.2f} in the jump week" + (f"; VIX z {own.loc[oe, 'z_vix_-4'].mean():+.2f} → {own.loc[oe, 'z_vix_+0'].mean():+.2f}" if "z_vix_-4" in own.columns else "") + f"; volatility was already above +1 before the jump in {own.loc[oe, 'share_z_vol_above1_before'].mean():.0%} of jumps" + (f", the VIX in {own.loc[oe, 'share_z_vix_above1_before'].mean():.0%}" if "share_z_vix_above1_before" in own.columns else "") + f" — the same 'together, not before' as the group, pair by pair, with the pair-level rise in volatility smaller than the group's (the group's jump weeks are the big sell-offs; a single pair's jumps include quieter re-couplings). Gold pairs: own vol z {own.loc[og, 'z_vol_+0'].mean():+.2f} in the jump week" + (f", VIX {own.loc[og, 'z_vix_+0'].mean():+.2f}" if "z_vix_+0" in own.columns else "") + " — a gold pair's correlation jumps are not volatility events; they are gold moving with or against the indices for its own reasons."))
    H.append(_img("5_lead_lag_blocks"))
    llb = _csv("5_lead_lag_blocks"); llv = _csv("5_lead_lag_verdicts")
    H.append(_tbl(llv.assign(pair=llv["pair"].str.replace("|", " × ", regex=False)), fmt="{:+.2f}", index=False, int_cols=("block_weeks", "own_vol_n", "VIX_n")))
    H.append(_tbl(llb.assign(pair=llb["pair"].str.replace("|", " × ", regex=False)), fmt="{:+.3f}", index=False, int_cols=("block_weeks", "n", "n_nonoverlap"), max_rows=200))
    H.append(_says(f"<b>The lead/lag test with no overlapping weeks (redone 2026-10-19).</b> X on the block of L weeks ending at t, Y on the block starting at t+1, L = {C.LL_BLOCKS[0]} and {C.LL_BLOCKS[-1]}; both directions, every pair, as changes and as levels; Newey-West t and the non-overlapping sub-sample's Pearson. Nothing leads: no VIX → correlation link passes in the index pairs; the links that pass are negative both ways (after a block in which both spiked, both ease). The sliding test below, whose windows overlap, is kept for reference — its VIX head start was the overlap."))
    H.append(_img("5_vol_corr_lead"))
    H.append(_tbl(vl, fmt="{:.3f}", int_cols=("weeks", "weeks_vix")))
    H.append(_img("5_vol_corr_lead_pairs"))
    vlp = _csv("5_vol_corr_lead_pairs"); vlp_show = vlp.pivot(index="k_weeks", columns="pair", values="corr_dvol_t_dcorr_t_plus_k")
    vlp_show = vlp_show[[p for p in PAIRS_ALL if p in vlp_show.columns]].rename(columns=lambda c: c.replace("|", " × "))
    H.append("<p class='small'>The same test pair by pair — the change in the pair's own volatility (the average of its two legs) at t against the change in the pair's correlation at t + k:</p>")
    H.append(_tbl(vlp_show, fmt="{:.2f}"))
    if "corr_dvix_t_dcorr_t_plus_k" in vlp.columns:
        vlx_show = vlp.pivot(index="k_weeks", columns="pair", values="corr_dvix_t_dcorr_t_plus_k")
        vlx_show = vlx_show[[p for p in PAIRS_ALL if p in vlx_show.columns]].rename(columns=lambda c: c.replace("|", " × "))
        H.append("<p class='small'>And the 13-week change in the log VIX at t against the change in the pair's correlation at t + k:</p>")
        H.append(_tbl(vlx_show, fmt="{:.2f}"))
    pk = {p: int(vlp_show[p].idxmax()) for p in vlp_show.columns if "Gold" not in p}
    pkx = {p: int(vlx_show[p].idxmax()) for p in vlx_show.columns if "Gold" not in p} if "corr_dvix_t_dcorr_t_plus_k" in vlp.columns else {}
    H.append(_says(f"<b>Pair by pair:</b> for the index pairs the realised-vol peak sits at k = 0 in {sum(1 for v in pk.values() if v == 0)} of {len(pk)} (" + ", ".join(f"{p} k={v:+d}" for p, v in pk.items() if v != 0) + ("" if all(v == 0 for v in pk.values()) else " the exceptions") + "): a pair's own volatility and its correlation change in the same week, as the group's do." + (f" The VIX's peak is at k = {', '.join(f'{v:+d}' for v in pkx.values())} for {', '.join(pkx.keys())} — a few weeks' head start in the large-cap pairs, none in the small-cap ones." if pkx else "") + " The gold pairs' bars are flat near zero: nothing in volatility, realised or implied, relates to when a gold pair couples or decouples."))
    lv = vl["corr_dvol_t_dcorr_t_plus_k"]
    lead_says = ""
    if has_vix and "corr_dvix_t_dcorr_t_plus_k" in vl.columns:
        lx = vl["corr_dvix_t_dcorr_t_plus_k"]; kmax = int(lx.idxmax())
        lead_says = f" <b>The VIX is different:</b> the change in the VIX correlates {lx.loc[0]:.2f} with the same week's correlation change but {lx.loc[kmax]:.2f} with the change {kmax:+d} weeks later — its peak sits at k = {kmax:+d}, not 0 — so a rise in implied volatility runs a few weeks <i>ahead</i> of the rise in realised correlation. That is the one place in this study where something moves before the correlation does; it is modest ({lx.loc[kmax]:.2f} against {lv.loc[0]:.2f} for realised vol at k = 0) and the VIX file ends {vix_last}. The levels: VIX vs G correlate {vlv.get('vix_level_corr', np.nan):.2f} (rank {vlv.get('vix_level_spearman', np.nan):.2f}), realised vol vs G {vlv.get('level_corr', np.nan):.2f} (rank {vlv.get('level_spearman', np.nan):.2f})."
    H.append(_says(f"<b>Realised vol:</b> the changes in N100's vol and in the group correlation correlate most in the same week ({lv.loc[0]:.2f} at k = 0) and fall away symmetrically on both sides ({lv.loc[-4]:.2f} at k = −4, {lv.loc[4]:.2f} at k = +4). Neither leads: a vol spike and a correlation spike are the same event seen through two windows.{lead_says}"))
    H.append(_not("'Correlation rises with vol' is measured on overlapping 13-week windows, so it is descriptive (what the regimes look like), not predictive (it does not say that this week's vol forecasts next quarter's correlation). The regime edges are quintiles over the whole sample, so Q5 is simply 'the wildest fifth of weeks since 2016' under that measure; for 'the pair's own vol' each pair has its own edges. The VIX-leads finding is a few weeks of lead in a noisy series and would need the refreshed VIX file (fetch_vix.py) and a pre-registered test before it is used."))

    # ============================================================ 4 history ==
    H.append("<h2>4. The historical pattern between correlation and returns</h2>")
    H.append(_ask("When the indices are tightly correlated, what is the market doing at the time — and what does it do next? Is a high correlation good news, bad news, or no news for the following quarter?",
                  f"The 13-week average correlation among the four indices (G), week by week. <b>(a) By year:</b> its mean, its low–high range and the number of weeks above 0.90 in each calendar year, next to each asset's return that year. <b>(b) By quintile:</b> every week is sorted by G into five equal buckets (Q1 the least coupled fifth of weeks … Q5 the most). Inside each bucket, each asset's return over the <i>same</i> 13 weeks the correlation was measured on, and over the <i>next</i> {C.FWD_WINDOW} weeks (t+1 … t+13), are averaged; <code>_hit</code> is the share of those quarters that were positive. <b>(c) Stretches:</b> every run of at least 8 weeks with G above 0.90, with what N100, SC100 and gold returned during it and over the 13 weeks after it ended.",
                  "<code>same_quarter_mean</code> answers 'what was the market doing while correlation sat at this level' — the correlation and the return are measured on the same weeks, so this is a description, not a forecast. <code>next_quarter_mean</code> is the forward-looking one: if Q5's next quarter beats Q1's, tightly-correlated weeks were followed by better returns. <code>corr_lo</code>–<code>corr_hi</code> is the range of G inside the bucket; <code>weeks</code> the count."))
    H.append("<h3>4.1 By year</h3>")
    H.append(_img("2_by_year"))
    H.append(_tbl(yr, fmt="{:.3f}", int_cols=("weeks", "weeks_corr_above_0.9"), pct_cols=tuple(c for c in yr.columns if c.startswith("ret_"))))
    ypr = _csv("2_by_year_pairs", index_col=0); ypr = ypr[[p for p in PAIRS_ALL if p in ypr.columns]].rename(columns=lambda c: c.replace("|", " × ")); ypr.insert(0, "group G", yr["mean_corr"])
    H.append("<p class='small'>Every pair's average 13-week correlation by calendar year (the lines on the chart):</p>")
    H.append(_tbl(ypr, fmt="{:.2f}"))
    yr_hi, yr_lo = yr["mean_corr"].idxmax(), yr["mean_corr"].idxmin()
    H.append(_says(f"<b>Reading the table:</b> in {yr.index[0]} the four indices averaged {yr['mean_corr'].iloc[0]:.2f} correlation, swinging between {yr['min_corr'].iloc[0]:.2f} and {yr['max_corr'].iloc[0]:.2f} within the year, with {int(yr['weeks_corr_above_0.9'].iloc[0])} weeks above 0.90; N100 returned {_pct(yr['ret_N100'].iloc[0])} and SC100 {_pct(yr['ret_SC100'].iloc[0])} that year. The yearly average has drifted up, from {yr['mean_corr'].iloc[0]:.2f} in {yr.index[0]} to {yr['mean_corr'].iloc[-1]:.2f} in {yr.index[-1]} (highest {yr.loc[yr_hi, 'mean_corr']:.2f} in {yr_hi}, lowest {yr.loc[yr_lo, 'mean_corr']:.2f} in {yr_lo}). The most-correlated years are not the worst years — {yr_hi} returned N100 {_pct(yr.loc[yr_hi, 'ret_N100'])} — because a year averages over both the sell-off and the recovery; the quintile table below separates the two."))
    H.append("<h3>4.2 By correlation quintile: the same quarter, and the next</h3>")
    H.append(_img("2_corr_quintiles"))
    H.append(_tbl(piv, fmt="{:+.3f}", pct_cols=tuple(piv.columns)))
    H.append("<p class='small'>N100's rows in full (the bucket edges, the counts and the hit rates):</p>")
    H.append(_tbl(lg[lg["asset"] == "N100"][["bucket", "corr_lo", "corr_hi", "weeks", "same_quarter_mean", "same_quarter_hit", "next_quarter_mean", "next_quarter_hit"]], fmt="{:.3f}", index=False, int_cols=("weeks",), pct_cols=("same_quarter_mean", "next_quarter_mean")))
    H.append(_says(f"<b>Reading Q5 for N100:</b> the {int(n100.loc['Q5', 'weeks'])} weeks in which G was between {n100.loc['Q5', 'corr_lo']:.2f} and {n100.loc['Q5', 'corr_hi']:.2f}. Over the quarter <i>ending</i> in those weeks N100 had returned {_pct(n100.loc['Q5', 'same_quarter_mean'])} on average and was up in only {n100.loc['Q5', 'same_quarter_hit'] * 100:.0f}% of them; over the quarter <i>after</i> them it returned {_pct(n100.loc['Q5', 'next_quarter_mean'])} and was up {n100.loc['Q5', 'next_quarter_hit'] * 100:.0f}% of the time. <b>Q1</b> (G {n100.loc['Q1', 'corr_lo']:.2f}–{n100.loc['Q1', 'corr_hi']:.2f}) is the mirror image: {_pct(n100.loc['Q1', 'same_quarter_mean'])} over the same quarter, up {n100.loc['Q1', 'same_quarter_hit'] * 100:.0f}% of the time, then {_pct(n100.loc['Q1', 'next_quarter_mean'])} the next. SC100 is the same with bigger numbers: {_pct(sc.loc['Q5', 'same_quarter_mean'])} / {_pct(sc.loc['Q1', 'same_quarter_mean'])} over the same quarter, {_pct(sc.loc['Q5', 'next_quarter_mean'])} / {_pct(sc.loc['Q1', 'next_quarter_mean'])} over the next. "
                   f"<b>So:</b> high correlation is a <i>picture of a sell-off in progress</i> — the market falls together, and the correlation measured on those weeks is high because of it — and the quarter after the most-correlated weeks has, on this history, been better than average, because sell-offs have ended. Low correlation is what a calm, rising, dispersed market looks like."))
    H.append("<h4>Pair by pair</h4>")
    qp = _csv("2_corr_quintile_returns_pairs")
    H.append(_img("2_corr_quintiles_pairs"))
    F4 = C.QUINT_FWD
    qp_show = qp[qp["pair"].isin([p for p in PAIRS_ALL if "Gold" not in p])][["pair", "leg", "bucket", "corr_lo", "corr_hi", "weeks", "same_quarter_mean", "same_quarter_hit", f"next_{F4}w_mean", f"next_{F4}w_hit", f"n_next_{F4}w", "next_quarter_mean", "next_quarter_hit", "n_next"]].assign(pair=lambda d: d["pair"].str.replace("|", " × ", regex=False))
    H.append(f"<p class='small'>The index pairs in full (gold pairs in <code>2_corr_quintile_returns_pairs.csv</code>); the chart's bars are the next {F4} weeks (changed 2026-10-19), the next-quarter columns are kept:</p>")
    H.append(_tbl(qp_show, fmt="{:.3f}", index=False, int_cols=("weeks", f"n_next_{F4}w", "n_next"), pct_cols=("same_quarter_mean", f"next_{F4}w_mean", "next_quarter_mean"), max_rows=60))
    H.append("<h4>The ratio study (added 2026-10-19)</h4>")
    H.append(_img("2_ratio_by_corr"))
    H.append(_img("2_ratio_by_state"))
    rsf = _csv("2_ratio_study")
    H.append(_tbl(rsf.assign(pair=rsf["pair"].str.replace("|", " ÷ ", regex=False)), fmt="{:+.3f}", index=False, int_cols=("weeks", "n_forward"), pct_cols=(f"past_{C.RATIO_BACK}w_mean", f"fwd_{C.RATIO_FWD}w_mean", f"fwd_{C.RATIO_FWD}w_median", f"fwd_{C.RATIO_FWD}w_sd"), max_rows=140))
    H.append(_says(f"<b>The ratio first leg ÷ second leg, forward {C.RATIO_FWD} weeks, by the pair's correlation quintile, the joint signal state and the sign of its last {C.RATIO_BACK} weeks.</b> Nothing sorts it in the full-history pairs: every bar within about ±1%, hit rates near a coin flip, rank correlations with the correlation level and with the past change both near zero. The Micro250 pairs show larger bars on a quarter of the history — too thin to read."))
    qe = qp[~qp["pair"].str.contains("Gold")]; qg = qp[qp["pair"].str.contains("Gold") & (qp["leg"] == "GoldMCX")]
    H.append(_says(f"<b>Pair by pair the shape holds.</b> Sorting the weeks by each pair's own correlation instead of G: across the six index pairs the legs returned {_pct(qe[qe['bucket'] == 'Q5']['same_quarter_mean'].mean())} over the quarter ending in the pair's most-correlated weeks and {_pct(qe[qe['bucket'] == 'Q5']['next_quarter_mean'].mean())} over the next, against {_pct(qe[qe['bucket'] == 'Q1']['same_quarter_mean'].mean())} and {_pct(qe[qe['bucket'] == 'Q1']['next_quarter_mean'].mean())} for the least-correlated weeks. The two Micro250 pairs, with one sell-off in their sample, show the same-quarter fall but not the better next quarter. In the gold pairs the bucket is 'how much gold happened to move with that index': gold's own next quarter is {_pct(qg[qg['bucket'] == 'Q1']['next_quarter_mean'].mean())} after Q1 and {_pct(qg[qg['bucket'] == 'Q5']['next_quarter_mean'].mean())} after Q5 — no pattern — while the index leg shows the usual sell-off shape in Q5."))
    H.append("<h3>4.3 The timeline, and the stretches above 0.90</h3>")
    H.append(_img("2_timeline"))
    H.append(_tbl(hc, fmt="{:+.3f}", index=False, int_cols=("weeks",), pct_cols=tuple(c for c in hc.columns if "_during" in c or "_next" in c)))
    hps = _csv("2_high_corr_pairs_summary", index_col=0); hps = hps.reindex([p for p in PAIRS_ALL if p in hps.index]); hpp = _csv("2_high_corr_episodes_pairs")
    H.append("<p class='small'>The stretches above 0.90 pair by pair — summary, then every stretch ('a' = the first leg, 'b' = the second):</p>")
    H.append(_tbl(hps.rename(index=lambda p: p.replace("|", " × ")), fmt="{:.1f}", int_cols=("weeks_above_0.90", "stretches"), pct_cols=("share_of_weeks_above_0.90", "a_next_13w_mean", "a_next_13w_positive", "b_next_13w_mean", "b_next_13w_positive")))
    H.append(_tbl(hpp.assign(pair=hpp["pair"].str.replace("|", " × ", regex=False)), fmt="{:+.3f}", index=False, int_cols=("weeks",), pct_cols=tuple(c for c in hpp.columns if "_during" in c or "_next" in c), max_rows=60))
    hps_top = hps["share_of_weeks_above_0.90"].idxmax(); hps_lo = hps.loc[[p for p in hps.index if "Gold" not in p], "share_of_weeks_above_0.90"].idxmin()
    H.append(_says(f"<b>Pair by pair:</b> {hps_top.replace('|', '×')} is above 0.90 in {hps.loc[hps_top, 'share_of_weeks_above_0.90']:.0%} of all weeks ({int(hps.loc[hps_top, 'stretches'])} stretches of 8+ weeks, median {hps.loc[hps_top, 'median_weeks']:.0f} weeks), {hps_lo.replace('|', '×')} in only {hps.loc[hps_lo, 'share_of_weeks_above_0.90']:.0%}; no gold pair ever reaches 0.90. The group's stretches are therefore the weeks the N100 pairs join the mid/small pairs up there."))
    hc_pos = int((hc["N100_next_13w"].dropna() > 0).sum()); hc_n = int(hc["N100_next_13w"].notna().sum())
    H.append(_says(f"{len(hc)} stretches of at least 8 weeks above 0.90 since {C.START}, lasting {int(hc['weeks'].min())}–{int(hc['weeks'].max())} weeks. During them N100 returned between {_pct(hc['N100_during'].min())} and {_pct(hc['N100_during'].max())}; in the 13 weeks after they ended, N100 was positive in {hc_pos} of {hc_n} ({_pct(hc['N100_next_13w'].mean())} on average), SC100 {_pct(hc['SC100_next_13w'].mean())}, gold {_pct(hc['GoldMCX_next_13w'].mean())}. The {hc['start'].iloc[0]} and {hc['start'].iloc[2] if len(hc) > 2 else ''} stretches were followed by strong quarters; the {hc['start'].iloc[3] if len(hc) > 3 else ''} one was not. A stretch of 0.90+ is a sell-off or its aftermath; what follows depends on which."))
    H.append(_not("'The quarter after Q5 was good' is an average over ten years that contain a handful of sharp recoveries (2020 above all); it is a tendency, not a timing rule, and the overlapping quarters mean the effective sample is far smaller than the week count suggests. Nothing here says that <i>this week's</i> correlation forecasts next quarter's return with any reliability — the §2–§3 finding that correlation moves after the signal, not before, applies."))


    # ============================================================ 5 who breaks correlation ==
    H.append("<h2>5. Who breaks the correlation first</h2>")
    H.append(_ask("When the market stops moving as one block, which index lets go of the others first — and which way does it go?",
                  f"(1) Each index's 'correlation with the others' <code>c</code>: its {C.CB_WINDOW}-week correlation with each of the other three size indices, averaged. (2) The group's correlation <code>G</code>: the average of all six pairs. (3) Each series as a z-score against its own {C.CB_NORM}-week history. (4) A break = the first week z ≤ {C.CB_Z:.0f} after having been above {C.CB_RESET:.1f}; applied to G it opens a decorrelation episode (which ends when z &gt; {C.CB_RESET:.1f}); applied to each c it marks that index letting go. (5) For each episode, each index's own break date is looked for from {C.CB_LOOKBACK} weeks before the group break to the episode's end; lead = its date − the group's, in weeks; ranked by date, with two tie-breakers (who was deepest below its norm when the group broke; whose correlation fell most). (6) Direction: at its break week the index's 13-week return minus the others' — positive is an up-break (it decoupled by outperforming). (7) What followed: returns over the next {C.CB_FWD[0]} and {C.CB_FWD[1]} weeks. (8) The same ranking re-run at window {C.CB_SENSITIVITY['window']} × threshold {C.CB_SENSITIVITY['z']}.",
                  "Negative lead = broke before the market. Ties by date are common because G is the average of the c's, so several cross together — that is why depth and drop are reported beside date. <code>share_up_breaks</code> says how often the index left the group by outperforming."))
    ex = cep.iloc[0]
    exb = ab[ab["episode"] == ex["episode"]].set_index("asset")
    H.append(f"<div class='ask'><b>A worked example — {ex['episode']}.</b> In the {C.CB_LOOKBACK} weeks before, the group's correlation averaged {ex['G_before']:.2f}. On {ex['start'].date()} its z-score crossed {C.CB_Z:.0f}: the market was decorrelating. It bottomed at {ex['G_min']:.2f} on {ex['G_min_date'].date()} (z {ex['z_min']:+.1f}) and the episode lasted {int(ex['weeks'])} weeks. "
             + " ".join(f"{a}'s own correlation with the others went from {exb.loc[a, 'corr_before']:.2f} to {exb.loc[a, 'corr_min']:.2f} (a drop of {abs(exb.loc[a, 'corr_drop']):.2f}, {exb.loc[a, 'drop_share'] * 100:.0f}% of the group's loss); it broke {_wk(exb.loc[a, 'lead_vs_group_weeks'])} the group{', ' + exb.loc[a, 'direction'] if isinstance(exb.loc[a, 'direction'], str) and exb.loc[a, 'direction'] else ''}." for a in exb.index)
             + " Ranked by date: " + ", ".join(f"{a} #{int(exb.loc[a, 'rank'])}" for a in exb.index if pd.notna(exb.loc[a, "rank"])) + ".</div>")
    H.append("<h3>5.1 The correlation structure over time</h3>")
    H.append(_img("3_corr_series"))
    H.append(_tbl(cep[["episode", "start", "end", "weeks", "G_before", "G_min", "G_min_date", "drop", "z_min"]], fmt="{:.2f}", index=False, int_cols=("weeks",)))
    H.append("<h3>5.2 The summary</h3>")
    H.append(_img("3_who_breaks_correlation"))
    cols = ["episodes_present", "episodes_broke", "share_first", "share_first_alone", "avg_rank", "median_lead_weeks", "share_deepest", "avg_rank_deepest", "share_biggest_drop", "avg_rank_drop", "avg_drop_share", "combined_rank", "share_up_breaks", "median_rel_13w_return_at_break", "verdict"]
    H.append(_tbl(sm[cols], fmt="{:.2f}", int_cols=("episodes_present", "episodes_broke")))
    H.append(_says(f"By date {date_first} leads ({sm.loc[date_first, 'share_first'] * 100:.0f}% of episodes incl. ties, alone first in {sm.loc[date_first, 'share_first_alone'] * 100:.0f}%); by depth {deep_first} ({sm.loc[deep_first, 'share_deepest'] * 100:.0f}%); by size of fall {drop_first} ({sm.loc[drop_first, 'share_biggest_drop'] * 100:.0f}%, losing on average {sm.loc[drop_first, 'avg_drop_share'] * 100:.0f}% of the group's total correlation). These describe one mechanism from three sides: a decorrelation here is the mid, small and micro caps running away from Nifty 100 — {sm.loc['MC100', 'share_up_breaks'] * 100:.0f}% of Midcap's and {sm.loc['SC100', 'share_up_breaks'] * 100:.0f}% of Smallcap's breaks are up-breaks, {100 - sm.loc['N100', 'share_up_breaks'] * 100:.0f}% of Nifty's are down-breaks. The index that starts running first crosses the line first; the index whose relationship collapses is the one everybody runs away from. <b>{date_first} tends to break first; {drop_first} ends up most broken.</b>"))
    H.append("<h3>5.3 Episode by episode</h3>")
    H.append(_img("3_corr_episodes"))
    for _, e in cep.iterrows():
        g = ab[ab["episode"] == e["episode"]].set_index("asset")
        order = g[g["break_date"].notna()].sort_values(["break_date", "z_at_group_break"])
        nob = g[g["break_date"].isna()].index.tolist()
        H.append(f"<div class='ep'><b>{e['episode']}</b> — {e['start'].date()} to {e['end'].date()}, {int(e['weeks'])} weeks; group {e['G_before']:.2f} → {e['G_min']:.2f} (z {e['z_min']:+.1f}). "
                 + "Order of breaking: " + "; ".join(f"<b>{a}</b> {_wk(g.loc[a, 'lead_vs_group_weeks'])} (z at the group break {g.loc[a, 'z_at_group_break']:+.1f}, own correlation {g.loc[a, 'corr_before']:.2f} → {g.loc[a, 'corr_min']:.2f}, {str(g.loc[a, 'direction']).split(' (')[0]}, 13-week relative return {_pct(g.loc[a, 'rel_13w_return_at_break'])})" for a in order.index)
                 + (f". Did not break: {', '.join(nob)}" if nob else "") + ".</div>")
    H.append("<h3>5.4 The three orderings, episode × asset</h3>")
    for name, title in (("date", "rank by break date (1 = earliest; ties share)"), ("deepest", "rank by z-score in the week the group broke (1 = furthest below its norm)"), ("drop", "rank by the fall in its own correlation (1 = biggest)")):
        H.append(f"<p class='small'>{title}:</p>")
        H.append(_tbl(_csv(f"3_rank_{name}", index_col=0), fmt="{:.0f}"))
    H.append("<p class='small'>Lead in weeks vs the group break (negative = earlier):</p>")
    H.append(_tbl(_csv("3_lead_weeks", index_col=0), fmt="{:+.0f}"))
    H.append("<h3>5.5 At pair level: which relationship goes first</h3>")
    H.append(_tbl(psm, fmt="{:.2f}"))
    H.append(_tbl(pb.pivot(index="episode", columns="pair", values="lead_vs_group_weeks"), fmt="{:+.0f}"))
    H.append("<h3>5.6 Does the correlation break before a price sell-off?</h3>")
    H.append(_tbl(bsf, fmt="{:.2f}", index=False))
    H.append(_says(f"The group decorrelated in the 13 weeks before N100's peak in {n_warn} of {n_sell} sell-offs ({', '.join(bsf[bsf['group_broke_before_peak']]['selloff'])}). Where it happened, every index crossed in the same week — a market-wide decoupling, not one asset leading. Where it did not ({', '.join(bsf[~bsf['group_broke_before_peak']]['selloff'])}), the fall came out of a correlated market. Rising correlation is not the warning either (§2.2)."))
    H.append("<h3>5.7 What happens after an index breaks away</h3>")
    H.append(_img("3_after_break"))
    H.append(_tbl(afs, fmt="{:+.3f}", index=False, int_cols=("n", "weeks_ahead"), pct_cols=("breaker", "rest_of_group", "gold", "breaker_minus_rest")))
    H.append(_says(f"Up-breaks are the common case ({int(up.loc[4, 'n'])} of {int(up.loc[4, 'n'] + dn.loc[4, 'n'])}). The index that decoupled by outperforming returned {_pct(up.loc[13, 'breaker'])} over the next 13 weeks against {_pct(up.loc[13, 'rest_of_group'])} for the rest — ahead only {up.loc[13, 'share_breaker_ahead'] * 100:.0f}% of the time: decoupling by outperformance has tended to be the end of a run, not the start of one. After a down-break the laggard returned {_pct(dn.loc[13, 'breaker'])} vs {_pct(dn.loc[13, 'rest_of_group'])} (ahead {dn.loc[13, 'share_breaker_ahead'] * 100:.0f}%, n {int(dn.loc[13, 'n'])}). Gold returned {_pct(up.loc[13, 'gold'])} and {_pct(dn.loc[13, 'gold'])} over the same quarters."))
    H.append("<h3>5.8 Does the answer depend on the settings?</h3>")
    H.append(f"<p class='small'>Episode counts by setting: {', '.join(f'{k}: {v}' for k, v in meta.get('sensitivity_episodes', {}).items())}.</p>")
    H.append(_tbl(sens, fmt="{:.2f}"))
    H.append(_not(f"The 'earliest by date' winner moves with the settings — the date ranking is dominated by ties and {len(cep)} episodes. The direction finding (smaller caps break upward, Nifty breaks downward) and 'Nifty loses the most correlation' do not depend on the threshold, because they come from the size and sign of the moves, not from who crossed a line first. No t-statistics here: ten episodes are too few for them to mean anything."))

    # ============================================================ 6 price/signal ==
    H.append("<h2>6. The other meaning: who breaks in price, and whose signal fires first</h2>")
    H.append(_ask("In a sell-off, which index falls first, whose supertrend fires first, and who recovers first?",
                  f"A sell-off = a {C.DD_THRESHOLD:.0%}+ fall in N100 from its running high on weekly closes. For each sell-off and asset: its own peak (looked for from {C.DD_PEAK_LOOKBACK} weeks before N100's peak), the week its supertrend flipped to SELL, the week it closed {C.DD_CROSS:.0%} below its own high, and the week it flipped back to BUY after the trough (within {C.RECOVERY_WINDOW} weeks) — each as a lead in weeks relative to N100's own date.",
                  "Negative lead = earlier than N100. Ranks within an episode share ties. <code>first_share_…</code> = the share of sell-offs the asset was first (ties counted)."))
    H.append(_tbl(pep, fmt="{:.3f}", index=False, int_cols=("weeks_peak_to_trough", "weeks_to_recover"), pct_cols=("depth",)))
    H.append(_img("4_who_breaks_first"))
    H.append(_tbl(sm4[["episodes", "first_share_signal_sell", "avg_rank_signal_sell", "median_lead_signal_sell_weeks", "first_share_cross_10pct", "avg_rank_cross_10pct", "median_lead_cross_10pct_weeks", "first_share_recover_buy", "avg_rank_recover_buy", "median_own_fall", "verdict"]], fmt="{:.2f}", int_cols=("episodes",)))
    H.append(_img("4_price_episodes"))
    for col, title in (("signal_sell", "supertrend flipped to SELL"), ("cross_10pct", "price crossed −10%"), ("recover_buy", "supertrend flipped back to BUY")):
        H.append(f"<p class='small'>Weeks relative to N100, {title} (negative = earlier):</p>")
        H.append(_tbl(_csv(f"4_lead_{col}", index_col=0), fmt="{:+.0f}"))
    H.append(_says(f"{first_cross} crosses −10% first in {sm4.loc[first_cross, 'first_share_cross_10pct'] * 100:.0f}% of sell-offs (median {-sm4.loc[first_cross, 'median_lead_cross_10pct_weeks']:.0f} weeks before N100) and falls most (median {_pct(sm4.loc[first_cross, 'median_own_fall'])} to the trough); {first_sig}'s supertrend fires first ({sm4.loc[first_sig, 'first_share_signal_sell'] * 100:.0f}%, average rank {sm4.loc[first_sig, 'avg_rank_signal_sell']:.1f}); {first_rec} turns back to BUY first ({sm4.loc[first_rec, 'first_share_recover_buy'] * 100:.0f}%). Price and signal give different leaders because the supertrend's band scales with each asset's own volatility: a 10% fall is a bigger event, in band terms, for a low-vol large cap than for a high-vol small cap."))
    H.append(_not(f"{len(pep)} sell-offs; Micro250 is present in {int(sm4.loc['Micro250', 'episodes']) if 'Micro250' in sm4.index else 0}. Ranks and medians only."))

    # ============================================================ 7 pairs ==
    H.append("<h2>7. Each pair separately</h2>")
    H.append(_ask("For each of the ten pairs: how far apart have the two legs drifted (the deviation), does the ratio pull back when it has drifted, how coupled and how volatile are the legs, and what happened on the pair's own SELL flips and pair trades?",
                  f"The <b>price ratio</b> ln(P<sub>a</sub>/P<sub>b</sub>) and its <b>deviation z</b> against the trailing {C.PAIR_WINDOW} weeks (§1.3). <b>Pull-back</b> is tested two ways: statistically — an ADF test on the log ratio (does it have a long-run level?), Engle-Granger both ways (is there a hedge ratio at which it does?), and the half-life of the ratio from an AR(1) fit (a half-life above {C.COINT_MAX_HALFLIFE} weeks counts as 'does not pull back in practice') — and empirically: every week is put into a z bucket ({_esc(' / '.join(B))}) and the ratio's change over the next {H4} and {H13} weeks is averaged inside each bucket, with a Newey-West t. The card also carries the pair's correlation (full sample, 13-week, 52-week), each leg's volatility, the beta of a on b, how often |z| exceeded 1 and 2, and the pair's rows from §8 and §9.",
                  "<b>Sign convention, everywhere:</b> z &gt; 0 means the <i>first-named</i> asset has run ahead. In the pull-back table, a pull-back after 'a lagged' (z &lt; 0) would show as a <i>positive</i> ratio change, and after 'a led' (z &gt; 0) as a negative one; <code>share_reverted</code> is the share of weeks the ratio moved back toward its mean at all. On each card: panel 1 each leg's cumulative return (both start at 100; log scale — the returns the ratio is made of); panel 2 the ratio with its 52-week mean and ±1/±2 sd bands; panel 3 the z-score with ±1/±2 lines, ▼ (top row) the first leg's SELL flips, ▲ (bottom row) the second leg's, ● pair-trade entries and ○ exits; panel 4 the 13-week correlation; panel 5 each leg's 13-week volatility" + (" with the VIX on the right axis" if has_vix else "") + ". Hover for the values; click a legend entry to hide or show it; drag to zoom, double-click to reset. Below each chart: the pair's facts, a <b>returns table</b> (each leg's total, annualised, last-13-week and last-52-week return, and its average annualised return in both-BUY / both-SELL / disagree weeks, with the spread a − b), the pull-back table, and the pair's own rows from §8 and §9."))
    H.append("<h3>7.1 Do deviations pull back? — all pairs</h3>")
    H.append("<h4>Statistically</h4>")
    H.append(_tbl(pco[["weeks", "adf_p_log_ratio", "eg_p_a_on_b", "eg_p_b_on_a", "halflife_weeks_full", "halflife_weeks_vs_52w_mean", "ratio_drift_per_year", "verdict"]], fmt="{:.2f}", int_cols=("weeks",), pct_cols=("ratio_drift_per_year",)))
    H.append(f"<p class='small'>p-values below 0.05 mean 'a long-run level is established'. <code>halflife_weeks_full</code> is how long a deviation from the full-sample level takes to halve; <code>halflife_weeks_vs_52w_mean</code> the same against the moving 52-week mean (always finite, because that mean follows the level — see the trap in §1.3). <code>ratio_drift_per_year</code> is the ratio's average yearly change: a steady negative number for N100|SC100 means Smallcap has out-run Nifty on average, year after year.</p>")
    H.append("<h4>Empirically — what the ratio did next, by this week's deviation</h4>")
    H.append(_img("6_z_forward"))
    H.append(_tbl(pooled[["horizon_weeks", "bucket", "weeks", "mean_ratio_change", "median_ratio_change", "share_reverted", "hac_t_ratio_change", "mean_ret_a", "mean_ret_b"]], fmt="{:.3f}", index=False, int_cols=("horizon_weeks", "weeks"), pct_cols=("mean_ratio_change", "median_ratio_change", "mean_ret_a", "mean_ret_b")))
    H.append(_img("6_path_after_2sd"))
    H.append(_says(f"Statistically, {len(coint_ok)} of 10 pairs ({', '.join(p.replace('|', '×') for p in coint_ok) if coint_ok else 'none'}) has a ratio with a long-run level; the index-vs-index ratios drift (the smaller index out-runs the larger one over the years, by {pco.loc['N100|SC100', 'ratio_drift_per_year'] * -100:.0f}% a year for N100|SC100 and {pco.loc['N100|MC100', 'ratio_drift_per_year'] * -100:.0f}% for N100|MC100), and so do the gold ratios. Empirically, across all ten pairs, a 1–2 sd deviation continued rather than reverted: after '{_esc(B[1])}' the ratio moved a further {_pct(pz.loc[B[1], 'mean_ratio_change'])} the same way over {H13} weeks (t {_t(pz.loc[B[1], 'hac_t_ratio_change'])}, reverted in {_p(pz.loc[B[1], 'share_reverted'])} of weeks), after '{_esc(B[3])}' a further {_pct(pz.loc[B[3], 'mean_ratio_change'])} (t {_t(pz.loc[B[3], 'hac_t_ratio_change'])}, reverted {_p(pz.loc[B[3], 'share_reverted'])}). Beyond 2 sd the ratio went nowhere ({_pct(pz.loc[B[0], 'mean_ratio_change'])} and {_pct(pz.loc[B[4], 'mean_ratio_change'])}, t {_t(pz.loc[B[0], 'hac_t_ratio_change'])} and {_t(pz.loc[B[4], 'hac_t_ratio_change'])}) — the run stopped, but it did not come back. The z-path chart shows the z falling from about 2.2 to about {pa[pa['weeks_after'] == C.PAIR_PATH_WEEKS]['mean_z'].mean():.1f} within {C.PAIR_PATH_WEEKS} weeks: that is the window catching up with the level, not the ratio returning — the empirical table is the one to trust."))
    H.append(_not("'Continued' is an average over all weeks in the bucket, with heavily overlapping windows; the t-stats carry the Newey-West correction but the effective sample is a few dozen independent episodes per pair. The finding is consistent with §5.7 (an index that decouples by outperforming tends to be near the end of its run) — moderate deviations keep going, extreme ones stall."))

    # ---- the ten cards
    zfp = zf[zf["pair"] != "ALL PAIRS"]
    for p in pst.index:
        a, b = p.split("|")
        s = pst.loc[p]; c = pco.loc[p]
        z13 = zfp[(zfp["pair"] == p) & (zfp["horizon_weeks"] == H13)].set_index("bucket")
        ptr = pts.loc[p] if p in pts.index else None
        ab_ = bp[(bp["seller"] == a) & (bp["candidate"] == b)]; ba_ = bp[(bp["seller"] == b) & (bp["candidate"] == a)]
        H.append(f"<div class='card'><h3>7.{2 + list(pst.index).index(p)} {a} × {b}</h3>")
        H.append(_img("6_pair", pair=p))
        H.append("<div class='kv'>"
                 f"<div><b>weeks</b> {int(s['weeks'])} ({s['first']} → {s['last']})</div>"
                 f"<div><b>correlation</b> full {s['corr_full']:.2f} · 13w now {s['corr_13w_now']:.2f} (median {s['corr_13w_median']:.2f}, range {s['corr_13w_min']:.2f}–{s['corr_13w_max']:.2f}) · 52w now {s['corr_52w_now']:.2f}</div>"
                 f"<div><b>volatility</b> {a} {_p(s['vol_a_full'])} / {b} {_p(s['vol_b_full'])} full · now {_p(s['vol_a_13w_now'])} / {_p(s['vol_b_13w_now'])} (ratio {s['vol_ratio_a_over_b_now']:.2f})</div>"
                 f"<div><b>beta of {a} on {b}</b> {s['beta_a_on_b_full']:.2f} full · {s['beta_a_on_b_52w_now']:.2f} last 52w</div>"
                 f"<div><b>deviation now</b> z {s['z_now']:+.2f} ({s['z_date']}): {s['reading_now']}</div>"
                 f"<div><b>|z| &gt; 1</b> {_p(s['share_weeks_abs_z_gt_1'])} of weeks · <b>|z| &gt; 2</b> {_p(s['share_weeks_abs_z_gt_2'])} · {int(s['n_2sd_deviations'])} separate 2-sd deviations · z range {s['z_min']:+.1f} … {s['z_max']:+.1f}</div>"
                 f"<div><b>ratio over the sample</b> {a} {'out' if s['ratio_total_change'] > 0 else 'under'}-performed {b} by {abs(s['ratio_total_change']) * 100:.0f}% in total</div>"
                 f"<div><b>pull-back</b> {c['verdict']} · ADF p {c['adf_p_log_ratio']:.2f} · EG p {min(c['eg_p_a_on_b'], c['eg_p_b_on_a']):.2f} · drift {c['ratio_drift_per_year'] * 100:+.1f}%/yr</div>"
                 "</div>")
        rr = rt.loc[p]
        rtab = pd.DataFrame({a: [rr["total_a"], rr["annualised_a"], rr["last_13w_a"], rr["last_52w_a"], rr["both_BUY_ann_a"], rr["both_SELL_ann_a"], rr["disagree_ann_a"], rr["vol_a"], rr["share_weeks_up_a"], rr["worst_week_a"]],
                             b: [rr["total_b"], rr["annualised_b"], rr["last_13w_b"], rr["last_52w_b"], rr["both_BUY_ann_b"], rr["both_SELL_ann_b"], rr["disagree_ann_b"], rr["vol_b"], rr["share_weeks_up_b"], rr["worst_week_b"]],
                             f"spread {a} − {b}": [rr["spread_total"], rr["spread_annualised"], rr["spread_last_13w"], rr["spread_last_52w"], rr["spread_both_BUY_ann"], rr["spread_both_SELL_ann"], rr["spread_disagree_ann"], np.nan, np.nan, np.nan]},
                            index=["total return over the common sample", "annualised", "last 13 weeks", "last 52 weeks", f"both BUY weeks ({int(rr['both_BUY_weeks'])}), annualised", f"both SELL weeks ({int(rr['both_SELL_weeks'])}), annualised", f"disagree weeks ({int(rr['disagree_weeks'])}), annualised", "volatility (full)", "share of weeks up", "worst week"])
        H.append(f"<h4>Returns of the two legs ({int(rr['weeks'])} common weeks, {rr['years']:.1f} years)</h4>")
        H.append(_tbl(rtab, fmt="{:.2f}", pct_cols=tuple(rtab.columns)))
        H.append("<h4>What the ratio did over the next 13 weeks, by this week's deviation</h4>")
        H.append(_tbl(z13.reset_index()[["bucket", "weeks", "mean_ratio_change", "share_reverted", "hac_t_ratio_change", "mean_ret_a", "mean_ret_b"]], fmt="{:.2f}", index=False, int_cols=("weeks",), pct_cols=("mean_ratio_change", "mean_ret_a", "mean_ret_b")))
        lag_b, lead_b = z13.loc[B[1]] if B[1] in z13.index else None, z13.loc[B[3]] if B[3] in z13.index else None
        lines = [f"<b>Returns:</b> over the {rr['years']:.1f} years both exist, {a} returned {_pct(rr['total_a'], 0)} ({_pct(rr['annualised_a'])} a year) and {b} {_pct(rr['total_b'], 0)} ({_pct(rr['annualised_b'])} a year), so {b if rr['spread_total'] < 0 else a} led by {abs(rr['spread_annualised']) * 100:.1f} points a year; {a} beat {b} in {_p(rr['share_weeks_a_beat_b'])} of weeks. In both-BUY weeks the two ran at {_pct(rr['both_BUY_ann_a'], 0)} and {_pct(rr['both_BUY_ann_b'], 0)} annualised, in both-SELL weeks at {_pct(rr['both_SELL_ann_a'], 0)} and {_pct(rr['both_SELL_ann_b'], 0)} — {b if rr['vol_b'] > rr['vol_a'] else a}, the higher-vol leg, {'gains more when both are BUY and loses more when both are SELL' if (rr['spread_both_BUY_ann'] < 0) == (rr['vol_b'] > rr['vol_a']) and (rr['spread_both_SELL_ann'] > 0) == (rr['vol_b'] > rr['vol_a']) else 'does not simply amplify the other leg in both states'}. Last 52 weeks: {_pct(rr['last_52w_a'])} vs {_pct(rr['last_52w_b'])}."]
        if lag_b is not None and lead_b is not None:
            lines.append(f"When {a} had lagged {b} by 1–2 sd, the ratio moved {_pct(lag_b['mean_ratio_change'])} over the next {H13} weeks ({'a pull-back' if lag_b['mean_ratio_change'] > 0 else 'continued the same way'}, t {_t(lag_b['hac_t_ratio_change'])}, n {int(lag_b['weeks'])}); when {a} had led by 1–2 sd, {_pct(lead_b['mean_ratio_change'])} ({'a pull-back' if lead_b['mean_ratio_change'] < 0 else 'continued'}, t {_t(lead_b['hac_t_ratio_change'])}, n {int(lead_b['weeks'])}).")
        if ptr is not None:
            lines.append(f"<b>Pair trade (§8):</b> {int(ptr['trades'])} trades, {_p(ptr['win_rate_net'])} winners, {_pct(ptr['mean_net'])} net per trade, {_pct(ptr['total_net'])} in total, average {ptr['avg_weeks']:.0f} weeks held, worst trade {_pct(ptr['worst_trade'])} — {ptr['verdict']}. Long-only: the cheap leg returned {_pct(ptr['mean_ret_cheap_leg'])} per trade vs the rich leg's {_pct(ptr['mean_ret_rich_leg'])}.")
        else:
            lines.append("<b>Pair trade (§8):</b> no trade was ever triggered.")
        for lab, d_, x, y in ((f"{a} flips to SELL → enter {b}", ab_, a, b), (f"{b} flips to SELL → enter {a}", ba_, b, a)):
            if len(d_):
                d_ = d_.iloc[0]
                lines.append(f"<b>{lab} (§9):</b> {int(d_['events'])} flips; {y} returned {_pct(d_[f'cand_mean_{H13}'])} over the next {H13} weeks (beat cash {_p(d_[f'share_cand_beat_cash_{H13}'])} of the time) against staying in {x} {_pct(d_[f'stay_mean_{H13}'])} and cash {_pct(d_[f'cash_mean_{H13}'])}; {y}'s own signal was BUY at the flip in {_p(d_['share_cand_in_BUY'])} of them; held until {y}'s own SELL (cap {C.RP_MAX_HOLD} wk): {_pct(d_['hold_mean'])} over {d_['hold_weeks_mean']:.0f} wk on average.")
        H.append(_says(" ".join(lines)))
        H.append("</div>")

    # ============================================================ 8 pair trade ==
    H.append("<h2>8. The pair-trading rule</h2>")
    H.append(_ask("If the book had traded every large deviation as a pair — buy the leg that has lagged, sell the leg that has run — what would it have made?",
                  f"The rule, from the deviation z of §7: at the close of week t, if |z| ≥ {C.PT_ENTRY:.0f} and no trade is open in that pair, go <b>long the cheap leg and short the rich leg</b>, one unit each. The trade earns side × (r<sub>a</sub> − r<sub>b</sub>) each week from t+1. Close at the first week |z| ≤ {C.PT_EXIT:.1f}, or after {C.PT_MAX_HOLD} weeks, or at the end of the data. Cost: {C.PT_COST_BPS} bp one-way per leg, i.e. {4 * C.PT_COST_BPS} bp per round trip. The same trade read long-only: the cheap leg's own return over the trade against the rich leg's — 'had the book simply held the lagging index instead of the leading one'. The rule is then re-run at entry {C.PT_SENSITIVITY['entry']} × exit {C.PT_SENSITIVITY['exit']} × with/without requiring the cheap leg's own signal to be BUY.",
                  "<code>mean_net</code> is the average P&L per trade in log-return units (+0.05 = a one-unit spread position made 5 points). <code>t_mean_net</code> is a plain t on the trade P&Ls (trades do not overlap within a pair). <code>share_exit_z</code> = trades that closed because the deviation came back inside; <code>share_exit_max_hold</code> = trades that were still open after 26 weeks. The verdict needs ≥ 5 trades, a positive mean, ≥ 60% winners and t ≥ 2 to say 'worked'."))
    H.append(_img("7_equity"))
    H.append(_img("7_summary"))
    H.append(_tbl(pts[["trades", "win_rate_net", "mean_net", "median_net", "total_net", "total_cost", "t_mean_net", "avg_weeks", "share_exit_z", "share_exit_max_hold", "mean_ret_cheap_leg", "mean_ret_rich_leg", "share_cheap_beat_rich", "worst_trade", "best_trade", "net_per_year", "equity_drawdown", "verdict"]], fmt="{:.3f}", int_cols=("trades",), pct_cols=("mean_net", "median_net", "total_net", "total_cost", "mean_ret_cheap_leg", "mean_ret_rich_leg", "worst_trade", "best_trade", "net_per_year", "equity_drawdown")))
    H.append("<h4>Sensitivity — the pooled result at other settings</h4>")
    H.append(_tbl(ptsens, fmt="{:.3f}", index=False, int_cols=("trades", "pairs_positive", "pairs_traded"), pct_cols=("mean_net", "total_net")))
    H.append("<h4>Every trade</h4>")
    H.append(_tbl(tr[["pair", "entry_date", "exit_date", "weeks", "cheap_leg", "rich_leg", "z_entry", "z_exit", "exit_reason", "gross", "net", "worst_point", "ret_cheap_leg", "ret_rich_leg", "cheap_leg_state_at_entry"]].assign(entry_date=lambda d: d["entry_date"].dt.date, exit_date=lambda d: d["exit_date"].dt.date), fmt="{:.2f}", index=False, max_rows=200, int_cols=("weeks",), pct_cols=("gross", "net", "worst_point", "ret_cheap_leg", "ret_rich_leg")))
    H.append(_says(f"<b>{ptall['verdict'].capitalize()}.</b> {int(ptall['trades'])} trades across the ten pairs, {_p(ptall['win_rate_net'])} winners, {_pct(ptall['mean_net'])} per trade net ({_pct(ptall['mean_net'] + ptall['total_cost'] / ptall['trades'])} before costs), t {_t(ptall['t_mean_net'])}; {n_pairs_neg} of 10 pairs lost money and none of the {n_pairs_judged} pairs with enough trades earns a 'worked'. Only {_p(ptall['share_exit_z'])} of trades closed because the deviation came back inside; {_p(ptall['share_exit_max_hold'])} hit the 26-week limit still open. Best pair {best_pt.replace('|', '×')} ({_pct(ptp.loc[best_pt, 'mean_net'])} per trade, {int(ptp.loc[best_pt, 'trades'])} trades), worst {worst_pt.replace('|', '×')} ({_pct(ptp.loc[worst_pt, 'mean_net'])}). The long-only view says the same thing: the leg the rule buys returned {_pct(ptall['mean_ret_cheap_leg'])} over the average trade, the leg it sells {_pct(ptall['mean_ret_rich_leg'])} — the laggard kept lagging in {_p(1 - ptall['share_cheap_beat_rich'])} of trades. Requiring the cheap leg's own signal to be BUY makes it worse, not better ({_pct(ptsens[(ptsens['entry'] == C.PT_ENTRY) & (ptsens['exit'] == C.PT_EXIT) & (ptsens['require_cheap_leg_BUY'])]['mean_net'].iloc[0])} per trade). The reason is §7.1: these ratios trend; the deviation is a trend in progress, and buying the lagging leg is betting against it."))
    H.append(_not(f"This is an evaluation of a rule, not a backtest of a strategy: one unit per leg, no sizing, no margin or borrow cost on the short leg, a flat {C.PT_COST_BPS} bp cost, and the pairs are not run as one portfolio. It also does not say that <i>momentum</i> in the ratio is tradable — that would be a separate, pre-registered test. What it does say is that the classic mean-reversion pair trade, at these settings, has had no edge in these five assets since 2016, and the sensitivity grid shows no setting that changes that."))

    # ============================================================ 9 replacement ==
    H.append("<h2>9. Who replaces a SELL</h2>")
    H.append(_ask("When an asset's supertrend flips to SELL, its money has to go somewhere. Which asset should take its place — the one that is correlated with it and has deviated, as proposed? — and how did each choice do?",
                  f"An <b>event</b> is a week an asset's state goes BUY → SELL (the <b>seller</b>). {n_ev} since {C.START}: " + ", ".join(f"{a} {flips_per[a]}" for a in C.ASSETS) + f". The <b>candidates</b> are the other four assets{' and cash (' + C.CASH + ')' if cash_ok else ''}. For each event and candidate, measured <i>at</i> the flip week: the {C.RP_CORR_WINDOW}-week correlation with the seller (<b>correlated</b> if ≥ {C.RP_CORR_MIN}); the deviation z of the candidate against the seller (<b>lagging</b> ≤ {-C.RP_Z_DEV:.0f}, <b>leading</b> ≥ {C.RP_Z_DEV:+.0f}, neutral between); the candidate's <b>own signal state</b>. Then, <i>after</i> the flip: the candidate's return over the next {H4} and {H13} weeks, the seller's own return (staying put), cash and gold over the same weeks, and the candidate's return if held until its <i>own</i> signal turned SELL (capped at {C.RP_MAX_HOLD} weeks). Twelve replacement <b>rules</b> are then applied to the same {n_ev} events, one pick per event, falling back to cash when a rule finds nobody.",
                  "In the cells table a row is a set of (event, candidate) pairs: <code>cand_mean</code> the candidate's average return, <code>stay_mean</code> the seller's, <code>cash_mean</code> cash, and the share of rows where the candidate beat each. In the rules table a row is one rule applied to every flip: <code>share_with_a_pick</code> is how often it found someone (the rest of the time it sat in cash), <code>when_pick_…</code> the result on those events only, <code>beat_cash</code> the share of flips it beat cash, <code>hac_t_minus_cash</code> the t of its edge over cash. <code>hold_mean</code> / <code>hold_weeks_mean</code> are the 'hold until the pick's own SELL' variant."))
    H.append("<h3>9.1 The events</h3>")
    H.append(_tbl(ev.assign(date=lambda d: d["date"].dt.date).pivot_table(index="seller", values="date", aggfunc=lambda s: ", ".join(str(x) for x in s)).reindex([a for a in C.ASSETS if a in ev["seller"].values]), fmt="{}"))
    H.append("<h3>9.2 By the candidate's deviation, its correlation with the seller, and its own signal</h3>")
    H.append(_img("8_cells"))
    show = c13[(c13["deviation"].isin(["lagging", "neutral", "leading", "any"]))][["deviation", "correlated", "cand_state", "rows", "events", "cand_mean", "cand_median", "cand_hit_positive", "stay_mean", "cash_mean", "cand_minus_stay", "cand_minus_cash", "share_cand_beat_stay", "share_cand_beat_cash"]]
    H.append(_tbl(show, fmt="{:.2f}", index=False, int_cols=("rows", "events"), pct_cols=("cand_mean", "cand_median", "stay_mean", "cash_mean", "cand_minus_stay", "cand_minus_cash")))
    H.append(_says(f"Over the next {H13} weeks (all {n_ev} flips pooled): a candidate whose own signal was <b>BUY</b> returned {_pct(c_buy['cand_mean'])} (beat cash {_p(c_buy['share_cand_beat_cash'])} of the time, n {int(c_buy['rows'])}); one in <b>SELL</b> {_pct(c_sellst['cand_mean'])} (beat cash {_p(c_sellst['share_cand_beat_cash'])}, n {int(c_sellst['rows'])}); the seller itself {_pct(c_all['stay_mean'])}; cash {_pct(c_all['cash_mean'])}. "
                   f"By deviation: a <b>lagging</b> candidate {_pct(cell('lagging', 'any', 'any')['cand_mean'])} (n {int(cell('lagging', 'any', 'any')['rows'])}), <b>neutral</b> {_pct(cell('neutral', 'any', 'any')['cand_mean'])}, <b>leading</b> {_pct(cell('leading', 'any', 'any')['cand_mean'])} (n {int(cell('leading', 'any', 'any')['rows'])}). "
                   f"The proposed cell — <b>correlated and lagging</b> — returned {_pct(c_lag_corr['cand_mean'])} (beat cash {_p(c_lag_corr['share_cand_beat_cash'])}, n {int(c_lag_corr['rows'])}), and every one of those candidates was itself in SELL ({int(c_lag_sell['rows'])} of {int(cell('lagging', 'any', 'any')['rows'])} lagging candidates were in SELL; the single lagging candidate in BUY is not a sample). Correlated and leading: {_pct(c_lead_corr['cand_mean'])} (n {int(c_lead_corr['rows'])}) — split by its own signal, {_pct(cell('leading', 'True', 'BUY')['cand_mean'])} in BUY (n {int(cell('leading', 'True', 'BUY')['rows'])}) against {_pct(cell('leading', 'True', 'SELL')['cand_mean'])} in SELL (n {int(cell('leading', 'True', 'SELL')['rows'])}). The <b>uncorrelated</b> candidate — in practice gold, or an index when gold is the seller — returned {_pct(c_lag_unc['cand_mean'])} / {_pct(c_neu_unc['cand_mean'])} / {_pct(c_lead_unc['cand_mean'])} (lagging / neutral / leading) and beat cash in {_p(cell('any', 'False', 'any')['share_cand_beat_cash'])} of rows. <b>The candidate's own signal is the variable that matters; its deviation from the seller matters only through it</b> (a correlated asset that has lagged a seller is almost always in SELL itself)."))
    H.append("<h3>9.3 The rules, head to head</h3>")
    H.append(_img("8_rules"))
    H.append(_tbl(R[["what it does", "events", "share_with_a_pick", f"mean_{H4}", f"beat_cash_{H4}", f"mean_{H13}", f"median_{H13}", f"hit_positive_{H13}", f"beat_cash_{H13}", f"beat_stay_{H13}", f"minus_cash_{H13}", f"hac_t_minus_cash_{H13}", f"worst_{H13}", f"when_pick_mean_{H13}", f"when_pick_beat_cash_{H13}", "hold_mean", "hold_weeks_mean"]], fmt="{:.2f}", int_cols=("events",), pct_cols=(f"mean_{H4}", f"mean_{H13}", f"median_{H13}", f"minus_cash_{H13}", f"worst_{H13}", f"when_pick_mean_{H13}", "hold_mean")))
    rk = rules_rank[[f"mean_{H13}"]].copy()
    H.append(_says(f"Ranked by the {H13}-week return over all {n_ev} flips: " + "; ".join(f"<b>{r}</b> {_pct(rules_rank.loc[r, f'mean_{H13}'])}" for r in rules_rank.index[:5]) + f" … <b>stay in the seller</b> {_pct(stay13)}, <b>cash</b> {_pct(cash13)}. "
                   f"<b>The proposed rule</b> (correlated & lagging) found a candidate in {int(lag_rule['events_with_a_pick'])} of {n_ev} flips and on those returned {_pct(lag_rule[f'when_pick_mean_{H13}'])}, beating cash {_p(lag_rule[f'when_pick_beat_cash_{H13}'])} of the time; adding 'and its own signal is BUY' leaves <b>{int(lagbuy_rule['events_with_a_pick'])} events in ten years</b>. "
                   f"<b>Most leading (any correlation)</b> — the asset that has pulled furthest ahead of the seller — had a pick in {_p(lead_rule['share_with_a_pick'])} of flips, returned {_pct(lead_rule[f'mean_{H13}'])} ({_pct(lead_rule[f'minus_cash_{H13}'])} over cash, t {_t(lead_rule[f'hac_t_minus_cash_{H13}'])}, beat cash {_p(lead_rule[f'beat_cash_{H13}'])}, beat staying {_p(lead_rule[f'beat_stay_{H13}'])}), and held until its own SELL made {_pct(lead_rule['hold_mean'])} over {lead_rule['hold_weeks_mean']:.0f} weeks on average. "
                   f"<b>Most correlated, in BUY</b> returned {_pct(corrbuy[f'mean_{H13}'])} (t {_t(corrbuy[f'hac_t_minus_cash_{H13}'])}; {_pct(corrbuy[f'when_pick_mean_{H13}'])} on the {int(corrbuy['events_with_a_pick'])} flips where some correlated asset was still in BUY). <b>Gold</b> returned {_pct(gold_rule[f'mean_{H13}'])} (t {_t(gold_rule[f'hac_t_minus_cash_{H13}'])}; beat cash {_p(gold_rule[f'beat_cash_{H13}'])}). <b>Best 13-week momentum in BUY</b> {_pct(mom[f'mean_{H13}'])} (t {_t(mom[f'hac_t_minus_cash_{H13}'])}). "
                   f"On the other side, the <b>most correlated asset regardless of its signal</b> returned {_pct(mc_rule[f'mean_{H13}'])} and the <b>most lagging</b> {_pct(mlag[f'mean_{H13}'])} — both below cash, because they are usually in SELL too. Over {H4} weeks the ordering is the same ({order4_str})."))
    H.append("<h3>9.4 By seller: which replacement worked for whose SELL</h3>")
    bpp = bp.pivot(index="seller", columns="candidate", values=f"cand_mean_{H13}").reindex([a for a in C.ASSETS if a in bp["seller"].values])
    bpp["stay"] = bp.groupby("seller")[f"stay_mean_{H13}"].first(); bpp["cash"] = bp.groupby("seller")[f"cash_mean_{H13}"].first(); bpp["flips"] = bp.groupby("seller")["events"].max()
    gold_seller_str = ", ".join(f"{c} {_pct(bpp.loc['GoldMCX', c])}" for c in C.EQUITY if "GoldMCX" in bpp.index and c in bpp.columns and pd.notna(bpp.loc["GoldMCX", c]))
    H.append(f"<p class='small'>Average return over the next {H13} weeks of each candidate (columns) after each seller's SELL flips (rows):</p>")
    H.append(_tbl(bpp, fmt="{:.3f}", int_cols=("flips",), pct_cols=tuple(c for c in bpp.columns if c != "flips")))
    H.append(_tbl(bp[["seller", "candidate", "events", "avg_corr_52w", "avg_z_cand_vs_seller", "share_lagging", "share_leading", "share_cand_in_BUY", f"cand_mean_{H4}", f"cand_mean_{H13}", f"stay_mean_{H13}", f"cash_mean_{H13}", f"share_cand_beat_cash_{H13}", "hold_mean", "hold_weeks_mean"]], fmt="{:.2f}", index=False, int_cols=("events",), pct_cols=(f"cand_mean_{H4}", f"cand_mean_{H13}", f"stay_mean_{H13}", f"cash_mean_{H13}", "hold_mean")))
    rs_show = Rs[Rs["rule"].isin(["most leading (any corr)", "most correlated, in BUY", "gold", "best 13w momentum, in BUY", "correlated & lagging", "stay in the seller", "cash"])].pivot(index="rule", columns="seller", values=f"mean_{H13}").reindex(["most leading (any corr)", "most correlated, in BUY", "gold", "best 13w momentum, in BUY", "correlated & lagging", "stay in the seller", "cash"])
    H.append(f"<p class='small'>The main rules by seller, {H13}-week return:</p>")
    H.append(_tbl(rs_show, fmt="{:.3f}", pct_cols=tuple(rs_show.columns)))
    best_by_seller = {a: bpp.loc[a, [c for c in bpp.columns if c not in ('stay', 'cash', 'flips')]].astype(float).idxmax() for a in bpp.index}
    H.append(_says("Best single replacement by seller: " + "; ".join(f"after <b>{a}</b> → {best_by_seller[a]} ({_pct(bpp.loc[a, best_by_seller[a]])} vs staying {_pct(bpp.loc[a, 'stay'])}, {int(bpp.loc[a, 'flips'])} flips)" for a in bpp.index) + f". When an <i>index</i> sells, the other indices are poor replacements (they are falling with it) and gold is the one that has worked; when <i>gold</i> sells, the indices have been fine replacements ({gold_seller_str}) because gold's SELL says nothing about them."))
    H.append("<h3>9.5 How the deviation z-score looked around the flips</h3>")
    H.append(_img("8_z_path"))
    zp = _csv("8_z_path"); zpa = zp[zp["candidate"] == "all candidates"].pivot(index="seller", columns="k", values="mean_z")
    H.append(_tbl(zpa[[k for k in (-13, -8, -4, -2, 0, 2, 4, 8, 13) if k in zpa.columns]], fmt="{:+.2f}"))
    H.append(_says("Averaged over all flips and all candidates (z of the candidate against the seller, + = candidate ahead): " + "; ".join(f"after <b>{a}</b> sells, the others go from {zpa.loc[a, -4]:+.2f} four weeks before to {zpa.loc[a, 0]:+.2f} at the flip and {zpa.loc[a, 13]:+.2f} thirteen weeks after" for a in zpa.index) + ". A path that keeps rising after week 0 means the seller kept falling behind the others after its signal fired — the deviation that exists at the flip is, on average, not the end of the move."))
    for a in [x for x in C.ASSETS if x in set(ev["seller"])]:
        H.append(f"<h4>Every SELL flip of {a}</h4>")
        H.append(_img("8_events", seller=a))
    H.append("<h3>9.6 Right now</h3>")
    H.append(_tbl(cur8, fmt="{:.2f}", index=False))
    H.append(_says((f"In SELL as of {cur8['date'].iloc[0]}: {', '.join(nz)}. " if nz else f"As of {cur8['date'].iloc[0]} every asset is in BUY, so no replacement is called for. ") + "The table shows, for each asset <i>if</i> it flipped this week, each candidate's correlation, deviation and own state — the inputs the rules of §9.3 would use."))
    H.append(_not(f"{n_ev} flips, of which Micro250 contributes {flips_per['Micro250']} and the others {min(flips_per[a] for a in C.ASSETS if a != 'Micro250')}–{max(flips_per[a] for a in C.ASSETS if a != 'Micro250')} each; several flips fall in the same weeks (2020-03 above all), so the rules' t-stats, though Newey-West corrected, rest on fewer independent events than {n_ev}. 'Gold' as a replacement carries gold's own 2016–2026 run; 'most leading' picked gold in many events, which is part of why they agree. The result is a ranking of simple rules on one history, not a strategy — the next step is to pre-register 'replace a SELL with the most-leading BUY-state asset' and test it inside the rotation framework with its real costs and cadence."))

    # ============================================================ 10 ==
    H.append("<h2>10. Reading it all together</h2><ul>")
    H.append(f"<li><b>SELL weeks are volatility weeks.</b> Correlation rises in them ({eqp['corr_buy'].mean():.2f} → {eqp['corr_sell'].mean():.2f}) and volatility rises more ({vs.loc[C.EQUITY, 'vol_ratio_SELL_over_BUY'].mean():.1f}×); {eqp['share_of_cov_change_from_vol'].mean() * 100:.0f}% of the extra covariance is volatility. Correlation and volatility move together (§3.4), both after the flip, not before (§2.2, §3.2). Neither is a warning.</li>")
    H.append(f"<li><b>Deviations in these pairs are trends, not spreads.</b> {10 - len(coint_ok)} of 10 ratios have no long-run level; a 1–2 sd deviation has continued (§7.1); the mean-reversion pair trade lost money in {n_pairs_neg} of 10 pairs (§8). The classic pair-trading strategy is not there to be found in these five assets.</li>")
    H.append(f"<li><b>Which asset replaces a SELL: the one whose own signal is BUY, and if it has deviated, the one that has deviated upward.</b> 'Correlated and lagging' is in practice 'also in SELL' and did no better than cash; 'most leading' ({_pct(lead_rule[f'mean_{H13}'])}), 'most correlated still in BUY' ({_pct(corrbuy[f'mean_{H13}'])}) and gold ({_pct(gold_rule[f'mean_{H13}'])}) beat cash ({_pct(cash13)}) and staying put ({_pct(stay13)}) on {n_ev} flips (§9). This is the same fact as §5.7's 'the breaker keeps going for a while' seen from the book's side.</li>")
    H.append(f"<li><b>Who breaks first</b> depends on the question: the correlation by date, {date_first}; by collapse, {drop_first}; the price, {first_cross}; the signal, {first_sig}. The direction is the robust part — smaller caps leave the group upward, Nifty downward.</li>")
    H.append(f"<li><b>Samples are small where it matters.</b> {len(cep)} correlation episodes, {len(pep)} sell-offs, {n_ev} SELL flips, {int(ptall['trades'])} pair trades. Ranks, medians and shares are reported where t-stats would mislead; t-stats where there are enough weeks, always Newey-West for overlapping windows.</li>")
    H.append("<li>Weekly closes, price indices (no dividends), log returns; Micro250 from 2022 only; the VIX file ends in 2025-09 and is used for one context line.</li></ul>")
    H.append(f"<p class='small'>Tables in <code>outputs/tables/</code> — prefix 1_ for §2, 5_ for §3, 2_ for §4, 3_ for §5, 4_ for §6, 6_ for §7, 7_ for §8, 8_ for §9 — interactive charts built by <code>figs.py</code> (static PNG copies in <code>outputs/charts/</code> with <code>python plots.py</code>), parameters in <code>config.py</code>, every term and column in <code>GLOSSARY.md</code>. Built in {meta.get('elapsed_s', '?')} s.</p></body></html>")
    return "\n".join(H)


def main() -> int:
    out = build()
    C.REPORT_FULL_HTML.write_text(out, encoding="utf-8")
    print(f"wrote {C.REPORT_FULL_HTML.relative_to(C.ROOT)} ({len(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
