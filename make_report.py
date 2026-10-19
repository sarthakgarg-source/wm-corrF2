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

    now_pairs_short = "; ".join(f"{p.replace('|', '×')} z {cur6.loc[p, 'z_now']:+.1f}" for p in ("N100|SC100", "N100|Micro250", "N100|MC100") if p in cur6.index)
    ex_d = ev[ev["seller"] == "N100"]["date"]
    ex_d = ex_d[ex_d.dt.year == 2020].iloc[0] if (ex_d.dt.year == 2020).any() else ex_d.iloc[0]
    pxw = _csv("px_weekly", index_col=0, parse_dates=True)
    ii = pxw.index.get_loc(ex_d)
    ex_c = pst.loc["N100|SC100", "corr_full"]
    exv = vs.loc["N100", "vol_all"]
    piv_af = afa[afa["event"] == "flip to SELL"].pivot(index="ref", columns="weeks_from_flip", values="mean_corr").reindex(refs_all)
    piv_af["n_flips"] = afa[afa["event"] == "flip to SELL"].groupby("ref")["n_events"].max().reindex(refs_all)
    piv_ab = afa[afa["event"] == "flip to BUY"].pivot(index="ref", columns="weeks_from_flip", values="mean_corr").reindex(refs_all)
    bf = af[af["event"] == "flip to BUY"].set_index("weeks_from_flip")["mean_corr"]
    eq_sell_str = "; ".join(f"{a} {_af_row(a, 'flip to SELL', -4):.2f} → {_af_row(a, 'flip to SELL', 0):.2f} → {_af_row(a, 'flip to SELL', 8):.2f}" for a in refs_all if a in C.EQUITY)
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
    cvi = cv.loc[idx_pairs]; cvg = cv.loc[gold_pairs]
    lv = vl["corr_dvol_t_dcorr_t_plus_k"]
    ex = cep.iloc[0]; exb = ab[ab["episode"] == ex["episode"]].set_index("asset")
    show = c13[(c13["deviation"].isin(["lagging", "neutral", "leading", "any"]))][["deviation", "correlated", "cand_state", "rows", "events", "cand_mean", "cand_median", "cand_hit_positive", "stay_mean", "cash_mean", "cand_minus_stay", "cand_minus_cash", "share_cand_beat_stay", "share_cand_beat_cash"]]
    bpp = bp.pivot(index="seller", columns="candidate", values=f"cand_mean_{H13}").reindex([a for a in C.ASSETS if a in bp["seller"].values])
    bpp["stay"] = bp.groupby("seller")[f"stay_mean_{H13}"].first(); bpp["cash"] = bp.groupby("seller")[f"cash_mean_{H13}"].first(); bpp["flips"] = bp.groupby("seller")["events"].max()
    gold_seller_str = ", ".join(f"{c} {_pct(bpp.loc['GoldMCX', c])}" for c in C.EQUITY if "GoldMCX" in bpp.index and c in bpp.columns and pd.notna(bpp.loc["GoldMCX", c]))
    best_by_seller = {a: bpp.loc[a, [c for c in bpp.columns if c not in ('stay', 'cash', 'flips')]].astype(float).idxmax() for a in bpp.index}
    rs_show = Rs[Rs["rule"].isin(["most leading (any corr)", "most correlated, in BUY", "gold", "best 13w momentum, in BUY", "correlated & lagging", "stay in the seller", "cash"])].pivot(index="rule", columns="seller", values=f"mean_{H13}").reindex(["most leading (any corr)", "most correlated, in BUY", "gold", "best 13w momentum, in BUY", "correlated & lagging", "stay in the seller", "cash"])
    zp = _csv("8_z_path"); zpa = zp[zp["candidate"] == "all candidates"].pivot(index="seller", columns="k", values="mean_z")
    css = """
    body{font-family:Inter,Helvetica,Arial,sans-serif;max-width:1320px;margin:0 auto;padding:24px 20px;color:#0b0b0b;background:#fcfcfb;line-height:1.55}
    h1{font-size:24px;margin:0 0 4px} h2{font-size:20px;margin:48px 0 10px;border-bottom:2px solid #e1e0d9;padding-bottom:4px} h3{font-size:15.5px;margin:26px 0 6px} h4{font-size:13.5px;margin:16px 0 4px;color:#52514e}
    .sub{color:#52514e;margin-bottom:18px} .fig{margin:12px 0;border:1px solid #e1e0d9;border-radius:4px;padding:4px 4px 2px;background:#fcfcfb} .fignote{font-size:12px;color:#52514e;margin:2px 6px 8px}
    .t{border-collapse:collapse;font-size:12.5px;margin:8px 0;display:block;overflow-x:auto;max-width:100%} .t th,.t td{padding:3px 9px;text-align:right;border-bottom:1px solid #eeede8} .t th{color:#52514e;font-weight:600} .t td:first-child,.t th:first-child{text-align:left}
    .box{background:#f3f3ef;border-left:4px solid #2a78d6;padding:10px 14px;margin:14px 0} .warn{background:#fdecea;border-left:4px solid #e34948;padding:10px 14px;margin:14px 0;font-weight:600}
    .lead{font-size:14px;color:#2b2b28;margin:6px 0 10px} .lead b{color:#0b0b0b} .read{font-size:12.8px;color:#52514e;margin:4px 0 8px}
    .says{background:#eef5fd;border-left:4px solid #2a78d6;padding:9px 14px;margin:12px 0;font-size:13.8px} .note{font-size:12.6px;color:#52514e;margin:6px 0 12px;padding-left:14px;border-left:3px solid #eda100}
    details{margin:10px 0 16px;border:1px dashed #c3c2b7;border-radius:4px;padding:4px 12px;background:#fbfbf7} summary{cursor:pointer;font-size:13px;color:#52514e;font-weight:600;padding:4px 0} details[open] summary{margin-bottom:6px}
    .card{border:1px solid #e1e0d9;border-radius:6px;padding:10px 16px;margin:22px 0;background:#fdfdfb} .kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:5px 18px;font-size:12.8px;margin:8px 0} .kv div{border-bottom:1px dotted #e1e0d9;padding:2px 0} .kv b{color:#52514e;font-weight:600}
    .q{font-size:14.5px;margin:8px 0 4px} .ans{font-size:15px;margin:4px 0 12px;padding:10px 14px;background:#eef5fd;border-left:4px solid #2a78d6} .walk{font-size:13.6px;margin:10px 0;padding:9px 14px;background:#f0faf5;border-left:4px solid #1baf7a}
    .tiles{display:flex;flex-wrap:wrap;gap:10px;margin:12px 0} .tile{flex:1 1 150px;border:1px solid #e1e0d9;border-radius:6px;padding:10px 12px;background:#fdfdfb} .tile .big{font-size:24px;font-weight:700;color:#0b0b0b;line-height:1.1} .tile .cap{font-size:12px;color:#52514e;margin-top:4px}
    .book{font-size:13.8px;margin:10px 0 14px;padding:8px 14px;border-left:4px solid #eda100;background:#fbf6ea} abbr{text-decoration:underline dotted #898781;cursor:help}
    .toc{columns:2;font-size:13px;list-style:none;padding-left:4px} .toc li{margin:1px 0} .blocks{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:10px 22px;margin:10px 0} .blocks div{font-size:13.3px;border-top:2px solid #e1e0d9;padding-top:6px} .blocks b{display:block;margin-bottom:2px}
    .missing{color:#e34948} .small{font-size:12.5px;color:#52514e} code{background:#f0efec;padding:1px 4px;border-radius:3px;font-size:12.5px} ul{margin:6px 0} li{margin:3px 0} ol li{margin:6px 0}
    """
    JS = """
    document.querySelectorAll('details').forEach(d => d.addEventListener('toggle', () => { if (d.open) d.querySelectorAll('.js-plotly-plot').forEach(p => { try { Plotly.Plots.resize(p); } catch (e) {} }); }));
    """

    def _lead(txt):
        return f"<p class='lead'>{txt}</p>"

    def _read(txt):
        return f"<p class='read'><b>How to read it.</b> {txt}</p>"

    def _note(txt):
        return f"<p class='note'>{txt}</p>"

    def _details(summary, inner):
        return f"<details><summary>{summary}</summary>{inner}</details>"


    H = [f"<!doctype html><html><head><meta charset='utf-8'><title>wm-BREAK · Study I</title><style>{css}</style><script>{PLOTLY_JS.read_text(encoding='utf-8')}</script></head><body>"]
    H.append("<h1>Study I — correlation: between the asset classes, with respect to the signal, with volatility, and with returns</h1>")
    H.append(f"<div class='sub'>wm-BREAK · weekly, {C.START} → {meta['last_session']} · N100 · MC100 · SC100 · Micro250 · GoldMCX{' · cash = ' + C.CASH if cash_ok else ''} · built {meta['run_at']}</div>")
    if meta.get("synthetic"):
        H.append("<div class='warn'>SYNTHETIC DEMO DATA — nothing below is a market result.</div>")
    H.append("<div class='box'><b>Part I</b> is the four things this study is about, one chapter each: <b>2</b> correlation between the asset classes · <b>3</b> correlation with respect to the signal · <b>4</b> correlation and volatility · <b>5</b> the historical pattern between correlation and returns. Each chapter: the <i>question</i> → the <i>answer</i> in one line → <i>how we know</i> (the chart, and what to look at on it) → the key numbers → <i>what it means for the book</i>. Hover any dotted-underlined word for its meaning. <b>Part II</b> (the further studies — who breaks first, each pair, the pair-trading rule, who replaces a SELL) is folded away at the end; each box shows its one-line finding and opens on a click. Charts are interactive: hover for values, click a legend entry to hide or show it, double-click to isolate it. Every number comes from <code>outputs/tables/</code>; every term is in <code>GLOSSARY.md</code>; <code>outputs/report_full.html</code> has everything expanded.</div>")
    H.append("<ul class='toc'><li>0. The answers</li><li>1. One sell-off, start to finish (the measures, on a week you remember)</li><li>2. Correlation between the asset classes</li><li>3. Correlation with respect to the signal</li><li>4. Correlation and volatility</li><li>5. The historical pattern: correlation and returns</li><li>Part II · 6. Who breaks the correlation first</li><li>Part II · 7. Who breaks in price and signal</li><li>Part II · 8. Each pair</li><li>Part II · 9. The pair-trading rule</li><li>Part II · 10. Who replaces a SELL</li><li>11. In one page</li></ul>")

    # ============================================================ shared ==
    M = _csv("0_corr_matrix_full", index_col=0); M52 = _csv("0_corr_matrix_52w", index_col=0); M13 = _csv("0_corr_matrix_13w", index_col=0)
    idx_names = [a for a in C.EQUITY if a in M.index]
    offdiag = M.loc[idx_names, idx_names].where(~np.eye(len(idx_names), dtype=bool))
    gold_row = M.loc["GoldMCX", idx_names] if "GoldMCX" in M.index else pd.Series(dtype=float)
    vix_row = M.loc[C.VIXCHG, idx_names] if C.VIXCHG in M.index else pd.Series(dtype=float)
    pair_min = offdiag.stack().idxmin(); pair_max = offdiag.stack().idxmax()
    pair_order = [p for p in pst.index if p in set(bs["pair"])]
    bs3 = bs[bs["state"].isin(["both BUY", "disagree", "both SELL", "all weeks"])].pivot(index="pair", columns="state", values="corr").reindex(pair_order)
    bsw = bs[bs["state"].isin(["both BUY", "disagree", "both SELL"])].pivot(index="pair", columns="state", values="weeks").reindex(pair_order)
    eq_rows = [p for p in bs3.index if "Gold" not in p]; gd_rows = [p for p in bs3.index if "Gold" in p]
    per = _csv("5_corr_rise_events"); pa = _csv("5_corr_rise_paths"); pav = pa.pivot(index="k", columns="series", values="mean_z"); n_jumps = int(len(per))
    cvi = cv.loc[idx_pairs]; cvg = cv.loc[gold_pairs]; lv = vl["corr_dvol_t_dcorr_t_plus_k"]
    TERMS = {
        "correlation": "a number from −1 to +1: how much two assets move in the same direction week to week. +1 always together, 0 no relation, −1 always opposite.",
        "13-week": "computed on the last 13 weeks (one quarter) only, and recomputed every week — the 'current' value.",
        "52-week": "computed on the last 52 weeks (one year) only.",
        "volatility": "how big an asset's weekly moves typically are, scaled to a year: 20% means a normal year sees swings of about ±20%.",
        "covariance": "correlation × swing size of a × swing size of b: the risk a pair contributes to a book. Same correlation with double the swings = four times the risk.",
        "VIX": "India VIX: what Nifty 50 option prices expect volatility to be over the next 30 days, annualised. 15 ≈ ±4% a month expected.",
        "supertrend": "the desk's weekly signal: a band 2 average-ranges around the mid-price; BUY above it, SELL below it; bands only tighten in the trend's direction.",
        "flip": "the week the signal changes state. A SELL flip comes after a sizeable fall, never at the top.",
        "z-score": "how unusual a value is for that series: (value − its average over the last two years) ÷ its usual spread. 0 = normal, +1 = unusually high, +2 = rare.",
        "quintile": "all weeks sorted by a measure and cut into five equal groups: Q1 the lowest fifth … Q5 the highest.",
        "annualised": "scaled to a yearly figure (returns × 52, volatility × √52).",
        "t": "the average divided by its own uncertainty, with a correction for overlapping windows (Newey-West). Beyond about ±2 the average is unlikely to be luck.",
        "decorrelation episode": "a stretch of weeks in which the correlation among the indices sat at least one standard deviation below its own two-year norm.",
        "group correlation": "the average of the six pairwise 13-week correlations among N100, MC100, SC100 and Micro250 (whichever exist that week): one number for 'how much is the market moving as one block'.",
        "deviation": "the z-score of a pair's price ratio against its own last year: z > 0 = the first asset has run ahead of the second, z < 0 = it has lagged.",
    }

    def _d(term, text=None):
        return '<abbr title="' + TERMS[term] + '">' + (text or term) + '</abbr>'

    def _tiles(items):
        return "<div class='tiles'>" + "".join(f"<div class='tile'><div class='big'>{v}</div><div class='cap'>{c}</div></div>" for v, c in items) + "</div>"

    def _q(txt):
        return f"<p class='q'><b>Question.</b> {txt}</p>"

    def _ans(txt):
        return f"<p class='ans'><b>Answer.</b> {txt}</p>"

    def _walk(txt):
        return f"<div class='walk'><b>How we know — look at the chart:</b> {txt}</div>"

    def _book(txt):
        return f"<p class='book'><b>What it means for the book.</b> {txt}</p>"

    # ============================================================ 0 ==
    H.append("<h2>0. The answers</h2><div class='box'><ol>")
    H.append(f"<li><b>The four indices move as one block; gold does not.</b> Whole-sample correlation {offdiag.stack().min():.2f}–{offdiag.stack().max():.2f} between the indices; gold {gold_row.min():+.2f} to {gold_row.max():+.2f} with all of them. <span class='small'>§2</span></li>")
    H.append(f"<li><b>A SELL signal does not make the indices move together much more — it makes them move much more.</b> Correlation {bs3.loc[eq_rows, 'both BUY'].mean():.2f} (both BUY) → {bs3.loc[eq_rows, 'disagree'].mean():.2f} (disagree) → {bs3.loc[eq_rows, 'both SELL'].mean():.2f} (both SELL); swings {vs.loc[C.EQUITY, 'vol_ratio_SELL_over_BUY'].mean():.1f}× bigger; {eqp['share_of_cov_change_from_vol'].mean() * 100:.0f}% of the extra risk is the swings. <span class='small'>§3, §4</span></li>")
    H.append(f"<li><b>Correlation rises after the flip, not before — and the rise is Nifty re-joining the block.</b> Around every asset's SELL flip the group correlation is flat in the weeks before and climbs about 0.1 in the two months after; pair by pair, the climb is in the N100 pairs, while the pairs among the smaller indices were already coupled. A consequence of the sell-off, not a warning. <span class='small'>§3</span></li>")
    H.append(f"<li><b>The wilder the market, the more the indices move together; gold never joins.</b> N100×SC100 {cvi.loc['N100|SC100', qcols[0]]:.2f} in the calmest fifth of weeks → {cvi.loc['N100|SC100', qcols[-1]]:.2f} in the wildest; gold's pairs stay within {cvg[qcols].min().min():+.2f} … {cvg[qcols].max().max():+.2f}. Volatility and correlation spike in the same week. <span class='small'>§4</span></li>")
    H.append(f"<li><b>High correlation is a sell-off in progress, and the quarter after it has been the best.</b> Most-correlated fifth of weeks: N100 {_pct(n100.loc['Q5', 'same_quarter_mean'])} over that quarter, then {_pct(n100.loc['Q5', 'next_quarter_mean'])} the next (positive {n100.loc['Q5', 'next_quarter_hit'] * 100:.0f}% of the time). A tendency, not a timing rule. <span class='small'>§5</span></li>")
    H.append(f"<li><b>Part II in one line each.</b> The smaller index breaks the correlation first by running ahead, Nifty is left behind (§6); in price {first_cross} falls first, in signal {first_sig} fires first (§7); deviations between the indices are trends, not spreads — no pair trade (§8, §9); replace a SELL with an asset whose own signal is BUY — the one that pulled ahead, the most correlated still in BUY, or gold — never the correlated laggard (§10).</li>")
    H.append(f"<li><b>Right now</b> ({cur8['date'].iloc[0]}): " + (f"in SELL: {', '.join(nz)}. " if nz else "every asset is in BUY. ") + f"The market is in a {_d('decorrelation episode')} (group correlation {last['G']:.2f}, z {last['z_G']:+.1f}); SC100 and Micro250 are about 2 sd ahead of N100.</li>")
    H.append("</ol></div>")

    # ============================================================ 1 anchor ==
    H.append("<h2>1. One sell-off, start to finish — the measures this report uses, on a week you remember</h2>")
    ev_n = ev[(ev["seller"] == "N100") & (ev["date"].dt.year == 2020)]
    d0 = ev_n["date"].iloc[0] if len(ev_n) else ev[ev["seller"] == "N100"]["date"].iloc[0]
    pxw = _csv("px_weekly", index_col=0, parse_dates=True); rwk = _csv("ret_weekly", index_col=0, parse_dates=True); Vr = _csv("5_vol_rolling", index_col=0, parse_dates=True); tg = _csv("5_together", index_col=0, parse_dates=True)
    i0 = pxw.index.get_loc(d0); j0 = rwk.index.get_loc(d0)
    stw = _csv("sig_states_weekly", index_col=0, parse_dates=True)
    s_after = stw["N100"].loc[d0:]; buy_d = s_after[s_after == 1].index[0] if (s_after == 1).any() else None
    same_week = [a for a in C.EQUITY if a in ev["seller"].values and (ev[(ev["seller"] == a) & (ev["date"] == d0)].shape[0] > 0)]
    rows = []
    for k, lab in ((-8, "8 weeks before"), (-4, "4 weeks before"), (0, "the flip week"), (4, "4 weeks after"), (8, "8 weeks after"), (13, "13 weeks after")):
        dt = pxw.index[i0 + k]
        rows.append({"week": lab, "date": dt.date(), "N100 close": pxw["N100"].iloc[i0 + k], "N100 since the flip": pxw["N100"].iloc[i0 + k] / pxw["N100"].iloc[i0] - 1,
                     "correlation among the indices": S["G"].iloc[S.index.get_loc(dt)] if dt in S.index else np.nan, "N100 volatility (13w)": Vr["N100"].iloc[Vr.index.get_loc(dt)] if dt in Vr.index else np.nan,
                     "India VIX": tg["vix"].iloc[tg.index.get_loc(dt)] if ("vix" in tg.columns and dt in tg.index) else np.nan, "gold since the flip": pxw["GoldMCX"].iloc[i0 + k] / pxw["GoldMCX"].iloc[i0] - 1})
    story = pd.DataFrame(rows).set_index("week")      # 'since the flip' = the close that week ÷ the flip-week close − 1: positive before the flip (the index was higher), negative after
    g4, g0, g4a, g8 = story.loc["4 weeks before", "correlation among the indices"], story.loc["the flip week", "correlation among the indices"], story.loc["4 weeks after", "correlation among the indices"], story.loc["8 weeks after", "correlation among the indices"]
    v4, v0, v4a, v8 = story.loc["4 weeks before", "N100 volatility (13w)"], story.loc["the flip week", "N100 volatility (13w)"], story.loc["4 weeks after", "N100 volatility (13w)"], story.loc["8 weeks after", "N100 volatility (13w)"]
    x4, x0, x4a = story.loc["4 weeks before", "India VIX"], story.loc["the flip week", "India VIX"], story.loc["4 weeks after", "India VIX"]
    fall_wk = pxw["N100"].iloc[i0] / pxw["N100"].iloc[i0 - 1] - 1
    n4, n13, gld4, gld13 = story.loc["4 weeks after", "N100 since the flip"], story.loc["13 weeks after", "N100 since the flip"], story.loc["4 weeks after", "gold since the flip"], story.loc["13 weeks after", "gold since the flip"]
    sc13 = np.exp(rwk["SC100"].iloc[j0 + 1:j0 + 14].sum()) - 1
    H.append(f"<p>Every measure in this report can be seen in one episode. Take the week ending <b>{d0.date()}</b>: N100 fell {fall_wk:.1%} in the week and its {_d('supertrend')} {_d('flip', 'flipped')} to SELL" + (f" — so did {' and '.join(a for a in same_week if a != 'N100')}, the same week" if len(same_week) > 1 else "") + ". The table follows the market from eight weeks before that flip to thirteen weeks after:</p>")
    n8b = story.loc["8 weeks before", "N100 since the flip"]
    H.append(_tbl(story.rename(columns={"N100 since the flip": "N100 vs the flip-week close", "gold since the flip": "gold vs the flip-week close"}), fmt="{:.2f}", pct_cols=("N100 vs the flip-week close", "N100 volatility (13w)", "gold vs the flip-week close")))
    H.append(f"<p class='read'>'vs the flip-week close' = that week's close ÷ the flip week's close − 1: positive before the flip (N100 was {n8b:+.1%} above the flip level eight weeks earlier), 0 in the flip week, negative after it while the fall went on. Correlation = the {_d('group correlation', 'average 13-week correlation among the four indices')}; volatility = N100's {_d('volatility', '13-week realised volatility')}, {_d('annualised')}.</p>")
    H.append(f"<p><b>Read it row by row.</b> Four weeks before the flip the market was calm: the {_d('group correlation', 'correlation among the four indices')} was {g4:.2f} (loosely coupled), N100's {_d('volatility')} {_p(v4)} (small swings)" + (f", the {_d('VIX')} {x4:.1f}" if pd.notna(x4) else "") + f". <b>In the flip week all three jumped together</b> — correlation {g0:.2f}, volatility {_p(v0)}" + (f", VIX {x0:.1f}" if pd.notna(x0) else "") + f" — in the same week as the fall, none of them earlier. Four weeks later correlation was {g4a:.2f} (the indices were one block), volatility {_p(v4a)}" + (f", VIX {x4a:.1f}" if pd.notna(x4a) else "") + f"; N100 had fallen a further {n4:.1%} and gold had risen {gld4:+.1%}. Thirteen weeks after the flip N100 was {n13:+.1%} from the flip, SC100 {sc13:+.1%}, gold {gld13:+.1%}" + (f"; the signal turned back to BUY on {buy_d.date()}, {(buy_d - d0).days // 7} weeks after it fired, with correlation still {g8:.2f}." if buy_d is not None else ".") + "</p>")
    H.append(f"<p><b>What this one episode shows, and what the rest of the report checks across the whole history:</b></p><ul>"
             f"<li>The indices fell <i>together</i> — correlation went to {g4a:.2f}. Were they always that coupled, or only in sell-offs? → <b>§2</b> (between the asset classes) and <b>§5</b> (correlation vs returns).</li>"
             f"<li>Correlation jumped <i>in the week the signal fired</i>, not before. Is that every time? → <b>§3</b> (correlation with respect to the signal).</li>"
             f"<li>Volatility went from {_p(v4)} to {_p(v8)} — far more than correlation went from {g4:.2f} to {g8:.2f}. Which of the two is the risk in a SELL state? Do they always move together? → <b>§4</b> (correlation and volatility).</li>"
             f"<li>Gold rose {gld13:+.1%} while the indices fell. Is gold always the one that doesn't follow? → <b>§2</b>, <b>§4</b>, and Part II <b>§10</b> (what replaces a SELL).</li></ul>")
    H.append("<h3>1.1 The words used (hover any underlined word anywhere in the report for its meaning)</h3>")
    H.append("<div class='blocks'>")
    H.append(f"<div><b>Weekly returns</b> One row per trading week (Friday close), {len(rwk)} weeks from {C.START}. A week's return is ln(close ÷ last close) — the ordinary % for normal moves; returns over several weeks add up. Price indices, no dividends. Micro250 exists only from 2022.</div>")
    H.append(f"<div><b>The signal and a flip</b> The desk's weekly {_d('supertrend')}: BUY or SELL for each asset each week. A {_d('flip')} is the week it changes. SELL flips since 2016: " + ", ".join(f"{a} {flips_per[a]}" for a in C.ASSETS) + ".</div>")
    H.append(f"<div><b>Correlation</b> {TERMS['correlation']} Rough scale: above 0.90 one asset; 0.70–0.90 move together; near 0 unrelated. '{_d('13-week')}' is the current value; '{_d('52-week')}' the last year.</div>")
    H.append(f"<div><b>Volatility, and the VIX</b> {TERMS['volatility']} The {_d('VIX')} is the option market's forecast of it; realised volatility looks back, the VIX looks forward.</div>")
    H.append(f"<div><b>Covariance</b> {TERMS['covariance']} Whenever this report says <i>risk</i> it means this.</div>")
    H.append(f"<div><b>z-score</b> {TERMS['z-score']} One trap: a z-score always drifts back to 0 as the window catches up, so 'z came back' never means 'the price came back'.</div>")
    H.append(f"<div><b>'After', and the t</b> Anything measured after a week uses the following weeks only, never the decision week. The {_d('t')} says whether an average could be luck. Where there are only {len(cep)} episodes or {len(pep)} sell-offs, no t is given — ranks and medians only.</div>")
    H.append("</div>")

    # ============================================================ 2 asset classes ==
    H.append("<h2>2. Correlation between the asset classes</h2>")
    H.append(_q("How much do large caps, mid caps, small caps, micro caps and gold move together — which pairs are one bet, and which are separate bets?"))
    H.append(_ans(f"The four indices are one block ({offdiag.stack().min():.2f}–{offdiag.stack().max():.2f}); gold is a separate bet ({gold_row.min():+.2f} to {gold_row.max():+.2f} with every index)."))
    H.append(_img("0_corr_heatmap"))
    cov0 = _csv("0_corr_matrix_coverage", index_col=0)
    m13_eq = M13.loc[idx_names, idx_names].where(~np.eye(len(idx_names), dtype=bool)).stack()
    d_now = m13_eq.mean() - offdiag.stack().mean()
    now_vs = "tighter than usual (bluer)" if d_now > 0.02 else ("looser than usual (paler)" if d_now < -0.02 else "about as tight as usual")
    H.append(_walk(f"The index block (top-left 4×4) is solid blue in the whole-sample panel — every index pair is {offdiag.stack().min():.2f} or above, and the closer two indices are in size the bluer the cell ({pair_max[0]}×{pair_max[1]} {offdiag.stack().max():.2f}, {pair_min[0]}×{pair_min[1]} {offdiag.stack().min():.2f}). The gold row and column are grey in every panel: gold is unrelated to all four." + (f" The VIX row is red: a rising VIX is a falling market." if len(vix_row) else "") + f" The two right-hand panels are 'now' against 'usual': over the last 13 weeks the block is {now_vs} — {m13_eq.mean():.2f} on average against {offdiag.stack().mean():.2f} over the whole sample."))
    H.append(_tiles([(f"{offdiag.stack().min():.2f}–{offdiag.stack().max():.2f}", "index pairs, whole sample"), (f"{(M.loc['MC100', 'SC100'] ** 2) * 100:.0f}%", "of Smallcap's weekly moves are 'the market' (MC100×SC100 squared)"), (f"{gold_row.min():+.2f} … {gold_row.max():+.2f}", "gold vs each index"), (f"{m13_eq.min():.2f}–{m13_eq.max():.2f}", "index pairs, last 13 weeks")]))
    H.append(_book("Holding two of the four indices is mostly one position with two names on it; the only holding that behaves differently is gold."))
    lab = {C.VIXCHG: "VIX (change)", C.CASH: "Cash (liquid)"}
    H.append("<h3>2.1 The same, over time</h3>")
    H.append(_img("0_corr_rolling"))
    ps_tab = pst[["corr_full", "corr_13w_median", "corr_13w_min", "corr_13w_max", "corr_52w_now", "corr_13w_now"]].rename(columns={"corr_full": "whole sample", "corr_13w_median": "typical (13-week median)", "corr_13w_min": "lowest 13-week", "corr_13w_max": "highest 13-week", "corr_52w_now": "last 52 weeks", "corr_13w_now": "last 13 weeks"})
    ps_tab.index = [p.replace("|", " × ") for p in ps_tab.index]
    H.append(_walk(f"The index lines (top) stay between about 0.6 and 1.0 and drop together in sharp Vs — those are the {_d('decorrelation episode', 'decorrelation episodes')} Part II §6 looks at. The gold lines (bottom) wander around zero with no trend and never climb into the index range."))
    H.append(_tbl(ps_tab, fmt="{:.2f}"))
    grk = _csv("0_gold_corr_rank", index_col=0); gq = _csv("0_gold_by_group_corr"); gper = _csv("0_gold_in_episodes")
    H.append("<h4>Gold against the indices — who gold follows, and what it does when the indices decouple</h4>")
    H.append(_img("0_gold_when_decoupled"))
    H.append(_walk(f"Two questions about gold. <b>Is gold closer to Smallcap than to Nifty?</b> No: Smallcap's 13-week correlation with gold is above Nifty's in {int(grk.loc['SC100', 'weeks_above_N100'])} of {int(grk.loc['SC100', 'weeks_compared'])} weeks ({grk.loc['SC100', 'share_above_N100']:.0%}), average gap {grk.loc['SC100', 'mean_gap_vs_N100']:+.2f} (the bottom panel of the rolling chart above); gold is most correlated with Nifty {grk.loc['N100', 'share_weeks_most_correlated_with_gold']:.0%} of the time, with Smallcap {grk.loc['SC100', 'share_weeks_most_correlated_with_gold']:.0%}. <b>When the indices stop moving together, does gold start moving with them?</b> No: the dotted lines (|correlation|) are flat — {gq[gq['bucket'] == 'Q1']['mean_abs_corr'].mean():.2f} in the weeks the indices were least coupled vs {gq[gq['bucket'] == 'Q5']['mean_abs_corr'].mean():.2f} in the most, {gper['abs_before'].mean():.2f} before a decorrelation episode vs {gper['abs_during'].mean():.2f} inside it. Only the sign drifts, from slightly negative before an episode to slightly positive in its first weeks."))
    grk_show = grk.rename(columns={"mean_gold_corr": "average corr with gold", "share_weeks_most_correlated_with_gold": "gold's most-correlated index", "share_weeks_largest_abs_corr_with_gold": "largest |corr| with gold", "weeks_above_N100": "weeks above N100×gold", "share_above_N100": "share above N100×gold", "mean_gap_vs_N100": "average gap vs N100×gold", "weeks_compared": "weeks compared"})
    H.append(_tbl(grk_show, fmt="{:+.2f}", int_cols=("weeks", "weeks above N100×gold", "weeks compared"), pct_cols=("gold's most-correlated index", "largest |corr| with gold", "share above N100×gold")))
    gvf = _csv("0_gold_vs_group_fit", index_col=0); ggs = _csv("0_gold_gap_stats").iloc[0]; gga = _csv("0_gold_gap_after_extreme"); gbe = _csv("0_gold_after_break_episodes")
    a4g = gga[(gga["horizon_weeks"] == C.GAP_FWD[0]) & gga["extreme"].str.contains("above")].iloc[0]
    em = gbe[gbe["pair"] == "Micro250|GoldMCX"]
    gvf_show = gvf[["n", "pearson", "spearman", "slope", "nw_t_slope", "r2", "r2_quadratic", "mean_y_G_lowest_fifth", "mean_y_G_highest_fifth"]].rename(columns={"n": "weeks", "nw_t_slope": "t (Newey-West)", "r2": "R²", "r2_quadratic": "R² (quadratic)", "mean_y_G_lowest_fifth": "mean, indices least coupled", "mean_y_G_highest_fifth": "mean, indices most coupled"})
    gvf_show.index = [i.replace("|", " × ").replace("gap_SC100_minus_N100", "the gap SC100×Gold − N100×Gold") for i in gvf_show.index]
    H.append(_details("More (2026-10-19): is gold's correlation related to how coupled the indices are? does the SC100×Gold − N100×Gold gap mean-revert, and can it be used? how long does a gold pair stay positive after a break?",
                      _img("0_gold_vs_group") + _walk(f"<b>Scatter and fit.</b> Week by week, how coupled the indices are (G, x-axis) against gold's correlation with each index (y) and against the gap. If one drove the other the dots would slope; they do not — Pearson {gvf.loc[[p for p in gvf.index if 'Gold' in p], 'pearson'].min():+.2f} to {gvf.loc[[p for p in gvf.index if 'Gold' in p], 'pearson'].max():+.2f}, R² at most {gvf['r2'].max():.2f}, no slope's |t| above {gvf['nw_t_slope'].abs().max():.1f}, and a curve fits no better. There is no inverse relationship and nothing to model.") + _tbl(gvf_show, fmt="{:+.3f}", int_cols=("weeks",)) +
                      _img("0_gold_gap_reversion") + _walk(f"<b>The gap's mean reversion.</b> Real (half-life {ggs['half_life_weeks']:.0f} weeks, ADF p {ggs['adf_p']:.3f}) but a property of the measurement: a 13-week rolling correlation of an unrelated series is noise that the window pulls back as weeks roll off. The gap is not a price, so the narrowing cannot be traded. An extreme gap says little about returns: after the {int(a4g['events'])} gaps beyond +{C.GAP_SD:g} sd Smallcap beat Nifty over the next {C.GAP_FWD[0]} weeks {a4g['SC100_minus_N100_hit']:.0%} of the time by {_pct(a4g['SC100_minus_N100_mean'])} (an ordinary {C.GAP_FWD[0]} weeks: {_pct(a4g['SC100_minus_N100_all_weeks_mean'])}) — a thin tilt on a small count.") +
                      _img("0_gold_after_break") + _walk(f"<b>After a break.</b> Micro250 × Gold was positive in the break week in {int(em['positive_at_break'].sum())} of its {len(em)} episodes (where it then stayed positive {int(em['longest_positive_run'].max())} weeks); in the rest it was negative at the break, and the median episode has it positive {em[f'weeks_positive_of_first_{C.GOLD_AFTER_BREAK}'].median():.0f} of the first {C.GOLD_AFTER_BREAK} weeks, scattered. No sustained positive correlation to time — for Micro250 or any gold pair.")))
    H.append(_details("More: the three matrices as tables", "<p class='small'>Whole sample:</p>" + _tbl(M.rename(index=lab, columns=lab), fmt="{:.2f}") + "<p class='small'>Last 52 weeks:</p>" + _tbl(M52.rename(index=lab, columns=lab), fmt="{:.2f}") + "<p class='small'>Last 13 weeks:</p>" + _tbl(M13.rename(index=lab, columns=lab), fmt="{:.2f}") + _tbl(cov0, fmt="{:.0f}")))

    # ============================================================ 3 signal ==
    H.append("<h2>3. Correlation with respect to the signal</h2>")
    H.append(_q("Does the correlation between two assets depend on what the supertrend says about them (both BUY, one BUY and one SELL, both SELL)? And does correlation rise before a signal fires — a warning — or after it?"))
    H.append(_ans(f"Index pairs are a little more correlated in SELL ({bs3.loc[eq_rows, 'both BUY'].mean():.2f} → {bs3.loc[eq_rows, 'disagree'].mean():.2f} → {bs3.loc[eq_rows, 'both SELL'].mean():.2f}), but the rise comes <b>after</b> the flip, for every asset — it is a consequence of the sell-off, not a warning. Gold stays near zero in every state."))
    H.append("<h3>3.1 By what the two signals said</h3>")
    H.append(_img("1_corr_by_state"))
    H.append(_walk(f"Left panel: for every index pair the red dot (both SELL) sits a little to the right of the blue dot (both BUY), with the grey diamond (disagree) between them — more coupled in SELL, but not by much. For the gold pairs all three markers sit at zero. Right panel: the same pairs' {_d('covariance', 'risk')} — here the red dots sit far to the right. The gap on the right is what a SELL state does; the gap on the left is small."))
    st_tab = pd.DataFrame({"both BUY": bs3["both BUY"], "weeks ": bsw["both BUY"], "disagree": bs3["disagree"], "weeks  ": bsw["disagree"], "both SELL": bs3["both SELL"], "weeks   ": bsw["both SELL"], "all weeks": bs3["all weeks"],
                           "risk in SELL ÷ risk in BUY": dec.set_index("pair")["cov_ratio_sell_over_buy"].reindex(bs3.index), "share of the extra risk that is bigger swings": dec.set_index("pair")["share_of_cov_change_from_vol"].reindex(bs3.index)})
    st_tab.loc[[p for p in st_tab.index if "Gold" in p], ["risk in SELL ÷ risk in BUY", "share of the extra risk that is bigger swings"]] = np.nan
    st_tab.index = [p.replace("|", " × ") for p in st_tab.index]
    H.append(_tbl(st_tab, fmt="{:.2f}", int_cols=("weeks ", "weeks  ", "weeks   ")))
    H.append(f"<p class='read'>One row per pair; the three states side by side with the weeks in each. The last two columns: how many times larger the pair's risk is in both-SELL weeks than in both-BUY weeks, and how much of that extra came from bigger swings rather than from tighter coupling. Gold's are blank: its risk is near zero either way.</p>")
    H.append(_tiles([(f"{bs3.loc[eq_rows, 'both BUY'].mean():.2f}", "index pairs, both BUY"), (f"{bs3.loc[eq_rows, 'disagree'].mean():.2f}", "signals disagree"), (f"{bs3.loc[eq_rows, 'both SELL'].mean():.2f}", "both SELL"), (f"{eqp['cov_ratio_sell_over_buy'].mean():.1f}×", "the pair's risk in SELL vs BUY"), (f"{eqp['share_of_cov_change_from_vol'].mean() * 100:.0f}%", "of that extra risk is bigger swings")]))
    H.append(_book("A SELL state does not call for 'a less correlated index' — there isn't one, except gold. It calls for less exposure, because the risk is in the size of the moves (§4)."))
    H.append("<h3>3.2 Before the flip, or after? — every asset's flips</h3>")
    H.append(_img("1_corr_around_flip"))
    H.append(_walk(f"Left panel, grey band = the eight weeks before the flip: every index line is flat there (N100 {_af_row('N100', 'flip to SELL', -8):.2f} → {_af_row('N100', 'flip to SELL', -4):.2f}). At the dashed line (the flip week) the lines step up and keep climbing to the right (N100 {_af_row('N100', 'flip to SELL', 0):.2f} → {_af_row('N100', 'flip to SELL', 8):.2f}). The yellow line — gold's flips — does nothing in either panel. Right panel: around BUY flips the lines are at their highest and fall away afterwards — a BUY flip comes out of the bottom of a sell-off. The menu (top right) swaps the mean for the median or for the mean without the {C.FLIP_EXCLUDE_YEAR} flips: the step at week 0 is in all three (N100 median {_af_row('N100', 'flip to SELL', -4, 'median_corr'):.2f} → {_af_row('N100', 'flip to SELL', 0, 'median_corr'):.2f} → {_af_row('N100', 'flip to SELL', 8, 'median_corr'):.2f}; without {C.FLIP_EXCLUDE_YEAR} {_af_row('N100', 'flip to SELL', -4, f'mean_corr_ex{C.FLIP_EXCLUDE_YEAR}'):.2f} → {_af_row('N100', 'flip to SELL', 0, f'mean_corr_ex{C.FLIP_EXCLUDE_YEAR}'):.2f} → {_af_row('N100', 'flip to SELL', 8, f'mean_corr_ex{C.FLIP_EXCLUDE_YEAR}'):.2f}), so it is not the COVID crash."))
    H.append(_tbl(piv_af[[k for k in (-8, -4, 0, 4, 8)] + ["n_flips"]].rename(columns={-8: "8 wk before", -4: "4 wk before", 0: "at the flip", 4: "4 wk after", 8: "8 wk after", "n_flips": "flips"}), fmt="{:.2f}", int_cols=("flips",)))
    H.append(_tiles([(f"{_af_row('N100', 'flip to SELL', -4):.2f}", "4 weeks before N100's SELL flip"), (f"{_af_row('N100', 'flip to SELL', 0):.2f}", "in the flip week"), (f"{_af_row('N100', 'flip to SELL', 8):.2f}", "8 weeks after"), (f"{int(afa[(afa['ref'] == 'N100') & (afa['event'] == 'flip to SELL')]['n_events'].max())}", "N100 SELL flips averaged")]))
    H.append(_book("Rising correlation cannot be used as an early warning: by the time it shows, the signal has already fired."))
    # pair by pair around the flips
    afp = _csv("1_corr_around_flip_pairs")
    afn = afp[(afp["ref"] == "N100") & (afp["event"] == "flip to SELL")].pivot(index="pair", columns="weeks_from_flip", values="mean_corr")
    afn = afn.reindex([p for p in pair_order if p in afn.index])
    afn_show = afn[[-8, -4, 0, 4, 8]].rename(columns={-8: "8 wk before", -4: "4 wk before", 0: "at the flip", 4: "4 wk after", 8: "8 wk after"})
    afn_show["change, −4 → +8"] = afn[8] - afn[-4]
    afn_show.index = [p.replace("|", " × ") for p in afn_show.index]; afn_show.index.name = None; afn_show.columns.name = None
    n100p = [p for p in afn.index if p.startswith("N100|") and "Gold" not in p]; othp = [p for p in afn.index if not p.startswith("N100|") and "Gold" not in p]; gldp = [p for p in afn.index if "Gold" in p]
    H.append("<h4>Pair by pair</h4>")
    H.append(_img("1_corr_around_flip_pairs"))
    H.append(_walk(f"The same event study with every pair drawn separately (the menu picks whose flips). Around N100's SELL flips the lines that climb are the <b>N100 pairs</b> ({', '.join(p.replace('|', '×') for p in n100p)}: on average {afn.loc[n100p, -4].mean():.2f} four weeks before → {afn.loc[n100p, 8].mean():.2f} eight weeks after); the pairs among the smaller indices were already coupled and barely move ({afn.loc[othp, -4].mean():.2f} → {afn.loc[othp, 8].mean():.2f}). So 'correlation rises after a SELL flip' means: Nifty re-joins the mid/small block it had drifted from. The gold pairs (click them on) dip <i>below</i> zero in the flip week ({afn.loc[gldp, 0].mean():+.2f}) — gold rose while the indices fell — and drift back."))
    H.append(_tbl(afn_show, fmt="{:.2f}"))
    H.append(f"<p class='read'>Each pair's 13-week correlation around N100's {int(afa[(afa['ref'] == 'N100') & (afa['event'] == 'flip to SELL')]['n_events'].max())} SELL flips, averaged. The other assets' flips are in the chart's menu and in <code>1_corr_around_flip_pairs.csv</code>.</p>")
    H.append("<h3>3.3 By how many signals are in SELL</h3>")
    bpd = _csv("1_corr_by_breadth_pairs", index_col=0)
    H.append(_img("1_corr_by_breadth"))
    bp_show = bpd[[p for p in pair_order if p in bpd.columns] + ["weeks"]].copy(); bp_show.columns = [c.replace("|", " × ") for c in bp_show.columns]
    bp_show.insert(0, "group G", bb["mean"].reindex(bpd.index))
    H.append(_walk(f"Every solid line (an index pair) rises from left to right — the more of the four signals are in SELL, the more every pair moves together; the black line (G) is their average, {bb.loc['none in SELL', 'mean']:.2f} with none in SELL → {bb.loc['all in SELL', 'mean']:.2f} with all four. The steepest lines are the N100 pairs (N100×SC100 {bpd.loc['none in SELL', 'N100|SC100']:.2f} → {bpd.loc['all in SELL', 'N100|SC100']:.2f}); the pairs among the smaller indices start high and gain little (MC100×SC100 {bpd.loc['none in SELL', 'MC100|SC100']:.2f} → {bpd.loc['all in SELL', 'MC100|SC100']:.2f}). The gold pairs (dashed, click to show) stay near zero in every group. The same fact as 3.1 from the breadth side."))
    H.append(_tbl(bp_show, fmt="{:.2f}", int_cols=("weeks",)))
    H.append(f"<p class='read'>The weeks grouped by how many of the four index signals were in SELL; each column is one pair's average 13-week correlation in those weeks, 'group G' their average over the index pairs.</p>")
    H.append(_details("More: the week-by-week chart; around the flips back to BUY; the full by-state table with each leg's volatility and return in every state",
                      _img("1_corr_timeseries") + _tbl(piv_ab[[k for k in (-8, -4, 0, 4, 8)]].rename(columns={-8: "8 wk before", -4: "4 wk before", 0: "at the BUY flip", 4: "4 wk after", 8: "8 wk after"}), fmt="{:.2f}") + _tbl(bs[["pair", "state", "weeks", "corr", "cov_ann", "vol_a_ann", "vol_b_ann", "mean_ret_a_ann", "mean_ret_b_ann"]], fmt="{:.3f}", index=False, int_cols=("weeks",), pct_cols=("vol_a_ann", "vol_b_ann", "mean_ret_a_ann", "mean_ret_b_ann"), max_rows=60)))

    # ============================================================ 4 volatility ==
    H.append("<h2>4. Correlation and volatility</h2>")
    H.append(_q("Is a SELL state a high-correlation state or a high-volatility state? Do the indices move together more when the market is wild? And which comes first — the swings or the coupling?"))
    H.append(_ans(f"A SELL state is a high-volatility state: swings {vs.loc[C.EQUITY, 'vol_ratio_SELL_over_BUY'].mean():.1f}× bigger, correlation only {bs3.loc[eq_rows, 'both SELL'].mean() - bs3.loc[eq_rows, 'both BUY'].mean():.2f} higher. The wilder the market, the more the indices move together — gold never joins. The swings and the coupling arrive in the same week."))
    H.append("<h3>4.1 Volatility by the asset's own signal state</h3>")
    H.append(_img("5_vol_by_state"))
    H.append(_walk(f"Left panel: for every index the red dot (SELL weeks) is far to the right of the blue dot (BUY weeks) — {vs.loc[C.EQUITY, 'vol_ratio_SELL_over_BUY'].min():.1f}–{vs.loc[C.EQUITY, 'vol_ratio_SELL_over_BUY'].max():.1f}× the swings; for gold the two dots sit on top of each other. Middle panel: the red bars are only mildly negative — SELL weeks are not mostly down weeks ({vs.loc[C.EQUITY, 'share_weeks_down_SELL'].mean() * 100:.0f}% down vs {vs.loc[C.EQUITY, 'share_weeks_down_BUY'].mean() * 100:.0f}% in BUY)." + (f" Right panel: the VIX is higher in SELL weeks too, but the gap is much smaller ({vs.loc['N100', 'vix_mean_BUY']:.1f} → {vs.loc['N100', 'vix_mean_SELL']:.1f} for N100)." if has_vix else "")))
    vcols = ["vol_BUY", "vol_SELL", "vol_ratio_SELL_over_BUY", "share_weeks_down_SELL", "worst_week_SELL"] + (["vix_mean_BUY", "vix_mean_SELL"] if has_vix else [])
    H.append(_tbl(vs[vcols].rename(columns={"vol_BUY": "swings in BUY weeks", "vol_SELL": "swings in SELL weeks", "vol_ratio_SELL_over_BUY": "SELL ÷ BUY", "share_weeks_down_SELL": "SELL weeks that were down", "worst_week_SELL": "worst SELL week", "vix_mean_BUY": "VIX in BUY weeks", "vix_mean_SELL": "VIX in SELL weeks"}), fmt="{:.2f}", pct_cols=("swings in BUY weeks", "swings in SELL weeks", "SELL weeks that were down", "worst SELL week")))
    H.append(_tiles([(f"{vs.loc[C.EQUITY, 'vol_ratio_SELL_over_BUY'].mean():.1f}×", "index swings in SELL vs BUY"), (f"{bs3.loc[eq_rows, 'both SELL'].mean() - bs3.loc[eq_rows, 'both BUY'].mean():+.2f}", "the correlation change, for comparison"), (f"{vs.loc[C.EQUITY, 'share_weeks_down_SELL'].mean() * 100:.0f}%", "of SELL weeks are down weeks"), (f"{vs.loc['GoldMCX', 'vol_ratio_SELL_over_BUY']:.2f}×", "gold's swings in SELL vs BUY")]))
    H.append(_book("A SELL week is a coin-flip week with double the stakes. The risk of a SELL state is the size of the moves, so the defence is less exposure — not a different index."))
    sdist = _csv("5_state_dist")
    def _sd(a, s, m, col_):
        g = sdist[(sdist["asset"] == a) & (sdist["state"] == s) & (sdist["measure"] == m)]
        return g[col_].iloc[0] if len(g) else np.nan
    H.append("<h4>The distribution behind the averages — and is the VIX leading or lagging the flip?</h4>")
    H.append(_img("5_vol_state_box"))
    H.append(_walk(f"Every week as a dot, so one outlier cannot hide behind an average. The medians say what the means say: N100's typical weekly move is {_sd('N100', 'BUY', 'abs_weekly_return', 'median'):.1%} in BUY weeks and {_sd('N100', 'SELL', 'abs_weekly_return', 'median'):.1%} in SELL weeks" + (f"; the VIX {_sd('N100', 'BUY', 'vix', 'median'):.1f} vs {_sd('N100', 'SELL', 'vix', 'median'):.1f}. The VIX is where the outlier lives — its SELL mean is {_sd('N100', 'SELL', 'vix', 'mean'):.1f} against a median of {_sd('N100', 'SELL', 'vix', 'median'):.1f} because one week reached {_sd('N100', 'SELL', 'vix', 'max'):.0f} (March 2020) — so the SELL-vs-BUY gap in the VIX is real but smaller than the means suggest." if has_vix else ".")))
    if has_vix:
        vd = sdist[sdist["measure"] == "vix"].pivot(index="asset", columns="state", values=["n", "min", "p25", "median", "mean", "p75", "max"]).reindex([a for a in C.ASSETS if a in vs.index])
        vd.columns = [f"{s}: {c}" for c, s in vd.columns]; vd = vd[[f"{s}: {c}" for s in ("BUY", "SELL") for c in ("n", "min", "p25", "median", "mean", "p75", "max")]]
        H.append(_tbl(vd, fmt="{:.1f}", int_cols=tuple(c for c in vd.columns if c.endswith(": n"))))
        H.append(f"<p class='read'>The India VIX level in each asset's own BUY and SELL weeks: the count, the minimum, the 25th percentile, the median, the mean, the 75th percentile and the maximum. The same for the VIX's weekly change and |change|, the swings and the realised vol is in <code>5_state_dist.csv</code>.</p>")
    if (T / "5_vix_flip_summary.csv").exists():
        vfs = _csv("5_vix_flip_summary", index_col=0); vfe = _csv("5_vix_flip_events"); vfp = _csv("5_vix_flip_paths").set_index("k"); eqv = [a for a in C.EQUITY if a in vfs.index]
        H.append(_img("5_vix_flip_leadlag"))
        H.append(_walk(f"Both directions. Left: the VIX lined up at every index's SELL flip is flat until about four weeks before, then rises into the flip — up {vfe[vfe['vix_data']]['vix_change_4w_before'].mean():+.0%} on average over those four weeks — and for the typical flip has peaked by the flip week (the median falls afterwards; the mean's rise to week 4 is 2020). Middle: lined up at each VIX surge (up ≥ {C.VIX_SURGE:.0%} in {C.VIX_SURGE_WEEKS} weeks), the index signals flip to SELL most often in the surge week itself ({vfp.loc[0, 'share_signals_flipping_to_SELL']:.0%} of signal-weeks). Right: per index, a surge preceded only {vfs.loc[eqv, 'share_flips_with_surge_before'].mean():.0%} of SELL flips within {C.VIX_FLIP_WINDOW} weeks, {vfs.loc[eqv, 'share_surges_already_in_SELL'].mean():.0%} of surges found the index already in SELL and only {vfs.loc[eqv, 'share_surges_followed_by_flip'].mean():.0%} were followed by its flip. So the VIX is a coincident-to-slightly-leading reading of the same sell-off — ahead by the few weeks the supertrend needs to cross its band — not an earlier warning; the flip remains the thing that dates the SELL state."))
        vfs_show = vfs[["sell_flips_with_vix_data", "mean_vix_change_4w_before_flip", "median_vix_change_4w_before_flip", "share_flips_with_surge_before", "share_flips_with_surge_in_week", "share_flips_with_surge_after", "surges", "share_surges_already_in_SELL", "share_surges_followed_by_flip"]].rename(columns={"sell_flips_with_vix_data": "SELL flips", "mean_vix_change_4w_before_flip": "VIX change, 4 wk before (mean)", "median_vix_change_4w_before_flip": "(median)", "share_flips_with_surge_before": "flips with a surge before", "share_flips_with_surge_in_week": "surge in the flip week", "share_flips_with_surge_after": "surge after", "surges": "VIX surges", "share_surges_already_in_SELL": "surges with the index already in SELL", "share_surges_followed_by_flip": "surges followed by its flip"})
        H.append(_tbl(vfs_show, fmt="{:.2f}", int_cols=("SELL flips", "VIX surges"), pct_cols=("VIX change, 4 wk before (mean)", "(median)", "flips with a surge before", "surge in the flip week", "surge after", "surges with the index already in SELL", "surges followed by its flip")))
    H.append("<h3>4.2 Correlation by volatility regime</h3>")
    H.append(_img("5_corr_by_regime"))
    H.append(_img("5_corr_by_regime_more"))
    reg_tab = pd.DataFrame(index=[p for p in pst.index if p in regw[regimes[0]].index])
    for rg in regimes:
        w = regw[rg]
        reg_tab[f"{rg}: calm"] = w["Q1"]; reg_tab[f"{rg}: wild"] = w["Q5"]
    reg_tab.index = [p.replace("|", " × ") for p in reg_tab.index]
    H.append(_walk(f"Every solid line (an index pair) climbs from left (the calmest fifth of weeks) to right (the wildest) — in all three panels of the first chart, so it does not matter whose volatility defines 'wild'; the climb is flattest in the VIX panel, a forecast sorting weeks less sharply than what actually happened. The second chart checks the two things a reader asks next: 'the pair's own vol' is the average of its two legs' 13-week vol, so panels 1–2 use each leg alone, and panels 3–4 swap the standard deviation for the desk's ATR (the supertrend's 5-bar average true range ÷ close) — the climb is the same under every definition. Every dashed line (a gold pair) stays flat near zero; the one addition is that when gold itself is at its wildest (second leg's vol, Q5) the gold pairs turn mildly negative."))
    H.append(_tbl(reg_tab, fmt="{:.2f}"))
    H.append(f"<p class='read'>For each pair, its correlation in the calmest fifth of weeks and in the wildest fifth, under the three definitions of 'wild'.</p>")
    H.append(_tiles([(f"{cvi.loc['N100|SC100', qcols[0]]:.2f} → {cvi.loc['N100|SC100', qcols[-1]]:.2f}", "N100×SC100, calm → wild"), (f"+{cvi['Q5_minus_Q1'].min():.2f} to +{cvi['Q5_minus_Q1'].max():.2f}", "the climb across the six index pairs"), (f"{cvg[qcols].min().min():+.2f} … {cvg[qcols].max().max():+.2f}", "gold pairs, in every regime")]))
    H.append(_book("The diversification between two indices is smallest exactly in the weeks the book needs it most. Gold's does not shrink."))
    rstx = _csv("5_regime_stretches"); stx_sum = rstx.groupby("pair").agg(stretches=("weeks", "count"), weeks_in_Q5=("weeks", "sum"), share_corr_higher_inside=("change", lambda s: float((s > 0).mean())), median_change=("change", "median"), mean_before=("corr_before", "mean"), mean_inside=("corr_inside", "mean")).reindex([p for p in pair_order if p in set(rstx["pair"])])
    ns_ = stx_sum.loc["N100|SC100"] if "N100|SC100" in stx_sum.index else stx_sum.iloc[0]
    stx_show = stx_sum.rename(columns={"stretches": "wild stretches", "weeks_in_Q5": "weeks in Q5", "share_corr_higher_inside": "correlation higher inside than before", "median_change": "median change", "mean_before": "before (mean)", "mean_inside": "inside (mean)"}); stx_show.index = [p.replace("|", " × ") for p in stx_show.index]
    H.append(_details("More (2026-10-19): every week behind the regime averages, and the hypothesis checked stretch by stretch on the timeline",
                      _img("5_regime_scatter") + _walk("Every week as a dot — the pair's own vol against its correlation — with the median inside each fifth as a diamond. In the index pairs the whole cloud slopes up and the medians climb with the means: the B8 rise is the bulk of the weeks, not a few wild ones. The gold clouds are flat bands around zero.") +
                      _img("5_regime_timeline") + _walk(f"The same hypothesis instance by instance: the menu picks a pair, the shading marks every stretch of at least {C.REGIME_STRETCH_MIN} weeks with the pair's own vol in its wildest fifth, and the bottom panel shows what the correlation did inside each one. N100 × SC100: {int(ns_['stretches'])} stretches, the correlation higher inside than in the 8 weeks before in {ns_['share_corr_higher_inside']:.0%} of them ({ns_['mean_before']:.2f} → {ns_['mean_inside']:.2f}); the exceptions are second legs of a sell-off that began at the ceiling, and the 2018-type wild stretches where small-caps fell alone (the decorrelation episodes of §6). Every stretch is listed in <code>5_regime_stretches.csv</code>.") + _tbl(stx_show, fmt="{:.2f}", int_cols=("wild stretches", "weeks in Q5"), pct_cols=("correlation higher inside than before",))))
    H.append("<h3>4.3 Which comes first — the swings or the coupling?</h3>")
    H.append(_img("5_together"))
    H.append(_img("5_corr_rise_paths"))
    vix4 = f", the VIX {pav.loc[-4, 'z_vix']:+.1f}" if "z_vix" in pav.columns else ""
    vix0 = f", the VIX {pav.loc[0, 'z_vix']:+.1f}" if "z_vix" in pav.columns else ""
    pps = _csv("5_corr_rise_pairs_summary", index_col=0).reindex([p for p in pair_order if p in _csv("5_corr_rise_pairs_summary", index_col=0).index])
    pn = [p for p in pps.index if p.startswith("N100|") and "Gold" not in p]; po = [p for p in pps.index if not p.startswith("N100|") and "Gold" not in p]
    H.append(_walk(f"Both charts put the series on one scale — each as a {_d('z-score')} against its own last two years, so '+1' means 'unusually high' for all of them. First chart: every spike of the black line (correlation) has a blue spike (volatility) and a red spike (VIX) at the same date. Second chart: the {n_jumps} weeks the correlation jumped, lined up at week 0 — to the left of it all three lines sit near their usual level (correlation {pav.loc[-4, 'z_corr']:+.1f}, volatility {pav.loc[-4, 'z_vol']:+.1f}{vix4} four weeks before); they cross the dashed +1 line in the same week (correlation {pav.loc[0, 'z_corr']:+.1f}, volatility {pav.loc[0, 'z_vol']:+.1f}{vix0}). If either volatility measure gave warning, its line would be above +1 to the left of week 0. None is. The table under them says which pairs made the group jump — the N100 pairs reached their own +1 in {pps.loc[pn, 'share_pair_also_reached_plus1'].mean():.0%} of the jumps, the pairs among the smaller indices in {pps.loc[po, 'share_pair_also_reached_plus1'].mean():.0%}: a correlation jump is Nifty re-joining a block that was already coupled."))
    pps_show = pps.rename(columns={"jumps": "jumps the pair existed in", "share_pair_also_reached_plus1": "share in which the pair's own z reached +1", "mean_z_at_jump": "the pair's z in the jump week"}); pps_show.index = [p.replace("|", " × ") for p in pps_show.index]
    pps_show["share in which the pair's own z reached +1"] = pps_show["share in which the pair's own z reached +1"].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
    H.append(_tbl(pps_show, fmt="{:.2f}", int_cols=("jumps the pair existed in",)))
    H.append(_tiles([(f"{n_jumps}", "correlation jumps since 2016"), (f"{pav.loc[-4, 'z_vol']:+.1f}", "volatility z, 4 weeks before a jump"), (f"{pav.loc[0, 'z_vol']:+.1f}", "volatility z in the jump week"), (f"{pav.loc[0, 'z_corr']:+.1f}", "correlation z in the jump week"), (f"{pps.loc[pn, 'share_pair_also_reached_plus1'].mean():.0%} / {pps.loc[po, 'share_pair_also_reached_plus1'].mean():.0%}", "jumps in which N100 pairs / the other index pairs also jumped")]))
    own = _csv("5_corr_rise_own_summary", index_col=0); own = own.reindex([p for p in pair_order if p in own.index])
    oe = [p for p in own.index if "Gold" not in p]; og = [p for p in own.index if "Gold" in p]
    vlp = _csv("5_vol_corr_lead_pairs")
    pk_v = {p: int(vlp[vlp["pair"] == p].set_index("k_weeks")["corr_dvol_t_dcorr_t_plus_k"].idxmax()) for p in own.index}
    pk_x = {p: int(vlp[vlp["pair"] == p].set_index("k_weeks")["corr_dvix_t_dcorr_t_plus_k"].idxmax()) for p in own.index} if "corr_dvix_t_dcorr_t_plus_k" in vlp.columns else {}
    H.append("<h4>Pair by pair — each pair's own correlation, its own volatility and the VIX</h4>")
    H.append(_img("5_together_pairs"))
    H.append(_img("5_corr_rise_paths_own"))
    own_show = pd.DataFrame({"jumps": own["jumps"], "own vol peaks at k": pd.Series(pk_v), "VIX peaks at k": pd.Series(pk_x) if pk_x else np.nan,
                             "corr z, 4 wk before → at the jump": [f"{a:+.1f} → {b:+.1f}" for a, b in zip(own["z_corr_-4"], own["z_corr_+0"])],
                             "own vol z, 4 wk before → at the jump": [f"{a:+.1f} → {b:+.1f}" for a, b in zip(own["z_vol_-4"], own["z_vol_+0"])],
                             "VIX z, 4 wk before → at the jump": [f"{a:+.1f} → {b:+.1f}" for a, b in zip(own["z_vix_-4"], own["z_vix_+0"])] if "z_vix_-4" in own.columns else "",
                             "jumps with vol above +1 BEFORE": own["share_z_vol_above1_before"].map(lambda v: "" if pd.isna(v) else f"{v:.0%}"),
                             "jumps with the VIX above +1 BEFORE": own["share_z_vix_above1_before"].map(lambda v: "" if pd.isna(v) else f"{v:.0%}") if "share_z_vix_above1_before" in own.columns else ""}, index=own.index)
    own_show.index = [p.replace("|", " × ") for p in own_show.index]
    H.append(_walk(f"Now every pair on its own — its own correlation, its own volatility (the average of its two legs' 13-week vol) and the VIX, all as z-scores, and the jumps defined on the pair's own correlation. First chart: in every index-pair panel the black, blue and red lines spike in the same weeks; in the gold-pair panels the black line spikes on its own — a gold pair's correlation moves for reasons that are not volatility. Second chart, lined up at each pair's own jumps: for the index pairs the blue line (own vol) is near its usual level four weeks before (z {own.loc[oe, 'z_vol_-4'].mean():+.1f}) and up in the jump week ({own.loc[oe, 'z_vol_+0'].mean():+.1f}); it was already above +1 before the jump in only {own.loc[oe, 'share_z_vol_above1_before'].mean():.0%} of jumps — no warning pair by pair either" + (f"; the VIX likewise ({own.loc[oe, 'z_vix_-4'].mean():+.1f} → {own.loc[oe, 'z_vix_+0'].mean():+.1f})" if "z_vix_-4" in own.columns else "") + f". For the gold pairs volatility does nothing at the jump (own vol z {own.loc[og, 'z_vol_+0'].mean():+.1f}). The sliding test (in More) agrees: the pair's own vol change peaks at k = 0 in {sum(1 for p in oe if pk_v[p] == 0)} of the {len(oe)} index pairs."))
    H.append(_tbl(own_show, fmt="{:.2f}", int_cols=("jumps", "own vol peaks at k", "VIX peaks at k")))
    H.append(f"<p class='read'>Per pair: its own correlation jumps (its z first reaching +1), where the sliding test peaks for its own vol and for the VIX (k = 0 means the same week; k &gt; 0 means volatility first), each series' z four weeks before and in the jump week, and how often volatility or the VIX was already above +1 before the jump.</p>")
    ll = _csv("5_lead_lag_blocks"); chg = ll[ll["kind"] == "change"]; L0, L1 = C.LL_BLOCKS[0], C.LL_BLOCKS[-1]
    eql = [p for p in pair_order if "Gold" not in p and p in set(chg["pair"])]
    def _okr(r_):
        return pd.notna(r_["nw_t"]) and abs(r_["nw_t"]) >= 2 and pd.notna(r_["pearson_nonoverlap"]) and np.sign(r_["pearson"]) == np.sign(r_["pearson_nonoverlap"])
    vxr = chg[(chg["direction"] == "VIX → correlation") & chg["pair"].isin(eql)]; vx_rel = int(sum(_okr(r_) for _, r_ in vxr.iterrows())); vx_n = len(vxr)
    vcr = chg[(chg["direction"] == "own vol → correlation") & chg["pair"].isin(eql)]; vc_pos = int(sum(_okr(r_) and r_["pearson"] > 0 for _, r_ in vcr.iterrows()))
    cv13 = chg[(chg["direction"] == "correlation → own vol") & (chg["block_weeks"] == L1) & chg["pair"].isin(eql)]["pearson"].mean(); vc13 = chg[(chg["direction"] == "own vol → correlation") & (chg["block_weeks"] == L1) & chg["pair"].isin(eql)]["pearson"].mean()
    H.append("<h4>The same test with no overlapping weeks (redone 2026-10-19)</h4>")
    H.append(_img("5_lead_lag_blocks"))
    H.append(_walk(f"The sliding test in More relates 13-week changes whose windows overlap (9 of 13 weeks at k = 4), which can manufacture a lead. Here X is measured on the block of L weeks ending at t and Y on the block starting at t+1 — no shared week — for L = {L0} and {L1}, both directions, every pair. Nothing leads: the VIX's change passes as a predictor of the next block's correlation change in {vx_rel} of {vx_n} index-pair tests, and the pair's own volatility never leads its correlation upward ({vc_pos} passes). The bars that do pass are negative both ways ({L1}-week blocks: corr → vol {cv13:+.2f}, vol → corr {vc13:+.2f}): after a block in which both spiked, both ease — the shape of a sell-off, not a lead. The VIX's small head start in the sliding test was the overlap. The guide (B9) checks each inference drawn from the first version against these numbers."))
    H.append(_book("Neither the swings nor the VIX warn that coupling is coming; they are the same sell-off seen through different windows. The §1 episode is typical. The 'head start' the overlapping test gave the VIX does not survive non-overlapping windows."))
    vlp_show = vlp.pivot(index="k_weeks", columns="pair", values="corr_dvol_t_dcorr_t_plus_k")
    vlp_show = vlp_show[[p for p in pair_order if p in vlp_show.columns]]; vlp_show.columns = [c.replace("|", " × ") for c in vlp_show.columns]
    vlx_show = None
    if "corr_dvix_t_dcorr_t_plus_k" in vlp.columns:
        vlx_show = vlp.pivot(index="k_weeks", columns="pair", values="corr_dvix_t_dcorr_t_plus_k")
        vlx_show = vlx_show[[p for p in pair_order if p in vlx_show.columns]]; vlx_show.columns = [c.replace("|", " × ") + " (VIX)" for c in vlx_show.columns]
    H.append(_details("More: the first lead/lag test (sliding the 13-week changes against each other — its windows overlap, superseded by the block test above), for the group and for every pair; volatility over time; volatility around every asset's flips; the full regime tables; the covariance split per pair",
                      _img("5_vol_corr_lead") + _tbl(vl, fmt="{:.3f}", int_cols=("weeks", "weeks_vix")) + "<p class='small'>Each bar: the correlation between the 13-week change in volatility at week t and the 13-week change in the group correlation at week t + k. Realised vol peaks at k = 0 (they change together); the VIX's bars rise slowly to the right — an artefact of the 9-week overlap between the two windows at k = 4, as the non-overlapping test above shows.</p>" + _img("5_vol_corr_lead_pairs") + "<p class='small'>The same numbers pair by pair — each pair's own volatility against its own correlation (realised), then against the VIX:</p>" + _tbl(vlp_show, fmt="{:.2f}") + (_tbl(vlx_show, fmt="{:.2f}") if vlx_show is not None else "") + _img("5_vol_timeseries") + _img("5_vol_around_flip") + _tbl(own_df, fmt="{:.1f}", int_cols=("n_flips",), pct_cols=tuple(c for c in own_df.columns if c.startswith("own vol"))) + "".join(f"<p class='small'>Regime = <b>{rg}</b>:</p>" + _tbl(regw[rg].assign(**{'Q5−Q1': regw[rg]['Q5'] - regw[rg]['Q1']}), fmt="{:.2f}") for rg in regimes) + _tbl(dec[["pair", "corr_buy", "corr_sell", "cov_buy", "cov_sell", "cov_ratio_sell_over_buy", "share_of_cov_change_from_corr", "share_of_cov_change_from_vol", "vol_a_buy", "vol_a_sell", "vol_b_buy", "vol_b_sell"]], fmt="{:.3f}", index=False, pct_cols=("vol_a_buy", "vol_a_sell", "vol_b_buy", "vol_b_sell"))))

    # ============================================================ 5 history ==
    H.append("<h2>5. The historical pattern between correlation and returns</h2>")
    H.append(_q("When the indices are tightly correlated, what is the market doing at the time — and what does it do next? Has high correlation been good news, bad news or no news for the following quarter?"))
    F4 = C.QUINT_FWD; nx4m, nx4h, nx4n = f"next_{F4}w_mean", f"next_{F4}w_hit", f"n_next_{F4}w"
    H.append(_ans(f"High correlation is a sell-off in progress (the most-correlated fifth of weeks: N100 {_pct(n100.loc['Q5', 'same_quarter_mean'])} over that quarter) — and what follows has been the best: N100 {_pct(n100.loc['Q5', nx4m])} over the next {F4} weeks (positive {n100.loc['Q5', nx4h]:.0%} of the time) and {_pct(n100.loc['Q5', 'next_quarter_mean'])} over the next quarter (positive {n100.loc['Q5', 'next_quarter_hit']:.0%}), because sell-offs end. A tendency, not a timing rule."))
    H.append(f"<h3>5.1 By correlation quintile: the same quarter, and the next {F4} weeks</h3>")
    H.append(_img("2_corr_quintiles"))
    H.append(_walk(f"Left panel: the bars step down from Q1 to Q5 — the more correlated the weeks, the worse the quarter that ended in them (N100 {_pct(n100.loc['Q1', 'same_quarter_mean'])} in Q1 weeks, {_pct(n100.loc['Q5', 'same_quarter_mean'])} in Q5 weeks; SC100 {_pct(sc.loc['Q1', 'same_quarter_mean'])} → {_pct(sc.loc['Q5', 'same_quarter_mean'])}). That is what a sell-off looks like in correlation terms: everything falls together. Right panel — the next {F4} weeks (the horizon asked for on 2026-10-19; the next-quarter numbers stay in the table): the Q5 bars are the tallest for the indices (N100 {_pct(n100.loc['Q5', nx4m])}, SC100 {_pct(sc.loc['Q5', nx4m])}, against {_pct(n100.loc['Q1', nx4m])} / {_pct(sc.loc['Q1', nx4m])} after Q1), though the gradient is flatter than over a full quarter ({_pct(n100.loc['Q5', 'next_quarter_mean'])} vs {_pct(n100.loc['Q1', 'next_quarter_mean'])}) — four weeks catch only the start of the recovery, and Q2 is nearly as good as Q5. Gold's yellow bars follow neither pattern."))
    q_tab = pd.DataFrame({"correlation range": [f"{n100.loc[b, 'corr_lo']:.2f}–{n100.loc[b, 'corr_hi']:.2f}" for b in n100.index], "weeks": n100["weeks"].values,
                          "N100 same quarter": n100["same_quarter_mean"].values, f"N100 next {F4} wk": n100[nx4m].values, f"positive": n100[nx4h].values, "N100 next quarter": n100["next_quarter_mean"].values, "positive ": n100["next_quarter_hit"].values,
                          "SC100 same quarter": sc["same_quarter_mean"].values, f"SC100 next {F4} wk": sc[nx4m].values, "SC100 next quarter": sc["next_quarter_mean"].values}, index=n100.index)
    H.append(_tbl(q_tab, fmt="{:.2f}", int_cols=("weeks",), pct_cols=("N100 same quarter", f"N100 next {F4} wk", "positive", "N100 next quarter", "positive ", "SC100 same quarter", f"SC100 next {F4} wk", "SC100 next quarter")))
    H.append(f"<p class='read'>All weeks sorted by the {_d('13-week')} correlation among the indices into five equal groups ({_d('quintile', 'quintiles')}), Q1 least coupled … Q5 most — equal counts by construction. 'Same quarter' = the return over the 13 weeks the correlation was measured on (what was happening); 'next {F4} wk' = over the following {F4} weeks (t+1 … t+{F4}); 'next quarter' = over the following 13. 'Positive' counts only the weeks that have a forward return.</p>")
    H.append(_tiles([(f"{_pct(n100.loc['Q5', 'same_quarter_mean'])}", "N100 over the quarter ending in the most-correlated weeks"), (f"{_pct(n100.loc['Q5', nx4m])}", f"N100 over the {F4} weeks after them"), (f"{n100.loc['Q5', nx4h]:.0%}", f"of those {F4}-week stretches were positive"), (f"{_pct(n100.loc['Q5', 'next_quarter_mean'])} / {n100.loc['Q5', 'next_quarter_hit']:.0%}", "over the quarter after them / positive"), (f"{_pct(n100.loc['Q1', 'same_quarter_mean'])}", "N100 over the quarter ending in the least-correlated weeks")]))
    H.append(_book("Tight correlation says 'you are in a sell-off', not 'a sell-off is coming'. Reading it as a sell signal would mean selling at the bottom more often than not."))
    # pair by pair
    qp = _csv("2_corr_quintile_returns_pairs")
    qrows = []
    for p in [p for p in pair_order if p in set(qp["pair"])]:
        a_, b_ = p.split("|"); g = qp[qp["pair"] == p]
        ga = g[g["leg"] == a_].set_index("bucket"); gb = g[g["leg"] == b_].set_index("bucket")
        qrows.append({"pair": p.replace("|", " × "), "Q5 correlation range": f"{ga.loc['Q5', 'corr_lo']:.2f}–{ga.loc['Q5', 'corr_hi']:.2f}", "first leg, same quarter, Q1 → Q5": f"{_pct(ga.loc['Q1', 'same_quarter_mean'])} → {_pct(ga.loc['Q5', 'same_quarter_mean'])}", f"first leg, next {F4} wk, Q1 → Q5": f"{_pct(ga.loc['Q1', nx4m])} → {_pct(ga.loc['Q5', nx4m])}", "first leg, next quarter, Q1 → Q5": f"{_pct(ga.loc['Q1', 'next_quarter_mean'])} → {_pct(ga.loc['Q5', 'next_quarter_mean'])}",
                      "second leg, same quarter, Q1 → Q5": f"{_pct(gb.loc['Q1', 'same_quarter_mean'])} → {_pct(gb.loc['Q5', 'same_quarter_mean'])}", f"second leg, next {F4} wk, Q1 → Q5": f"{_pct(gb.loc['Q1', nx4m])} → {_pct(gb.loc['Q5', nx4m])}", "second leg, next quarter, Q1 → Q5": f"{_pct(gb.loc['Q1', 'next_quarter_mean'])} → {_pct(gb.loc['Q5', 'next_quarter_mean'])}", f"next {F4} wk positive after Q5 (first / second leg)": f"{ga.loc['Q5', nx4h]:.0%} / {gb.loc['Q5', nx4h]:.0%}"})
    qpt = pd.DataFrame(qrows).set_index("pair")
    qe = qp[~qp["pair"].str.contains("Gold")]; qg = qp[qp["pair"].str.contains("Gold") & (qp["leg"] == "GoldMCX")]
    H.append("<h4>Pair by pair</h4>")
    H.append(_img("2_corr_quintiles_pairs"))
    def _q51(p, what):
        g = qp[qp["pair"] == p]
        return all(g[(g["leg"] == leg) & (g["bucket"] == "Q5")][what].iloc[0] > g[(g["leg"] == leg) & (g["bucket"] == "Q1")][what].iloc[0] for leg in p.split("|"))
    eqp_list = [p for p in pair_order if p in set(qp["pair"]) and "Gold" not in p]
    fall_ok = [p for p in eqp_list if not _q51(p, "same_quarter_mean")]; next_ok = [p for p in eqp_list if _q51(p, nx4m)]; next_no = [p for p in eqp_list if p not in next_ok]; nq_ok = [p for p in eqp_list if _q51(p, "next_quarter_mean")]
    H.append(_walk(f"One panel per pair, the weeks cut by <i>that pair's</i> correlation. In {'every' if len(fall_ok) == len(eqp_list) else str(len(fall_ok)) + ' of the six'} index pair{'s' if len(fall_ok) != 1 else ''} the dotted lines (the same quarter) fall from Q1 to Q5, and in {len(next_ok)} of the six the bars (the next {F4} weeks) are taller at Q5 than at Q1 for both legs" + (f" — not in {', '.join(p.replace('|', '×') for p in next_no)}" if next_no else "") + f" ({len(nq_ok)} of six over the next quarter). The pattern is not an artefact of averaging six pairs into G: across the six index pairs the legs made {_pct(qe[qe['bucket'] == 'Q5']['same_quarter_mean'].mean())} over the quarter ending in their most-correlated weeks and {_pct(qe[qe['bucket'] == 'Q5'][nx4m].mean())} over the next {F4} weeks ({_pct(qe[qe['bucket'] == 'Q5']['next_quarter_mean'].mean())} over the next quarter), against {_pct(qe[qe['bucket'] == 'Q1']['same_quarter_mean'].mean())}, {_pct(qe[qe['bucket'] == 'Q1'][nx4m].mean())} and {_pct(qe[qe['bucket'] == 'Q1']['next_quarter_mean'].mean())} for the least-correlated. In the gold pairs (bottom row) gold's own bars are flat across the buckets ({_pct(qg[qg['bucket'] == 'Q1'][nx4m].mean())} after Q1, {_pct(qg[qg['bucket'] == 'Q5'][nx4m].mean())} after Q5): how much gold happens to be moving with an index says nothing about gold's next weeks."))
    H.append(_tbl(qpt))
    H.append(f"<p class='read'>Per pair: the correlation range of its most-correlated fifth, and each leg's average return over the same quarter, the next {F4} weeks and the next quarter, in the least-correlated weeks (Q1) → the most-correlated (Q5). Full detail per bucket in <code>2_corr_quintile_returns_pairs.csv</code>.</p>")
    # the ratio study (2026-10-19)
    rs = _csv("2_ratio_study"); FR, BK = C.RATIO_FWD, C.RATIO_BACK
    allw = rs[rs["cut"] == "all weeks"].set_index("pair"); qq = rs[rs["cut"] == "correlation quintile"]
    fullp = [p for p in pair_order if p in allw.index and "Micro250" not in p]; mxq = qq[qq["pair"].isin(fullp)][f"fwd_{FR}w_mean"].abs().max()
    H.append("<h4>The ratio — one number per pair: does the correlation, the signal or its own last month say which leg wins next?</h4>")
    H.append(_img("2_ratio_by_corr"))
    H.append(_img("2_ratio_by_state"))
    H.append(_walk(f"Instead of two returns, the ratio first leg ÷ second leg — its log change is the first leg's return minus the second's. For every pair and week: the ratio's change over the next {FR} weeks (t+1 … t+{FR}), cut by the pair's correlation quintile at t, by the joint signal state at t, and by who led over the last {BK} weeks. Nothing sorts it: in the full-history pairs every bar is within ±{mxq:.1%}, the hit rates sit at a coin flip, and the rank correlation of the forward change with the correlation level is {allw.loc[fullp, 'spearman_fwd_vs_corr'].min():+.2f} to {allw.loc[fullp, 'spearman_fwd_vs_corr'].max():+.2f}, with the ratio's own last {BK} weeks {allw.loc[fullp, 'spearman_fwd_vs_past'].min():+.2f} to {allw.loc[fullp, 'spearman_fwd_vs_past'].max():+.2f} — neither momentum nor reversal. The Micro250 pairs show bigger bars on a quarter of the history, too thin to read. Correlation and the signal describe how two legs move together, not which one wins — the same answer the deviation z-score gives in §8, reached more simply."))
    ratio_show = pd.DataFrame({"next 4 wk, Q1 → Q5 (first leg ahead)": [f"{_pct(qq[(qq['pair'] == p) & (qq['group'] == 'Q1')][f'fwd_{FR}w_mean'].iloc[0])} → {_pct(qq[(qq['pair'] == p) & (qq['group'] == 'Q5')][f'fwd_{FR}w_mean'].iloc[0])}" for p in allw.index],
                            "hit rate Q1 / Q5": [f"{qq[(qq['pair'] == p) & (qq['group'] == 'Q1')][f'fwd_{FR}w_hit'].iloc[0]:.0%} / {qq[(qq['pair'] == p) & (qq['group'] == 'Q5')][f'fwd_{FR}w_hit'].iloc[0]:.0%}" for p in allw.index],
                            "rank corr: forward vs correlation": allw["spearman_fwd_vs_corr"].values, f"rank corr: forward vs last {BK} wk": allw["spearman_fwd_vs_past"].values, "weeks": allw["n_forward"].values}, index=[p.replace("|", " ÷ ") for p in allw.index])
    H.append(_tbl(ratio_show.reindex([p.replace("|", " ÷ ") for p in pair_order if p in allw.index]), fmt="{:+.2f}", int_cols=("weeks",)))
    H.append(f"<p class='read'>Per pair (first leg ÷ second leg): the ratio's average change over the next {FR} weeks in the least- and most-correlated fifths with the share of weeks the first leg was ahead, and the rank correlations over all weeks. Every cut in <code>2_ratio_study.csv</code>.</p>")
    H.append("<h3>5.2 By calendar year, the timeline, and the stretches above 0.90</h3>")
    H.append(_img("2_by_year"))
    H.append(_img("2_timeline"))
    yr_hi, yr_lo = yr["mean_corr"].idxmax(), yr["mean_corr"].idxmin()
    hc_pos = int((hc["N100_next_13w"].dropna() > 0).sum()); hc_n = int(hc["N100_next_13w"].notna().sum())
    ypr = _csv("2_by_year_pairs", index_col=0); ypr = ypr[[p for p in pair_order if p in ypr.columns]]
    hps = _csv("2_high_corr_pairs_summary", index_col=0).reindex([p for p in pair_order if p in _csv("2_high_corr_pairs_summary", index_col=0).index]); hpp = _csv("2_high_corr_episodes_pairs")
    hps_top = hps["share_of_weeks_above_0.90"].idxmax(); hps_lo = hps.loc[[p for p in hps.index if "Gold" not in p], "share_of_weeks_above_0.90"].idxmin()
    H.append(_walk(f"By year: the bars (G) drift up over the decade ({yr['mean_corr'].iloc[0]:.2f} in {yr.index[0]} → {yr['mean_corr'].iloc[-1]:.2f} in {yr.index[-1]}; highest {yr.loc[yr_hi, 'mean_corr']:.2f} in {yr_hi}), and the pair lines keep their order every year — {hps_top.replace('|', '×')} on top, the N100 pairs underneath — so a year's level is the same story for every pair. The most-correlated years are not the worst years ({yr_hi}: N100 {_pct(yr.loc[yr_hi, 'ret_N100'])}) because a year averages the sell-off and the recovery — the quintile chart above separates the two. Timeline: every spike of the black line sits on a trough of the 13-week return line and a spike of the VIX — three views of the same sell-offs — and the thin pair lines spike with it. The stretches above 0.90: for the group {len(hc)} since {C.START} ({int(hc['weeks'].min())}–{int(hc['weeks'].max())} weeks each), followed by a positive N100 quarter {hc_pos} times out of {hc_n}; pair by pair, {hps_top.replace('|', '×')} is above 0.90 {hps.loc[hps_top, 'share_of_weeks_above_0.90']:.0%} of all weeks and {hps_lo.replace('|', '×')} only {hps.loc[hps_lo, 'share_of_weeks_above_0.90']:.0%}; no gold pair ever gets there."))
    yr2 = yr.rename(columns={"mean_corr": "average correlation", "min_corr": "lowest", "max_corr": "highest", "weeks_corr_above_0.9": "weeks above 0.90"}).rename(columns={c: c.replace("ret_", "return ") for c in yr.columns if c.startswith("ret_")})
    H.append(_tbl(yr2, fmt="{:.2f}", int_cols=("weeks", "weeks above 0.90"), pct_cols=tuple(c for c in yr2.columns if c.startswith("return "))))
    hps_show = hps.rename(columns={"weeks_above_0.90": "weeks above 0.90", "share_of_weeks_above_0.90": "share of all weeks", "stretches": "stretches of 8+ weeks", "median_weeks": "median length", "a_next_13w_mean": "first leg, next quarter", "a_next_13w_positive": "positive", "b_next_13w_mean": "second leg, next quarter", "b_next_13w_positive": "positive "}); hps_show.index = [p.replace("|", " × ") for p in hps_show.index]
    H.append(_tbl(hps_show, fmt="{:.1f}", int_cols=("weeks above 0.90", "stretches of 8+ weeks"), pct_cols=("share of all weeks", "first leg, next quarter", "positive", "second leg, next quarter", "positive ")))
    H.append(f"<p class='read'>Each pair above 0.90: how often, how many stretches of 8+ weeks, and what each leg did over the quarter after a stretch ended.</p>")
    ypr_show = ypr.copy(); ypr_show.columns = [c.replace("|", " × ") for c in ypr_show.columns]; ypr_show.insert(0, "group G", yr["mean_corr"])
    H.append(_details("More: the stretches above 0.90 one by one (the group, then every pair); every pair's average correlation by year",
                      _tbl(hc.rename(columns={"peak_corr": "peak correlation"}), fmt="{:+.3f}", index=False, int_cols=("weeks",), pct_cols=tuple(c for c in hc.columns if "_during" in c or "_next" in c)) +
                      "<p class='small'>Pair by pair ('a' = the first leg, 'b' = the second):</p>" + _tbl(hpp.assign(pair=hpp["pair"].str.replace("|", " × ", regex=False)), fmt="{:+.3f}", index=False, int_cols=("weeks",), pct_cols=tuple(c for c in hpp.columns if "_during" in c or "_next" in c), max_rows=60) +
                      "<p class='small'>Average 13-week correlation by calendar year, the group and every pair:</p>" + _tbl(ypr_show, fmt="{:.2f}")))

    # ============================================================ Part II ==
    H.append("<h2>Part II — the further studies (click a box to open it)</h2>")
    H.append("<p class='small'>Each box shows its one-line finding. Inside: the same chart → table → what-it-says shape as Part I, with its own 'More…' boxes for the detail.</p>")

    # ---------------------------------------------------------- Part II: 6. Who breaks the correlation first
    S = []
    # ============================================================ 5 ==
    S.append("<h2>6. Who breaks the correlation first</h2>")
    S.append(_lead(f"<b>What it answers:</b> when the market stops moving as one block, which index lets go of the others first — and which way does it go? <b>Method in four lines:</b> each index's 13-week correlation with the other three (<code>c</code>) and the group average (<code>G</code>); each turned into a z-score against its own {C.CB_NORM}-week history; a <i>break</i> = z falls to {C.CB_Z:.0f} (having been above {C.CB_RESET:.1f}); a group break opens a <i>decorrelation episode</i>, and each index's own break date is compared with the group's (negative lead = earlier). Because G is the average of the c's, several indices often cross in the same week, so two tie-breakers are kept: who was deepest below its norm when the group broke, and whose correlation fell most."))
    S.append(f"<p class='small'><b>Worked example — {ex['episode']}:</b> the group's correlation went from {ex['G_before']:.2f} to {ex['G_min']:.2f} over {int(ex['weeks'])} weeks. " + " ".join(f"{a}: {exb.loc[a, 'corr_before']:.2f} → {exb.loc[a, 'corr_min']:.2f}, broke {_wk(exb.loc[a, 'lead_vs_group_weeks'])} the group, {str(exb.loc[a, 'direction']).split(' (')[0]}." for a in exb.index) + "</p>")
    S.append(_img("3_corr_series"))
    S.append(_tbl(cep[["episode", "start", "end", "weeks", "G_before", "G_min", "drop"]], fmt="{:.2f}", index=False, int_cols=("weeks",)))
    S.append(_img("3_who_breaks_correlation"))
    sm_tab = sm[["episodes_present", "share_first", "share_deepest", "share_biggest_drop", "avg_drop_share", "combined_rank", "share_up_breaks"]].rename(columns={"episodes_present": "episodes", "share_first": "first by date", "share_deepest": "deepest at the break", "share_biggest_drop": "biggest collapse", "avg_drop_share": "its share of the group's loss", "combined_rank": "combined rank (low = first)", "share_up_breaks": "broke upward"})
    S.append(_read("Shares of episodes. 'First by date' counts ties. 'Deepest at the break' = furthest below its own norm in the week the group broke. 'Biggest collapse' = its own correlation with the others fell most. 'Broke upward' = it decoupled by outperforming the others over the prior 13 weeks."))
    S.append(_tbl(sm_tab, fmt="{:.2f}", int_cols=("episodes",), pct_cols=("first by date", "deepest at the break", "biggest collapse", "its share of the group's loss", "broke upward")))
    S.append(_says(f"By date <b>{date_first}</b> breaks first most often ({sm.loc[date_first, 'share_first'] * 100:.0f}%, alone first only {sm.loc[date_first, 'share_first_alone'] * 100:.0f}% — ties dominate); by size of its own collapse <b>{drop_first}</b> ({sm.loc[drop_first, 'share_biggest_drop'] * 100:.0f}%, losing {sm.loc[drop_first, 'avg_drop_share'] * 100:.0f}% of the group's correlation on average). Both describe one mechanism: <b>a decorrelation here is the mid, small and micro caps running away from Nifty</b> — {sm.loc['MC100', 'share_up_breaks'] * 100:.0f}% of Midcap's and {sm.loc['SC100', 'share_up_breaks'] * 100:.0f}% of Smallcap's breaks are upward, {100 - sm.loc['N100', 'share_up_breaks'] * 100:.0f}% of Nifty's downward. The runner crosses the line first; the index everybody runs away from is the one whose relationship collapses. The 'first by date' winner changes with the settings (see More); the direction does not."))
    S.append("<h4>What happens after an index breaks away</h4>")
    S.append(_tbl(afs[afs["weeks_ahead"] == 13][["direction", "n", "breaker", "rest_of_group", "gold", "share_breaker_ahead"]], fmt="{:.2f}", index=False, int_cols=("n",), pct_cols=("breaker", "rest_of_group", "gold")))
    S.append(_says(f"Up-breaks are the common case ({int(up.loc[4, 'n'])} of {int(up.loc[4, 'n'] + dn.loc[4, 'n'])}). The index that decoupled by outperforming returned {_pct(up.loc[13, 'breaker'])} over the next 13 weeks against {_pct(up.loc[13, 'rest_of_group'])} for the others, ahead only {up.loc[13, 'share_breaker_ahead'] * 100:.0f}% of the time. <b>Decoupling by outperformance has been the end of a run, not the start of one.</b> As an early warning of a price sell-off, a correlation break worked {n_warn} times in {n_sell}."))
    more5 = _img("3_corr_episodes") + "".join(f"<div class='ep'><b>{e['episode']}</b> — {e['start'].date()} to {e['end'].date()}, {int(e['weeks'])} weeks; group {e['G_before']:.2f} → {e['G_min']:.2f}. " + "; ".join(f"<b>{a}</b> {_wk(g.loc[a, 'lead_vs_group_weeks'])} ({str(g.loc[a, 'direction']).split(' (')[0]}, own correlation {g.loc[a, 'corr_before']:.2f} → {g.loc[a, 'corr_min']:.2f})" for a in g[g['break_date'].notna()].sort_values(['break_date', 'z_at_group_break']).index) + ".</div>" for e, g in ((e, ab[ab["episode"] == e["episode"]].set_index("asset")) for _, e in cep.iterrows()))
    more5 += "<p class='small'>Rank by break date / by depth / by size of fall, and the lead in weeks:</p>" + _tbl(_csv("3_rank_date", index_col=0), fmt="{:.0f}") + _tbl(_csv("3_rank_deepest", index_col=0), fmt="{:.0f}") + _tbl(_csv("3_rank_drop", index_col=0), fmt="{:.0f}") + _tbl(_csv("3_lead_weeks", index_col=0), fmt="{:+.0f}")
    more5 += "<p class='small'>Pair level — which relationship goes first:</p>" + _tbl(psm, fmt="{:.2f}") + "<p class='small'>Before each price sell-off — did the group decorrelate in the 13 weeks before N100's peak?</p>" + _tbl(bsf, fmt="{:.2f}", index=False) + _img("3_after_break") + _tbl(afs, fmt="{:+.3f}", index=False, int_cols=("n", "weeks_ahead"), pct_cols=("breaker", "rest_of_group", "gold", "breaker_minus_rest")) + f"<p class='small'>Sensitivity — the by-date ranking at window {C.CB_SENSITIVITY['window']} × threshold {C.CB_SENSITIVITY['z']} (episode counts: {', '.join(f'{k}: {v}' for k, v in meta.get('sensitivity_episodes', {}).items())}):</p>" + _tbl(sens, fmt="{:.2f}")
    S.append(_details("More: every episode, the three rankings, pair level, the early-warning test, the after-break detail, sensitivity", more5))

    H.append(_details(f"<b>6. Who breaks the correlation first.</b> {len(cep)} decorrelation episodes: the smaller index leaves by running ahead (by date {date_first}), Nifty is the one left behind (by collapse {drop_first}); an index that decoupled by outperforming returned {_pct(up.loc[13, 'breaker'])} the next quarter. Click to open.", ''.join(S)))

    # ---------------------------------------------------------- Part II: 7. Who breaks in price and signal
    S = []
    # ============================================================ 6 ==
    S.append("<h2>7. Who breaks in price and signal</h2>")
    S.append(_lead(f"<b>What it answers:</b> the other meaning of 'breaks first' — in a sell-off ({C.DD_THRESHOLD:.0%}+ fall in N100), which index falls {C.DD_CROSS:.0%} first, whose supertrend fires first, and who recovers first?"))
    S.append(_img("4_who_breaks_first"))
    sm6 = sm4[["episodes", "first_share_cross_10pct", "median_lead_cross_10pct_weeks", "first_share_signal_sell", "median_lead_signal_sell_weeks", "first_share_recover_buy", "median_own_fall"]].rename(columns={"first_share_cross_10pct": "first to fall 10%", "median_lead_cross_10pct_weeks": "lead (wk)", "first_share_signal_sell": "signal fires first", "median_lead_signal_sell_weeks": "lead (wk) ", "first_share_recover_buy": "first back to BUY", "median_own_fall": "median own fall"})
    S.append(_read("Shares of the sell-offs the asset was first (ties counted); leads in weeks relative to N100 (negative = earlier)."))
    S.append(_tbl(sm6, fmt="{:.2f}", int_cols=("episodes",), pct_cols=("first to fall 10%", "signal fires first", "first back to BUY", "median own fall")))
    S.append(_says(f"<b>{first_cross}</b> crosses −10% first ({sm4.loc[first_cross, 'first_share_cross_10pct'] * 100:.0f}% of the {len(pep)} sell-offs, median {-sm4.loc[first_cross, 'median_lead_cross_10pct_weeks']:.0f} weeks before N100) and falls most ({_pct(sm4.loc[first_cross, 'median_own_fall'])} median); <b>{first_sig}</b>'s supertrend fires first ({sm4.loc[first_sig, 'first_share_signal_sell'] * 100:.0f}%) and turns back to BUY first. Price and signal disagree because the band scales with each asset's own volatility: a 10% fall is a bigger event, in band terms, for a low-vol large cap. Six events — ranks and medians only."))
    S.append(_details("More: the six sell-offs, the price charts, the lead tables", _tbl(pep, fmt="{:.3f}", index=False, int_cols=("weeks_peak_to_trough", "weeks_to_recover"), pct_cols=("depth",)) + _img("4_price_episodes") + "".join(f"<p class='small'>Weeks relative to N100, {title}:</p>" + _tbl(_csv(f"4_lead_{col}", index_col=0), fmt="{:+.0f}") for col, title in (("signal_sell", "supertrend flipped to SELL"), ("cross_10pct", "price crossed −10%"), ("recover_buy", "supertrend flipped back to BUY")))))

    H.append(_details(f"<b>7. Who breaks in price and signal.</b> In the {len(pep)} sell-offs {first_cross} falls 10% first and falls most; {first_sig}'s supertrend fires first. Click to open.", ''.join(S)))

    # ---------------------------------------------------------- Part II: 8. Each pair
    S = []
    # ============================================================ 7 ==
    S.append("<h2>8. Each pair</h2>")
    S.append(_lead(f"<b>What it answers:</b> for each of the ten pairs — how far apart have the two legs drifted (the deviation z of the price ratio vs its trailing {C.PAIR_WINDOW} weeks), does the ratio pull back when it has drifted, and how coupled and how volatile are the legs? <b>Sign convention:</b> z &gt; 0 = the first-named asset has run ahead."))
    S.append("<h3>8.1 Do deviations pull back? — all pairs</h3>")
    S.append(_img("6_z_forward"))
    S.append(_read("Weeks bucketed by the pair's deviation z; the ratio's change over the next 13 weeks averaged inside each bucket. After 'a lagged b' (z &lt; 0) a pull-back would be a <i>positive</i> change; after 'a led' (z &gt; 0) a negative one. Bars = all pairs pooled; dots = each pair (click the legend)."))
    co_tab = pco[["halflife_weeks_full", "ratio_drift_per_year", "adf_p_log_ratio", "verdict"]].rename(columns={"halflife_weeks_full": "half-life (wk)", "ratio_drift_per_year": "drift per year", "adf_p_log_ratio": "ADF p"})
    S.append(_tbl(co_tab, fmt="{:.2f}", pct_cols=("drift per year",)))
    S.append(_says(f"Statistically, only <b>{', '.join(p.replace('|', '×') for p in coint_ok) if coint_ok else 'none'}</b> has a ratio with a long-run level; the other {10 - len(coint_ok)} drift — the smaller index out-runs the larger one year after year (N100|MC100 {pco.loc['N100|MC100', 'ratio_drift_per_year'] * 100:+.1f}% a year). Empirically, a 1–2 sd deviation <b>continued</b>: after 'a lagged by 1–2 sd' the ratio moved a further {_pct(pz.loc[B[1], 'mean_ratio_change'])} the same way (t {_t(pz.loc[B[1], 'hac_t_ratio_change'])}), after 'a led by 1–2 sd' a further {_pct(pz.loc[B[3], 'mean_ratio_change'])} (t {_t(pz.loc[B[3], 'hac_t_ratio_change'])}); beyond 2 sd it went nowhere. <b>These ratios are trends, not spreads.</b> The z-score itself always decays (see More) — that is the window catching up, not the ratio returning."))
    S.append(_details("More: the full pull-back tables and the z-path after a 2-sd deviation", _tbl(pco, fmt="{:.2f}", int_cols=("weeks",), pct_cols=("ratio_drift_per_year",)) + _tbl(pooled[["horizon_weeks", "bucket", "weeks", "mean_ratio_change", "share_reverted", "hac_t_ratio_change", "mean_ret_a", "mean_ret_b"]], fmt="{:.3f}", index=False, int_cols=("horizon_weeks", "weeks"), pct_cols=("mean_ratio_change", "mean_ret_a", "mean_ret_b")) + _img("6_path_after_2sd")))
    S.append("<h3>8.2 The ten pairs</h3>")
    S.append(_read("Each card: (1) both legs' cumulative return; (2) the price ratio with its 52-week mean and ±1/±2 sd bands; (3) the deviation z with ▼ the first leg's SELL flips, ▲ the second's, ● pair-trade entries and ○ exits; (4) the 13-week correlation; (5) each leg's 13-week vol" + (" and the VIX" if has_vix else "") + ". Then the pair's facts and a two-line reading; the tables are in 'More'."))
    zfp = zf[zf["pair"] != "ALL PAIRS"]
    for k_, p in enumerate(pst.index):
        a, b = p.split("|")
        s = pst.loc[p]; c = pco.loc[p]; rr = rt.loc[p]
        z13 = zfp[(zfp["pair"] == p) & (zfp["horizon_weeks"] == H13)].set_index("bucket")
        ptr = pts.loc[p] if p in pts.index else None
        ab_ = bp[(bp["seller"] == a) & (bp["candidate"] == b)]; ba_ = bp[(bp["seller"] == b) & (bp["candidate"] == a)]
        S.append(f"<div class='card'><h3>8.2.{k_ + 1} {a} × {b}</h3>")
        S.append(_img("6_pair", pair=p))
        S.append("<div class='kv'>"
                 f"<div><b>correlation</b> full {s['corr_full']:.2f} · now {s['corr_13w_now']:.2f} (13w) / {s['corr_52w_now']:.2f} (52w)</div>"
                 f"<div><b>returns, annualised</b> {a} {_pct(rr['annualised_a'])} · {b} {_pct(rr['annualised_b'])} · last 52w {_pct(rr['last_52w_a'])} / {_pct(rr['last_52w_b'])}</div>"
                 f"<div><b>volatility</b> {a} {_p(s['vol_a_full'])} · {b} {_p(s['vol_b_full'])} · beta of {a} on {b} {s['beta_a_on_b_full']:.2f}</div>"
                 f"<div><b>in both-SELL weeks, annualised</b> {a} {_pct(rr['both_SELL_ann_a'], 0)} · {b} {_pct(rr['both_SELL_ann_b'], 0)}</div>"
                 f"<div><b>deviation now</b> z {s['z_now']:+.1f}: {s['reading_now']} · |z| &gt; 2 in {_p(s['share_weeks_abs_z_gt_2'])} of weeks</div>"
                 f"<div><b>pull-back</b> {c['verdict'].split(' (')[0]}</div>"
                 "</div>")
        lag_b = z13.loc[B[1]] if B[1] in z13.index else None; lead_b = z13.loc[B[3]] if B[3] in z13.index else None
        lines = [f"<b>Returns:</b> {b if rr['spread_annualised'] < 0 else a} led by {abs(rr['spread_annualised']) * 100:.1f} points a year over {rr['years']:.1f} years; in both-SELL weeks the two ran at {_pct(rr['both_SELL_ann_a'], 0)} and {_pct(rr['both_SELL_ann_b'], 0)} annualised."]
        if lag_b is not None and lead_b is not None:
            lines.append(f"<b>Deviation:</b> after {a} lagged by 1–2 sd the ratio moved {_pct(lag_b['mean_ratio_change'])} over 13 weeks ({'a pull-back' if lag_b['mean_ratio_change'] > 0 else 'kept going'}, t {_t(lag_b['hac_t_ratio_change'])}); after {a} led, {_pct(lead_b['mean_ratio_change'])} ({'a pull-back' if lead_b['mean_ratio_change'] < 0 else 'kept going'}, t {_t(lead_b['hac_t_ratio_change'])}).")
        if ptr is not None:
            lines.append(f"<b>Pair trade:</b> {int(ptr['trades'])} trades, {_pct(ptr['mean_net'])} net each — {ptr['verdict']}.")
        rp_bits = []
        for lab, d_, x, y in ((f"after {a} sells, {b}", ab_, a, b), (f"after {b} sells, {a}", ba_, b, a)):
            if len(d_):
                d_ = d_.iloc[0]
                rp_bits.append(f"{lab} returned {_pct(d_[f'cand_mean_{H13}'])} over 13 weeks (staying in {x} {_pct(d_[f'stay_mean_{H13}'])}, cash {_pct(d_[f'cash_mean_{H13}'])}; {y} was in BUY at {_p(d_['share_cand_in_BUY'])} of those flips)")
        if rp_bits:
            lines.append("<b>Replacement:</b> " + "; ".join(rp_bits) + ".")
        S.append(_says(" ".join(lines)))
        rtab = pd.DataFrame({a: [rr["total_a"], rr["annualised_a"], rr["last_13w_a"], rr["last_52w_a"], rr["both_BUY_ann_a"], rr["both_SELL_ann_a"], rr["disagree_ann_a"], rr["vol_a"], rr["worst_week_a"]],
                             b: [rr["total_b"], rr["annualised_b"], rr["last_13w_b"], rr["last_52w_b"], rr["both_BUY_ann_b"], rr["both_SELL_ann_b"], rr["disagree_ann_b"], rr["vol_b"], rr["worst_week_b"]],
                             f"spread {a} − {b}": [rr["spread_total"], rr["spread_annualised"], rr["spread_last_13w"], rr["spread_last_52w"], rr["spread_both_BUY_ann"], rr["spread_both_SELL_ann"], rr["spread_disagree_ann"], np.nan, np.nan]},
                            index=["total return", "annualised", "last 13 weeks", "last 52 weeks", f"both BUY weeks ({int(rr['both_BUY_weeks'])}), annualised", f"both SELL weeks ({int(rr['both_SELL_weeks'])}), annualised", f"disagree weeks ({int(rr['disagree_weeks'])}), annualised", "volatility", "worst week"])
        S.append(_details("More: the returns table; what the ratio did next by deviation; the pair's trades",
                          _tbl(rtab, fmt="{:.2f}", pct_cols=tuple(rtab.columns)) + _tbl(z13.reset_index()[["bucket", "weeks", "mean_ratio_change", "share_reverted", "hac_t_ratio_change", "mean_ret_a", "mean_ret_b"]], fmt="{:.2f}", index=False, int_cols=("weeks",), pct_cols=("mean_ratio_change", "mean_ret_a", "mean_ret_b"))
                          + (_tbl(tr[tr["pair"] == p][["entry_date", "exit_date", "weeks", "cheap_leg", "z_entry", "z_exit", "exit_reason", "net", "ret_cheap_leg", "ret_rich_leg"]].assign(entry_date=lambda d: d["entry_date"].dt.date, exit_date=lambda d: d["exit_date"].dt.date), fmt="{:.2f}", index=False, int_cols=("weeks",), pct_cols=("net", "ret_cheap_leg", "ret_rich_leg")) if len(tr) and (tr["pair"] == p).any() else "<p class='small'>no trades</p>")))
        S.append("</div>")

    H.append(_details(f"<b>8. Each pair — deviation, pull-back, the ten cards.</b> {10 - len(coint_ok)} of 10 price ratios drift; a 1–2 sd deviation continued rather than reverted. Click to open.", ''.join(S)))

    # ---------------------------------------------------------- Part II: 9. The pair-trading rule
    S = []
    # ============================================================ 8 ==
    S.append("<h2>9. The pair-trading rule</h2>")
    S.append(_lead(f"<b>What it answers:</b> had the book traded every large deviation as a pair — at |z| ≥ {C.PT_ENTRY:.0f}, long the leg that lagged, short the leg that ran, out at |z| ≤ {C.PT_EXIT:.1f} or after {C.PT_MAX_HOLD} weeks, {C.PT_COST_BPS} bp per leg per side — what would it have made? The same trade read long-only: the cheap leg's return vs the rich leg's, i.e. 'hold the laggard instead of the leader'."))
    S.append(_img("7_summary"))
    pt_tab = pts[["trades", "win_rate_net", "mean_net", "total_net", "avg_weeks", "share_exit_z", "mean_ret_cheap_leg", "mean_ret_rich_leg", "verdict"]].rename(columns={"win_rate_net": "winners", "mean_net": "net per trade", "total_net": "total net", "avg_weeks": "weeks held", "share_exit_z": "closed by z", "mean_ret_cheap_leg": "cheap leg's return", "mean_ret_rich_leg": "rich leg's return"})
    S.append(_read("<code>net per trade</code> in log-return units (+0.05 = a unit spread position made 5 points). <code>closed by z</code> = the share of trades that ended because the deviation came back inside; the rest hit the 26-week limit. A verdict of 'worked' needs ≥ 5 trades, a positive mean, ≥ 60% winners and t ≥ 2."))
    S.append(_tbl(pt_tab, fmt="{:.2f}", int_cols=("trades",), pct_cols=("net per trade", "total net", "cheap leg's return", "rich leg's return")))
    S.append(_says(f"<b>{ptall['verdict'].capitalize()}.</b> {int(ptall['trades'])} trades, {_p(ptall['win_rate_net'])} winners, {_pct(ptall['mean_net'])} per trade net (t {_t(ptall['t_mean_net'])}); {n_pairs_neg} of 10 pairs lost money. Only {_p(ptall['share_exit_z'])} of trades closed because the deviation came back. Long-only says the same: the leg the rule buys returned {_pct(ptall['mean_ret_cheap_leg'])} over the average trade, the leg it sells {_pct(ptall['mean_ret_rich_leg'])} — <b>the laggard kept lagging.</b> No setting in the sensitivity grid fixes it (best: entry {sens_best['entry']:.1f} / exit {sens_best['exit']:.1f}, {_pct(sens_best['mean_net'])} per trade, t {_t(sens_best['t_mean_net'])}). The reason is §8.1: buying the lagging leg is betting against a trend in progress. An evaluation of a rule, not a backtest — one unit per leg, flat costs, no sizing."))
    S.append(_details("More: the cumulative P&L chart, the sensitivity grid, every trade", _img("7_equity") + _tbl(ptsens, fmt="{:.3f}", index=False, int_cols=("trades", "pairs_positive", "pairs_traded"), pct_cols=("mean_net", "total_net")) + _tbl(tr[["pair", "entry_date", "exit_date", "weeks", "cheap_leg", "rich_leg", "z_entry", "z_exit", "exit_reason", "gross", "net", "ret_cheap_leg", "ret_rich_leg"]].assign(entry_date=lambda d: d["entry_date"].dt.date, exit_date=lambda d: d["exit_date"].dt.date), fmt="{:.2f}", index=False, max_rows=200, int_cols=("weeks",), pct_cols=("gross", "net", "ret_cheap_leg", "ret_rich_leg"))))

    H.append(_details(f"<b>9. The pair-trading rule.</b> {ptall['verdict'].capitalize()}: {int(ptall['trades'])} trades, {_pct(ptall['mean_net'])} each (t {_t(ptall['t_mean_net'])}), {n_pairs_neg} of 10 pairs lost. Click to open.", ''.join(S)))

    # ---------------------------------------------------------- Part II: 10. Who replaces a SELL
    S = []
    # ============================================================ 9 ==
    S.append("<h2>10. Who replaces a SELL</h2>")
    S.append(_lead(f"<b>What it answers:</b> when an asset's signal flips to SELL, its money has to go somewhere — which asset should take its place, and how did each choice do? {n_ev} SELL flips since {C.START} (" + ", ".join(f"{a} {flips_per[a]}" for a in C.ASSETS) + f"). For each flip, every other asset is a candidate, described <i>at</i> the flip by its {C.RP_CORR_WINDOW}-week correlation with the seller (correlated if ≥ {C.RP_CORR_MIN}), its deviation against the seller (lagging z ≤ {-C.RP_Z_DEV:.0f} / leading z ≥ {C.RP_Z_DEV:+.0f}) and its own signal; then measured <i>after</i> it over {H4} and {H13} weeks against staying put, cash and gold."))
    S.append("<h3>10.1 The rules, head to head</h3>")
    S.append(_img("8_rules"))
    r_tab = R[[f"share_with_a_pick", f"mean_{H13}", f"beat_cash_{H13}", f"beat_stay_{H13}", f"hac_t_minus_cash_{H13}", f"when_pick_mean_{H13}", "hold_mean", "hold_weeks_mean"]].rename(columns={"share_with_a_pick": "had a pick", f"mean_{H13}": f"return, {H13} wk", f"beat_cash_{H13}": "beat cash", f"beat_stay_{H13}": "beat staying", f"hac_t_minus_cash_{H13}": "t vs cash", f"when_pick_mean_{H13}": "when it had a pick", "hold_mean": "held to its own SELL", "hold_weeks_mean": "weeks held"})
    S.append(_read(f"Twelve rules applied to the same {n_ev} flips, one pick each; a rule with no qualifying candidate sits in cash that time (<code>had a pick</code> = how often it found one). <code>return</code> = the average over all flips; <code>when it had a pick</code> = over the flips where it actually chose someone; <code>held to its own SELL</code> = the pick held until its own signal turned, capped at {C.RP_MAX_HOLD} weeks."))
    S.append(_tbl(r_tab, fmt="{:.2f}", pct_cols=("had a pick", f"return, {H13} wk", "beat cash", "beat staying", "when it had a pick", "held to its own SELL")))
    S.append(_says(f"<b>The proposed rule — enter the correlated asset that has lagged the seller — </b>found a candidate in {int(lag_rule['events_with_a_pick'])} of {n_ev} flips and returned {_pct(lag_rule[f'when_pick_mean_{H13}'])} on them (cash {_pct(cash13)}); with 'and its own signal is BUY' it <b>never had a candidate</b>: a correlated asset that has lagged a seller is itself in SELL. <b>What worked:</b> the asset that had pulled furthest <i>ahead</i> of the seller ({_pct(lead_rule[f'mean_{H13}'])}, beat cash {_p(lead_rule[f'beat_cash_{H13}'])}, t {_t(lead_rule[f'hac_t_minus_cash_{H13}'])}), the most correlated asset <i>still in BUY</i> ({_pct(corrbuy[f'mean_{H13}'])}, t {_t(corrbuy[f'hac_t_minus_cash_{H13}'])}), and gold ({_pct(gold_rule[f'mean_{H13}'])}, t {_t(gold_rule[f'hac_t_minus_cash_{H13}'])}). Staying in the seller {_pct(stay13)}; the most correlated asset regardless of its signal {_pct(mc_rule[f'mean_{H13}'])}; the most lagging {_pct(mlag[f'mean_{H13}'])}."))
    S.append("<h3>10.2 Why: the candidate's own signal is what matters</h3>")
    S.append(_img("8_cells"))
    cells_tab = pd.DataFrame([{"candidate's deviation": d, "own signal": st_, "rows": int(cell(d, 'any', st_)['rows']), f"candidate return, {H13} wk": cell(d, 'any', st_)['cand_mean'], "beat cash": cell(d, 'any', st_)['share_cand_beat_cash'], "the seller made": cell(d, 'any', st_)['stay_mean']} for d in ("lagging", "neutral", "leading") for st_ in ("BUY", "SELL")] + [{"candidate's deviation": "any", "own signal": st_, "rows": int(cell('any', 'any', st_)['rows']), f"candidate return, {H13} wk": cell('any', 'any', st_)['cand_mean'], "beat cash": cell('any', 'any', st_)['share_cand_beat_cash'], "the seller made": cell('any', 'any', st_)['stay_mean']} for st_ in ("BUY", "SELL")])
    S.append(_tbl(cells_tab, fmt="{:.2f}", index=False, int_cols=("rows",), pct_cols=(f"candidate return, {H13} wk", "the seller made")))
    S.append(_says(f"A candidate in <b>BUY</b> returned {_pct(c_buy['cand_mean'])} over {H13} weeks (beat cash {_p(c_buy['share_cand_beat_cash'])}), one in <b>SELL</b> {_pct(c_sellst['cand_mean'])}; the seller itself {_pct(c_all['stay_mean'])}, cash {_pct(c_all['cash_mean'])}. Deviation matters only through the signal: 'leading' candidates made {_pct(c_lead_buy['cand_mean'])} in BUY and {_pct(c_lead_sell['cand_mean'])} in SELL; {int(c_lag_sell['rows'])} of {int(cell('lagging', 'any', 'any')['rows'])} lagging candidates were in SELL. <b>Replace a SELL with something whose own signal is BUY.</b>"))
    S.append("<h3>10.3 By seller</h3>")
    S.append(_read(f"Average return over the next {H13} weeks of each candidate (columns) after each seller's flips (rows), with staying put and cash alongside."))
    S.append(_tbl(bpp, fmt="{:.3f}", int_cols=("flips",), pct_cols=tuple(c for c in bpp.columns if c != "flips")))
    S.append(_says("After any <b>index</b> sells, the other indices are poor replacements — they are falling with it — and gold has worked (" + "; ".join(f"after {a} → {best_by_seller[a]} {_pct(bpp.loc[a, best_by_seller[a]])} vs staying {_pct(bpp.loc[a, 'stay'])}" for a in bpp.index if a != "GoldMCX") + f"). After <b>gold</b> sells, the indices have been fine replacements ({gold_seller_str}) because gold's SELL says nothing about them."))
    S.append("<h3>10.4 What the deviation looked like around the flips</h3>")
    S.append(_img("8_z_path"))
    S.append(_says("The z of each candidate against the seller, averaged from 13 weeks before the flip to 13 after (+ = the candidate ahead). " + "; ".join(f"after <b>{a}</b> sells the others go {zpa.loc[a, -4]:+.2f} → {zpa.loc[a, 0]:+.2f} → {zpa.loc[a, 13]:+.2f}" for a in zpa.index) + ". A line still rising after week 0 means the seller kept falling behind — <b>the deviation at the flip is not the end of the move.</b>"))
    S.append("<h3>10.5 Right now</h3>")
    S.append(_tbl(cur8, fmt="{:.2f}", index=False))
    S.append(_says((f"In SELL as of {cur8['date'].iloc[0]}: {', '.join(nz)}. " if nz else f"As of {cur8['date'].iloc[0]} every asset is in BUY — no replacement is called for. ") + "For each asset, the table shows what the rules of 10.1 would see if it flipped this week."))
    S.append(_details("More: the full cells table; the rules by seller; every flip's z-path", _tbl(show, fmt="{:.2f}", index=False, int_cols=("rows", "events"), pct_cols=("cand_mean", "cand_median", "stay_mean", "cash_mean", "cand_minus_stay", "cand_minus_cash")) + _tbl(rs_show, fmt="{:.3f}", pct_cols=tuple(rs_show.columns)) + _tbl(bp[["seller", "candidate", "events", "avg_corr_52w", "avg_z_cand_vs_seller", "share_cand_in_BUY", f"cand_mean_{H4}", f"cand_mean_{H13}", f"stay_mean_{H13}", f"cash_mean_{H13}", "hold_mean", "hold_weeks_mean"]], fmt="{:.2f}", index=False, int_cols=("events",), pct_cols=(f"cand_mean_{H4}", f"cand_mean_{H13}", f"stay_mean_{H13}", f"cash_mean_{H13}", "hold_mean")) + "".join(f"<h4>Every SELL flip of {a}</h4>" + _img("8_events", seller=a) for a in [x for x in C.ASSETS if x in set(ev["seller"])])))
    S.append(_note(f"{n_ev} flips, several in the same weeks (2020-03 above all), so the t-stats rest on fewer independent events than {n_ev}; 'gold' carries gold's own 2016–2026 run. A ranking of simple rules on one history, not a strategy — the next step is to pre-register 'replace a SELL with the most-leading BUY-state asset' and test it inside the rotation framework with real costs and cadence."))

    H.append(_details(f"<b>10. Who replaces a SELL.</b> On {n_ev} flips: the asset whose own signal is BUY — most leading {_pct(lead_rule[f'mean_{H13}'])}, most correlated still in BUY {_pct(corrbuy[f'mean_{H13}'])}, gold {_pct(gold_rule[f'mean_{H13}'])}; the correlated laggard never had a BUY-state candidate. Click to open.", ''.join(S)))

    # ============================================================ 11 ==
    H.append("<h2>11. In one page</h2><ul>")
    H.append(f"<li><b>Between the asset classes:</b> the four indices are one block ({offdiag.stack().min():.2f}–{offdiag.stack().max():.2f}); gold is unrelated ({gold_row.min():+.2f} to {gold_row.max():+.2f}).</li>")
    H.append(f"<li><b>With respect to the signal:</b> {bs3.loc[eq_rows, 'both BUY'].mean():.2f} (both BUY) → {bs3.loc[eq_rows, 'disagree'].mean():.2f} (disagree) → {bs3.loc[eq_rows, 'both SELL'].mean():.2f} (both SELL); the rise comes after the flip, not before, and it is the N100 pairs that rise — the smaller indices were already coupled. Not a warning.</li>")
    H.append(f"<li><b>With volatility:</b> SELL weeks are volatility weeks — {vs.loc[C.EQUITY, 'vol_ratio_SELL_over_BUY'].mean():.1f}× the swings, {eqp['share_of_cov_change_from_vol'].mean() * 100:.0f}% of the extra risk is volatility; correlation rises with volatility in every index pair, gold's does not; the two spike together, neither warns. Cut exposure; don't hunt for a less-correlated index.</li>")
    H.append(f"<li><b>With returns:</b> the most-correlated weeks are sell-offs in progress (N100 {_pct(n100.loc['Q5', 'same_quarter_mean'])} that quarter) and have been followed by better-than-average quarters ({_pct(n100.loc['Q5', 'next_quarter_mean'])}); a tendency, not a rule.</li>")
    H.append(f"<li><b>Part II:</b> the smaller index breaks the correlation first by running ahead and Nifty is left behind; deviations between the indices are trends, not spreads (no pair trade); replace a SELL with an asset whose own signal is BUY — the one that pulled ahead ({_pct(lead_rule[f'mean_{H13}'])}), the most correlated still in BUY ({_pct(corrbuy[f'mean_{H13}'])}), or gold ({_pct(gold_rule[f'mean_{H13}'])}).</li>")
    H.append(f"<li><b>Small samples where it matters:</b> {len(cep)} episodes, {len(pep)} sell-offs, {n_ev} flips, {int(ptall['trades'])} trades — ranks and medians there, Newey-West t-stats where there are enough weeks.</li></ul>")
    H.append(f"<p class='small'>Tables in <code>outputs/tables/</code> (0_ §2, 1_ §3, 5_ §4, 2_ §5, 3_ §6, 4_ §7, 6_ §8, 7_ §9, 8_ §10), charts by <code>figs.py</code>, parameters in <code>config.py</code>, every column in <code>GLOSSARY.md</code>; the full version is <code>outputs/report_full.html</code>. Built in {meta.get('elapsed_s', '?')} s.</p>")
    H.append(f"<script>{JS}</script></body></html>")
    return "\n".join(H)


def main() -> int:
    out = build()
    C.REPORT_HTML.write_text(out, encoding="utf-8")
    print(f"wrote {C.REPORT_HTML.relative_to(C.ROOT)} ({len(out) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
