"""Build outputs/episodes_<index pair>_minus_<gold pair>.html — every ±1 sd crossing of one difference series (B2's
extreme follow-ups), one episode at a time: the difference around the crossing, its two components, and the assets'
cumulative returns. Default series: N100 × SC100 minus SC100 × GoldMCX.

POINT-IN-TIME THRESHOLD (fix of 2026-11-02): the ±GAP_SD sd lines are an EXPANDING mean ± sd of the difference — at every
week they use only the data up to that week (a warm-up of C.EP_WARMUP weeks first). The earlier whole-history mean/sd was a
look-ahead: it flagged and placed the 2016–17 crossings against a 2016–2026 baseline they could not have known. The lines
are now time-varying, so a crossing is "the first week the difference went outside its own expanding ±1 sd band." The
difference series itself and the forward returns were always trailing / t+1…t+h; only the band changed. This is the same
point-in-time band histcorr.extreme_followups and histcorr.gap_reversion use for B2.

    python make_episodes.py                       # the default series (C.PAIR_VS_GOLD_DEFAULT)
    python make_episodes.py "N100|MC100" "MC100|GoldMCX"

Self-contained: loads the panel and computes the two rolling correlations (sigcorr.pair_corrs) and the crossings here, so it
needs only the raw data in data/, not outputs/tables/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as C
import data
import sigcorr

PLOTLY_JS = (Path(__import__("plotly").__file__).parent / "package_data" / "plotly.min.js")
COL = {"N100": "#2a78d6", "MC100": "#eb6834", "SC100": "#1baf7a", "Micro250": "#4a3aa7", "GoldMCX": "#eda100"}
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
FONT = "Inter, Helvetica, Arial, sans-serif"
CONFIG = {"responsive": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}
BEFORE, AFTER = 8, C.EXTREME_PATH          # 8 weeks before (matches histcorr.extreme_followups' path window) … C.EXTREME_PATH after
EP_WARMUP = C.EP_WARMUP                     # weeks of history before the expanding band is defined (no lines, no episodes before this)


def _nm(p: str) -> str:
    return p.replace("|", " × ")


_DEF = C.PAIR_VS_GOLD_DEFAULT


def build(a: str = _DEF[0], b: str = _DEF[1]) -> Path:
    panel = data.load_panel(verbose=False)
    px = data.close_panel(panel)
    r = data.returns(px, "weekly")
    PC = sigcorr.pair_corrs(r)
    for p in (a, b):
        if p not in PC.columns:
            raise SystemExit(f"{p} not among the pair correlations: {list(PC.columns)}")
    d = (PC[a] - PC[b]).rename("d")
    idx = d.index
    # expanding (point-in-time) band: mean and sd of the difference using only data up to each week
    m_exp = d.expanding(min_periods=EP_WARMUP).mean()
    sd_exp = d.expanding(min_periods=EP_WARMUP).std()
    up = m_exp + C.GAP_SD * sd_exp
    lo = m_exp - C.GAP_SD * sd_exp
    hit_up = (d > up).fillna(False)
    hit_lo = (d < lo).fillna(False)
    starts_up = idx[hit_up & ~hit_up.shift(1, fill_value=False)]
    starts_lo = idx[hit_lo & ~hit_lo.shift(1, fill_value=False)]
    ev = pd.DataFrame([{"date": t, "side": "above +1 sd"} for t in starts_up] + [{"date": t, "side": "below -1 sd"} for t in starts_lo]).sort_values("date").reset_index(drop=True)
    if ev.empty:
        raise SystemExit("no crossings")
    legs = list(dict.fromkeys(a.split("|") + b.split("|")))
    ks = list(range(-BEFORE, AFTER + 1))
    r2 = r.reindex(idx)
    m_now, sd_now = float(m_exp.iloc[-1]), float(sd_exp.iloc[-1])

    def window(series: pd.Series, t: pd.Timestamp) -> pd.Series:
        i = idx.get_loc(t); out = pd.Series(np.nan, index=ks)
        for k in ks:
            j = i + k
            if 0 <= j < len(idx):
                out[k] = series.iloc[j]
        return out

    def cum_path(asset: str, t: pd.Timestamp) -> pd.Series:
        i = idx.get_loc(t); rr = r2[asset]; out = pd.Series(np.nan, index=ks)
        for k in ks:
            j = i + k
            if not (0 <= j < len(idx)):
                continue
            seg = rr.iloc[i + 1:j + 1] if k >= 0 else rr.iloc[j + 1:i + 1]
            if seg.notna().all():
                out[k] = seg.sum() if k >= 0 else -seg.sum()
        return out

    def fwd(asset, t, h):
        i = idx.get_loc(t); seg = r2[asset].iloc[i + 1:i + 1 + h]
        return seg.sum() if len(seg) == h and seg.notna().all() else np.nan

    def dchg(t, h):
        i = idx.get_loc(t); j = i + h
        return (d.iloc[j] - d.iloc[i]) if j < len(idx) and pd.notna(d.iloc[j]) else np.nan

    def weeks_back(t):
        i = idx.get_loc(t); sign = np.sign(d.iloc[i] - m_exp.iloc[i])
        for k in range(1, AFTER + 1):
            if i + k < len(idx) and pd.notna(m_exp.iloc[i + k]) and np.sign(d.iloc[i + k] - m_exp.iloc[i + k]) != sign:
                return k
        return np.nan

    def layout(fig, title, height):
        kw = {}
        if title:
            kw["title"] = {"text": f"<b>{title}</b>", "x": 0, "xanchor": "left", "font": {"size": 15}}
        fig.update_layout(height=height, margin={"l": 60, "r": 30, "t": 70 if title else 40, "b": 95}, paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                          font={"family": FONT, "size": 12.5, "color": INK}, hovermode="x unified",
                          legend={"orientation": "h", "yanchor": "top", "y": -0.24, "x": 0, "font": {"size": 11.5}}, **kw)
        for ann in fig.layout.annotations:
            ann.font = {"size": 13, "color": INK}
        fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont={"color": MUTED})
        fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, tickfont={"color": MUTED})
        return fig

    # ------------------------------------------------------------ per-episode
    figs_html, toc, rows = [], [], []
    for n, e in ev.iterrows():
        t = pd.Timestamp(e["date"]); side = e["side"]; above = side.startswith("above")
        i0 = idx.get_loc(t)
        wd, wa, wb = window(d, t), window(PC[a], t), window(PC[b], t)
        wup, wlo, wm = window(up, t), window(lo, t), window(m_exp, t)
        fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=["The difference, with its expanding ±1 sd band", "Its two components", "Cumulative return (0 at the crossing)"])
        fig.add_trace(go.Scatter(x=ks, y=wup.values, mode="lines", name="+1 sd (expanding)", line={"color": "#e34948", "dash": "dash", "width": 1}, hovertemplate="%{y:+.2f}<extra>+1 sd</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ks, y=wlo.values, mode="lines", name="−1 sd (expanding)", line={"color": "#2a78d6", "dash": "dash", "width": 1}, hovertemplate="%{y:+.2f}<extra>−1 sd</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ks, y=wm.values, mode="lines", name="mean (expanding)", line={"color": MUTED, "dash": "dot", "width": 1}, hovertemplate="%{y:+.2f}<extra>mean</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=ks, y=wd.values, mode="lines+markers", name="difference", line={"color": INK, "width": 2.2}, marker={"size": 4}, hovertemplate="%{y:+.2f}<extra>difference</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=[0], y=[d.iloc[i0]], mode="markers", name="the crossing", marker={"color": "#e34948" if above else "#2a78d6", "size": 11, "symbol": "triangle-down" if above else "triangle-up"}, hovertemplate=f"{t.date()}: %{{y:+.2f}}<extra>crossing</extra>"), row=1, col=1)
        fig.add_vline(x=0, line={"color": MUTED, "dash": "dash", "width": 1}, row=1, col=1)
        fig.add_trace(go.Scatter(x=ks, y=wa.values, mode="lines+markers", name=_nm(a), line={"color": "#4a3aa7", "width": 2}, marker={"size": 4}, hovertemplate="%{y:+.2f}<extra>" + _nm(a) + "</extra>"), row=1, col=2)
        fig.add_trace(go.Scatter(x=ks, y=wb.values, mode="lines+markers", name=_nm(b), line={"color": "#9a7b1a", "width": 2}, marker={"size": 4}, hovertemplate="%{y:+.2f}<extra>" + _nm(b) + "</extra>"), row=1, col=2)
        fig.add_hline(y=0, line={"color": GRID, "width": 1}, row=1, col=2); fig.add_vline(x=0, line={"color": MUTED, "dash": "dash", "width": 1}, row=1, col=2)
        for x in legs:
            fig.add_trace(go.Scatter(x=ks, y=cum_path(x, t).values, mode="lines", name=x, line={"color": COL[x], "width": 2 if x != "N100" else 1.6}, hovertemplate="%{y:+.1%}<extra>" + x + "</extra>"), row=1, col=3)
        fig.add_hline(y=0, line={"color": GRID, "width": 1}, row=1, col=3); fig.add_vline(x=0, line={"color": MUTED, "dash": "dash", "width": 1}, row=1, col=3)
        for c_ in (1, 2, 3):
            fig.update_xaxes(title_text="weeks from the crossing", row=1, col=c_)
        fig.update_yaxes(title_text="difference (correlation units)", row=1, col=1); fig.update_yaxes(title_text="13-week correlation", range=[-1, 1.05], row=1, col=2); fig.update_yaxes(tickformat="+.0%", title_text="cumulative return", row=1, col=3)
        layout(fig, None, 400)
        wk = weeks_back(t); back = f"back to its mean in {int(wk)} weeks" if pd.notna(wk) else f"not back to its mean within {AFTER} weeks"
        sc, gd, nf = fwd("SC100", t, 13), fwd("GoldMCX", t, 13), fwd("N100", t, 13)
        band = f"band that week {lo.iloc[i0]:+.2f} / {up.iloc[i0]:+.2f} (mean {m_exp.iloc[i0]:+.2f})"
        head = (f"<h3>Episode {n + 1} · {t.date()} · {side}</h3>"
                f"<p class='key'><span>difference <b>{d.iloc[i0]:+.2f}</b> ({_nm(a)} {wa[0]:+.2f}, {_nm(b)} {wb[0]:+.2f})</span><span>{band}</span><span>{back}</span>"
                f"<span>next 13 weeks: SC100 <b>{sc:+.1%}</b> · gold <b>{gd:+.1%}</b> · N100 <b>{nf:+.1%}</b></span></p>")
        figs_html.append(f"<section id='ep{n + 1}'>" + head + fig.to_html(full_html=False, include_plotlyjs=False, config=CONFIG, div_id=f"ep_{n + 1}") + "</section>")
        toc.append(f"<a href='#ep{n + 1}'>{n + 1} · {t.date()} · {'+' if above else '−'}</a>")
        rows.append({"#": n + 1, "date": t.date(), "side": side, "difference at crossing": d.iloc[i0], "band that week": f"{lo.iloc[i0]:+.2f} / {up.iloc[i0]:+.2f}", _nm(a): wa[0], _nm(b): wb[0], "weeks to mean": wk,
                     "difference change, 13 wk": dchg(t, 13), "SC100 4 wk": fwd("SC100", t, 4), "SC100 13 wk": sc, "gold 13 wk": gd, "N100 13 wk": nf})

    # ------------------------------------------------------------ overlays + aggregates
    overlays, agg = [], {}
    for side_name, sub in ev.groupby("side", sort=False):
        above = side_name.startswith("above")
        fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=["The difference", "The two components", "SC100 cumulative return"])
        D, A, B, R = [], [], [], []
        for _, e in sub.iterrows():
            t = pd.Timestamp(e["date"]); D.append(window(d, t)); A.append(window(PC[a], t)); B.append(window(PC[b], t)); R.append(cum_path("SC100", t))
            fig.add_trace(go.Scatter(x=ks, y=D[-1].values, mode="lines", line={"color": INK, "width": 0.8}, opacity=0.35, showlegend=False, hovertemplate=f"{t.date()}: %{{y:+.2f}}<extra></extra>"), row=1, col=1)
            fig.add_trace(go.Scatter(x=ks, y=A[-1].values, mode="lines", line={"color": "#4a3aa7", "width": 0.8}, opacity=0.35, showlegend=False, hoverinfo="skip"), row=1, col=2)
            fig.add_trace(go.Scatter(x=ks, y=B[-1].values, mode="lines", line={"color": "#9a7b1a", "width": 0.8}, opacity=0.35, showlegend=False, hoverinfo="skip"), row=1, col=2)
            fig.add_trace(go.Scatter(x=ks, y=R[-1].values, mode="lines", line={"color": COL["SC100"], "width": 0.8}, opacity=0.35, showlegend=False, hovertemplate=f"{t.date()}: %{{y:+.1%}}<extra></extra>"), row=1, col=3)
        for arr, col, colr, nm, pctfmt in ((D, 1, INK, "average difference", False), (A, 2, "#4a3aa7", f"average {_nm(a)}", False), (B, 2, "#9a7b1a", f"average {_nm(b)}", False), (R, 3, COL["SC100"], "average SC100 return", True)):
            mean = pd.concat(arr, axis=1).mean(axis=1)
            fig.add_trace(go.Scatter(x=ks, y=mean.values, mode="lines", name=nm, line={"color": colr, "width": 3}, hovertemplate=("%{y:+.1%}" if pctfmt else "%{y:+.2f}") + "<extra>" + nm + "</extra>"), row=1, col=col)
        fig.add_hline(y=0, line={"color": GRID, "width": 1}, row=1, col=2); fig.add_vline(x=0, line={"color": MUTED, "dash": "dash", "width": 1}, row=1, col=2)
        fig.add_hline(y=0, line={"color": GRID, "width": 1}, row=1, col=3); fig.add_vline(x=0, line={"color": MUTED, "dash": "dash", "width": 1}, row=1, col=3)
        fig.add_vline(x=0, line={"color": MUTED, "dash": "dash", "width": 1}, row=1, col=1)
        for c_ in (1, 2, 3):
            fig.update_xaxes(title_text="weeks from the crossing", row=1, col=c_)
        fig.update_yaxes(title_text="difference", row=1, col=1); fig.update_yaxes(title_text="13-week correlation", range=[-1, 1.05], row=1, col=2); fig.update_yaxes(tickformat="+.0%", title_text="cumulative return", row=1, col=3)
        layout(fig, f"All {len(sub)} episodes {side_name}, overlaid", 440)
        overlays.append(fig.to_html(full_html=False, include_plotlyjs=False, config=CONFIG, div_id=f"ov_{'up' if above else 'lo'}"))
        s13 = [fwd("SC100", pd.Timestamp(x), 13) for x in sub["date"]]; s13 = [v for v in s13 if pd.notna(v)]
        g13 = [fwd("GoldMCX", pd.Timestamp(x), 13) for x in sub["date"]]; g13 = [v for v in g13 if pd.notna(v)]
        agg[side_name] = {"n": len(sub), "sc_mean": np.mean(s13), "sc_hit": np.mean([v > 0 for v in s13]), "sc_med": np.median(s13), "gold_mean": np.mean(g13)}
    base13 = r2["SC100"].rolling(13).sum().shift(-13)
    base_mean, base_hit = float(base13.mean()), float((base13 > 0).mean())

    # ------------------------------------------------------------ the page
    tab = pd.DataFrame(rows)
    fmt = tab.copy()
    for c_ in fmt.columns:
        if c_ in ("SC100 4 wk", "SC100 13 wk", "gold 13 wk", "N100 13 wk"):
            fmt[c_] = fmt[c_].map(lambda v: "" if pd.isna(v) else f"{v * 100:+.1f}%")
        elif c_ == "weeks to mean":
            fmt[c_] = fmt[c_].map(lambda v: f"not within {AFTER}" if pd.isna(v) else f"{int(v)}")
        elif pd.api.types.is_float_dtype(fmt[c_]):
            fmt[c_] = fmt[c_].map(lambda v: "" if pd.isna(v) else f"{v:+.2f}")
    fmt["#"] = fmt["#"].map(lambda v: f"<a href='#ep{v}'>{v}</a>")
    table_html = fmt.to_html(index=False, border=0, classes="t", escape=False)
    n_up, n_lo = int((ev["side"].str.startswith("above")).sum()), int((ev["side"].str.startswith("below")).sum())
    au, al = agg["above +1 sd"], agg["below -1 sd"]
    css = """
    body{font-family:Inter,Helvetica,Arial,sans-serif;max-width:1320px;margin:0 auto;padding:24px 20px;color:#0b0b0b;background:#fcfcfb;line-height:1.5}
    h1{font-size:22px;margin:0 0 4px} h2{font-size:18px;margin:36px 0 8px;border-bottom:2px solid #e1e0d9;padding-bottom:4px}
    h3{font-size:15px;margin:8px 8px 2px} .key{margin:0 8px 6px;font-size:12.5px;color:#52514e;display:flex;flex-wrap:wrap;gap:4px 22px} .key b{color:#0b0b0b}
    .sub{color:#52514e;margin-bottom:14px;font-size:13.5px} .box{background:#f3f3ef;border-left:4px solid #2a78d6;padding:10px 14px;margin:14px 0;font-size:13.5px}
    .warn{background:#fdf3e7;border-left:4px solid #eb6834;padding:10px 14px;margin:14px 0;font-size:13.5px}
    .toc{display:flex;flex-wrap:wrap;gap:6px 12px;font-size:12.5px;margin:8px 0 18px} .toc a{text-decoration:none;color:#2a78d6;border:1px solid #e1e0d9;border-radius:3px;padding:2px 7px;background:#fff}
    section{margin:18px 0;border:1px solid #e1e0d9;border-radius:4px;padding:4px;background:#fcfcfb}
    table.sum{border-collapse:collapse;font-size:13px;margin:8px 0} table.sum th,table.sum td{padding:4px 12px;border-bottom:1px solid #e1e0d9;text-align:right} table.sum td:first-child,table.sum th:first-child{text-align:left}
    .t{border-collapse:collapse;font-size:12px;margin:8px 0;display:block;overflow-x:auto;max-width:100%} .t th,.t td{padding:3px 8px;text-align:right;border-bottom:1px solid #eeede8;white-space:nowrap} .t th{color:#52514e;font-weight:600} .t td:first-child,.t th:first-child,.t td:nth-child(2),.t td:nth-child(3){text-align:left}
    .top{position:fixed;right:16px;bottom:16px;background:#fff;border:1px solid #e1e0d9;border-radius:4px;padding:6px 10px;font-size:12px;text-decoration:none;color:#2a78d6}
    """
    H = [f"<!doctype html><html><head><meta charset='utf-8'><title>Episodes · {_nm(a)} − {_nm(b)}</title><style>{css}</style><script>{PLOTLY_JS.read_text(encoding='utf-8')}</script></head><body id='top'>"]
    H.append(f"<h1>Every ±{C.GAP_SD:g} sd crossing of {_nm(a)} − {_nm(b)}, one episode at a time</h1>")
    H.append(f"<div class='sub'>Study I, B2 · point-in-time (expanding) band · {len(ev)} episodes: {n_up} above, {n_lo} below · the band today is mean {m_now:+.2f} ± {sd_now:.2f} → {m_now - sd_now:+.2f} / {m_now + sd_now:+.2f} · weekly, {idx[0].date()} → {idx[-1].date()}</div>")
    H.append(f"<div class='warn'><b>No look-ahead in the band.</b> The ±{C.GAP_SD:g} sd lines are an <b>expanding</b> mean ± sd of the difference: at every week they use only the data up to that week (after a {EP_WARMUP}-week warm-up, so there are no lines and no episodes before then). "
             f"The earlier version used the whole-history mean and sd, which placed and flagged the 2016–17 crossings against a 2016–2026 baseline they could not have known. Correcting it drops a few early episodes and shrinks the forward edge (see the summary). The difference series itself (two 13-week correlations) and the forward returns (t+1 … t+h) were always trailing; only the band changed.</div>")
    H.append("<h2>What followed, by side (point-in-time episodes)</h2>")
    H.append("<table class='sum'><tr><th>side</th><th>episodes</th><th>SC100 next 13 wk, mean</th><th>positive</th><th>median</th><th>gold next 13 wk</th></tr>"
             f"<tr><td>above +1 sd</td><td>{au['n']}</td><td>{au['sc_mean'] * 100:+.1f}%</td><td>{au['sc_hit'] * 100:.0f}%</td><td>{au['sc_med'] * 100:+.1f}%</td><td>{au['gold_mean'] * 100:+.1f}%</td></tr>"
             f"<tr><td>below −1 sd</td><td>{al['n']}</td><td>{al['sc_mean'] * 100:+.1f}%</td><td>{al['sc_hit'] * 100:.0f}%</td><td>{al['sc_med'] * 100:+.1f}%</td><td>{al['gold_mean'] * 100:+.1f}%</td></tr>"
             f"<tr><td>any week (baseline)</td><td>—</td><td>{base_mean * 100:+.1f}%</td><td>{base_hit * 100:.0f}%</td><td>—</td><td>—</td></tr></table>")
    H.append(f"<p class='sub'>Above +{C.GAP_SD:g} sd = the indices tight with each other while gold moves against Smallcap (the sell-off signature); below −{C.GAP_SD:g} sd = the indices loose while Smallcap moves with gold. Read the <i>side</i> as a state; the difference's own return to the mean is mechanical (the 13-week window rolling off) and carries no signal. {au['n']} / {al['n']} episodes, overlapping windows, one decade — tendencies, not rules.</p>")
    H.append("<div class='box'><b>What an episode is.</b> The difference = the 13-week correlation of " + _nm(a) + " minus the 13-week correlation of " + _nm(b) + ", week by week. An episode starts in the first week the difference goes outside its own expanding ±" + f"{C.GAP_SD:g}" + " sd band. For each: the difference from " + str(BEFORE) + " weeks before to " + str(AFTER) + " after (with the band, which drifts slightly as it is re-estimated); its two components; and each leg's cumulative log return rebased to zero at the crossing week. Hover for values; click a legend entry to hide a line.</div>")
    H.append("<h2>All episodes at a glance</h2>")
    H.append(table_html)
    H.append("<h2>The episodes overlaid, by side</h2>")
    H.append("<p class='sub'>Each thin line is one episode; the thick line is their average. The spread of the thin lines is how alike the episodes really are. Middle panel: purple is " + _nm(a) + ", brown is " + _nm(b) + ".</p>")
    H.extend(overlays)
    H.append("<h2>One episode at a time</h2>")
    H.append(f"<p class='sub'>Every figure has the same three panels. <b>Left:</b> the difference from {BEFORE} weeks before to {AFTER} after, with its expanding mean (dotted) and ±{C.GAP_SD:g} sd band (red/blue dashed) as they stood each week; the triangle is the crossing. <b>Middle:</b> the two correlations that make the difference, {_nm(a)} (purple) and {_nm(b)} (brown). <b>Right:</b> each leg's cumulative log return, zero at the crossing week. x-axis is weeks from the crossing.</p>")
    H.append("<div class='toc'>" + "".join(toc) + "</div>")
    H.extend(figs_html)
    H.append("<a class='top' href='#top'>↑ top</a>")
    H.append("<p class='sub'>Built by make_episodes.py from the raw data (weekly W-FRI closes, log returns, 13-week rolling correlations). The band is point-in-time (expanding mean ± sd), the same one histcorr's B2 extreme follow-ups use.</p></body></html>")
    out = C.OUT / f"episodes_{a.replace('|', 'x')}_minus_{b.replace('|', 'x')}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(H), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB): {len(ev)} episodes ({n_up} above, {n_lo} below)")
    print(f"  above: SC100 +13w {au['sc_mean'] * 100:+.1f}% (hit {au['sc_hit'] * 100:.0f}%)   below: {al['sc_mean'] * 100:+.1f}% (hit {al['sc_hit'] * 100:.0f}%)   baseline {base_mean * 100:+.1f}%")
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    build(*args) if len(args) == 2 else build()
