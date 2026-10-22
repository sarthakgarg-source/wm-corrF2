"""
Interactive charts for the report (Plotly), built from outputs/tables/.

Every figure is a function returning a plotly Figure; `html(name)` renders it as a <div>
(the plotly library itself is embedded once by make_report.py, so the report works offline).
Click a legend entry to hide / show that trace; double-click to isolate it; drag to zoom;
double-click the plot to reset. The same design system as plots.py (static PNGs).
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config as C

T = C.TABLES
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
COL = {"N100": "#2a78d6", "MC100": "#eb6834", "SC100": "#1baf7a", "Micro250": "#4a3aa7", "GoldMCX": "#eda100", "VIX": "#e34948", "Liquid": "#898781"}
BUY, SELL = "#2a78d6", "#e34948"
FONT = "Inter, Helvetica, Arial, sans-serif"
CONFIG = {"responsive": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "toImageButtonOptions": {"format": "png", "scale": 2}}


def _csv(name, **kw):
    return pd.read_csv(T / f"{name}.csv", **kw)


PAIR_COL = {"N100|MC100": "#0b6e99", "N100|SC100": "#7a4fd1", "N100|Micro250": "#c2185b", "MC100|SC100": "#e07b00", "MC100|Micro250": "#5c7a1f", "SC100|Micro250": "#8c564b",
            "N100|GoldMCX": "#b8860b", "MC100|GoldMCX": "#d4a017", "SC100|GoldMCX": "#9a7b1a", "Micro250|GoldMCX": "#6f5a00"}
PAIR_DASH = {"N100|GoldMCX": "dash", "MC100|GoldMCX": "dot", "SC100|GoldMCX": "dashdot", "Micro250|GoldMCX": "longdash"}


def _pair_col(pair: str) -> str:
    """One colour per pair, the same in every chart (gold pairs in gold tones, told apart by their dash)."""
    if pair in PAIR_COL:
        return PAIR_COL[pair]
    a, b = pair.split("|")
    return COL[a] if "Gold" not in pair else COL[b if a == "GoldMCX" else a]


def _pair_line(pair: str, width: float = 1.3) -> dict:
    return {"color": _pair_col(pair), "width": width, "dash": PAIR_DASH.get(pair, "solid")}


def _pair_name(pair: str) -> str:
    return pair.replace("|", " × ")


PAIR_ORDER = [f"{a}|{b}" for a, b in itertools.combinations(C.ASSETS, 2)]


def _add_pairs(fig, x, df, row=None, col=None, width: float = 1.1, gold: str = "legendonly", showlegend: bool = True, fmt: str = "%{y:.2f}", mode: str = "lines", suffix: str = ""):
    """Add every pair column of `df` ('a|b' columns, in the study's pair order) as one line each. Gold pairs are dashed and, by default, off until clicked."""
    cols = [c for c in df.columns if "|" in c]
    for p in [p for p in PAIR_ORDER if p in cols] + [c for c in cols if c not in PAIR_ORDER]:
        is_gold = "Gold" in p
        fig.add_trace(go.Scatter(x=x, y=df[p], mode=mode, line=_pair_line(p, width), marker={"size": 4}, name=_pair_name(p) + suffix, legendgroup="pair_" + p, showlegend=showlegend,
                                 visible=gold if is_gold else True, hovertemplate=fmt + "<extra>" + _pair_name(p) + "</extra>"), row=row, col=col)


def _layout(fig: go.Figure, title: str, height: int, note: str | None = None, legend_h: bool = True, subtitle: str | None = None, **kw) -> go.Figure:
    """Title = the finding, in bold; `subtitle` = the method, in grey under it (Plotly's native subtitle, so nothing is clipped).
    The top margin grows with the subtitle and again when the figure has subplot titles, so the plot area keeps `height`."""
    has_panel_titles = any((a.yref == "paper") and ((a.y or 0) >= 0.999) for a in (fig.layout.annotations or ()))
    extra = 0
    lines_t, rest_t = [], title
    while len(rest_t) > 118:                               # a long finding wraps onto further lines (Plotly does not wrap titles itself)
        cut = -1
        for sep, lo in ((" — ", 60), ("; ", 60), (", ", 60), (" ", 40)):     # a clause break if there is one late enough, else the last space
            i = rest_t.rfind(sep, lo, 118)
            if i > 0:
                lines_t.append(rest_t[:i] + sep.rstrip()); rest_t = rest_t[i + len(sep):]; cut = i
                break
        if cut < 0:
            break
        extra += 20
    lines_t.append(rest_t)
    title = "<br>".join(lines_t)
    ttl = {"text": f"<b>{title}</b>", "x": 0.0, "xanchor": "left", "yanchor": "top", "y": 1.0, "pad": {"t": 10}, "font": {"size": 15}}
    if subtitle:
        lines, rest = [], subtitle
        while len(rest) > 165:                                   # the method line wraps too (Plotly wraps neither)
            cut = max(rest.rfind(". ", 60, 165), rest.rfind("; ", 60, 165), rest.rfind(" — ", 60, 165), rest.rfind(", ", 60, 165), rest.rfind(" ", 60, 165))
            if cut < 60:
                break
            lines.append(rest[:cut + 1].rstrip()); rest = rest[cut + 1:].lstrip()
        lines.append(rest)
        ttl["subtitle"] = {"text": "<br>".join(lines), "font": {"size": 12, "color": INK2}}
        extra += 22 + 16 * (len(lines) - 1) + (20 if has_panel_titles else 0)
    if fig.layout.updatemenus:                             # a menu row gets its own band between the subtitle and the panel titles
        extra += 44
        plot_h = height - 140
        for m in fig.layout.updatemenus:
            m.y = 1 + (84 if has_panel_titles else 64) / plot_h; m.yanchor = "top"
    fig.update_layout(title=ttl, height=height + extra, margin={"l": 60, "r": 30, "t": 70 + extra, "b": 70},
                      paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font={"family": FONT, "size": 12.5, "color": INK},
                      hovermode="x unified" if kw.pop("unified", True) else "closest",
                      legend={"orientation": "h", "yanchor": "top", "y": kw.pop("legend_y", -0.08), "x": 0, "font": {"size": 12}} if legend_h else {"font": {"size": 12}},
                      **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont={"color": MUTED})
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=AXIS, tickfont={"color": MUTED})
    if note:
        fig.update_layout(meta={"note": note})
    return fig


def html(name: str, **kw) -> str:
    fig = FIGS[name](**kw)
    suffix = "".join(f"_{str(v).replace('|', '_')}" for v in kw.values())
    note = (fig.layout.meta or {}).get("note") if fig.layout.meta else None
    out = fig.to_html(full_html=False, include_plotlyjs=False, config=CONFIG, div_id=f"fig_{name}{suffix}")
    return out + (f"<p class='fignote'>{note}</p>" if note else "")


def _shade(fig, idx, mask, row=None, col=None, color=SELL, opacity=0.08):
    """Shade the stretches where `mask` is True (vrect per stretch)."""
    m = pd.Series(mask.values, index=idx).fillna(False).astype(bool)
    on, start = False, None
    for d, v in m.items():
        if v and not on:
            on, start = True, d
        elif not v and on:
            fig.add_vrect(x0=start, x1=d, fillcolor=color, opacity=opacity, line_width=0, row=row, col=col); on = False
    if on:
        fig.add_vrect(x0=start, x1=m.index[-1], fillcolor=color, opacity=opacity, line_width=0, row=row, col=col)


# ============================================================ 1. corr vs signal ==
def corr_by_state():
    bs = _csv("1_corr_by_signal_state")
    bs = bs[bs["state"].isin(["both BUY", "disagree", "both SELL"])]
    pairs = list(dict.fromkeys(bs["pair"]))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Correlation of the pair, by what the two signals said", "Covariance of the pair (annualised) — the risk"), horizontal_spacing=0.16)
    for c, what in ((1, "corr"), (2, "cov_ann")):
        b = bs[bs["state"] == "both BUY"].set_index("pair").reindex(pairs)[what]
        d = bs[bs["state"] == "disagree"].set_index("pair").reindex(pairs)[what]
        s = bs[bs["state"] == "both SELL"].set_index("pair").reindex(pairs)[what]
        for p in pairs:
            fig.add_trace(go.Scatter(x=[b[p], s[p]], y=[p, p], mode="lines", line={"color": GRID, "width": 4}, showlegend=False, hoverinfo="skip"), row=1, col=c)
        fig.add_trace(go.Scatter(x=b, y=pairs, mode="markers", marker={"color": BUY, "size": 11}, name="both signals BUY", showlegend=c == 1, legendgroup="buy", hovertemplate="%{y}: %{x:.3f}<extra>both BUY</extra>"), row=1, col=c)
        fig.add_trace(go.Scatter(x=d, y=pairs, mode="markers", marker={"color": MUTED, "size": 9, "symbol": "diamond"}, name="signals disagree (one BUY, one SELL)", showlegend=c == 1, legendgroup="dis", hovertemplate="%{y}: %{x:.3f}<extra>disagree</extra>"), row=1, col=c)
        fig.add_trace(go.Scatter(x=s, y=pairs, mode="markers", marker={"color": SELL, "size": 11}, name="both signals SELL", showlegend=c == 1, legendgroup="sell", hovertemplate="%{y}: %{x:.3f}<extra>both SELL</extra>"), row=1, col=c)
    fig.update_yaxes(autorange="reversed", ticktext=[p.replace("|", " × ") for p in pairs], tickvals=pairs)
    fig.update_xaxes(range=[-0.4, 1.02], row=1, col=1)
    eqp = [p for p in pairs if "Gold" not in p]
    cb = bs[bs["state"] == "both BUY"].set_index("pair").loc[eqp, "corr"].mean(); cs = bs[bs["state"] == "both SELL"].set_index("pair").loc[eqp, "corr"].mean()
    vb = bs[bs["state"] == "both BUY"].set_index("pair").loc[eqp, "cov_ann"].mean(); vsl = bs[bs["state"] == "both SELL"].set_index("pair").loc[eqp, "cov_ann"].mean()
    dec = _csv("1_cov_decomposition").set_index("pair")
    ratio = dec.loc[eqp, "cov_ratio_sell_over_buy"].mean()        # the same number the report's tiles use: the average over index pairs of SELL-risk ÷ BUY-risk
    fig.add_annotation(x=cs, y=eqp[1], xref="x", yref="y", text=f"red is a little right of blue:<br>index pairs {cb:.2f} → {cs:.2f}", showarrow=True, arrowhead=2, arrowcolor=INK2, ax=-200, ay=-28, font={"size": 11, "color": INK2}, bgcolor="rgba(252,252,251,0.92)", bordercolor=GRID)
    fig.add_annotation(x=vsl, y=eqp[1], xref="x2", yref="y2", text=f"red is far right of blue:<br>the risk is {ratio:.1f}× larger in SELL", showarrow=True, arrowhead=2, arrowcolor=INK2, ax=-120, ay=70, font={"size": 11, "color": INK2}, bgcolor="rgba(252,252,251,0.92)", bordercolor=GRID)
    gp = [p for p in pairs if "Gold" in p]
    fig.add_annotation(x=bs[bs["state"] == "both SELL"].set_index("pair").loc[gp[0], "corr"], y=gp[0], xref="x", yref="y", text="gold: near zero<br>in every state", showarrow=True, arrowhead=2, arrowcolor=INK2, ax=130, ay=30, font={"size": 11, "color": INK2}, bgcolor="rgba(252,252,251,0.92)", bordercolor=GRID)
    return _layout(fig, f"When both signals say SELL, index pairs move together a little more ({cb:.2f} → {cs:.2f}) but carry {ratio:.1f}× the risk", 540, "Covariance = correlation × swing size of a × swing size of b. The left panel barely moves; the right panel moves a lot — SELL weeks are about bigger swings, not tighter coupling. Disagree weeks (one leg already SELL) sit between.", unified=False,
                   subtitle="Each pair's correlation (left) and covariance (right) in the weeks both supertrend signals were BUY (blue), one BUY one SELL (grey diamond), both SELL (red)")


def corr_heatmap():
    M = _csv("0_corr_matrix_full", index_col=0)
    M52 = _csv("0_corr_matrix_52w", index_col=0); M13 = _csv("0_corr_matrix_13w", index_col=0)
    labels = {"VIXchg": "VIX (change)", "Liquid": "Cash (liquid)"}
    names = [labels.get(c, c) for c in M.columns]
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Whole sample (2016 →)", "Last 52 weeks", "Last 13 weeks"), horizontal_spacing=0.11)
    for c_, (m, _) in enumerate(((M, "full"), (M52, "52w"), (M13, "13w")), start=1):
        z = m.values.copy()
        txt = [["" if np.isnan(v) else f"{v:.2f}" for v in row] for row in z]
        fig.add_trace(go.Heatmap(z=z, x=names, y=names, zmin=-1, zmax=1, colorscale=[[0, "#e34948"], [0.5, "#f3f3ef"], [1, "#2a78d6"]], text=txt, texttemplate="%{text}", textfont={"size": 11}, hoverongaps=False,
                                 showscale=c_ == 3, colorbar={"title": "corr", "thickness": 12}, hovertemplate="%{y} × %{x}: %{z:.2f}<extra></extra>"), row=1, col=c_)
        fig.update_yaxes(autorange="reversed", row=1, col=c_)
        fig.update_xaxes(tickangle=-35, row=1, col=c_)
    fig.update_layout(height=520)
    idx = [a for a in C.EQUITY if a in M.index]
    off = M.loc[idx, idx].where(~np.eye(len(idx), dtype=bool)).stack()
    gold = M.loc["GoldMCX", idx] if "GoldMCX" in M.index else pd.Series(dtype=float)
    return _layout(fig, f"The four indices are one block ({off.min():.2f}–{off.max():.2f}); gold is unrelated to all of them ({gold.min():+.2f} to {gold.max():+.2f})", 540, "Blue = move together, red = move opposite, grey = unrelated. VIX (change) and cash are reference rows, not assets of the study. A blank cell = too few weeks in that window (the VIX file ends early).", unified=False, legend_h=False,
                   subtitle="The correlation of weekly returns between every two series — over the whole sample, the last 52 weeks and the last 13 weeks")


def corr_rolling():
    PC = _csv("6_pair_corr13", index_col=0, parse_dates=True)
    gr = _csv("0_gold_corr_rank", index_col=0)
    sh = gr.loc["SC100", "share_above_N100"] if "SC100" in gr.index and "share_above_N100" in gr.columns else np.nan
    nw = int(gr.loc["SC100", "weeks_above_N100"]) if "SC100" in gr.index else 0; nc = int(gr.loc["SC100", "weeks_compared"]) if "SC100" in gr.index else 0
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.46, 0.32, 0.22], vertical_spacing=0.07,
                        subplot_titles=(f"Index pairs: {C.CORR_WINDOW}-week rolling correlation", "Gold vs each index", f"SC100×Gold minus N100×Gold — above zero in {nw} of {nc} weeks ({sh:.0%}): gold follows Smallcap no more than it follows Nifty"))
    for p in PC.columns:
        gold = "Gold" in p
        a, b = p.split("|")
        fig.add_trace(go.Scatter(x=PC.index, y=PC[p], mode="lines", line={"color": _pair_col(p) if not gold else COL[a if a != "GoldMCX" else b], "width": 1.3, "dash": "solid" if (a == "N100" or gold) else "dot"}, name=p.replace("|", " × "), hovertemplate="%{y:.2f}<extra>" + p + "</extra>"), row=2 if gold else 1, col=1)
    if "SC100|GoldMCX" in PC.columns and "N100|GoldMCX" in PC.columns:
        d = PC["SC100|GoldMCX"] - PC["N100|GoldMCX"]
        fig.add_trace(go.Scatter(x=d.index, y=d.clip(lower=0), mode="lines", line={"width": 0}, fill="tozeroy", fillcolor="rgba(27,175,122,0.35)", name="SC100×Gold above N100×Gold", hovertemplate="%{y:+.2f}<extra>SC100×Gold − N100×Gold</extra>"), row=3, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d.clip(upper=0), mode="lines", line={"width": 0}, fill="tozeroy", fillcolor="rgba(227,73,72,0.35)", name="SC100×Gold below N100×Gold", hovertemplate="%{y:+.2f}<extra>SC100×Gold − N100×Gold</extra>"), row=3, col=1)
        fig.add_trace(go.Scatter(x=d.index, y=d, mode="lines", line={"color": INK, "width": 0.9}, name="the difference", hovertemplate="%{y:+.2f}<extra>difference</extra>"), row=3, col=1)
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=3, col=1)
    fig.update_yaxes(range=[-0.2, 1.02], row=1, col=1); fig.update_yaxes(range=[-0.7, 0.7], row=2, col=1); fig.update_yaxes(range=[-0.8, 0.8], row=3, col=1)
    fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=2, col=1)
    return _layout(fig, f"The index pairs live between about 0.6 and 1.0 and dip together in the decorrelation episodes; gold wanders around zero with every index — Smallcap's gold correlation is above Nifty's only {sh:.0%} of the time", 900, "Top: solid = pairs with N100, dotted = the other index pairs. Middle: each index against gold. Bottom: the gap between Smallcap's and Nifty's correlation with gold (green = Smallcap's higher). Click a pair in the legend to isolate it (double-click).", legend_y=-0.05,
                   subtitle="Each pair's 13-week rolling correlation, week by week")


def gold_when_decoupled():
    """What gold's correlation with each index does when the indices stop moving together: by group-correlation quintile, and around the start of each decorrelation episode."""
    gq = _csv("0_gold_by_group_corr"); gpth = _csv("0_gold_around_decoupling"); gper = _csv("0_gold_in_episodes")
    gps = [p for p in PAIR_ORDER if p in set(gq["pair"])]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("by how coupled the indices were (Q1 least … Q5 most)", f"around the start of the {gper['episode'].nunique()} decorrelation episodes (week 0)"), horizontal_spacing=0.08)
    for p in gps:
        g = gq[gq["pair"] == p].set_index("bucket")
        fig.add_trace(go.Scatter(x=g.index, y=g["mean_corr"], mode="lines+markers", line=_pair_line(p, 2.0), marker={"size": 6}, name=_pair_name(p), legendgroup="pair_" + p, customdata=np.c_[g["G_lo"], g["G_hi"], g["weeks"]], hovertemplate="%{x}: corr %{y:+.2f} (G %{customdata[0]:.2f}–%{customdata[1]:.2f}, n %{customdata[2]})<extra>" + _pair_name(p) + "</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=g.index, y=g["mean_abs_corr"], mode="lines+markers", line={"color": _pair_col(p), "width": 1.0, "dash": "dot"}, marker={"size": 4, "symbol": "diamond"}, name=_pair_name(p) + " |corr|", legendgroup="pair_" + p + "abs", hovertemplate="%{x}: |corr| %{y:.2f}<extra>" + _pair_name(p) + " |corr|</extra>"), row=1, col=1)
        h = gpth[gpth["pair"] == p]
        fig.add_trace(go.Scatter(x=h["k"], y=h["mean_corr"], mode="lines+markers", line=_pair_line(p, 2.0), marker={"size": 4}, name=_pair_name(p), legendgroup="pair_" + p, showlegend=False, hovertemplate="k=%{x}: corr %{y:+.2f}<extra>" + _pair_name(p) + "</extra>"), row=1, col=2)
        fig.add_trace(go.Scatter(x=h["k"], y=h["mean_abs_corr"], mode="lines", line={"color": _pair_col(p), "width": 1.0, "dash": "dot"}, name=_pair_name(p) + " |corr|", legendgroup="pair_" + p + "abs", showlegend=False, hovertemplate="k=%{x}: |corr| %{y:.2f}<extra>" + _pair_name(p) + " |corr|</extra>"), row=1, col=2)
    for c in (1, 2):
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=1, col=c)
    fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}, row=1, col=2)
    fig.update_xaxes(title_text="weeks from the start of the decorrelation episode", dtick=2, row=1, col=2)
    fig.update_yaxes(range=[-0.5, 0.6])
    q1 = gq[gq["bucket"] == "Q1"]["mean_abs_corr"].mean(); q5 = gq[gq["bucket"] == "Q5"]["mean_abs_corr"].mean()
    ab, ad = gper["abs_before"].mean(), gper["abs_during"].mean()
    pre = gpth[gpth["k"].between(-8, -1)].groupby("pair")["mean_corr"].mean().mean(); post = gpth[gpth["k"].between(0, 3)].groupby("pair")["mean_corr"].mean().mean()
    return _layout(fig, f"When the indices decouple, gold does not couple to any of them: |corr| with gold is {q1:.2f} in the least-coupled weeks vs {q5:.2f} in the most, and {ab:.2f} before a decorrelation episode vs {ad:.2f} inside it; the sign merely drifts from {pre:+.2f} to {post:+.2f}", 520,
                   "Solid: the average correlation (sign kept). Dotted: the average absolute correlation — the right measure for 'does gold start moving with or against an index', since a swing to −0.4 is as much coupling as +0.4. If gold took over when the indices let go of each other, the dotted lines would rise to the left (Q1) and after week 0 (right). They stay near 0.2–0.3; the small lift 10+ weeks after week 0 is 9 episodes and the tail of their sell-offs, not the decoupling itself.", unified=False, legend_y=-0.2,
                   subtitle="Left: each gold pair's 13-week correlation averaged inside quintiles of the group correlation G. Right: the same series lined up at the week each decorrelation episode began, averaged over the episodes")


def corr_around_flip():
    af = _csv("1_corr_around_flip")
    refs = [a for a in C.ASSETS if a in set(af["ref"])]
    exc = f"mean_corr_ex{C.FLIP_EXCLUDE_YEAR}"
    stats_ = [("mean_corr", "mean over the flips", "mean"), ("median_corr", "median over the flips", "median"), (exc, f"mean without the {C.FLIP_EXCLUDE_YEAR} flips", f"ex-{C.FLIP_EXCLUDE_YEAR}")]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("around the asset's flip to SELL", "around the asset's flip to BUY"), shared_yaxes=True, horizontal_spacing=0.06)
    vis = {s: [] for s, _, _ in stats_}
    for c, ev in ((1, "flip to SELL"), (2, "flip to BUY")):
        for ref in refs:
            sub = af[(af["ref"] == ref) & (af["event"] == ev)]
            n = int(sub["n_events"].max()); nx = int(sub[f"n_events_ex{C.FLIP_EXCLUDE_YEAR}"].max())
            for s, _, _ in stats_:
                nn = nx if s == exc else n
                fig.add_trace(go.Scatter(x=sub["weeks_from_flip"], y=sub[s], mode="lines+markers", line={"color": COL[ref], "width": 2.4 if ref == C.BENCHMARK else 1.6}, marker={"size": 5},
                                         name=f"{ref}'s flips (n {nn}) — group correlation G", legendgroup=ref, showlegend=c == 1, visible=(s == "mean_corr"), hovertemplate="k=%{x}: G %{y:.3f}<extra>" + ref + "</extra>"), row=1, col=c)
                for s2 in vis:
                    vis[s2].append(s2 == s)
            fig.add_trace(go.Scatter(x=sub["weeks_from_flip"], y=sub["ref_own_corr"], mode="lines", line={"color": COL[ref], "width": 1.2, "dash": "dot"},
                                     name=f"{ref}'s own correlation with the other indices (mean)", legendgroup=ref + "own", showlegend=c == 1, visible="legendonly", hovertemplate="k=%{x}: own %{y:.3f}<extra>" + ref + " own</extra>"), row=1, col=c)
            for s2 in vis:
                vis[s2].append("legendonly")
        fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}, row=1, col=c)
        fig.add_vrect(x0=-8.5, x1=0, fillcolor=GRID, opacity=0.25, line_width=0, row=1, col=c)
    n100 = af[(af["ref"] == C.BENCHMARK) & (af["event"] == "flip to SELL")].set_index("weeks_from_flip")
    fig.add_annotation(x=-4, y=0.93, xref="x", yref="y", text="BEFORE the flip: flat — no warning", showarrow=False, font={"size": 11.5, "color": INK2})
    fig.add_annotation(x=4.5, y=0.93, xref="x", yref="y", text="AFTER: climbs ~0.1 over two months", showarrow=False, font={"size": 11.5, "color": INK2})
    fig.add_annotation(x=0, y=0.93, xref="x2", yref="y2", text="BUY flips come at the peak of coupling, then it loosens", showarrow=False, font={"size": 11.5, "color": INK2})
    base = list(fig.layout.annotations)
    def ann(s, label):
        return base + [go.layout.Annotation(x=0, y=n100.loc[0, s], xref="x", yref="y", text=f"N100 flips to SELL ({label}):<br>{n100.loc[-4, s]:.2f} four weeks before → {n100.loc[0, s]:.2f} here → {n100.loc[8, s]:.2f} eight weeks after", showarrow=True, arrowhead=2, arrowcolor=INK2, ax=-10, ay=120, font={"size": 11, "color": INK2}, bgcolor="rgba(252,252,251,0.92)", bordercolor=GRID)]
    fig.update_layout(annotations=ann("mean_corr", "mean"))
    fig.update_layout(updatemenus=[{"type": "dropdown", "x": 1.0, "xanchor": "right", "y": 1.13, "yanchor": "top", "showactive": True, "bgcolor": SURFACE, "bordercolor": GRID, "font": {"size": 12},
                                    "buttons": [{"label": label, "method": "update", "args": [{"visible": vis[s]}, {"annotations": ann(s, short)}]} for s, label, short in stats_]}])
    fig.update_xaxes(title_text="weeks from the flip (0 = the flip week)", dtick=2)
    fig.update_yaxes(range=[0.65, 0.96])
    m, md, mx = n100["mean_corr"], n100["median_corr"], n100[exc]
    return _layout(fig, f"Correlation rises AFTER a SELL flip, not before — for every asset's flips, and not because of {C.FLIP_EXCLUDE_YEAR}: N100's step at the flip is {m.loc[-4]:.2f} → {m.loc[0]:.2f} on the mean, {md.loc[-4]:.2f} → {md.loc[0]:.2f} on the median, {mx.loc[-4]:.2f} → {mx.loc[0]:.2f} without the {C.FLIP_EXCLUDE_YEAR} flips", 580,
                   f"Grey band = the 8 weeks before the flip. Solid lines: the group correlation around that asset's flips. The menu (top right) switches the statistic: the mean over the flips, the median (immune to one extreme flip), or the mean with the {C.FLIP_EXCLUDE_YEAR} flips (the COVID crash, when correlation went to 1) left out. Dotted (off by default — click the legend): the flipping asset's own correlation with the other indices. Gold's flips (yellow) move nothing.", legend_y=-0.18,
                   subtitle="The average correlation among the four indices from 8 weeks before to 8 weeks after each asset's supertrend flips, averaged over that asset's flips — mean, median, or mean without one year's flips")


def corr_around_flip_pairs():
    """Every pair's correlation around one asset's flips; the menu picks the flipping asset (N100 first)."""
    af = _csv("1_corr_around_flip_pairs")
    refs = [a for a in C.ASSETS if a in set(af["ref"])]
    prs = list(dict.fromkeys(af["pair"]))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("around the asset's flip to SELL", "around the asset's flip to BUY"), shared_yaxes=True, horizontal_spacing=0.06)
    vis_by_ref = {ref: [] for ref in refs}
    for ref in refs:
        for c, ev in ((1, "flip to SELL"), (2, "flip to BUY")):
            for p in prs:
                sub = af[(af["ref"] == ref) & (af["event"] == ev) & (af["pair"] == p)]
                n = int(sub["n_events"].max()) if len(sub) else 0
                is_gold = "Gold" in p
                fig.add_trace(go.Scatter(x=sub["weeks_from_flip"], y=sub["mean_corr"], mode="lines+markers", line=_pair_line(p, 1.4 if is_gold else 1.9), marker={"size": 4},
                                         name=_pair_name(p) + f" (n {n})", legendgroup="pair_" + p, showlegend=c == 1, visible=(("legendonly" if is_gold else True) if ref == refs[0] else False),
                                         hovertemplate="k=%{x}: %{y:.3f}<extra>" + _pair_name(p) + "</extra>"), row=1, col=c)
                for r_ in refs:
                    vis_by_ref[r_].append((("legendonly" if is_gold else True) if r_ == ref else False))
    for c in (1, 2):
        fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}, row=1, col=c)
        fig.add_vrect(x0=-8.5, x1=0, fillcolor=GRID, opacity=0.25, line_width=0, row=1, col=c)
    fig.update_layout(updatemenus=[{"type": "dropdown", "x": 1.0, "xanchor": "right", "y": 1.13, "yanchor": "top", "showactive": True, "bgcolor": SURFACE, "bordercolor": GRID, "font": {"size": 12},
                                    "buttons": [{"label": f"{ref}'s flips", "method": "update", "args": [{"visible": vis_by_ref[ref]}]} for ref in refs]},
                                   {"type": "buttons", "direction": "right", "x": 0.0, "xanchor": "left", "y": 1.13, "yanchor": "top", "showactive": True, "bgcolor": SURFACE, "bordercolor": GRID, "font": {"size": 11},
                                    "buttons": [{"label": "y-axis 0.5 – 1 (index pairs)", "method": "relayout", "args": [{"yaxis.range": [0.5, 1.0]}]},
                                                {"label": "y-axis −0.5 – 1 (gold pairs too)", "method": "relayout", "args": [{"yaxis.range": [-0.5, 1.0]}]}]}])
    fig.update_xaxes(title_text="weeks from the flip (0 = the flip week)", dtick=2)
    fig.update_yaxes(range=[0.5, 1.0])
    n100 = af[(af["ref"] == C.BENCHMARK) & (af["event"] == "flip to SELL")]
    nm = n100[n100["pair"] == "N100|MC100"].set_index("weeks_from_flip")["mean_corr"]; ms = n100[n100["pair"] == "MC100|SC100"].set_index("weeks_from_flip")["mean_corr"]
    title = f"Around N100's SELL flips it is the N100 pairs that climb (N100×MC100 {nm.loc[-4]:.2f} → {nm.loc[8]:.2f}); MC100×SC100 was already coupled ({ms.loc[-4]:.2f} → {ms.loc[8]:.2f})" if len(nm) and len(ms) else "Every pair's correlation around each asset's flips"
    return _layout(fig, title, 560, "Grey band = the 8 weeks before the flip. The y-axis starts at 0.5 so the climb is visible; the buttons at the top left widen it to −0.5 for the gold pairs, which are off by default — click them in the legend. The menu at the top right picks whose flips the lines are drawn around.", legend_y=-0.18, unified=False,
                   subtitle="Each pair's 13-week correlation from 8 weeks before to 8 weeks after the chosen asset's supertrend flips, averaged over that asset's flips")


def corr_by_breadth():
    bb = _csv("1_corr_by_breadth", index_col=0); bp = _csv("1_corr_by_breadth_pairs", index_col=0)
    x = [f"{b}<br>{int(bp.loc[b, 'weeks'])} wks" for b in bp.index]
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    cols = [c for c in bp.columns if "|" in c]
    for p in [p for p in PAIR_ORDER if p in cols]:                        # index pairs on the left axis (0.5–1), gold pairs on the right axis (their own scale around zero)
        is_gold = "Gold" in p
        fig.add_trace(go.Scatter(x=x, y=bp[p], mode="lines+markers", line=_pair_line(p, 2.2 if p == "SC100|GoldMCX" else 1.6), marker={"size": 4}, name=_pair_name(p) + (" (right axis)" if is_gold else ""), legendgroup="pair_" + p,
                                 visible=(True if (not is_gold or p == "SC100|GoldMCX") else "legendonly"), hovertemplate="%{y:.2f}<extra>" + _pair_name(p) + "</extra>"), secondary_y=is_gold)
    fig.add_trace(go.Scatter(x=x, y=bb["mean"].reindex(bp.index), mode="lines+markers", line={"color": INK, "width": 2.6}, marker={"size": 7}, name="group correlation G", hovertemplate="%{y:.2f}<extra>G</extra>"), secondary_y=False)
    fig.update_xaxes(title_text="how many of the four index signals were in SELL that week")
    fig.update_yaxes(range=[0.5, 1.0], title_text="index pairs and G", secondary_y=False)
    fig.update_yaxes(range=[-0.5, 0.5], title_text="gold pairs (dashed) — own scale", secondary_y=True, showgrid=False, zeroline=True, zerolinecolor=AXIS, tickfont={"color": PAIR_COL["SC100|GoldMCX"]})
    eqp = [p for p in bp.columns if "|" in p and "Gold" not in p]
    lo, hi = bp.loc["none in SELL", eqp].mean(), bp.loc["all in SELL", eqp].mean()
    return _layout(fig, f"The more signals are in SELL, the more every index pair moves together (average {lo:.2f} with none in SELL → {hi:.2f} with all four); gold pairs stay near zero", 480,
                   "The black line is the group correlation G (the average of the six index pairs); the x-axis labels carry the number of weeks in each group. The left axis runs 0.5–1 so the climb is visible; the gold pairs sit near zero, so SC100 × Gold (drawn by default, for contrast) and the other gold pairs (click to show) are on the right-hand axis, −0.5 to +0.5.", unified=False, legend_y=-0.3,
                   subtitle="Each pair's average 13-week correlation in the weeks grouped by how many of the four index supertrends were in SELL")


def corr_timeseries():
    ts = _csv("1_corr_timeseries", index_col=0, parse_dates=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06,
                        subplot_titles=(f"{C.CORR_WINDOW}-week average correlation among the four indices — shaded where N100's signal was in SELL", "How many of the four index signals were in SELL"))
    _add_pairs(fig, ts.index, ts, row=1, col=1, width=0.9)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["mean_corr_13w"], mode="lines", line={"color": INK, "width": 2.2}, name="group correlation G (13w)", hovertemplate="%{y:.2f}<extra>G</extra>"), row=1, col=1)
    _shade(fig, ts.index, ts["N100_state"] == -1, row=1, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["n_sell"], mode="lines", line={"color": SELL, "width": 0, "shape": "hv"}, fill="tozeroy", fillcolor="rgba(227,73,72,0.35)", name="signals in SELL", hovertemplate="%{y}"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["n_live"], mode="lines", line={"color": MUTED, "width": 1, "dash": "dash"}, name="signals that exist", hovertemplate="%{y}"), row=2, col=1)
    fig.update_yaxes(range=[-0.5, 1.02], row=1, col=1); fig.update_yaxes(range=[0, 4.3], row=2, col=1)
    return _layout(fig, "The correlation lines are highest inside the shaded SELL stretches — the coupling and the signal rise together, and the coupling does not lead", 680,
                   "Shaded = weeks N100's supertrend was in SELL. Black = the group correlation G; the thin lines are the six index pairs (gold pairs: click to show). The red area below counts how many of the four index signals were in SELL that week.", legend_y=-0.1,
                   subtitle="Top: the 13-week group correlation and every pair's 13-week correlation, week by week. Bottom: the number of index signals in SELL.")


# ============================================================ 2. history ==
def by_year():
    y = _csv("2_by_year", index_col=0)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.52, 0.48], vertical_spacing=0.08,
                        subplot_titles=("Average 13-week correlation by year — the group (bars, with its lowest and highest week) and every pair (lines)", "Return of each asset in that year (price index, no dividends)"))
    fig.add_trace(go.Bar(x=y.index.astype(str), y=y["mean_corr"], marker_color="rgba(42,120,214,0.35)", name="group correlation G (year average)",
                         error_y={"type": "data", "symmetric": False, "array": y["max_corr"] - y["mean_corr"], "arrayminus": y["mean_corr"] - y["min_corr"], "color": INK2, "thickness": 1.2, "width": 4}, hovertemplate="%{x}: %{y:.3f}<extra>G</extra>"), row=1, col=1)
    yp = _csv("2_by_year_pairs", index_col=0)
    _add_pairs(fig, yp.index.astype(str), yp, row=1, col=1, width=1.5, mode="lines+markers")
    for c in [c for c in y.columns if c.startswith("ret_")]:
        lab = c.replace("ret_", "")
        fig.add_trace(go.Bar(x=y.index.astype(str), y=y[c], marker_color=COL[lab], name=lab, hovertemplate="%{x}: %{y:.1%}<extra>" + lab + "</extra>"), row=2, col=1)
    fig.update_yaxes(range=[-0.3, 1.05], row=1, col=1); fig.update_yaxes(tickformat=".0%", row=2, col=1)
    fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=1, col=1)
    fig.update_layout(barmode="group")
    hi = y["mean_corr"].idxmax()
    return _layout(fig, f"Every year averages {y['mean_corr'].min():.2f}–{y['mean_corr'].max():.2f}: the block never loosens for long, and the most-correlated year ({hi}) was not a bad year", 820,
                   "A year averages the sell-off and the recovery together, so by-year numbers blur what the quintile chart separates. Bars = the group correlation G with its low–high range; lines = each index pair's own year average (gold pairs: click to show). The last year is partial.", legend_y=-0.08,
                   subtitle="Top: the average 13-week correlation in each calendar year — the group (bars, with the lowest and highest week) and every pair (lines). Bottom: each asset's return in that year.")


def quintiles():
    lg = _csv("2_corr_quintile_returns_long")
    buckets = [f"Q{i + 1}" for i in range(C.N_BUCKETS)]
    assets = [a for a in C.ASSETS if a in set(lg["asset"])]
    rng = lg[lg["asset"] == assets[0]].set_index("bucket")
    labels = [f"{b}<br>corr {rng.loc[b, 'corr_lo']:.2f}–{rng.loc[b, 'corr_hi']:.2f}" for b in buckets]
    F = C.QUINT_FWD; nx = f"next_{F}w_mean"; nh = f"next_{F}w_hit"; nn = f"n_next_{F}w"
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Return over the SAME quarter the correlation was measured on", f"Return over the NEXT {F} weeks (t+1 … t+{F})"), horizontal_spacing=0.07)
    for c, what in ((1, "same_quarter_mean"), (2, nx)):
        for a in assets:
            s = lg[lg["asset"] == a].set_index("bucket").reindex(buckets)
            cd = np.c_[s["weeks"], s[nn], s[nh], s[f"next_{F}w_median"]] if c == 2 else np.c_[s["weeks"], s["weeks"], s["same_quarter_hit"], s["same_quarter_mean"]]
            fig.add_trace(go.Bar(x=labels, y=s[what].values, marker_color=COL[a], name=a, legendgroup=a, showlegend=c == 1, customdata=cd,
                                 hovertemplate="%{x}: %{y:.1%} (positive %{customdata[2]:.0%} of n %{customdata[1]}" + (", median %{customdata[3]:.1%})" if c == 2 else ")") + "<extra>" + a + "</extra>"), row=1, col=c)
    fig.update_yaxes(tickformat=".0%", row=1, col=1); fig.update_yaxes(tickformat=".1%", row=1, col=2); fig.update_layout(barmode="group")
    q = lg[lg["asset"] == "N100"].set_index("bucket")
    fig.add_annotation(x=labels[-1], y=q.loc["Q5", "same_quarter_mean"], xref="x", yref="y", text=f"most-correlated weeks:<br>the market had been FALLING<br>(N100 {q.loc['Q5', 'same_quarter_mean']:+.1%} that quarter)", showarrow=True, arrowhead=2, arrowcolor=INK2, ax=-50, ay=-220, font={"size": 11, "color": INK2}, bgcolor="rgba(252,252,251,0.92)", bordercolor=GRID)
    fig.add_annotation(x=labels[-1], y=q.loc["Q5", nx], xref="x2", yref="y2", text=f"…and the {F} weeks AFTER them<br>were the best (N100 {q.loc['Q5', nx]:+.1%},<br>positive {q.loc['Q5', nh]:.0%} of the time)", showarrow=True, arrowhead=2, arrowcolor=INK2, ax=-130, ay=-60, font={"size": 11, "color": INK2}, bgcolor="rgba(252,252,251,0.92)", bordercolor=GRID)
    return _layout(fig, f"High correlation is a sell-off in progress — and the {F} weeks after it have been the best (N100 {q.loc['Q5', nx]:+.1%}, positive {q.loc['Q5', nh]:.0%} of the time; after the least-correlated weeks {q.loc['Q1', nx]:+.1%})", 560,
                   f"Left: what was happening while correlation sat at that level (the correlation and the return are measured on the same weeks). Right: what came over the next {F} weeks — the horizon asked for in the 2026-10-19 review; the next-quarter (13-week) numbers stay in the table. Hover gives the number of weeks, the share positive and the median.", unified=False,
                   subtitle=f"All weeks sorted by the 13-week correlation among the indices into five equal groups (Q1 least coupled … Q5 most); each asset's average return over the same quarter (left) and the next {F} weeks (right)")


def quintiles_pairs():
    """The quintile study for every pair: one panel per pair, the weeks cut by THAT pair's correlation, each leg's return over the next quarter (bars) and the same quarter (dotted)."""
    lg = _csv("2_corr_quintile_returns_pairs")
    F = C.QUINT_FWD; nx = f"next_{F}w_mean"; nh = f"next_{F}w_hit"; nn = f"n_next_{F}w"
    buckets = [f"Q{i + 1}" for i in range(C.N_BUCKETS)]
    prs = list(dict.fromkeys(lg["pair"])); prs = [p for p in prs if "Gold" not in p] + [p for p in prs if "Gold" in p]      # index pairs first, gold pairs last
    ncol = 5; nrow = int(np.ceil(len(prs) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, shared_yaxes=True, subplot_titles=[_pair_name(p) for p in prs], horizontal_spacing=0.03, vertical_spacing=0.16)
    seen = set()
    for i, p in enumerate(prs):
        r_, c_ = i // ncol + 1, i % ncol + 1
        a, b = p.split("|")
        g = lg[lg["pair"] == p]
        rng = g[g["leg"] == a].set_index("bucket")
        labels = [f"{bk}<br>{rng.loc[bk, 'corr_lo']:.2f}–{rng.loc[bk, 'corr_hi']:.2f}" if bk in rng.index else bk for bk in buckets]
        for leg in (a, b):
            s = g[g["leg"] == leg].set_index("bucket").reindex(buckets)
            fig.add_trace(go.Bar(x=labels, y=s[nx].values, marker_color=COL[leg], name=f"{leg} — next {F} weeks", legendgroup=leg, showlegend=leg not in seen, customdata=np.c_[s[nn], s[nh]],
                                 hovertemplate="%{x}: %{y:.1%} (positive %{customdata[1]:.0%} of n %{customdata[0]})<extra>" + leg + f" next {F} weeks</extra>"), row=r_, col=c_)
            fig.add_trace(go.Scatter(x=labels, y=s["same_quarter_mean"].values, mode="lines+markers", line={"color": COL[leg], "width": 1.2, "dash": "dot"}, marker={"size": 5, "symbol": "diamond"}, name=f"{leg} — same quarter", legendgroup=leg + "same", showlegend=leg not in seen,
                                     hovertemplate="%{x}: %{y:.1%}<extra>" + leg + " same quarter</extra>"), row=r_, col=c_)
            seen.add(leg)
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=r_, col=c_)
        fig.update_xaxes(tickfont={"size": 9}, row=r_, col=c_)
    fig.update_yaxes(tickformat=".0%"); fig.update_layout(barmode="group")
    # the finding: over the index pairs, the next-quarter return of the legs in Q5 vs Q1
    eq = lg[~lg["pair"].str.contains("Gold")]
    q5 = eq[eq["bucket"] == "Q5"][nx].mean(); q1 = eq[eq["bucket"] == "Q1"][nx].mean()
    s5 = eq[eq["bucket"] == "Q5"]["same_quarter_mean"].mean(); s1 = eq[eq["bucket"] == "Q1"]["same_quarter_mean"].mean()
    return _layout(fig, f"Pair by pair the same shape: the most-correlated weeks end a fall (legs {s5:+.1%} that quarter) and precede better {F} weeks ({q5:+.1%}, vs {q1:+.1%} after the least-correlated)", 300 * nrow + 120,
                   f"Each panel sorts the weeks by that pair's own 13-week correlation (Q1 least coupled … Q5 most; the range under each label). Bars: each leg's average return over the following {F} weeks (hover: n and the share positive). Dotted: over the same quarter (the least-correlated weeks sit in rallies: {s1:+.1%}). Index pairs first, gold pairs last.", unified=False, legend_y=-0.12,
                   subtitle=f"The quintile study repeated for every pair, with the two legs of the pair as the assets; the forward return is the next {F} weeks")


def timeline():
    tl = _csv("2_timeline", index_col=0, parse_dates=True)
    has_vix = "VIX" in tl.columns and tl["VIX"].notna().any()
    n = 3 if has_vix else 2
    titles = ["13-week correlation — the group (black) and every pair (dashed = 0.90)", "Trailing 13-week return"] + (["India VIX level (file ends " + str(tl['VIX'].dropna().index[-1].date()) + ")"] if has_vix else [])
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.06, subplot_titles=titles)
    _add_pairs(fig, tl.index, tl, row=1, col=1, width=0.9)
    fig.add_trace(go.Scatter(x=tl.index, y=tl["mean_corr_13w"], mode="lines", line={"color": INK, "width": 2.2}, name="group correlation G", hovertemplate="%{y:.2f}<extra>G</extra>"), row=1, col=1)
    fig.add_hline(y=0.9, line={"color": SELL, "dash": "dash", "width": 1}, row=1, col=1)
    for a in ("N100", "SC100"):
        fig.add_trace(go.Scatter(x=tl.index, y=tl[f"{a}_ret_13w"], mode="lines", line={"color": COL[a], "width": 1.5}, name=f"{a} 13w return", hovertemplate="%{y:.1%}"), row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=2, col=1); fig.update_yaxes(range=[-0.5, 1.02], row=1, col=1)
    if has_vix:
        fig.add_trace(go.Scatter(x=tl.index, y=tl["VIX"], mode="lines", line={"color": COL["VIX"], "width": 1.4}, fill="tozeroy", fillcolor="rgba(227,73,72,0.15)", name="India VIX", hovertemplate="%{y:.1f}"), row=3, col=1)
    return _layout(fig, "Every spike of the correlation line sits on a trough of the return line" + (" and a spike of the VIX" if has_vix else "") + " — three views of the same sell-offs", 260 * n + 120,
                   "Read the panels vertically: pick a correlation peak and look straight down. Black = the group correlation G; thin lines = the six index pairs (gold pairs: click to show).", legend_y=-0.06,
                   subtitle="Top: the 13-week group correlation and every pair's (dashed = 0.90). Middle: the trailing 13-week return of N100 and SC100." + (" Bottom: the India VIX level." if has_vix else ""))


# ============================================================ 3. correlation breaks ==
def cb_series():
    S = _csv("3_corr_series", index_col=0, parse_dates=True)
    ep = _csv("3_corr_episodes", parse_dates=["start", "end", "G_min_date"])
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, row_heights=[0.3, 0.3, 0.26, 0.14], vertical_spacing=0.05,
                        subplot_titles=("Each index's 13-week correlation with the other indices, and the group average — shaded: decorrelation episodes",
                                        f"The same, as z-scores against each series' own {C.CB_NORM}-week norm — dashed: the break line (z = {C.CB_Z:.0f})",
                                        "Every pair on its own: the pair's 13-week correlation (gold pairs: click to show)",
                                        "Gold's 13-week correlation with the four indices (reference; not ranked)"))
    for a in [x for x in C.EQUITY if f"c_{x}" in S.columns]:
        fig.add_trace(go.Scatter(x=S.index, y=S[f"c_{a}"], mode="lines", line={"color": COL[a], "width": 1.4}, name=a, legendgroup=a, hovertemplate="%{y:.2f}"), row=1, col=1)
        fig.add_trace(go.Scatter(x=S.index, y=S[f"z_c_{a}"], mode="lines", line={"color": COL[a], "width": 1.2}, name=a, legendgroup=a, showlegend=False, hovertemplate="%{y:+.1f}"), row=2, col=1)
    fig.add_trace(go.Scatter(x=S.index, y=S["G"], mode="lines", line={"color": INK, "width": 2.2}, name="group (all pairs)", legendgroup="G", hovertemplate="%{y:.2f}"), row=1, col=1)
    fig.add_trace(go.Scatter(x=S.index, y=S["z_G"], mode="lines", line={"color": INK, "width": 2.0}, name="group", legendgroup="G", showlegend=False, hovertemplate="%{y:+.1f}"), row=2, col=1)
    pair_cols = [c for c in S.columns if "|" in c and not c.startswith("z_")]
    _add_pairs(fig, S.index, S[pair_cols], row=3, col=1, width=1.1)
    for _, e in ep.iterrows():
        for r_ in (1, 2, 3):
            fig.add_vrect(x0=e["start"], x1=e["end"], fillcolor=SELL, opacity=0.10, line_width=0, row=r_, col=1)
    fig.add_hline(y=C.CB_Z, line={"color": SELL, "dash": "dash", "width": 1}, row=2, col=1); fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=2, col=1)
    fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=3, col=1)
    if f"c_{C.GOLD}" in S.columns:
        fig.add_trace(go.Scatter(x=S.index, y=S[f"c_{C.GOLD}"], mode="lines", line={"color": COL["GoldMCX"], "width": 1.4}, name="gold vs the indices", hovertemplate="%{y:.2f}"), row=4, col=1)
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=4, col=1)
    fig.update_yaxes(range=[0, 1.02], row=1, col=1); fig.update_yaxes(range=[-4, 3], row=2, col=1); fig.update_yaxes(range=[-0.5, 1.02], row=3, col=1); fig.update_yaxes(range=[-0.6, 0.6], row=4, col=1)
    return _layout(fig, "In every decorrelation episode the index lines fall together and cross the break line in the same weeks — and the pair panel shows which pair opened up", 1120,
                   "Shaded = the decorrelation episodes (the group's z at or below −1 until it is back above −0.5). Third panel: each of the ten pairs on its own, so an episode can be traced to the pair that loosened.",
                   subtitle="Top: each index's average 13-week correlation with the other three, and the group. Second: the same as z-scores. Third: every pair. Bottom: gold.")


def cb_episodes():
    S = _csv("3_corr_series", index_col=0, parse_dates=True)
    ep = _csv("3_corr_episodes", parse_dates=["start", "end", "G_min_date"])
    ab = _csv("3_asset_breaks", parse_dates=["break_date"])
    n = len(ep); cols = 4; rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"{e['episode']}: group {e['G_before']:.2f} → {e['G_min']:.2f}" for _, e in ep.iterrows()], vertical_spacing=0.10, horizontal_spacing=0.05)
    for k, (_, e) in enumerate(ep.iterrows()):
        r_, c_ = k // cols + 1, k % cols + 1
        i = S.index.get_loc(e["start"])
        seg = S.iloc[max(0, i - 12):min(len(S), i + 13)]
        x = np.arange(len(seg)) - (i - max(0, i - 12))
        for a in [x_ for x_ in C.EQUITY if f"z_c_{x_}" in S.columns]:
            fig.add_trace(go.Scatter(x=x, y=seg[f"z_c_{a}"], mode="lines", line={"color": COL[a], "width": 1.4}, name=a, legendgroup=a, showlegend=k == 0, hovertemplate="k=%{x}: z %{y:+.1f}<extra>" + a + "</extra>"), row=r_, col=c_)
            b = ab[(ab["episode"] == e["episode"]) & (ab["asset"] == a)]
            if len(b) and pd.notna(b["break_date"].iloc[0]) and b["break_date"].iloc[0] in seg.index:
                pos = seg.index.get_loc(b["break_date"].iloc[0])
                fig.add_trace(go.Scatter(x=[pos - (i - max(0, i - 12))], y=[seg[f"z_c_{a}"].iloc[pos]], mode="markers", marker={"color": COL[a], "size": 11, "symbol": "triangle-down", "line": {"color": SURFACE, "width": 1}}, name=f"{a} own break", legendgroup=a, showlegend=False, hovertemplate="own break<extra>" + a + "</extra>"), row=r_, col=c_)
        fig.add_trace(go.Scatter(x=x, y=seg["z_G"], mode="lines", line={"color": INK, "width": 2}, name="group", legendgroup="G", showlegend=k == 0, hovertemplate="k=%{x}: z %{y:+.1f}<extra>group</extra>"), row=r_, col=c_)
        fig.add_hline(y=C.CB_Z, line={"color": SELL, "dash": "dash", "width": 1}, row=r_, col=c_); fig.add_vline(x=0, line={"color": AXIS, "dash": "dot", "width": 1}, row=r_, col=c_)
        fig.update_yaxes(range=[-4, 2.5], row=r_, col=c_)
    fig.update_xaxes(title_text="weeks from the group break", title_font={"size": 10})
    return _layout(fig, "Every decorrelation episode: each index's z-score in the 12 weeks before and after the group break (▼ = its own break week)", 330 * rows + 120, unified=False)


def cb_summary():
    sm = _csv("3_summary", index_col=0)
    assets = list(sm.index)
    specs = (("share_first", "avg_rank", "Earliest by date (ties count)"), ("share_deepest", "avg_rank_deepest", "Deepest below its norm when the market broke"),
             ("share_biggest_drop", "avg_rank_drop", "Biggest fall in its own correlation"), ("share_up_breaks", None, "Broke by OUTperforming (up-break)"))
    fig = make_subplots(rows=1, cols=4, subplot_titles=[s[2] for s in specs], shared_yaxes=True, horizontal_spacing=0.03)
    for c, (col, rk, _) in enumerate(specs, start=1):
        txt = [f"avg rank {sm.loc[a, rk]:.1f}" if rk else f"n {int(sm.loc[a, 'episodes_broke'])}" for a in assets]
        fig.add_trace(go.Bar(y=assets, x=sm[col], orientation="h", marker_color=[COL[a] for a in assets], text=txt, textposition="outside", showlegend=False, hovertemplate="%{y}: %{x:.0%}<extra></extra>"), row=1, col=c)
        fig.update_xaxes(tickformat=".0%", range=[0, 1.45], row=1, col=c)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "Who breaks the correlation first — share of episodes, by three measures, and which way they broke", 420, "Micro250 exists only from 2022 and is present in fewer episodes.", unified=False, legend_h=False)


def cb_after():
    afs = _csv("3_after_break_summary")
    cats = [(d, h) for d in afs["direction"].unique() for h in sorted(afs["weeks_ahead"].unique())]
    labels = [f"{d.split(' (')[0]}<br>next {h} wk (n {int(afs[(afs['direction'] == d) & (afs['weeks_ahead'] == h)]['n'].iloc[0])})" for d, h in cats]
    fig = go.Figure()
    for col, lab, c in (("breaker", "the asset that broke", CAT[0]), ("rest_of_group", "the other indices", MUTED), ("gold", "gold", COL["GoldMCX"])):
        vals = [afs[(afs["direction"] == d) & (afs["weeks_ahead"] == h)][col].iloc[0] for d, h in cats]
        fig.add_trace(go.Bar(x=labels, y=vals, name=lab, marker_color=c, hovertemplate="%{x}: %{y:.1%}<extra>" + lab + "</extra>"))
    fig.update_yaxes(tickformat=".1%"); fig.update_layout(barmode="group")
    return _layout(fig, "After an index breaks away from the group: what it, the others and gold returned next", 460, "Up-break = it decoupled by outperforming over the prior 13 weeks; down-break = by underperforming.", unified=False)


# ============================================================ 4. price / signal ==
def breaks_leads():
    sm = _csv("4_summary", index_col=0)
    assets = [a for a in C.ASSETS if a in sm.index]
    specs = (("signal_sell", "Supertrend flips to SELL"), ("cross_10pct", "Price crosses −10% from its own high"), ("recover_buy", "Supertrend flips back to BUY after the trough"))
    fig = make_subplots(rows=1, cols=3, subplot_titles=[f"{t}: share of sell-offs it was FIRST" for _, t in specs], shared_yaxes=True, horizontal_spacing=0.03)
    for c, (col, _) in enumerate(specs, start=1):
        txt = [f"avg rank {sm.loc[a, f'avg_rank_{col}']:.1f} · median lead {sm.loc[a, f'median_lead_{col}_weeks']:+.0f} wk · n {int(sm.loc[a, f'n_{col}'])}" for a in assets]
        fig.add_trace(go.Bar(y=assets, x=sm.loc[assets, f"first_share_{col}"], orientation="h", marker_color=[COL[a] for a in assets], text=txt, textposition="outside", textfont={"size": 10}, showlegend=False, hovertemplate="%{y}: %{x:.0%}<extra></extra>"), row=1, col=c)
        fig.update_xaxes(tickformat=".0%", range=[0, 1.6], row=1, col=c)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, "In a price sell-off, who breaks first", 420, "Lead = weeks relative to N100's own date (negative = earlier). Ties share first place. Micro250 exists only from 2022.", unified=False, legend_h=False)


def breaks_episodes():
    ep = _csv("4_price_episodes", parse_dates=["peak_date", "trough_date", "end_date"])
    ld = _csv("4_dates_and_leads", parse_dates=["peak", "signal_sell", "cross_10pct", "recover_buy"])
    px = _csv("px_weekly", index_col=0, parse_dates=True)
    n = len(ep); cols = 3; rows = int(np.ceil(n / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[f"{e['episode']}: N100 {e['depth']:.0%} over {int(e['weeks_peak_to_trough'])} wk" for _, e in ep.iterrows()], vertical_spacing=0.12, horizontal_spacing=0.05)
    for k, (_, e) in enumerate(ep.iterrows()):
        r_, c_ = k // cols + 1, k % cols + 1
        i0 = max(0, px.index.get_loc(e["peak_date"]) - 8); i1 = min(len(px) - 1, px.index.get_loc(e["trough_date"]) + 16)
        seg = px.iloc[i0:i1 + 1]
        for a in [x for x in C.ASSETS if x in seg.columns and x != "GoldMCX"]:
            s = seg[a].dropna()
            if len(s) < 4:
                continue
            base = s.loc[:e["peak_date"]].max() if (s.index <= e["peak_date"]).any() else s.iloc[0]
            fig.add_trace(go.Scatter(x=s.index, y=s / base - 1, mode="lines", line={"color": COL[a], "width": 1.6}, name=a, legendgroup=a, showlegend=k == 0, hovertemplate="%{y:.1%}<extra>" + a + "</extra>"), row=r_, col=c_)
            d = ld[(ld["episode"] == e["episode"]) & (ld["asset"] == a)]
            if len(d) and pd.notna(d["signal_sell"].iloc[0]) and d["signal_sell"].iloc[0] in s.index:
                dt = d["signal_sell"].iloc[0]
                fig.add_trace(go.Scatter(x=[dt], y=[s.loc[dt] / base - 1], mode="markers", marker={"color": COL[a], "size": 11, "symbol": "triangle-down", "line": {"color": SURFACE, "width": 1}}, name=f"{a} SELL flip", legendgroup=a, showlegend=False, hovertemplate="SELL flip<extra>" + a + "</extra>"), row=r_, col=c_)
        fig.add_hline(y=-C.DD_CROSS, line={"color": SELL, "dash": "dash", "width": 1}, row=r_, col=c_)
        fig.add_vline(x=e["trough_date"], line={"color": AXIS, "dash": "dot", "width": 1}, row=r_, col=c_)
        fig.update_yaxes(tickformat=".0%", row=r_, col=c_)
    return _layout(fig, "Each sell-off: every index from N100's peak, ▼ = the week its supertrend flipped to SELL, dashed = −10%, dotted = N100's trough", 330 * rows + 120, unified=False)


# ============================================================ 5. volatility ==
def vol_timeseries():
    ts = _csv("5_vol_timeseries", index_col=0, parse_dates=True)
    has_vix = "VIX" in ts.columns and ts["VIX"].notna().any()
    n = 3 if has_vix else 2
    titles = [f"Each asset's {C.VOL_WINDOW}-week realised volatility (annualised) — shaded where N100's signal was in SELL"] + (["India VIX level (the market's implied volatility; file ends " + str(ts['VIX'].dropna().index[-1].date()) + ")"] if has_vix else []) + ["The group correlation G, for comparison"]
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.06, row_heights=[0.45, 0.25, 0.3] if has_vix else [0.6, 0.4], subplot_titles=titles)
    for a in [x for x in C.ASSETS if f"vol_{x}" in ts.columns]:
        fig.add_trace(go.Scatter(x=ts.index, y=ts[f"vol_{a}"], mode="lines", line={"color": COL[a], "width": 1.4, "dash": "dash" if a == "GoldMCX" else "solid"}, name=a, hovertemplate="%{y:.0%}"), row=1, col=1)
    _shade(fig, ts.index, ts["N100_state"] == -1, row=1, col=1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    if has_vix:
        fig.add_trace(go.Scatter(x=ts.index, y=ts["VIX"], mode="lines", line={"color": COL["VIX"], "width": 1.4}, fill="tozeroy", fillcolor="rgba(227,73,72,0.12)", name="India VIX", hovertemplate="%{y:.1f}"), row=2, col=1)
    fig.add_trace(go.Scatter(x=ts.index, y=ts["G"], mode="lines", line={"color": INK, "width": 1.8}, name="group correlation G", hovertemplate="%{y:.2f}"), row=n, col=1)
    fig.update_yaxes(range=[0.3, 1.0], row=n, col=1)
    return _layout(fig, "Volatility — realised and implied — against the group correlation", 300 * n + 100, "Volatility spikes and correlation spikes line up: both are features of the same sell-off weeks.")


def vol_by_state():
    vs = _csv("5_vol_by_state", index_col=0)
    assets = [a for a in C.ASSETS if a in vs.index]
    has_vix = "vix_mean_BUY" in vs.columns
    fig = make_subplots(rows=1, cols=3 if has_vix else 2, subplot_titles=("Realised volatility by the asset's own signal state", "Average weekly return in each state") + (("Average VIX level in the asset's BUY vs SELL weeks",) if has_vix else ()), horizontal_spacing=0.08)
    for a in assets:
        fig.add_trace(go.Scatter(x=[vs.loc[a, "vol_BUY"], vs.loc[a, "vol_SELL"]], y=[a, a], mode="lines", line={"color": GRID, "width": 4}, showlegend=False, hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=vs.loc[assets, "vol_BUY"], y=assets, mode="markers", marker={"color": BUY, "size": 12}, name="in its own BUY weeks", legendgroup="buy", hovertemplate="%{y}: %{x:.1%}<extra>BUY weeks</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=vs.loc[assets, "vol_SELL"], y=assets, mode="markers+text", marker={"color": SELL, "size": 12}, text=[f"  ×{vs.loc[a, 'vol_ratio_SELL_over_BUY']:.2f}" for a in assets], textposition="middle right", textfont={"size": 11, "color": INK2}, name="in its own SELL weeks", legendgroup="sell", hovertemplate="%{y}: %{x:.1%}<extra>SELL weeks</extra>"), row=1, col=1)
    fig.add_trace(go.Bar(x=assets, y=vs.loc[assets, "mean_weekly_ret_BUY"], marker_color=BUY, name="BUY weeks", legendgroup="buy", showlegend=False, hovertemplate="%{x}: %{y:.2%}<extra>BUY</extra>"), row=1, col=2)
    fig.add_trace(go.Bar(x=assets, y=vs.loc[assets, "mean_weekly_ret_SELL"], marker_color=SELL, name="SELL weeks", legendgroup="sell", showlegend=False, hovertemplate="%{x}: %{y:.2%}<extra>SELL</extra>"), row=1, col=2)
    if has_vix:
        for a in assets:
            fig.add_trace(go.Scatter(x=[vs.loc[a, "vix_mean_BUY"], vs.loc[a, "vix_mean_SELL"]], y=[a, a], mode="lines", line={"color": GRID, "width": 4}, showlegend=False, hoverinfo="skip"), row=1, col=3)
        fig.add_trace(go.Scatter(x=vs.loc[assets, "vix_mean_BUY"], y=assets, mode="markers", marker={"color": BUY, "size": 12}, name="BUY", legendgroup="buy", showlegend=False, hovertemplate="%{y}: VIX %{x:.1f}<extra>BUY weeks</extra>"), row=1, col=3)
        fig.add_trace(go.Scatter(x=vs.loc[assets, "vix_mean_SELL"], y=assets, mode="markers+text", marker={"color": SELL, "size": 12}, text=[f"  ×{vs.loc[a, 'vix_ratio_SELL_over_BUY']:.2f}" for a in assets], textposition="middle right", textfont={"size": 11, "color": INK2}, name="SELL", legendgroup="sell", showlegend=False, hovertemplate="%{y}: VIX %{x:.1f}<extra>SELL weeks</extra>"), row=1, col=3)
        fig.update_xaxes(range=[12, 22], row=1, col=3); fig.update_yaxes(autorange="reversed", row=1, col=3)
    lab = {a: f"{a}<br><span style='font-size:10px;color:{INK2}'>BUY {int(vs.loc[a, 'weeks_BUY'])} wk · SELL {int(vs.loc[a, 'weeks_SELL'])} wk</span>" for a in assets}
    fig.update_xaxes(tickformat=".0%", range=[0, 0.40], row=1, col=1); fig.update_yaxes(autorange="reversed", tickvals=assets, ticktext=[lab[a] for a in assets], row=1, col=1); fig.update_yaxes(tickformat=".1%", row=1, col=2)

    fig.update_layout(barmode="group")
    eqv = [a for a in assets if a != "GoldMCX"]
    fig.add_annotation(x=vs.loc[eqv[-1], "vol_SELL"], y=eqv[-1], xref="x", yref="y", text=f"index swings {vs.loc[eqv, 'vol_ratio_SELL_over_BUY'].min():.1f}–{vs.loc[eqv, 'vol_ratio_SELL_over_BUY'].max():.1f}× bigger in SELL", showarrow=True, arrowhead=2, arrowcolor=INK2, ax=15, ay=35, font={"size": 11, "color": INK2}, bgcolor="rgba(252,252,251,0.92)", bordercolor=GRID)
    if "GoldMCX" in vs.index:
        fig.add_annotation(x=vs.loc["GoldMCX", "vol_SELL"], y="GoldMCX", xref="x", yref="y", text="gold: no change", showarrow=True, arrowhead=2, arrowcolor=INK2, ax=120, ay=0, font={"size": 11, "color": INK2}, bgcolor="rgba(252,252,251,0.92)", bordercolor=GRID)
    return _layout(fig, f"In its own SELL weeks each index swings {vs.loc[eqv, 'vol_ratio_SELL_over_BUY'].mean():.1f}× as much — but only about half of those weeks are down weeks", 490, "SELL weeks are bigger-swing weeks far more than they are falling weeks. Gold's swings do not change with its signal. The VIX (right) separates the states much less than realised vol does — the option market prices part of it in advance.", unified=False,
                   subtitle="Realised volatility = the standard deviation of the asset's weekly log returns in those weeks × √52 (annualised). Left: in the asset's own BUY weeks (blue) vs SELL weeks (red), with the ratio and the number of weeks under each name. Middle: the average weekly return in each. Right: the average India VIX level in each.")


def vol_around_flip():
    af = _csv("5_vol_around_flip")
    refs = [a for a in C.ASSETS if a in set(af["ref"])]
    has_vix = "vix_level" in af.columns and af["vix_level"].notna().any()
    cols = 3 if has_vix else 2
    titles = []
    for ref in refs:
        titles += [f"{ref} flips to SELL: each index's vol", f"{ref} flips to BUY: each index's vol"] + ([f"{ref}'s flips: VIX level (SELL solid, BUY dotted)"] if has_vix else [])
    fig = make_subplots(rows=len(refs), cols=cols, subplot_titles=titles, vertical_spacing=0.06, horizontal_spacing=0.05)
    for r_, ref in enumerate(refs, start=1):
        for c_, ev in ((1, "flip to SELL"), (2, "flip to BUY")):
            sub = af[(af["ref"] == ref) & (af["event"] == ev)]
            for a in [x for x in C.ASSETS if f"vol_{x}" in sub.columns]:
                fig.add_trace(go.Scatter(x=sub["weeks_from_flip"], y=sub[f"vol_{a}"], mode="lines+markers", line={"color": COL[a], "width": 2.2 if a == ref else 1.3, "dash": "dash" if a == "GoldMCX" else "solid"}, marker={"size": 4},
                                         name=a, legendgroup=a, showlegend=(r_ == 1 and c_ == 1), hovertemplate="k=%{x}: %{y:.0%}<extra>" + a + "</extra>"), row=r_, col=c_)
            fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}, row=r_, col=c_)
            fig.update_yaxes(tickformat=".0%", row=r_, col=c_)
        if has_vix:
            for ev, dash in (("flip to SELL", "solid"), ("flip to BUY", "dot")):
                sub = af[(af["ref"] == ref) & (af["event"] == ev)]
                fig.add_trace(go.Scatter(x=sub["weeks_from_flip"], y=sub["vix_level"], mode="lines+markers", line={"color": COL["VIX"], "width": 1.8, "dash": dash}, marker={"size": 4}, name=f"VIX around the {ev}", legendgroup="vix" + ev, showlegend=r_ == 1, hovertemplate="k=%{x}: VIX %{y:.1f}<extra>" + ev + "</extra>"), row=r_, col=3)
            fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}, row=r_, col=3)
    fig.update_xaxes(dtick=4)
    return _layout(fig, "Volatility in the 8 weeks before and after EACH asset's signal flips (averaged over that asset's flips)", 260 * len(refs) + 140, "Row = the asset whose signal flipped (its own line is thicker). Volatility is already rising before a SELL flip — the flip is caused by a big down move — and keeps rising for weeks after it.", legend_y=-0.04)


def corr_by_regime():
    lg = _csv("5_corr_by_regime_long")
    regimes = [r for r in dict.fromkeys(lg["regime"]) if r in (f"{C.BENCHMARK} vol", "pair's own vol", "VIX level")]
    fig = make_subplots(rows=1, cols=len(regimes), shared_yaxes=True, subplot_titles=[f"regime = {r}" for r in regimes], horizontal_spacing=0.04)
    pairs = list(dict.fromkeys(lg["pair"]))
    for c_, reg in enumerate(regimes, start=1):
        sub = lg[lg["regime"] == reg]
        for p in pairs:
            g = sub[sub["pair"] == p]
            if not len(g):
                continue
            gold = "Gold" in p
            fig.add_trace(go.Scatter(x=g["bucket"], y=g["mean_corr"], mode="lines+markers", line={"color": _pair_col(p), "width": 1.3 if gold else 2.0, "dash": "dash" if gold else "solid"}, marker={"size": 6},
                                     name=p.replace("|", " × "), legendgroup=p, showlegend=c_ == 1,
                                     customdata=np.c_[g["lo"], g["hi"], g["weeks"]], hovertemplate="%{x}: corr %{y:.2f} (regime %{customdata[0]:.3g}–%{customdata[1]:.3g}, n %{customdata[2]})<extra>" + p + "</extra>"), row=1, col=c_)
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=1, col=c_)
    fig.update_yaxes(range=[-0.25, 1.02])
    fig.update_xaxes(title_text="calm (Q1) → wild (Q5)")
    g0 = lg[(lg["regime"] == regimes[0]) & (lg["pair"] == "N100|SC100")].set_index("bucket")["mean_corr"]
    fig.add_annotation(x="Q5", y=g0.get("Q5", np.nan), xref="x", yref="y", text=f"index pairs climb calm → wild<br>(N100×SC100 {g0.get('Q1', np.nan):.2f} → {g0.get('Q5', np.nan):.2f})", showarrow=True, arrowhead=2, arrowcolor=INK2, ax=-140, ay=90, font={"size": 11, "color": INK2}, bgcolor="rgba(252,252,251,0.92)", bordercolor=GRID)
    fig.add_annotation(x="Q3", y=0.0, xref="x", yref="y", text="gold pairs: flat near zero in every regime", showarrow=True, arrowhead=2, arrowcolor=INK2, ax=0, ay=-110, font={"size": 11, "color": INK2}, bgcolor="rgba(252,252,251,0.92)", bordercolor=GRID)
    return _layout(fig, "The wilder the market, the more the indices move together — whichever way 'wild' is measured; gold never joins in", 580, "Solid: index pairs. Dashed: gold pairs. Q1 = the calmest fifth of weeks by that measure, Q5 the wildest; hover shows each bucket's range.", legend_y=-0.16, unified=False,
                   subtitle="Each pair's 13-week correlation averaged inside five volatility regimes (equal fifths of the weeks; hover gives each fifth's range and its number of weeks), the regime defined three ways: N100's realised vol · the pair's own vol (the average of its two legs' 13-week realised vol — each leg on its own is in the next chart) · the India VIX level")


def corr_by_regime_more():
    """The regime study again with (a) each leg of the pair separately — the two components of 'the pair's own vol' — and (b) the desk's ATR instead of the standard deviation."""
    lg = _csv("5_corr_by_regime_long")
    want = ["first leg's vol", "second leg's vol", f"{C.BENCHMARK} ATR%", "pair's own ATR%"]
    regimes = [r for r in want if r in set(lg["regime"])]
    fig = make_subplots(rows=1, cols=len(regimes), shared_yaxes=True, subplot_titles=[f"regime = {r}" for r in regimes], horizontal_spacing=0.04)
    pairs = [p for p in PAIR_ORDER if p in set(lg["pair"])]
    for c_, reg in enumerate(regimes, start=1):
        sub = lg[lg["regime"] == reg]
        for p in pairs:
            g = sub[sub["pair"] == p]
            if not len(g):
                continue
            gold = "Gold" in p
            fig.add_trace(go.Scatter(x=g["bucket"], y=g["mean_corr"], mode="lines+markers", line={"color": _pair_col(p), "width": 1.3 if gold else 2.0, "dash": "dash" if gold else "solid"}, marker={"size": 6},
                                     name=_pair_name(p), legendgroup=p, showlegend=c_ == 1,
                                     customdata=np.c_[g["lo"], g["hi"], g["weeks"]], hovertemplate="%{x}: corr %{y:.2f} (regime %{customdata[0]:.3g}–%{customdata[1]:.3g}, n %{customdata[2]})<extra>" + p + "</extra>"), row=1, col=c_)
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=1, col=c_)
    fig.update_yaxes(range=[-0.25, 1.02]); fig.update_xaxes(title_text="calm (Q1) → wild (Q5)")
    def q15(reg, p):
        g = lg[(lg["regime"] == reg) & (lg["pair"] == p)].set_index("bucket")["mean_corr"]
        return g.get("Q1", np.nan), g.get("Q5", np.nan)
    a1, a5 = q15("first leg's vol", "N100|SC100"); b1, b5 = q15("second leg's vol", "N100|SC100"); t1, t5 = q15(f"{C.BENCHMARK} ATR%", "N100|SC100")
    return _layout(fig, f"The same climb whichever component or measure defines 'wild': N100×SC100 {a1:.2f} → {a5:.2f} by Nifty's own vol, {b1:.2f} → {b5:.2f} by Smallcap's, {t1:.2f} → {t5:.2f} by Nifty's ATR", 580,
                   "Left two panels: the pair's volatility split into its two components — the first-named leg's 13-week realised vol, then the second's. Right two panels: volatility measured by the desk's ATR (the 5-bar linear-weighted average true range on weekly bars, as a share of the close — the same ATR the supertrend uses; it sees the week's high-low range, not just the close-to-close move) — N100's, and the pair's own (the average of its two legs'). Hover gives each fifth's range and number of weeks.", unified=False, legend_y=-0.3,
                   subtitle="Each pair's 13-week correlation averaged inside five volatility regimes — the regime defined by each leg separately, and by ATR instead of the standard deviation")


def vol_corr_lead():
    vl = _csv("5_vol_corr_lead", index_col=0)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=vl.index, y=vl["corr_dvol_t_dcorr_t_plus_k"], name="realised vol (N100, 13w)", marker_color=CAT[0], text=[f"{v:.2f}" for v in vl["corr_dvol_t_dcorr_t_plus_k"]], textposition="outside", hovertemplate="k=%{x}: %{y:.3f}<extra>realised</extra>"))
    if "corr_dvix_t_dcorr_t_plus_k" in vl.columns:
        fig.add_trace(go.Bar(x=vl.index, y=vl["corr_dvix_t_dcorr_t_plus_k"], name="India VIX (13w log change)", marker_color=COL["VIX"], text=[f"{v:.2f}" for v in vl["corr_dvix_t_dcorr_t_plus_k"]], textposition="outside", hovertemplate="k=%{x}: %{y:.3f}<extra>VIX</extra>"))
    fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"})
    fig.update_xaxes(title_text="k (weeks): correlation between the change in volatility at week t and the change in the group correlation at week t + k", dtick=1)
    fig.update_layout(barmode="group"); fig.update_yaxes(range=[0, 0.65])
    kv = int(vl["corr_dvix_t_dcorr_t_plus_k"].idxmax()) if "corr_dvix_t_dcorr_t_plus_k" in vl.columns else 0
    return _layout(fig, f"The group: realised vol and correlation change in the same week (peak at k = 0, {vl.loc[0, 'corr_dvol_t_dcorr_t_plus_k']:.2f})" + (f"; the VIX's bars peak at k = {kv:+d} — a small head start" if kv > 0 else ""), 460,
                   "A peak at k = 0 means they move together; a peak at k > 0 means volatility leads (the correlation change comes k weeks later); at k < 0, correlation leads.", unified=False,
                   subtitle="The correlation between the 13-week change in volatility at week t and the 13-week change in the group correlation G at week t + k, for k from −4 to +4")


def _lead_panel(fig, vl, r_, c_, showlegend):
    fig.add_trace(go.Bar(x=vl.index, y=vl["corr_dvol_t_dcorr_t_plus_k"], name="the pair's own realised vol (13w)", marker_color=CAT[0], legendgroup="vol", showlegend=showlegend, text=[f"{v:.2f}" for v in vl["corr_dvol_t_dcorr_t_plus_k"]], textposition="outside", textfont={"size": 9}, hovertemplate="k=%{x}: %{y:.3f}<extra>own vol</extra>"), row=r_, col=c_)
    if "corr_dvix_t_dcorr_t_plus_k" in vl.columns:
        fig.add_trace(go.Bar(x=vl.index, y=vl["corr_dvix_t_dcorr_t_plus_k"], name="India VIX (13w log change)", marker_color=COL["VIX"], legendgroup="vix", showlegend=showlegend, text=[f"{v:.2f}" for v in vl["corr_dvix_t_dcorr_t_plus_k"]], textposition="outside", textfont={"size": 9}, hovertemplate="k=%{x}: %{y:.3f}<extra>VIX</extra>"), row=r_, col=c_)
    fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}, row=r_, col=c_); fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=r_, col=c_)


def vol_corr_lead_pairs():
    """One panel per pair: the lead/lag bars with the PAIR's own volatility and the VIX against the pair's own correlation."""
    vp = _csv("5_vol_corr_lead_pairs")
    prs = [p for p in PAIR_ORDER if p in set(vp["pair"])]
    ncol = 5; nrow = int(np.ceil(len(prs) / ncol))
    titles = []
    for p in prs:
        g = vp[vp["pair"] == p].set_index("k_weeks")
        kv = int(g["corr_dvol_t_dcorr_t_plus_k"].idxmax()) if g["corr_dvol_t_dcorr_t_plus_k"].notna().any() else 0
        kx = int(g["corr_dvix_t_dcorr_t_plus_k"].idxmax()) if "corr_dvix_t_dcorr_t_plus_k" in g.columns and g["corr_dvix_t_dcorr_t_plus_k"].notna().any() else None
        titles.append(f"{_pair_name(p)}<br><span style='font-size:10px;color:{INK2}'>own vol peaks at k = {kv:+d}" + (f" · VIX at k = {kx:+d}" if kx is not None else "") + "</span>")
    fig = make_subplots(rows=nrow, cols=ncol, shared_yaxes=True, subplot_titles=titles, horizontal_spacing=0.03, vertical_spacing=0.2)
    for i, p in enumerate(prs):
        r_, c_ = i // ncol + 1, i % ncol + 1
        _lead_panel(fig, vp[vp["pair"] == p].set_index("k_weeks"), r_, c_, i == 0)
        fig.update_xaxes(dtick=1, tickfont={"size": 9}, row=r_, col=c_)
    fig.update_layout(barmode="group"); fig.update_yaxes(range=[-0.25, 0.9])
    fig.update_annotations(font={"size": 11})
    eq = [p for p in prs if "Gold" not in p]
    at0 = sum(1 for p in eq if int(vp[vp["pair"] == p].set_index("k_weeks")["corr_dvol_t_dcorr_t_plus_k"].idxmax()) == 0)
    return _layout(fig, f"Pair by pair: the pair's own volatility and its correlation change in the same week in {at0} of the {len(eq)} index pairs; the gold pairs' correlation does not move with volatility at all", 300 * nrow + 120,
                   "Each panel is one pair on its own. Blue: the change in the pair's own realised vol (the average of its two legs' 13-week vol) at week t against the change in the pair's 13-week correlation at week t + k. Red: the same with the 13-week change in the log VIX. Index pairs first, gold pairs last. The panel title names where each peaks.", unified=False, legend_y=-0.1,
                   subtitle="The 'which moves first' test run separately for every pair — no group number involved")


def together_overlay():
    tg = _csv("5_together", index_col=0, parse_dates=True)
    per = _csv("5_corr_rise_events", parse_dates=["event"])
    has_vix = "z_vix" in tg.columns
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tg.index, y=tg["z_corr"], mode="lines", line={"color": INK, "width": 2.0}, name="correlation among the indices (G)", hovertemplate="z %{y:+.1f}<extra>correlation</extra>"))
    fig.add_trace(go.Scatter(x=tg.index, y=tg["z_vol"], mode="lines", line={"color": CAT[0], "width": 1.4}, name="realised volatility (N100, 13w)", hovertemplate="z %{y:+.1f}<extra>realised vol</extra>"))
    if has_vix:
        fig.add_trace(go.Scatter(x=tg.index, y=tg["z_vix"], mode="lines", line={"color": COL["VIX"], "width": 1.4}, name="India VIX", hovertemplate="z %{y:+.1f}<extra>VIX</extra>"))
    for _, e in per.iterrows():
        fig.add_vline(x=e["event"], line={"color": AXIS, "width": 1, "dash": "dot"})
    fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}); fig.add_hline(y=1, line={"color": SELL, "width": 0.8, "dash": "dash"})
    fig.update_yaxes(title_text="z-score vs its own 2-year history", range=[-3, 6])
    return _layout(fig, "The group: correlation, volatility and the VIX spike in the same weeks — three views of the same sell-offs", 540, "Dotted verticals = the weeks the group correlation jumped (crossed +1).",
                   subtitle="Each line is that series as a z-score against its own trailing two years, so all three fit one axis: 0 = its usual self, +1 (dashed) = unusually high")


def together_pairs():
    """One panel per pair: the pair's own correlation, the pair's own realised vol and the VIX, each as a z-score; dotted = the pair's own correlation jumps."""
    tp = _csv("5_together_pairs", index_col=0, parse_dates=True)
    evo = _csv("5_corr_rise_own_events", parse_dates=["event"])
    prs = [p for p in PAIR_ORDER if p in tp.columns]
    has_vix = "z_vix" in tp.columns
    ncol = 2; nrow = int(np.ceil(len(prs) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, shared_xaxes=True, shared_yaxes=True, subplot_titles=[_pair_name(p) for p in prs], horizontal_spacing=0.04, vertical_spacing=0.05)
    for i, p in enumerate(prs):
        r_, c_ = i // ncol + 1, i % ncol + 1
        fig.add_trace(go.Scatter(x=tp.index, y=tp[f"z_{p}"], mode="lines", line={"color": INK, "width": 1.6}, name="the pair's correlation", legendgroup="c", showlegend=i == 0, hovertemplate="z %{y:+.1f}<extra>" + _pair_name(p) + " corr</extra>"), row=r_, col=c_)
        fig.add_trace(go.Scatter(x=tp.index, y=tp[f"zv_{p}"], mode="lines", line={"color": CAT[0], "width": 1.1}, name="the pair's own realised vol", legendgroup="v", showlegend=i == 0, hovertemplate="z %{y:+.1f}<extra>" + _pair_name(p) + " vol</extra>"), row=r_, col=c_)
        if has_vix:
            fig.add_trace(go.Scatter(x=tp.index, y=tp["z_vix"], mode="lines", line={"color": COL["VIX"], "width": 1.0}, name="India VIX", legendgroup="x", showlegend=i == 0, hovertemplate="z %{y:+.1f}<extra>VIX</extra>"), row=r_, col=c_)
        for d in evo[evo["pair"] == p]["event"]:
            fig.add_vline(x=d, line={"color": AXIS, "width": 1, "dash": "dot"}, row=r_, col=c_)
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.6}, row=r_, col=c_); fig.add_hline(y=1, line={"color": SELL, "width": 0.8, "dash": "dash"}, row=r_, col=c_)
    fig.update_yaxes(range=[-3.5, 5.5])
    fig.update_annotations(font={"size": 12})
    return _layout(fig, "Pair by pair: in the index pairs the three z-scores still spike together; in the gold pairs the correlation moves on its own — its spikes are not volatility events", 210 * nrow + 140,
                   "One panel per pair. Black: the pair's 13-week correlation as a z-score against its own two years; blue: the pair's own realised vol (the average of its two legs) as a z-score; red: the India VIX as a z-score. Dotted verticals = the weeks THAT pair's correlation z first reached +1. Dashed = +1.", legend_y=-0.04,
                   subtitle="The overlay chart repeated separately for every pair, with the pair's own volatility")


def corr_rise_paths():
    pa = _csv("5_corr_rise_paths")
    names = {"z_corr": ("correlation among the indices", INK, 2.2), "z_vol": ("realised volatility (N100)", CAT[0], 1.6), "z_vix": ("India VIX", COL["VIX"], 1.6)}
    fig = go.Figure()
    for s, (nm, col, w) in names.items():
        g = pa[pa["series"] == s]
        if len(g):
            fig.add_trace(go.Scatter(x=g["k"], y=g["mean_z"], mode="lines+markers", line={"color": col, "width": w}, marker={"size": 5}, name=f"{nm} (n {int(g['n'].max())})", hovertemplate="k=%{x}: z %{y:+.2f}<extra>" + nm + "</extra>"))
    fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}); fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}); fig.add_hline(y=1, line={"color": SELL, "width": 0.8, "dash": "dash"})
    fig.update_xaxes(title_text="weeks from the correlation jump (0 = the week the correlation's z first reached +1)", dtick=2)
    fig.update_yaxes(title_text="z-score vs its own 2-year history")
    fig.add_annotation(x=-4, y=1.25, xref="x", yref="y", text="before the jump: all three near their usual level — no warning", showarrow=False, font={"size": 11.5, "color": INK2})
    fig.add_annotation(x=0, y=1.0, xref="x", yref="y", text="all three cross +1<br>in the same week", showarrow=True, arrowhead=2, arrowcolor=INK2, ax=70, ay=60, font={"size": 11, "color": INK2}, bgcolor="rgba(252,252,251,0.92)", bordercolor=GRID)
    return _layout(fig, "The group: around the weeks correlation jumped, volatility and the VIX were still at their usual level — then all three rose together", 500, "If volatility or the VIX gave warning, its line would already be above the dashed +1 line to the LEFT of week 0.", legend_y=-0.22,
                   subtitle="The three series lined up at the week the group correlation's z-score first reached +1, averaged over those weeks")


def corr_rise_paths_own():
    """One panel per pair: around the PAIR's own correlation jumps, the pair's correlation z, the pair's own vol z and the VIX z."""
    pa = _csv("5_corr_rise_paths_own"); per = _csv("5_corr_rise_own_summary", index_col=0)
    prs = [p for p in PAIR_ORDER if p in set(pa["pair"])]
    names = {"z_corr": ("the pair's correlation", INK, 2.0), "z_vol": ("the pair's own realised vol", CAT[0], 1.5), "z_vix": ("India VIX", COL["VIX"], 1.5)}
    ncol = 5; nrow = int(np.ceil(len(prs) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, shared_yaxes=True, subplot_titles=[f"{_pair_name(p)} (n {int(per.loc[p, 'jumps'])})" for p in prs], horizontal_spacing=0.03, vertical_spacing=0.16)
    for i, p in enumerate(prs):
        r_, c_ = i // ncol + 1, i % ncol + 1
        g = pa[pa["pair"] == p]
        for s_, (nm, col, w) in names.items():
            gg = g[g["series"] == s_]
            if len(gg):
                fig.add_trace(go.Scatter(x=gg["k"], y=gg["mean_z"], mode="lines+markers", line={"color": col, "width": w}, marker={"size": 4}, name=nm, legendgroup=s_, showlegend=i == 0, hovertemplate="k=%{x}: z %{y:+.2f}<extra>" + nm + "</extra>"), row=r_, col=c_)
        fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}, row=r_, col=c_); fig.add_hline(y=0, line={"color": AXIS, "width": 0.6}, row=r_, col=c_); fig.add_hline(y=1, line={"color": SELL, "width": 0.8, "dash": "dash"}, row=r_, col=c_)
        fig.update_xaxes(dtick=4, tickfont={"size": 9}, row=r_, col=c_)
    fig.update_yaxes(range=[-1.6, 2.2])
    fig.update_annotations(font={"size": 11})
    eq = [p for p in prs if "Gold" not in p]; gd = [p for p in prs if "Gold" in p]
    v0e, v0g = per.loc[eq, "z_vol_+0"].mean(), per.loc[gd, "z_vol_+0"].mean()
    warn = per.loc[eq, "share_z_vol_above1_before"].mean()
    return _layout(fig, f"Pair by pair: when an index pair's correlation jumps, its own volatility is up with it (z {v0e:+.1f} in the jump week) and was not up before ({warn:.0%} of jumps); when a gold pair's correlation jumps, volatility does nothing (z {v0g:+.1f})", 300 * nrow + 120,
                   "One panel per pair, lined up at the week THAT pair's correlation z first reached +1. A warning would be the blue or red line above the dashed +1 line to the LEFT of week 0. Index pairs first, gold pairs last; n = the pair's own jumps.", unified=False, legend_y=-0.1,
                   subtitle="The correlation-jump event study run separately for every pair, on the pair's own correlation, its own volatility and the VIX")


# ============================================================ 6. pairs ==
def pair_card(pair: str):
    a, b = pair.split("|")
    Z = _csv("6_ratio_z", index_col=0, parse_dates=True); B_ = _csv("6_ratio_bands", index_col=0, parse_dates=True)
    PC = _csv("6_pair_corr13", index_col=0, parse_dates=True); V = _csv("5_vol_rolling", index_col=0, parse_dates=True)
    px = _csv("px_weekly", index_col=0, parse_dates=True); st = _csv("sig_states_weekly", index_col=0, parse_dates=True)
    ts = _csv("5_vol_timeseries", index_col=0, parse_dates=True)
    tr = _csv("7_trades", parse_dates=["entry_date", "exit_date"]) if (T / "7_trades.csv").stat().st_size > 10 else pd.DataFrame()
    has_vix = "VIX" in ts.columns and ts["VIX"].notna().any()
    fig = make_subplots(rows=5, cols=1, shared_xaxes=True, row_heights=[0.2, 0.2, 0.26, 0.15, 0.19], vertical_spacing=0.045, specs=[[{}], [{}], [{}], [{}], [{"secondary_y": True}]],
                        subplot_titles=(f"1. Each leg's cumulative return (both start at 100 on the pair's first common week; log scale) — the returns the ratio is made of",
                                        f"2. The price ratio ln({a}/{b}) with its trailing {C.PAIR_WINDOW}-week mean and ±1 / ±2 sd bands (up = {a} doing better than {b})",
                                        f"3. The deviation z-score of that ratio · ▼ {a} flips to SELL (top row) · ▲ {b} flips to SELL (bottom row) · ● pair-trade entries, ○ exits",
                                        f"4. The pair's {C.CORR_WINDOW}-week correlation",
                                        f"5. Each leg's {C.VOL_WINDOW}-week volatility (left axis)" + (" and the VIX level (right axis)" if has_vix else "")))
    # 1 cumulative returns
    both = px[[a, b]].dropna()
    for leg in (a, b):
        fig.add_trace(go.Scatter(x=both.index, y=both[leg] / both[leg].iloc[0] * 100, mode="lines", line={"color": COL[leg], "width": 1.6}, name=leg, legendgroup=leg, hovertemplate="%{y:.0f}<extra>" + leg + "</extra>"), row=1, col=1)
    fig.update_yaxes(type="log", row=1, col=1)
    # 2 ratio + bands
    lr, m, s = B_[f"{pair}__lr"], B_[f"{pair}__mean"], B_[f"{pair}__sd"]
    for k_, (lo, hi, name, fc) in enumerate(((m - 2 * s, m + 2 * s, "±2 sd band", "rgba(225,224,217,0.55)"), (m - s, m + s, "±1 sd band", "rgba(195,194,183,0.5)"))):
        fig.add_trace(go.Scatter(x=lr.index, y=hi, mode="lines", line={"width": 0}, showlegend=False, hoverinfo="skip", legendgroup=name), row=2, col=1)
        fig.add_trace(go.Scatter(x=lr.index, y=lo, mode="lines", line={"width": 0}, fill="tonexty", fillcolor=fc, name=name, legendgroup=name, hoverinfo="skip"), row=2, col=1)
    fig.add_trace(go.Scatter(x=lr.index, y=m, mode="lines", line={"color": MUTED, "width": 1, "dash": "dash"}, name=f"{C.PAIR_WINDOW}-week mean", hovertemplate="%{y:.3f}<extra>mean</extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=lr.index, y=lr, mode="lines", line={"color": INK, "width": 1.6}, name=f"ln({a} / {b})", hovertemplate="%{y:.3f}<extra>log ratio</extra>"), row=2, col=1)
    # 3 z
    z = Z[pair]
    fig.add_trace(go.Scatter(x=z.index, y=z, mode="lines", line={"color": CAT[0], "width": 1.5}, name="deviation z", hovertemplate="z %{y:+.2f}<extra></extra>"), row=3, col=1)
    for lvl in C.PAIR_Z_LINES:
        for sgn in (1, -1):
            fig.add_hline(y=sgn * lvl, line={"color": SELL if lvl >= 2 else AXIS, "dash": "dash", "width": 1}, row=3, col=1)
    fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=3, col=1)
    for leg, y_, sym in ((a, 3.4, "triangle-down"), (b, -3.4, "triangle-up")):
        if leg in st.columns:
            sl = st[leg].reindex(z.index)
            flips = z.index[((sl == -1) & (sl.shift(1) == 1)).fillna(False)]
            fig.add_trace(go.Scatter(x=flips, y=[y_] * len(flips), mode="markers", marker={"color": COL[leg], "size": 10, "symbol": sym, "line": {"color": SURFACE, "width": 1}}, name=f"{leg} flips to SELL", legendgroup=leg + "flip",
                                     customdata=z.reindex(flips).values, hovertemplate="%{x|%Y-%m-%d}: z %{customdata:+.2f}<extra>" + leg + " → SELL</extra>"), row=3, col=1)
    if len(tr):
        g = tr[tr["pair"] == pair]
        if len(g):
            fig.add_trace(go.Scatter(x=g["entry_date"], y=g["z_entry"], mode="markers", marker={"color": INK, "size": 9}, name="pair-trade entry", customdata=np.c_[g["cheap_leg"], g["net"], g["weeks"]], hovertemplate="%{x|%Y-%m-%d}: z %{y:+.2f}, long %{customdata[0]}, net %{customdata[1]:+.1%} over %{customdata[2]} wk<extra>entry</extra>"), row=3, col=1)
            fig.add_trace(go.Scatter(x=g["exit_date"], y=g["z_exit"], mode="markers", marker={"color": SURFACE, "size": 9, "line": {"color": INK, "width": 1.5}}, name="pair-trade exit", customdata=g["exit_reason"], hovertemplate="%{x|%Y-%m-%d}: z %{y:+.2f} (%{customdata})<extra>exit</extra>"), row=3, col=1)
    fig.update_yaxes(range=[-3.9, 3.9], dtick=1, row=3, col=1)
    # 4 corr
    fig.add_trace(go.Scatter(x=PC.index, y=PC[pair], mode="lines", line={"color": CAT[0], "width": 1.5}, name=f"{C.CORR_WINDOW}w correlation", hovertemplate="%{y:.2f}<extra>corr</extra>"), row=4, col=1)
    fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=4, col=1)
    fig.update_yaxes(range=[-0.6 if "Gold" in pair else 0.0, 1.02], row=4, col=1)
    # 5 vol + VIX
    for leg in (a, b):
        if leg in V.columns:
            fig.add_trace(go.Scatter(x=V.index, y=V[leg], mode="lines", line={"color": COL[leg], "width": 1.5}, name=f"{leg} vol", legendgroup=leg, showlegend=False, hovertemplate="%{y:.0%}<extra>" + leg + " vol</extra>"), row=5, col=1, secondary_y=False)
    if has_vix:
        fig.add_trace(go.Scatter(x=ts.index, y=ts["VIX"], mode="lines", line={"color": COL["VIX"], "width": 1.2, "dash": "dot"}, name="India VIX (right axis)", hovertemplate="VIX %{y:.1f}<extra></extra>"), row=5, col=1, secondary_y=True)
        fig.update_yaxes(showgrid=False, range=[5, 45], dtick=10, title_text="VIX", row=5, col=1, secondary_y=True)
    fig.update_yaxes(tickformat=".0%", row=5, col=1, secondary_y=False)
    fig.update_layout(hovermode="x unified")
    return _layout(fig, f"{a} × {b}", 1500, legend_y=-0.03)


def z_forward():
    zf = _csv("6_z_forward")
    buckets = list(dict.fromkeys(zf["bucket"]))
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=[f"Change in ln(a/b) over the next {h} weeks, by this week's deviation z" for h in C.PAIR_FWD], horizontal_spacing=0.05)
    for c_, h in enumerate(C.PAIR_FWD, start=1):
        pooled = zf[(zf["pair"] == "ALL PAIRS") & (zf["horizon_weeks"] == h)].set_index("bucket").reindex(buckets)
        labels = [f"{b}<br>t {pooled.loc[b, 'hac_t_ratio_change']:+.1f} · n {int(pooled.loc[b, 'weeks'])}" for b in buckets]
        fig.add_trace(go.Bar(x=labels, y=pooled["mean_ratio_change"], marker_color=[CAT[1] if i < 2 else (MUTED if i == 2 else CAT[0]) for i in range(len(buckets))], name="all pairs pooled", showlegend=c_ == 1, hovertemplate="%{y:.2%}<extra>pooled</extra>"), row=1, col=c_)
        per = zf[(zf["pair"] != "ALL PAIRS") & (zf["horizon_weeks"] == h)]
        for p in list(dict.fromkeys(per["pair"])):
            g = per[per["pair"] == p].set_index("bucket").reindex(buckets)
            fig.add_trace(go.Scatter(x=labels, y=g["mean_ratio_change"], mode="markers", marker={"color": _pair_col(p), "size": 7, "symbol": "circle" if "Gold" not in p else "diamond", "line": {"color": INK2, "width": 0.5}}, name=p.replace("|", " × "), legendgroup=p, showlegend=c_ == 1,
                                     customdata=np.c_[g["weeks"], g["hac_t_ratio_change"]], hovertemplate="%{y:.2%} (n %{customdata[0]}, t %{customdata[1]:+.1f})<extra>" + p + "</extra>"), row=1, col=c_)
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=1, col=c_)
    fig.update_yaxes(tickformat=".1%")
    return _layout(fig, "Did a deviation pull back? If it had, the left bars would be positive and the right bars negative. Bars = all pairs pooled; dots = each pair", 560, "z < 0 means a has lagged b. A pull-back would then need the ratio to RISE (positive). Click a pair in the legend to hide or show its dots.", unified=False, legend_y=-0.2)


def path_after():
    pa = _csv("6_path_after_2sd")
    fig = go.Figure()
    for p, g in pa.groupby("pair"):
        gold = "Gold" in p
        fig.add_trace(go.Scatter(x=g["weeks_after"], y=g["mean_z"], mode="lines", line={"color": _pair_col(p), "width": 1.3 if gold else 1.9, "dash": "dash" if gold else "solid"}, name=f"{p.replace('|', ' × ')} (n {int(g['n_events'].iloc[0])})",
                                 customdata=np.c_[g["share_back_inside_1sd"], g["share_crossed_0"]], hovertemplate="week %{x}: z %{y:.2f} · back inside 1 sd %{customdata[0]:.0%} · crossed 0 %{customdata[1]:.0%}<extra>" + p + "</extra>"))
    for y_, c in ((2, SELL), (1, AXIS), (0, AXIS)):
        fig.add_hline(y=y_, line={"color": c, "dash": "dash" if y_ else "solid", "width": 1})
    fig.update_xaxes(title_text="weeks after |z| first reached 2"); fig.update_yaxes(range=[-0.5, 2.8])
    return _layout(fig, "The average z-score after a 2-sd deviation, sign-flipped so every event starts near +2", 520, "The z-score always drifts back toward 0 as the 52-week window absorbs the move — this shows the SHAPE; the chart above shows whether the RATIO itself came back.", legend_y=-0.14)


# ============================================================ 7. pair trade ==
def pt_equity():
    eq = _csv("7_equity", index_col=0, parse_dates=True)
    fig = go.Figure()
    for p in [c for c in eq.columns if c != "ALL PAIRS"]:
        gold = "Gold" in p
        fig.add_trace(go.Scatter(x=eq.index, y=eq[p], mode="lines", line={"color": _pair_col(p), "width": 1.2 if gold else 1.6, "dash": "dash" if gold else "solid"}, name=p.replace("|", " × "), hovertemplate="%{y:+.1%}<extra>" + p + "</extra>"))
    fig.add_trace(go.Scatter(x=eq.index, y=eq["ALL PAIRS"], mode="lines", line={"color": INK, "width": 2.6}, name="ALL PAIRS", hovertemplate="%{y:+.1%}<extra>all</extra>"))
    fig.add_hline(y=0, line={"color": AXIS, "width": 0.8})
    fig.update_yaxes(tickformat=".0%")
    return _layout(fig, f"Cumulative net P&L of the pair-trading rule, one unit per leg (enter |z| ≥ {C.PT_ENTRY:.0f}, exit |z| ≤ {C.PT_EXIT:.1f} or {C.PT_MAX_HOLD} weeks, cost {C.PT_COST_BPS} bp per leg per side)", 560, "Log-return units: +10% = a unit long/short position gained 10 points of spread. Solid: index pairs. Dashed: gold pairs.", legend_y=-0.12)


def pt_summary():
    sm = _csv("7_summary", index_col=0)
    pairs = [p for p in sm.index if p != "ALL PAIRS"]
    labels = [p.replace("|", " × ") for p in pairs]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Average net P&L per trade", "Average return of each leg over the trade (the long-only view)"), horizontal_spacing=0.04)
    txt = [f"n {int(sm.loc[p, 'trades'])} · wins {sm.loc[p, 'win_rate_net']:.0%} · {sm.loc[p, 'avg_weeks']:.0f} wk" for p in pairs]
    fig.add_trace(go.Bar(y=labels, x=sm.loc[pairs, "mean_net"], orientation="h", marker_color=[BUY if v > 0 else SELL for v in sm.loc[pairs, "mean_net"]], text=txt, textposition="outside", textfont={"size": 10}, name="net per trade", showlegend=False, hovertemplate="%{y}: %{x:+.1%}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Bar(y=labels, x=sm.loc[pairs, "mean_ret_cheap_leg"], orientation="h", marker_color=BUY, name="the cheap leg (the one the rule buys)", hovertemplate="%{y}: %{x:+.1%}<extra>cheap leg</extra>"), row=1, col=2)
    fig.add_trace(go.Bar(y=labels, x=sm.loc[pairs, "mean_ret_rich_leg"], orientation="h", marker_color=MUTED, name="the rich leg (the one it sells)", hovertemplate="%{y}: %{x:+.1%}<extra>rich leg</extra>"), row=1, col=2)
    fig.update_xaxes(tickformat=".0%"); fig.update_xaxes(range=[min(sm.loc[pairs, "mean_net"].min() * 1.3, -0.02), 0.1], row=1, col=1); fig.update_yaxes(autorange="reversed"); fig.update_layout(barmode="group")
    return _layout(fig, "The pair-trading rule, pair by pair", 520, "If the cheap leg's bar is shorter than the rich leg's, the lagging index kept lagging: holding the laggard instead of the leader lost money.", unified=False)


# ============================================================ 8. replacement ==
def rp_rules():
    R = _csv("8_rules", index_col=0)
    rules = list(R.index)
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=[f"Average return over the next {h} weeks" for h in C.RP_FWD], horizontal_spacing=0.04)
    for c_, h in enumerate(C.RP_FWD, start=1):
        v = R[f"mean_{h}"]; cash = R.loc["cash", f"mean_{h}"]
        colors = [COL["GoldMCX"] if r == "gold" else (MUTED if r in ("cash", "stay in the seller") else (BUY if x > cash else SELL)) for r, x in zip(rules, v)]
        txt = [f"beat cash {R.loc[r, f'beat_cash_{h}']:.0%} · had a pick {R.loc[r, 'share_with_a_pick']:.0%}" + (f" · t {R.loc[r, f'hac_t_minus_cash_{h}']:+.1f}" if pd.notna(R.loc[r, f"hac_t_minus_cash_{h}"]) else "") for r in rules]
        fig.add_trace(go.Bar(y=rules, x=v, orientation="h", marker_color=colors, text=txt, textposition="outside", textfont={"size": 10}, showlegend=False,
                             customdata=np.c_[R[f"median_{h}"], R[f"beat_stay_{h}"], R[f"when_pick_mean_{h}"]], hovertemplate="%{y}: mean %{x:+.1%}, median %{customdata[0]:+.1%}, beat staying %{customdata[1]:.0%}, when it had a pick %{customdata[2]:+.1%}<extra></extra>"), row=1, col=c_)
        fig.add_vline(x=cash, line={"color": INK, "dash": "dash", "width": 1}, row=1, col=c_)
        fig.update_xaxes(tickformat=".0%", range=[min(v.min() * 1.3, -0.01), max(v.max() * 2.1, 0.09)], row=1, col=c_)
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, f"What each replacement rule earned after a SELL flip (n = {int(R['events'].iloc[0])} flips of the five assets)", 560, "Dashed line = cash. Every rule is applied to the same SELL flips; a rule with no candidate that week falls back to cash, so 'had a pick' says how often it actually did something.", unified=False, legend_h=False)


def rp_cells():
    ce = _csv("8_cells")
    h = C.RP_FWD[-1]
    devs = ["lagging", "neutral", "leading"]
    labels = [f"candidate {d}<br>(z vs seller {'≤ −1' if d == 'lagging' else ('≥ +1' if d == 'leading' else 'between')})" for d in devs]
    allrow = ce[(ce["horizon_weeks"] == h) & (ce["deviation"] == "any") & (ce["correlated"] == "any") & (ce["cand_state"] == "any")].iloc[0]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=(f"Candidate's return over the next {h} weeks, by the candidate's OWN signal", f"… by whether it was correlated with the seller (52w corr ≥ {C.RP_CORR_MIN})"), horizontal_spacing=0.05)
    sub = ce[(ce["horizon_weeks"] == h) & (ce["correlated"] == "any") & (ce["cand_state"].isin(["BUY", "SELL"])) & (ce["deviation"].isin(devs))]
    sub2 = ce[(ce["horizon_weeks"] == h) & (ce["correlated"].isin(["True", "False"])) & (ce["cand_state"] == "any") & (ce["deviation"].isin(devs))]
    for c_, (frame, key, vals, cols_, names) in enumerate(((sub, "cand_state", ["BUY", "SELL"], [BUY, SELL], ["candidate in BUY", "candidate in SELL"]), (sub2, "correlated", ["True", "False"], [CAT[0], MUTED], ["correlated", "not correlated"])), start=1):
        for val, col, nm in zip(vals, cols_, names):
            s = frame[frame[key] == val].set_index("deviation").reindex(devs)
            fig.add_trace(go.Bar(x=labels, y=s["cand_mean"], marker_color=col, name=nm, text=[f"n {int(x)}" if pd.notna(x) else "" for x in s["rows"]], textposition="outside", textfont={"size": 10},
                                 customdata=np.c_[s["rows"], s["share_cand_beat_cash"], s["stay_mean"]], hovertemplate="%{y:+.1%} (n %{customdata[0]}, beat cash %{customdata[1]:.0%}; the seller made %{customdata[2]:+.1%})<extra>" + nm + "</extra>"), row=1, col=c_)
        fig.add_hline(y=allrow["cash_mean"], line={"color": INK, "dash": "dash", "width": 1}, annotation_text="cash", annotation_position="top right", row=1, col=c_)
        fig.add_hline(y=allrow["stay_mean"], line={"color": SELL, "dash": "dot", "width": 1}, annotation_text="stay in the seller", annotation_position="bottom right", row=1, col=c_)
    fig.update_yaxes(tickformat=".1%"); fig.update_layout(barmode="group")
    return _layout(fig, "After a SELL flip: how the candidate did, by its deviation from the seller, its own signal, and its correlation", 540, "One bar = all (SELL flip, candidate) pairs in that cell. n counts candidate rows, not distinct flips.", unified=False)


def rp_z_path():
    zp = _csv("8_z_path")
    sellers = [a for a in C.ASSETS if a in set(zp["seller"])]
    ev = _csv("8_events")
    fig = make_subplots(rows=1, cols=len(sellers), shared_yaxes=True, subplot_titles=[f"{a} flips to SELL (n {int((ev['seller'] == a).sum())})" for a in sellers], horizontal_spacing=0.03)
    for c_, a in enumerate(sellers, start=1):
        sub = zp[zp["seller"] == a]
        for b, g in sub.groupby("candidate"):
            if b == "all candidates":
                fig.add_trace(go.Scatter(x=g["k"], y=g["mean_z"], mode="lines", line={"color": INK, "width": 2.6}, name="all candidates (average)", legendgroup="all", showlegend=c_ == 1, hovertemplate="k=%{x}: z %{y:+.2f}<extra>all</extra>"), row=1, col=c_)
            else:
                fig.add_trace(go.Scatter(x=g["k"], y=g["mean_z"], mode="lines", line={"color": COL[b], "width": 1.4}, name=b, legendgroup=b, showlegend=c_ == 1, hovertemplate="k=%{x}: z %{y:+.2f}<extra>" + b + "</extra>"), row=1, col=c_)
        fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}, row=1, col=c_); fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=1, col=c_)
        fig.update_xaxes(title_text="weeks from the flip", title_font={"size": 10}, row=1, col=c_)
    fig.update_yaxes(range=[-1.6, 1.6]); fig.update_yaxes(title_text="z of the candidate vs the seller (+ = candidate ahead)", row=1, col=1)
    return _layout(fig, "The deviation of every other asset against the seller, before and after the seller's SELL flip (average over flips)", 480, "A line that keeps rising after week 0 means the candidate kept pulling ahead of the seller after the seller's signal fired.", unified=False, legend_y=-0.22)


def rp_events(seller: str):
    ep = _csv("8_event_paths", parse_dates=["date"]); P = _csv("8_rule_picks", parse_dates=["date"])
    h = C.RP_FWD[-1]
    sub = ep[ep["seller"] == seller]
    dates = sorted(sub["date"].unique()); n = len(dates); cols = 4; rows = int(np.ceil(n / cols))
    titles = []
    for d in dates:
        pk = P[(P["seller"] == seller) & (P["date"] == d)].set_index("rule")
        stay = pk.loc["stay in the seller", f"ret_{h}"] if "stay in the seller" in pk.index else np.nan
        cash = pk.loc["cash", f"ret_{h}"] if "cash" in pk.index else np.nan
        lead = pk.loc["most leading (any corr)"] if "most leading (any corr)" in pk.index else None
        titles.append(f"{pd.Timestamp(d).date()} · next {h} wk: stay {stay:+.0%}, cash {cash:+.0%}" + (f", most-leading pick {lead['pick']} {lead[f'ret_{h}']:+.0%}" if lead is not None and pd.notna(lead[f"ret_{h}"]) else ""))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, shared_yaxes=True, vertical_spacing=0.12 / max(1, rows / 3), horizontal_spacing=0.04)
    for k, d in enumerate(dates):
        r_, c_ = k // cols + 1, k % cols + 1
        g = sub[sub["date"] == d]
        for b, gg in g.groupby("candidate"):
            fig.add_trace(go.Scatter(x=gg["k"], y=gg["z"], mode="lines", line={"color": COL[b], "width": 1.5}, name=b, legendgroup=b, showlegend=k == 0, hovertemplate="k=%{x}: z %{y:+.2f}<extra>" + b + "</extra>"), row=r_, col=c_)
        fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}, row=r_, col=c_); fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=r_, col=c_)
        for lvl in (1, -1):
            fig.add_hline(y=lvl, line={"color": GRID, "width": 0.8}, row=r_, col=c_)
        fig.update_yaxes(range=[-3.5, 3.5], row=r_, col=c_)
    for ann in fig.layout.annotations:
        ann.font.size = 10.5
    return _layout(fig, f"Every SELL flip of {seller}: the z-score of each other asset against {seller} from {C.RP_PATH_WEEKS} weeks before to {C.RP_PATH_WEEKS} after (+ = that asset ahead of {seller})", 300 * rows + 130, unified=False, legend_y=-0.03)


# ============================================================ the review of 2026-10-19 ==
def gold_vs_group():
    """B2: is there a relationship between how coupled the indices are (G) and gold's correlation with each index? A scatter, a straight line, and the decile means."""
    wk = _csv("0_gold_vs_group_weekly", index_col=0, parse_dates=True); ft = _csv("0_gold_vs_group_fit", index_col=0); bn = _csv("0_gold_vs_group_bins")
    gps = [p for p in PAIR_ORDER if p in wk.columns]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("G against each gold pair's correlation (every week a dot)", "G against the gap SC100×Gold − N100×Gold"), horizontal_spacing=0.08)
    xs = np.linspace(wk["G"].min(), wk["G"].max(), 20)
    for p in gps:
        d = wk[["G", p]].dropna()
        fig.add_trace(go.Scatter(x=d["G"], y=d[p], mode="markers", marker={"color": _pair_col(p), "size": 4, "opacity": 0.35}, name=_pair_name(p), legendgroup="pair_" + p, hovertemplate="G %{x:.2f}: corr %{y:+.2f}<extra>" + _pair_name(p) + "</extra>"), row=1, col=1)
        f = ft.loc[p]
        fig.add_trace(go.Scatter(x=xs, y=f["intercept"] + f["slope"] * xs, mode="lines", line={"color": _pair_col(p), "width": 2.2, "dash": PAIR_DASH.get(p, "solid")}, name=_pair_name(p) + f" fit (slope {f['slope']:+.2f}, t {f['nw_t_slope']:+.1f})", legendgroup="pair_" + p, showlegend=False, hoverinfo="skip"), row=1, col=1)
        b = bn[bn["y"] == p]
        fig.add_trace(go.Scatter(x=(b["G_lo"] + b["G_hi"]) / 2, y=b["mean_y"], mode="markers", marker={"color": _pair_col(p), "size": 10, "symbol": "diamond", "line": {"color": INK, "width": 1}}, name=_pair_name(p) + " — decile means", legendgroup="pair_" + p, showlegend=False, customdata=np.c_[b["G_lo"], b["G_hi"], b["weeks"]], hovertemplate="G %{customdata[0]:.2f}–%{customdata[1]:.2f} (n %{customdata[2]}): mean corr %{y:+.2f}<extra>" + _pair_name(p) + " decile</extra>"), row=1, col=1)
    if "gap_SC100_minus_N100" in wk.columns:
        d = wk[["G", "gap_SC100_minus_N100"]].dropna(); f = ft.loc["gap_SC100_minus_N100"]; b = bn[bn["y"] == "gap_SC100_minus_N100"]
        fig.add_trace(go.Scatter(x=d["G"], y=d["gap_SC100_minus_N100"], mode="markers", marker={"color": INK2, "size": 4, "opacity": 0.35}, name="the gap, every week", hovertemplate="G %{x:.2f}: gap %{y:+.2f}<extra>gap</extra>"), row=1, col=2)
        fig.add_trace(go.Scatter(x=xs, y=f["intercept"] + f["slope"] * xs, mode="lines", line={"color": INK, "width": 2.2}, name=f"gap fit (slope {f['slope']:+.2f}, t {f['nw_t_slope']:+.1f}, R² {f['r2']:.2f})", hoverinfo="skip"), row=1, col=2)
        fig.add_trace(go.Scatter(x=(b["G_lo"] + b["G_hi"]) / 2, y=b["mean_y"], mode="markers", marker={"color": INK, "size": 10, "symbol": "diamond", "line": {"color": INK, "width": 1}}, name="gap — decile means", customdata=np.c_[b["G_lo"], b["G_hi"], b["weeks"]], hovertemplate="G %{customdata[0]:.2f}–%{customdata[1]:.2f} (n %{customdata[2]}): mean gap %{y:+.2f}<extra>decile</extra>"), row=1, col=2)
    for c in (1, 2):
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=1, col=c)
        fig.update_xaxes(title_text="the group correlation G (how coupled the four indices were that week)", row=1, col=c)
    fig.update_yaxes(range=[-0.8, 0.8])
    pr = ft.loc[gps, "pearson"]; tmax = ft.loc[gps + (["gap_SC100_minus_N100"] if "gap_SC100_minus_N100" in ft.index else []), "nw_t_slope"].abs().max()
    return _layout(fig, f"No relationship to fit: G explains none of gold's correlations (Pearson {pr.min():+.2f} to {pr.max():+.2f}, R² ≤ {ft['r2'].max():.2f}, every slope's |t| at most {np.ceil(tmax * 10) / 10:.1f}; a curve fits no better)", 560,
                   "If the gold pairs moved opposite to the index block — coupling to the indices when they decouple — the dots would slope down to the right and the diamonds (the mean inside each tenth of G) would fall from left to right. They do not: the lines are flat, the diamonds wander. Hover a diamond for its G range and week count; the legend names each fit's slope and Newey-West t.", unified=False, legend_y=-0.2,
                   subtitle="Each week's group correlation G (x) against the same week's 13-week gold-pair correlations (y, left) and the SC100×Gold − N100×Gold gap (right); a straight line fitted to each, with the means inside the deciles of G")


def gap_reversion():
    """B2: the SC100×Gold − N100×Gold gap — does it mean-revert, and what followed its extremes?"""
    wk = _csv("0_gold_gap_weekly", index_col=0, parse_dates=True); st = _csv("0_gold_gap_stats").iloc[0]; ae = _csv("0_gold_gap_after_extreme")
    fig = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]], row_heights=[0.56, 0.44], vertical_spacing=0.16, horizontal_spacing=0.1,
                        subplot_titles=(f"The gap week by week — y = SC100×Gold corr minus N100×Gold corr; mean {st['mean']:+.2f}, sd {st['sd']:.2f} over the whole history, so the lines sit at {st['mean'] + C.GAP_SD * st['sd']:+.2f} and {st['mean'] - C.GAP_SD * st['sd']:+.2f}", "Did the gap narrow afterwards?", "What Smallcap − Nifty and gold did afterwards (t+1 …)"))
    g = wk["gap"]
    fig.add_trace(go.Scatter(x=g.index, y=g, mode="lines", line={"color": INK, "width": 1.2}, name="gap = SC100×Gold − N100×Gold", hovertemplate="%{y:+.2f}<extra>gap</extra>"), row=1, col=1)
    up_v, lo_v = float(wk["upper"].iloc[0]), float(wk["lower"].iloc[0])
    x2 = [g.index[0], g.index[-1]]
    fig.add_trace(go.Scatter(x=x2, y=[up_v] * 2, mode="lines", line={"color": SELL, "width": 1, "dash": "dash"}, name=f"+{C.GAP_SD:g} sd = {up_v:+.2f}", hovertemplate=f"+{C.GAP_SD:g} sd line: {up_v:+.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x2, y=[st["mean"]] * 2, mode="lines", line={"color": MUTED, "width": 1, "dash": "dot"}, name=f"mean = {st['mean']:+.2f}", hovertemplate=f"mean: {st['mean']:+.3f}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x2, y=[lo_v] * 2, mode="lines", line={"color": BUY, "width": 1, "dash": "dash"}, name=f"−{C.GAP_SD:g} sd = {lo_v:+.2f}", hovertemplate=f"−{C.GAP_SD:g} sd line: {lo_v:+.3f}<extra></extra>"), row=1, col=1)
    for yv, txt, col_ in ((up_v, f"+{C.GAP_SD:g} sd = {up_v:+.2f}", SELL), (st["mean"], f"mean {st['mean']:+.2f}", INK2), (lo_v, f"−{C.GAP_SD:g} sd = {lo_v:+.2f}", BUY)):
        fig.add_annotation(x=g.index[-1], y=yv, xref="x", yref="y", text=txt, showarrow=False, xanchor="left", xshift=4, font={"size": 10.5, "color": col_})
    up = (g > wk["upper"]).fillna(False); dn = (g < wk["lower"]).fillna(False)
    su = g.index[up & ~up.shift(1, fill_value=False)]; sd_ = g.index[dn & ~dn.shift(1, fill_value=False)]
    fig.add_trace(go.Scatter(x=su, y=g.loc[su], mode="markers", marker={"color": SELL, "size": 8, "symbol": "triangle-down"}, name=f"first week above +{C.GAP_SD:g} sd", hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f}<extra>above</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sd_, y=g.loc[sd_], mode="markers", marker={"color": BUY, "size": 8, "symbol": "triangle-up"}, name=f"first week below −{C.GAP_SD:g} sd", hovertemplate="%{x|%Y-%m-%d}: %{y:+.2f}<extra>below</extra>"), row=1, col=1)
    fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=1, col=1)
    labs = {e: (f"above +{C.GAP_SD:g} sd" if "above" in e else f"below −{C.GAP_SD:g} sd") for e in ae["extreme"].unique()}
    for h, col_, op in zip(C.GAP_FWD, (INK2, INK), (0.55, 0.9)):
        s = ae[ae["horizon_weeks"] == h]
        fig.add_trace(go.Bar(x=[labs[e] for e in s["extreme"]], y=s["share_gap_narrowed"], marker_color=col_, opacity=op, name=f"share that narrowed within {h} weeks", customdata=s["events"], text=[f"{v:.0%}" for v in s["share_gap_narrowed"]], textposition="outside", hovertemplate="%{x}: %{y:.0%} of %{customdata} events<extra>" + f"{h} wk</extra>"), row=2, col=1)
    for h, op in zip(C.GAP_FWD, (0.55, 0.9)):
        s = ae[ae["horizon_weeks"] == h]
        x = [labs[e] for e in s["extreme"]]
        if "SC100_minus_N100_mean" in s.columns:
            fig.add_trace(go.Bar(x=x, y=s["SC100_minus_N100_mean"], marker_color=COL["SC100"], opacity=op, name=f"SC100 − N100 over the next {h} wk", customdata=np.c_[s["SC100_minus_N100_hit"], s["SC100_minus_N100_all_weeks_mean"], s["events"]], text=[f"{v:+.1%}" for v in s["SC100_minus_N100_mean"]], textposition="outside", textfont={"size": 9}, hovertemplate="%{x}: %{y:+.1%} (SC100 ahead %{customdata[0]:.0%} of %{customdata[2]} events; all weeks %{customdata[1]:+.1%})<extra>" + f"SC100−N100 {h} wk</extra>"), row=2, col=2)
        if "gold_mean" in s.columns:
            fig.add_trace(go.Bar(x=x, y=s["gold_mean"], marker_color=COL["GoldMCX"], opacity=op, name=f"gold over the next {h} wk", customdata=np.c_[s["gold_hit"], s["gold_all_weeks_mean"], s["events"]], text=[f"{v:+.1%}" for v in s["gold_mean"]], textposition="outside", textfont={"size": 9}, hovertemplate="%{x}: %{y:+.1%} (positive %{customdata[0]:.0%} of %{customdata[2]} events; all weeks %{customdata[1]:+.1%})<extra>" + f"gold {h} wk</extra>"), row=2, col=2)
    n_up = int(ae[(ae["extreme"].str.contains("above")) & (ae["horizon_weeks"] == C.GAP_FWD[0])]["events"].iloc[0]); n_dn = int(ae[(ae["extreme"].str.contains("below")) & (ae["horizon_weeks"] == C.GAP_FWD[0])]["events"].iloc[0])
    fig.layout.annotations[1].text = f"Did the gap narrow afterwards? ({n_up} runs above, {n_dn} below)"; fig.layout.annotations[2].text = f"What Smallcap − Nifty and gold did afterwards (t+1 …; same {n_up} / {n_dn} runs)"
    fig.update_yaxes(range=[-0.9, 0.9], title_text="gap (correlation units)", dtick=0.25, row=1, col=1); fig.update_yaxes(tickformat=".0%", range=[0, 1.15], row=2, col=1); fig.update_yaxes(tickformat=".0%", row=2, col=2)
    fig.update_layout(barmode="group", margin={"r": 110})
    a4 = ae[(ae["horizon_weeks"] == C.GAP_FWD[0])]
    nar = a4["share_gap_narrowed"].mean()
    rel = a4["SC100_minus_N100_mean"].iloc[0] if "SC100_minus_N100_mean" in a4.columns and len(a4) else np.nan; relh = a4["SC100_minus_N100_hit"].iloc[0] if "SC100_minus_N100_hit" in a4.columns and len(a4) else np.nan
    return _layout(fig, f"The gap reverts (half-life {st['half_life_weeks']:.0f} weeks) because a rolling correlation must — it narrowed within {C.GAP_FWD[0]} weeks {nar:.0%} of the time; in returns, a thin tilt: after a +{C.GAP_SD:g} sd gap Smallcap beat Nifty over {C.GAP_FWD[0]} weeks {relh:.0%} of the time ({int(a4['events'].iloc[0])} runs, {rel:+.1%})", 760,
                   f"The gap is the difference of two 13-week rolling correlations of a series (gold) that is unrelated to both — so it is noise around zero that the window itself pulls back (autocorrelation {st['autocorr_1w']:.2f} at one week, {st['autocorr_13w']:+.2f} at 13; ADF p {st['adf_p']:.3f}). The narrowing is therefore guaranteed and is not a trade: the gap is not a price. The only use would be if an extreme gap said something about returns; the bottom-right panel tests that on the first week of each run beyond ±{C.GAP_SD:g} sd — {int(ae['events'].max())} or fewer events, read as a tendency, no t-stat.", unified=False, legend_y=-0.1,
                   subtitle="Top: the gap week by week with its ±1 sd lines and the first week of each run beyond them. Bottom left: how often the gap had moved back toward zero 4 and 13 weeks later. Bottom right: Smallcap minus Nifty, and gold, over the same forward weeks")


def gold_after_break():
    """B2: after the index correlation structure breaks (a decorrelation episode starts), how long does each gold pair's correlation stay positive?"""
    avg = _csv("0_gold_after_break_avg"); pl = _csv("0_gold_after_break_paths"); per = _csv("0_gold_after_break_episodes")
    gps = [p for p in PAIR_ORDER if p in set(avg["pair"])]
    mp = "Micro250|GoldMCX"
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Each gold pair, averaged over the episodes it exists in (week 0 = the episode's first week)", f"Micro250 × Gold in each of its {per[per['pair'] == mp]['episode'].nunique()} episodes"), horizontal_spacing=0.08)
    for p in gps:
        a = avg[avg["pair"] == p]
        fig.add_trace(go.Scatter(x=a["k"], y=a["mean_corr"], mode="lines+markers", line=_pair_line(p, 2.4 if p == mp else 1.6), marker={"size": 4}, name=_pair_name(p) + f" (n {int(a['n'].max())} episodes)", legendgroup="pair_" + p, customdata=np.c_[a["share_positive"], a["n"], a["median_corr"]], hovertemplate="week %{x}: mean %{y:+.2f}, median %{customdata[2]:+.2f}, positive in %{customdata[0]:.0%} of %{customdata[1]}<extra>" + _pair_name(p) + "</extra>"), row=1, col=1)
    ep = per[per["pair"] == mp]
    for i, (_, e) in enumerate(ep.iterrows()):
        s = pl[(pl["pair"] == mp) & (pl["episode"] == e["episode"])]
        fig.add_trace(go.Scatter(x=s["k"], y=s["corr"], mode="lines", line={"color": CAT[i % len(CAT)], "width": 1.8}, name=f"{e['episode']}: positive {int(e['weeks_positive_of_first_' + str(C.GOLD_AFTER_BREAK)])} of {int(e['weeks_available'])} wk" + (f", first ≤ 0 at week {int(e['first_week_at_or_below_zero'])}" if pd.notna(e["first_week_at_or_below_zero"]) else ", never ≤ 0"), legendgroup="ep" + str(i), hovertemplate="week %{x}: %{y:+.2f}<extra>" + str(e["episode"]) + "</extra>"), row=1, col=2)
    for c in (1, 2):
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=1, col=c)
        fig.update_xaxes(title_text="weeks after the decorrelation episode began", dtick=2, row=1, col=c)
    fig.update_yaxes(range=[-0.8, 0.9])
    pos0 = ep["positive_at_break"].mean() if len(ep) else np.nan; medpos = ep["weeks_positive_of_first_" + str(C.GOLD_AFTER_BREAK)].median() if len(ep) else np.nan
    long = ep.sort_values("longest_positive_run", ascending=False).iloc[0] if len(ep) else None
    return _layout(fig, f"There is no consistent positive Micro250 × Gold correlation after a break to sustain: it was positive at the break in {int(ep['positive_at_break'].sum())} of its {len(ep)} episodes ({long['episode']}, where it then stayed positive {int(long['longest_positive_run'])} weeks); the median episode has it positive {medpos:.0f} of the first {C.GOLD_AFTER_BREAK} weeks", 540,
                   f"Left: each gold pair's 13-week correlation from the first week of a decorrelation episode out to week {C.GOLD_AFTER_BREAK}, averaged over the episodes the pair exists in (hover: the share of episodes in which it was positive that week). Right: Micro250 × Gold in each of its own episodes — the components of its average; the legend counts the positive weeks. Micro250 starts in 2022, so it is in the later episodes only.", unified=False, legend_y=-0.14,
                   subtitle="The gold pairs' correlation after each decorrelation episode begins — the average path, and Micro250 × Gold episode by episode")


def index_pair_vs_gold_pair():
    """B2 (review of 2026-10-22): do an index pair's correlation and a gold pair's correlation move in opposite directions? Both series on one panel, their difference underneath, a menu for every combination."""
    PC = _csv("6_pair_corr13", index_col=0, parse_dates=True); tb = _csv("0_index_vs_gold_pairs")
    ips = [p for p in PAIR_ORDER if p in PC.columns and "Gold" not in p]; gps = [p for p in PAIR_ORDER if p in PC.columns and "Gold" in p]
    combos = [(a, b) for a in ips for b in gps]
    d0 = tuple(C.PAIR_VS_GOLD_DEFAULT) if tuple(C.PAIR_VS_GOLD_DEFAULT) in combos else combos[0]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45], vertical_spacing=0.09, subplot_titles=("The two correlations, week by week", "Their difference (index pair − gold pair), with its mean and ±1 sd lines"))
    n_per = 6
    for a, b in combos:
        vis = (a, b) == d0
        row = tb[(tb["index_pair"] == a) & (tb["gold_pair"] == b)].iloc[0]
        diff = PC[a] - PC[b]; m, s = row["diff_mean"], row["diff_sd"]
        fig.add_trace(go.Scatter(x=PC.index, y=PC[a], mode="lines", line=_pair_line(a, 1.6), name=_pair_name(a), visible=vis, hovertemplate="%{y:.2f}<extra>" + _pair_name(a) + "</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=PC.index, y=PC[b], mode="lines", line={"color": _pair_col(b), "width": 1.6, "dash": "solid"}, name=_pair_name(b), visible=vis, hovertemplate="%{y:.2f}<extra>" + _pair_name(b) + "</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=PC.index, y=diff, mode="lines", line={"color": INK, "width": 1.2}, name=f"difference: {_pair_name(a)} − {_pair_name(b)}", visible=vis, hovertemplate="%{y:+.2f}<extra>difference</extra>"), row=2, col=1)
        x2 = [PC.index[0], PC.index[-1]]                                   # a flat line needs two points, not 547
        fig.add_trace(go.Scatter(x=x2, y=[m, m], mode="lines", line={"color": MUTED, "width": 1, "dash": "dot"}, name=f"mean of the difference = {m:+.2f}", visible=vis, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x2, y=[m + s] * 2, mode="lines", line={"color": SELL, "width": 1, "dash": "dash"}, name=f"+1 sd = {m + s:+.2f}", visible=vis, hoverinfo="skip"), row=2, col=1)
        fig.add_trace(go.Scatter(x=x2, y=[m - s] * 2, mode="lines", line={"color": BUY, "width": 1, "dash": "dash"}, name=f"−1 sd = {m - s:+.2f}", visible=vis, hoverinfo="skip"), row=2, col=1)
    def _lab(a, b):
        row = tb[(tb["index_pair"] == a) & (tb["gold_pair"] == b)].iloc[0]
        return f"{_pair_name(a)} vs {_pair_name(b)}  (corr {row['pearson_levels']:+.2f})"
    buttons = [{"label": _lab(a, b), "method": "update", "args": [{"visible": [(i // n_per) == k for i in range(n_per * len(combos))]}]} for k, (a, b) in enumerate(combos)]
    fig.update_layout(updatemenus=[{"type": "dropdown", "x": 1.0, "xanchor": "right", "y": 1.12, "yanchor": "top", "showactive": True, "bgcolor": SURFACE, "bordercolor": GRID, "font": {"size": 11}, "active": combos.index(d0), "buttons": buttons}])
    fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=1, col=1); fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=2, col=1)
    fig.update_yaxes(range=[-0.8, 1.05], title_text="13-week correlation", row=1, col=1); fig.update_yaxes(range=[-0.8, 1.8], title_text="difference", row=2, col=1)
    r0 = tb[(tb["index_pair"] == d0[0]) & (tb["gold_pair"] == d0[1])].iloc[0]
    strongest = tb.loc[tb["pearson_levels"].idxmin()]
    return _layout(fig, f"{_pair_name(d0[0])} and {_pair_name(d0[1])} do not move in opposite directions: their week-by-week correlation is {r0['pearson_levels']:+.2f} (t {r0['nw_t_levels']:+.1f}), their 4-week changes {r0['pearson_4w_changes']:+.2f}, their 13-week changes {r0['pearson_13w_changes']:+.2f}; the most opposite combination anywhere is {_pair_name(strongest['index_pair'])} vs {_pair_name(strongest['gold_pair'])} at {strongest['pearson_levels']:+.2f}", 680,
                   f"If the two moved opposite, the top panel's lines would mirror each other and the difference below would swing far beyond its ±1 sd lines whenever one of them moved. What the eye catches instead is the handful of V-shaped dips in the index pair (the decorrelation episodes) during which the gold pair happened to be rising — a few moments, not a relationship: across all {r0['weeks']} weeks the two series are unrelated. The menu switches to any index pair × gold pair; the number in each menu entry is that combination's week-by-week correlation, and the table under the chart has every combination's numbers.", unified=True, legend_y=-0.1,
                   subtitle="One index pair's 13-week correlation against one gold pair's, week by week, and the difference between them")


def vol_state_box():
    """B7: the distribution behind the averages — every week as a point, with the box (p25 / median / p75), the whiskers (min / max) and the mean (dashed), by the asset's own signal state."""
    pts = _csv("5_state_points", parse_dates=["date"])
    assets = [a for a in C.ASSETS if a in set(pts["asset"])]
    has_vix = "vix" in pts.columns and pts["vix"].notna().any()
    measures = [("abs_weekly_return", "|weekly return| — the components of the realised vol", ".1%"), ("vol_13w", "13-week realised vol, week by week", ".0%")] + ([("vix", "India VIX level", ".1f")] if has_vix else [])
    fig = make_subplots(rows=1, cols=len(measures), subplot_titles=[m[1] for m in measures], horizontal_spacing=0.06)
    for c, (m, _, fmt) in enumerate(measures, start=1):
        for stt, col_ in (("BUY", BUY), ("SELL", SELL)):
            sub = pts[pts["state"] == stt].dropna(subset=[m])
            fig.add_trace(go.Box(x=sub["asset"], y=sub[m], name=f"the asset's own {stt} weeks", legendgroup=stt, showlegend=c == 1, marker={"color": col_, "size": 2.5, "opacity": 0.35}, line={"color": col_, "width": 1.2}, fillcolor="rgba(0,0,0,0)",
                                 boxpoints="all", jitter=0.5, pointpos=0, boxmean=True, offsetgroup=stt, customdata=sub["date"].dt.strftime("%Y-%m-%d"), hovertemplate="%{customdata}: %{y:" + fmt + "}<extra>" + stt + "</extra>"), row=1, col=c)
        fig.update_yaxes(tickformat=fmt if fmt != ".1f" else None, row=1, col=c)
        fig.update_xaxes(categoryorder="array", categoryarray=assets, row=1, col=c)
    fig.update_layout(boxmode="group")
    sd = _csv("5_state_dist")
    def _m(a, s, m, col_):
        g = sd[(sd["asset"] == a) & (sd["state"] == s) & (sd["measure"] == m)]
        return g[col_].iloc[0] if len(g) else np.nan
    vb, vs_ = _m("N100", "BUY", "vix", "median"), _m("N100", "SELL", "vix", "median"); mb, ms = _m("N100", "BUY", "vix", "mean"), _m("N100", "SELL", "vix", "mean"); mx = _m("N100", "SELL", "vix", "max")
    ab, as_ = _m("N100", "BUY", "abs_weekly_return", "median"), _m("N100", "SELL", "abs_weekly_return", "median")
    title = (f"The medians say what the means say: N100's typical weekly move is {ab:.1%} in BUY vs {as_:.1%} in SELL" + (f", the VIX {vb:.1f} vs {vs_:.1f} (means {mb:.1f} vs {ms:.1f} — the SELL mean is pulled up by one week at {mx:.0f}, 2020)" if has_vix else ""))
    return _layout(fig, title, 560,
                   "Each dot is one week. The box runs from the 25th to the 75th percentile with the median as the line inside; the dashed line is the mean; the whiskers reach the furthest weeks. Where the mean sits well above the median (the VIX in SELL weeks) a few extreme weeks are pulling it. Hover a dot for its date. The table under the chart has the same numbers (min / p25 / median / mean / p75 / max) for each asset and state.", unified=False, legend_y=-0.12,
                   subtitle="Every week's values by the asset's own signal state: |weekly return|, the 13-week realised vol, and the India VIX level — the distribution behind B7's averages")


def vix_flip_leadlag():
    """B7: is the VIX a leading or a lagging indicator of the SELL flip? The VIX around SELL flips, the flips around VIX surges, and the shares both ways."""
    pth = _csv("5_vix_flip_paths"); sm = _csv("5_vix_flip_summary", index_col=0); sg = _csv("5_vix_surges"); fe = _csv("5_vix_flip_events")
    W = C.VIX_FLIP_WINDOW
    fig = make_subplots(rows=1, cols=3, subplot_titles=(f"The VIX around the {int(pth['n_flips'].max())} SELL flips (ratio to the flip week)", f"Signals flipping to SELL around the {len(sg)} VIX surges", f"Both ways, per index (within {W} weeks)"), horizontal_spacing=0.07)
    fig.add_trace(go.Scatter(x=pth["k"], y=pth["vix_ratio_to_flip_week_mean"], mode="lines+markers", line={"color": COL["VIX"], "width": 2.2}, marker={"size": 5}, name="VIX ÷ VIX in the flip week — mean", hovertemplate="k=%{x}: %{y:.2f}<extra>mean</extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=pth["k"], y=pth["vix_ratio_to_flip_week_median"], mode="lines+markers", line={"color": COL["VIX"], "width": 1.6, "dash": "dot"}, marker={"size": 4}, name="— median", hovertemplate="k=%{x}: %{y:.2f}<extra>median</extra>"), row=1, col=1)
    fig.add_hline(y=1, line={"color": AXIS, "width": 0.8}, row=1, col=1); fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}, row=1, col=1)
    fig.add_vrect(x0=-W - 0.5, x1=0, fillcolor=GRID, opacity=0.25, line_width=0, row=1, col=1)
    fig.add_trace(go.Bar(x=pth["k"], y=pth["share_signals_flipping_to_SELL"], marker_color=SELL, name="share of index signals flipping to SELL that week", customdata=pth["n_signal_weeks"], hovertemplate="k=%{x}: %{y:.1%} of %{customdata} signal-weeks<extra>flips</extra>"), row=1, col=2)
    fig.add_vline(x=0, line={"color": AXIS, "dash": "dash"}, row=1, col=2)
    fig.update_xaxes(title_text="weeks from the SELL flip", dtick=2, row=1, col=1); fig.update_xaxes(title_text="weeks from the VIX surge (0 = the surge week)", dtick=2, row=1, col=2)
    fig.update_yaxes(tickformat=".0%", row=1, col=2)
    assets = [a for a in C.EQUITY if a in sm.index]
    for col_, name, c_ in (("share_flips_with_surge_before", f"SELL flips with a VIX surge in the {W} weeks BEFORE (VIX led)", COL["VIX"]), ("share_flips_with_surge_in_week", "… with the surge in the flip week itself (same event)", MUTED), ("share_flips_with_surge_after", f"… with a surge in the {W} weeks AFTER (VIX lagged)", BUY),
                          ("share_surges_followed_by_flip", f"VIX surges followed by this index's SELL flip within {W} weeks", "#8c564b"), ("share_surges_already_in_SELL", "VIX surges with this index already in SELL", "#c3c2b7")):
        fig.add_trace(go.Bar(x=assets, y=sm.loc[assets, col_], name=name, marker_color=c_, hovertemplate="%{x}: %{y:.0%}<extra>" + name + "</extra>"), row=1, col=3)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.05], row=1, col=3)
    fig.update_layout(barmode="group")
    p0 = pth.set_index("k")
    n1 = sm.loc[C.BENCHMARK] if C.BENCHMARK in sm.index else sm.iloc[0]
    peak_k = int(p0["share_signals_flipping_to_SELL"].idxmax())
    up4 = fe[fe["vix_data"]]["vix_change_4w_before"].mean(); up4m = fe[fe["vix_data"]]["vix_change_4w_before"].median()
    return _layout(fig, f"The VIX and the SELL flip are one event, the VIX a few weeks ahead: it is up {up4:+.0%} (median {up4m:+.0%}) over the 4 weeks before a flip and the flip week is the commonest surge week (k = {peak_k:+d}); but a surge alone is a weak warning — {n1['share_surges_already_in_SELL']:.0%} of surges find N100 already in SELL and only {n1['share_surges_followed_by_flip']:.0%} are followed by its flip", 540,
                   f"A VIX surge = the first week the VIX is up at least {C.VIX_SURGE:.0%} over {C.VIX_SURGE_WEEKS} weeks ({len(sg)} since 2016). Left: the VIX level as a ratio to its level in the flip week, lined up at every index's SELL flips (grey band = the {W} weeks before) — rising into the flip (the supertrend needs the price to cross its band, which takes a few weeks of selling), mean still rising after it (2020), median falling. Middle: the share of the four index signals that flipped to SELL in each week around a surge — the spike at week 0 says the surge and the flip usually land in the same week. Right: the same relationship counted per index, both ways.", unified=False, legend_y=-0.14,
                   subtitle="The VIX against the supertrend's SELL flips, both ways: the VIX around flips, the flips around VIX surges, and the shares per index")


def regime_scatter():
    """B8: the components of the regime averages — every week as a dot, the pair's own vol against its correlation, with the median inside each fifth."""
    rw = _csv("5_regime_weekly", index_col=0, parse_dates=True)
    prs = [p for p in PAIR_ORDER if p in rw.columns and f"q_{p}" in rw.columns]
    ncol = 5; nrow = int(np.ceil(len(prs) / ncol))
    buckets = [f"Q{i + 1}" for i in range(C.VOL_BUCKETS)]
    fig = make_subplots(rows=nrow, cols=ncol, subplot_titles=[_pair_name(p) for p in prs], horizontal_spacing=0.04, vertical_spacing=0.14, shared_yaxes=True)
    qcol = {"Q1": "#2a78d6", "Q2": "#7fb3e6", "Q3": "#c3c2b7", "Q4": "#f0a27a", "Q5": "#e34948"}
    for i, p in enumerate(prs):
        r_, c_ = i // ncol + 1, i % ncol + 1
        d = rw[[p, f"v_{p}", f"q_{p}"]].dropna()
        for b in buckets:
            s = d[d[f"q_{p}"] == b]
            fig.add_trace(go.Scatter(x=s[f"v_{p}"], y=s[p], mode="markers", marker={"color": qcol[b], "size": 4, "opacity": 0.45}, name=f"{b} weeks" + (" (calmest fifth)" if b == "Q1" else (" (wildest fifth)" if b == "Q5" else "")), legendgroup=b, showlegend=i == 0, customdata=s.index.strftime("%Y-%m-%d"), hovertemplate="%{customdata}: vol %{x:.0%}, corr %{y:.2f}<extra>" + b + "</extra>"), row=r_, col=c_)
        med = d.groupby(f"q_{p}", observed=True).agg(v=(f"v_{p}", "median"), c=(p, "median"), m=(p, "mean"), n=(p, "count")).reindex(buckets)
        fig.add_trace(go.Scatter(x=med["v"], y=med["c"], mode="lines+markers", line={"color": INK, "width": 1.6}, marker={"size": 8, "symbol": "diamond", "color": INK}, name="median correlation inside each fifth (joined)", legendgroup="med", showlegend=i == 0, customdata=np.c_[med["m"], med["n"]], hovertemplate="median corr %{y:.2f} (mean %{customdata[0]:.2f}, n %{customdata[1]})<extra>fifth</extra>"), row=r_, col=c_)
        fig.update_xaxes(tickformat=".0%", tickfont={"size": 9}, row=r_, col=c_)
    fig.update_yaxes(range=[-0.8, 1.05])
    fig.update_xaxes(title_text="the pair's own 13-week vol (annualised)", row=nrow, col=3)
    fig.update_annotations(font={"size": 11})
    return _layout(fig, "Every week behind B8's averages: in the index pairs the whole cloud slopes up and the medians (diamonds) climb with the means — the rise is not a few wild weeks; in the gold pairs the cloud is a flat band around zero", 300 * nrow + 140,
                   "Each dot is one week: x = the pair's own volatility that week (the average of its two legs' 13-week vol), y = the pair's 13-week correlation; colour = which fifth of the pair's own vol the week belongs to (blue calm → red wild). The black diamonds are the median correlation inside each fifth, joined — compare them with the means on the B8 lines: if a handful of extreme weeks drove the average, the diamonds would sit well below the line. Hover any dot for its date; the timeline chart below shows the same weeks in order.", unified=False, legend_y=-0.1,
                   subtitle="The pair's own volatility against its correlation, one dot per week, for every pair — the components of the regime averages")


def regime_timeline():
    """B8: the same hypothesis on a time-series basis — the pair's own vol with its wildest-fifth weeks shaded, and the pair's correlation underneath; the menu picks the pair."""
    rw = _csv("5_regime_weekly", index_col=0, parse_dates=True); stx = _csv("5_regime_stretches", parse_dates=["start", "end"])
    prs = [p for p in PAIR_ORDER if p in rw.columns and f"q_{p}" in rw.columns]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.45, 0.55], vertical_spacing=0.08, subplot_titles=("The pair's own 13-week vol — shaded where it sat in its wildest fifth (Q5)", "The pair's 13-week correlation, same shading"))
    shapes_by = {}
    for i, p in enumerate(prs):
        v = rw[f"v_{p}"]; c = rw[p]; q5 = rw[f"q_{p}"] == "Q5"
        edge = v[q5].min()
        fig.add_trace(go.Scatter(x=v.index, y=v, mode="lines", line={"color": _pair_col(p), "width": 1.4}, name=f"{_pair_name(p)} — own vol", visible=i == 0, hovertemplate="%{y:.0%}<extra>own vol</extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=v.index, y=[edge] * len(v), mode="lines", line={"color": SELL, "width": 1, "dash": "dash"}, name=f"the Q5 line ({edge:.0%})", visible=i == 0, hoverinfo="skip"), row=1, col=1)
        fig.add_trace(go.Scatter(x=c.index, y=c, mode="lines", line={"color": _pair_col(p), "width": 1.6}, name=f"{_pair_name(p)} — correlation", visible=i == 0, hovertemplate="%{y:.2f}<extra>corr</extra>"), row=2, col=1)
        m = c.rolling(52, min_periods=26).median()
        fig.add_trace(go.Scatter(x=c.index, y=m, mode="lines", line={"color": MUTED, "width": 1, "dash": "dot"}, name="its trailing 52-week median", visible=i == 0, hovertemplate="%{y:.2f}<extra>52w median</extra>"), row=2, col=1)
        sh = []
        for _, e in stx[stx["pair"] == p].iterrows():
            for yref in ("y", "y2"):
                sh.append({"type": "rect", "xref": "x", "yref": yref + " domain", "x0": e["start"], "x1": e["end"], "y0": 0, "y1": 1, "fillcolor": SELL, "opacity": 0.12, "line": {"width": 0}, "layer": "below"})
        shapes_by[p] = sh
    n = 4
    fig.update_layout(shapes=shapes_by[prs[0]], updatemenus=[{"type": "dropdown", "x": 1.0, "xanchor": "right", "y": 1.12, "yanchor": "top", "showactive": True, "bgcolor": SURFACE, "bordercolor": GRID, "font": {"size": 12},
                                                                  "buttons": [{"label": _pair_name(p), "method": "update", "args": [{"visible": [j // n == i for j in range(n * len(prs))]}, {"shapes": shapes_by[p], "yaxis2.range": ([-0.8, 1.05] if "Gold" in p else [0.0, 1.05])}]} for i, p in enumerate(prs)]}])
    fig.update_yaxes(tickformat=".0%", row=1, col=1); fig.update_yaxes(range=([-0.8, 1.05] if "Gold" in prs[0] else [0.0, 1.05]), row=2, col=1)
    fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=2, col=1)
    e0 = stx[stx["pair"] == prs[0]]
    up = (e0["change"] > 0).mean() if len(e0) else np.nan
    return _layout(fig, f"Instance by instance, not on average: in {_pair_name(prs[0])} the correlation was higher inside the wild stretch than in the 8 weeks before it in {up:.0%} of its {len(e0)} stretches (table below lists every one)", 640,
                   f"The menu picks the pair. Shaded = every stretch of at least {C.REGIME_STRETCH_MIN} weeks in which the pair's own vol sat in its wildest fifth (above the dashed line). Read the bottom panel inside each shaded stretch against the weeks just before it: the hypothesis 'wild weeks are coupled weeks' holds when the correlation line is high or rising inside the shading. The dotted line is the correlation's trailing one-year median, so a stretch can be judged against the pair's own recent normal. The table under the chart lists every stretch with the correlation before, inside and the change.", legend_y=-0.1,
                   subtitle="The B8 hypothesis checked historically: when the pair's own vol was in its wildest fifth, what its correlation did — one pair at a time")


def lead_lag_blocks():
    """B9 redone: non-overlapping blocks — one panel per pair, the four directions as bars, 4-week and 13-week blocks side by side."""
    ll = _csv("5_lead_lag_blocks"); ch = ll[ll["kind"] == "change"]
    prs = [p for p in PAIR_ORDER if p in set(ch["pair"])]
    dirs = [d for d in ("correlation → own vol", "own vol → correlation", "correlation → VIX", "VIX → correlation") if d in set(ch["direction"])]
    xl = {"correlation → own vol": "corr<br>→ vol", "own vol → correlation": "vol<br>→ corr", "correlation → VIX": "corr<br>→ VIX", "VIX → correlation": "VIX<br>→ corr"}
    ncol = 5; nrow = int(np.ceil(len(prs) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, shared_yaxes=True, subplot_titles=[_pair_name(p) for p in prs], horizontal_spacing=0.03, vertical_spacing=0.18)
    bcol = {C.LL_BLOCKS[0]: INK, C.LL_BLOCKS[1]: "#9a9890"} if len(C.LL_BLOCKS) > 1 else {C.LL_BLOCKS[0]: INK}
    for i, p in enumerate(prs):
        r_, c_ = i // ncol + 1, i % ncol + 1
        for L in C.LL_BLOCKS:
            g = ch[(ch["pair"] == p) & (ch["block_weeks"] == L)].set_index("direction").reindex(dirs)
            ok = (g["nw_t"].abs() >= 2) & (np.sign(g["pearson"]) == np.sign(g["pearson_nonoverlap"]))
            fig.add_trace(go.Bar(x=[xl[d] for d in dirs], y=g["pearson"], name=f"{L}-week blocks", marker={"color": bcol[L], "line": {"color": SELL, "width": [2.2 if o else 0 for o in ok]}}, legendgroup=str(L), showlegend=i == 0,
                                 text=[f"t {t:+.1f}" if pd.notna(t) else "" for t in g["nw_t"]], textposition="outside", textfont={"size": 8.5}, customdata=np.c_[g["nw_t"], g["spearman"], g["pearson_nonoverlap"], g["n"], g["n_nonoverlap"]],
                                 hovertemplate="%{x}: Pearson %{y:+.2f}, NW t %{customdata[0]:+.1f}, Spearman %{customdata[1]:+.2f}; non-overlapping sample %{customdata[2]:+.2f} (n %{customdata[4]} of %{customdata[3]})<extra>" + f"{L}-week blocks</extra>"), row=r_, col=c_)
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=r_, col=c_)
        fig.update_xaxes(tickfont={"size": 9}, row=r_, col=c_)
    fig.update_layout(barmode="group"); fig.update_yaxes(range=[-0.62, 0.5])
    fig.update_annotations(font={"size": 11})
    eq = [p for p in prs if "Gold" not in p]
    vx = ch[(ch["direction"] == "VIX → correlation") & ch["pair"].isin(eq)]
    rel = int(((vx["nw_t"].abs() >= 2) & (np.sign(vx["pearson"]) == np.sign(vx["pearson_nonoverlap"]))).sum())
    cv = ch[(ch["direction"] == "correlation → own vol") & (ch["block_weeks"] == C.LL_BLOCKS[-1]) & ch["pair"].isin(eq)]["pearson"].mean()
    vc = ch[(ch["direction"] == "own vol → correlation") & (ch["block_weeks"] == C.LL_BLOCKS[-1]) & ch["pair"].isin(eq)]["pearson"].mean()
    return _layout(fig, f"With no shared weeks, nothing leads: the VIX's change predicts the next block's correlation change in {rel} of {len(vx)} index-pair tests; what remains is negative both ways in the index pairs (13-week blocks: corr → vol {cv:+.2f}, vol → corr {vc:+.2f}) — after a block in which both rose, both fall back", 300 * nrow + 140,
                   f"Each bar is the correlation between a CHANGE on the block ending at week t (X) and a CHANGE on the block starting at t+1 (Y): corr → vol = the pair's correlation change this block vs its own-vol change next block; vol → corr the other way round; the same with the VIX's log change over the block. Dark = {C.LL_BLOCKS[0]}-week blocks, light = {C.LL_BLOCKS[-1]}-week. The t above each bar is Newey-West (the blocks step weekly, so neighbours overlap; the hover also gives the Pearson on the strictly non-overlapping sub-sample). A bar outlined in red passed both tests (|t| ≥ 2 and the same sign on the non-overlapping sample). A positive bar at 'vol → corr' would mean volatility leads coupling; nothing of the kind is there.", unified=False, legend_y=-0.1,
                   subtitle=f"The lead/lag test with NO overlapping weeks: X measured on weeks t−L+1 … t, Y on weeks t+1 … t+L, for L = {C.LL_BLOCKS[0]} and {C.LL_BLOCKS[-1]}; one panel per pair, the four directions as bars")


def ratio_by_corr():
    """The ratio study: the ratio's change over the next 4 weeks, by the pair's correlation quintile — one panel per pair."""
    rs = _csv("2_ratio_study"); F = C.RATIO_FWD
    q = rs[rs["cut"] == "correlation quintile"]
    prs = [p for p in PAIR_ORDER if p in set(q["pair"])]
    ncol = 5; nrow = int(np.ceil(len(prs) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, shared_yaxes=True, subplot_titles=[_pair_name(p).replace(" × ", " ÷ ") for p in prs], horizontal_spacing=0.03, vertical_spacing=0.18)
    for i, p in enumerate(prs):
        r_, c_ = i // ncol + 1, i % ncol + 1
        g = q[q["pair"] == p].set_index("group")
        a, b = p.split("|")
        x = [f"{bk}<br>{g.loc[bk, 'corr_lo']:.2f}–{g.loc[bk, 'corr_hi']:.2f}" for bk in g.index]
        fig.add_trace(go.Bar(x=x, y=g[f"fwd_{F}w_mean"], marker_color=COL[a], name=f"mean change of the ratio over the next {F} weeks (+ = the first-named leg ahead)", legendgroup="mean", showlegend=i == 0, text=[f"{h:.0%}" for h in g[f"fwd_{F}w_hit"]], textposition="outside", textfont={"size": 8.5},
                             customdata=np.c_[g[f"fwd_{F}w_hit"], g["n_forward"], g[f"fwd_{F}w_median"]], hovertemplate="%{x}: mean %{y:+.1%}, median %{customdata[2]:+.1%}, first leg ahead %{customdata[0]:.0%} of n %{customdata[1]}<extra>" + f"{a} ÷ {b}</extra>"), row=r_, col=c_)
        fig.add_trace(go.Scatter(x=x, y=g[f"fwd_{F}w_median"], mode="markers", marker={"color": INK, "size": 6, "symbol": "diamond"}, name="median", legendgroup="med", showlegend=i == 0, hovertemplate="%{x}: median %{y:+.1%}<extra>median</extra>"), row=r_, col=c_)
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=r_, col=c_)
        fig.update_xaxes(tickfont={"size": 8.5}, row=r_, col=c_)
    fig.update_yaxes(tickformat=".1%", range=[-0.04, 0.04])
    fig.update_annotations(font={"size": 11})
    allw = rs[rs["cut"] == "all weeks"].set_index("pair")
    full = [p for p in prs if "Micro250" not in p]; micro = [p for p in prs if "Micro250" in p]
    sp = allw.loc[full, "spearman_fwd_vs_corr"]
    mx_full = q[q["pair"].isin(full)][f"fwd_{F}w_mean"].abs().max()
    mg = q[q["pair"] == "Micro250|GoldMCX"].set_index("group")[f"fwd_{F}w_mean"] if "Micro250|GoldMCX" in micro else None
    tail = (f"; only Micro250 ÷ Gold shows a slope (Q1 {mg['Q1']:+.1%} → Q5 {mg['Q5']:+.1%}, rank corr {allw.loc['Micro250|GoldMCX', 'spearman_fwd_vs_corr']:+.2f}) — on about {int(q[q['pair'] == 'Micro250|GoldMCX']['n_forward'].median())} weeks a bucket, all since 2022" if mg is not None else "")
    return _layout(fig, f"The pair's correlation does not say which leg wins next: in the full-history pairs the ratio's next-{F}-week change stays within ±{mx_full:.1%} in every quintile and ranks against the correlation at {sp.min():+.2f} to {sp.max():+.2f}" + tail, 300 * nrow + 140,
                   f"One number per pair instead of two returns: the ratio = first leg ÷ second leg, whose log change is the first leg's return minus the second's. Each panel sorts the weeks by that pair's own 13-week correlation (the range under each label); the bar is the average change of the ratio over the following {F} weeks (t+1 … t+{F}), the diamond the median, the number on the bar the share of weeks the first leg came out ahead. If high correlation favoured one leg, the bars would step up or down from Q1 to Q5. They do not.", unified=False, legend_y=-0.1,
                   subtitle=f"The forward {F}-week change of each pair's price ratio by the pair's correlation quintile at week t")


def ratio_by_state():
    """The ratio study, cut by the joint signal state and by the sign of the ratio's own last 4 weeks."""
    rs = _csv("2_ratio_study"); F, Bk = C.RATIO_FWD, C.RATIO_BACK
    cuts = ["joint signal state", f"sign of the last {Bk} weeks' change"]
    q = rs[rs["cut"].isin(cuts)]
    prs = [p for p in PAIR_ORDER if p in set(q["pair"])]
    ncol = 5; nrow = int(np.ceil(len(prs) / ncol))
    fig = make_subplots(rows=nrow, cols=ncol, shared_yaxes=True, subplot_titles=[_pair_name(p).replace(" × ", " ÷ ") for p in prs], horizontal_spacing=0.03, vertical_spacing=0.18)
    stc = {"both BUY": BUY, "disagree": MUTED, "both SELL": SELL}
    for i, p in enumerate(prs):
        r_, c_ = i // ncol + 1, i % ncol + 1
        a, b = p.split("|")
        g = q[q["pair"] == p]
        s1 = g[g["cut"] == cuts[0]].set_index("group").reindex(["both BUY", "disagree", "both SELL"]).dropna(subset=["weeks"])
        fig.add_trace(go.Bar(x=[f"{s}<br>n {int(n)}" for s, n in zip(s1.index, s1["n_forward"])], y=s1[f"fwd_{F}w_mean"], marker_color=[stc[s] for s in s1.index], name="by the joint signal state at t", legendgroup="st", showlegend=i == 0, text=[f"{h:.0%}" for h in s1[f"fwd_{F}w_hit"]], textposition="outside", textfont={"size": 8.5},
                             customdata=np.c_[s1[f"fwd_{F}w_hit"], s1[f"fwd_{F}w_median"]], hovertemplate="%{x}: mean %{y:+.1%}, median %{customdata[1]:+.1%}, first leg ahead %{customdata[0]:.0%}<extra>" + f"{a} ÷ {b}</extra>"), row=r_, col=c_)
        s2 = g[g["cut"] == cuts[1]].set_index("group")
        s2 = s2.reindex([x for x in (f"{a} ahead over the last {Bk} wk", f"{b} ahead over the last {Bk} wk") if x in s2.index])
        fig.add_trace(go.Bar(x=[f"{a} led<br>n {int(s2['n_forward'].iloc[0])}" if j == 0 else f"{b} led<br>n {int(s2['n_forward'].iloc[1])}" for j in range(len(s2))], y=s2[f"fwd_{F}w_mean"], marker_color=[COL[a], COL[b]][:len(s2)], name=f"by who was ahead over the last {Bk} weeks (continuation = same sign as the label's leg)", legendgroup="past", showlegend=i == 0, text=[f"{h:.0%}" for h in s2[f"fwd_{F}w_hit"]], textposition="outside", textfont={"size": 8.5},
                             customdata=np.c_[s2[f"fwd_{F}w_hit"], s2[f"fwd_{F}w_median"]], hovertemplate="%{x}: mean %{y:+.1%}, median %{customdata[1]:+.1%}, first leg ahead %{customdata[0]:.0%}<extra>" + f"{a} ÷ {b}</extra>"), row=r_, col=c_)
        fig.add_hline(y=0, line={"color": AXIS, "width": 0.8}, row=r_, col=c_)
        fig.update_xaxes(tickfont={"size": 8}, row=r_, col=c_)
    fig.update_yaxes(tickformat=".1%", range=[-0.04, 0.04])
    fig.update_annotations(font={"size": 11})
    allw = rs[rs["cut"] == "all weeks"].set_index("pair")
    full = [p for p in prs if "Micro250" not in p]
    sp = allw.loc[full, "spearman_fwd_vs_past"]; spm = allw.loc[[p for p in prs if "Micro250" in p], "spearman_fwd_vs_past"]
    mx_full = q[q["pair"].isin(full)][f"fwd_{F}w_mean"].abs().max()
    return _layout(fig, f"Neither the two signals nor the ratio's own last {Bk} weeks say which leg wins next: in the full-history pairs every bar is within ±{mx_full:.1%} and the last-{Bk}-week change ranks against the next {F} weeks at {sp.min():+.2f} to {sp.max():+.2f} (the Micro250 pairs, on a quarter of the history: {spm.min():+.2f} to {spm.max():+.2f})", 300 * nrow + 140,
                   f"Same ratio as the chart above. First three bars: the average change of the ratio over the next {F} weeks when both supertrends were BUY (blue), disagreed (grey), both SELL (red) at week t. Last two bars: when the first leg had been ahead over the last {Bk} weeks (its colour) vs when the second had — a bar with the same sign as its label's leg is continuation, the opposite sign is reversal. The number on each bar is the share of weeks the first-named leg came out ahead; the counts are under the labels.", unified=False, legend_y=-0.1,
                   subtitle=f"The forward {F}-week change of each pair's price ratio by the joint signal state and by the sign of its own last {Bk} weeks")


FIGS = {
    "0_corr_heatmap": corr_heatmap, "0_corr_rolling": corr_rolling, "0_gold_when_decoupled": gold_when_decoupled, "0_gold_vs_group": gold_vs_group, "0_gold_gap_reversion": gap_reversion, "0_gold_after_break": gold_after_break, "0_index_vs_gold_pair": index_pair_vs_gold_pair,
    "2_ratio_by_corr": ratio_by_corr, "2_ratio_by_state": ratio_by_state, "5_vol_state_box": vol_state_box, "5_vix_flip_leadlag": vix_flip_leadlag, "5_regime_scatter": regime_scatter, "5_regime_timeline": regime_timeline, "5_lead_lag_blocks": lead_lag_blocks,
    "1_corr_by_state": corr_by_state, "1_corr_around_flip": corr_around_flip, "1_corr_around_flip_pairs": corr_around_flip_pairs, "1_corr_by_breadth": corr_by_breadth, "1_corr_timeseries": corr_timeseries,
    "2_by_year": by_year, "2_corr_quintiles": quintiles, "2_corr_quintiles_pairs": quintiles_pairs, "2_timeline": timeline,
    "3_corr_series": cb_series, "3_corr_episodes": cb_episodes, "3_who_breaks_correlation": cb_summary, "3_after_break": cb_after,
    "4_who_breaks_first": breaks_leads, "4_price_episodes": breaks_episodes,
    "5_vol_timeseries": vol_timeseries, "5_vol_by_state": vol_by_state, "5_vol_around_flip": vol_around_flip, "5_corr_by_regime": corr_by_regime, "5_corr_by_regime_more": corr_by_regime_more, "5_vol_corr_lead": vol_corr_lead, "5_vol_corr_lead_pairs": vol_corr_lead_pairs, "5_together": together_overlay, "5_together_pairs": together_pairs, "5_corr_rise_paths": corr_rise_paths, "5_corr_rise_paths_own": corr_rise_paths_own,
    "6_pair": pair_card, "6_z_forward": z_forward, "6_path_after_2sd": path_after,
    "7_equity": pt_equity, "7_summary": pt_summary,
    "8_rules": rp_rules, "8_cells": rp_cells, "8_z_path": rp_z_path, "8_events": rp_events,
}


if __name__ == "__main__":
    import time
    t0 = time.time()
    for k, f in FIGS.items():
        if k == "6_pair":
            f("N100|SC100")
        elif k == "8_events":
            f("N100")
        else:
            f()
        print(f"  {k} ok")
    print(f"all figures build in {time.time() - t0:.0f}s")
