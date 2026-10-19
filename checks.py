"""
Regression checks.   python checks.py   -> exit 0 = all pass

  A. DATA      no back-fill, nothing carried past a series' last print, weekly bar = last close
  B. MATHS     cov = corr x vol x vol inside each state cell; decomposition shares sum to 1;
               quintile 'next quarter' uses t+1..t+13 only; a planted sell-off is detected with the
               right peak/trough; a planted early-breaker gets a negative lead and rank 1
  C. SIGNAL    ATR weights, state alphabet
  D. HYGIENE   password in no file; .env ignored; report present and stamped correctly
  E. PAIRS     ratio z-score = manual; z_of is antisymmetric; forward = t+1..t+h; a planted mean-reverting
               ratio is called 'pulls back' and its deviations revert in z_forward, a planted random walk is
               not; the pair-trade rule enters at |z| >= 2, exits at |z| <= 0.5 or 26 weeks, never overlaps,
               its P&L is the spread over t+1..exit, and it makes money on the planted reverting ratio;
               vol by state partitions the weeks; rolling vol = sd x sqrt(52)
  F. REPLACE   events = BUY->SELL flips; candidate forward return = t+1..t+h; 'hold until own SELL' stops at
               the candidate's first SELL week; a rule with no candidate falls back to cash; z vs seller
               is minus the pair's z
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

import numpy as np
import pandas as pd

import breaks
import config as C
import corrbreak
import data
import histcorr
import pairs as PR
import pairtrade
import replace
import sigcorr
import supertrend as ST
import vol

P, F, S = [], [], []


def check(name, cond, detail=""):
    (P if bool(cond) else F).append(name)
    print(f"  {'ok  ' if cond else 'FAIL'} {name}{('  - ' + detail) if detail and not cond else ''}")


def main() -> int:
    panel = data.load_panel(verbose=False)
    px = data.close_panel(panel)
    px_w = data.resample_close(px, "W-FRI")
    rw = data.returns(px, "weekly")
    lvl = data.vix_level(px, "weekly") if C.VIX in panel else None
    st = sigcorr.states(panel)

    print("A. data")
    for lab in px.columns:
        s, first, last = px[lab], panel[lab].index[0], panel[lab].index[-1]
        check(f"A no back-fill / no carry-forward: {lab}", s[s.index < first].isna().all() and s[s.index > last].isna().all())
    check("A weekly bar = last close of the week", np.isclose(px_w["N100"].iloc[10], px["N100"].loc[:px_w.index[10]].dropna().iloc[-1]))

    print("B. maths")
    bs, dec = sigcorr.by_state(st, rw)
    row = bs[(bs["pair"] == "N100|SC100") & (bs["state"] == "both SELL")].iloc[0]
    check("B cov = corr x vol_a x vol_b inside a state cell", np.isclose(row["cov_ann"], row["corr"] * row["vol_a_ann"] * row["vol_b_ann"], rtol=1e-6))
    d = dec[dec["pair"] == "N100|SC100"].iloc[0]
    check("B decomposition shares sum to 1", np.isclose(d["share_of_cov_change_from_corr"] + d["share_of_cov_change_from_vol"], 1.0))
    check("B state cells partition the weeks", int(bs[(bs['pair'] == 'N100|SC100') & (bs['state'] != 'all weeks')]['weeks'].sum()) == int(bs[(bs['pair'] == 'N100|SC100') & (bs['state'] == 'all weeks')]['weeks'].iloc[0]))
    # next-quarter alignment
    idx = pd.date_range("2016-01-08", periods=80, freq="W-FRI")
    r1 = pd.Series(np.arange(80, dtype=float), index=idx)
    nxt = r1.shift(-1).rolling(13).sum().shift(-12)
    check("B 'next quarter' at t = sum of r[t+1..t+13]", np.isclose(nxt.iloc[10], sum(range(11, 24))) and nxt.iloc[-13:].isna().all())
    # planted sell-off
    rng = np.random.default_rng(0)
    w = pd.date_range("2016-01-08", periods=200, freq="W-FRI")
    base = np.ones(200)
    base[60:72] = np.linspace(1.0, 0.82, 12)      # -18% over 12 weeks
    base[72:100] = np.linspace(0.82, 1.02, 28)    # recovers
    base[100:] = 1.02
    n100 = pd.Series(base * 100, index=w)
    early = n100.copy(); early.iloc[56:72] = early.iloc[56:72] * np.linspace(1.0, 0.85, 16)   # starts falling 4 weeks earlier, falls more
    pxx = pd.DataFrame({"N100": n100, "MC100": n100, "SC100": early, "Micro250": n100, "GoldMCX": 100 + np.arange(200) * 0.01})
    ep = breaks.episodes(pxx)
    check("B planted sell-off detected once, with the right peak and trough", len(ep) == 1 and ep.iloc[0]["peak_date"] == w[60] and ep.iloc[0]["trough_date"] == w[71], f"{ep[['peak_date', 'trough_date', 'depth']].to_dict('records') if len(ep) else 'none'}")
    stx = pd.DataFrame({c: pd.Series(1, index=w) for c in C.ASSETS})
    stx.loc[w[66]:, "N100"] = -1; stx.loc[w[66]:, "MC100"] = -1; stx.loc[w[62]:, "SC100"] = -1; stx.loc[w[66]:, "Micro250"] = -1
    for c in C.ASSETS:
        stx.loc[w[90]:, c] = 1
    dp = breaks.dates_per_asset(ep, pxx, stx)
    ld = breaks.leads(dp)
    sc = ld[ld["asset"] == "SC100"].iloc[0]
    check("B planted early breaker: SELL flip lead = -4 weeks, -10% cross earlier than N100", sc["lead_signal_sell"] == -4 and sc["lead_cross_10pct"] < 0, f"{sc[['lead_signal_sell', 'lead_cross_10pct']].to_dict()}")
    smx = breaks.summary(ld)
    check("B planted early breaker ranks first on the signal", smx.loc["SC100", "avg_rank_signal_sell"] == 1.0 and smx.loc["SC100", "first_share_signal_sell"] == 1.0)

    # correlation breaks: a planted panel where one index decouples 3 weeks before the others
    rng2 = np.random.default_rng(5)
    wk = pd.date_range("2016-01-08", periods=300, freq="W-FRI")
    f = rng2.normal(0, 0.02, 300)
    panelx = {a: f + rng2.normal(0, 0.006, 300) for a in ("N100", "MC100", "SC100", "Micro250")}
    panelx["SC100"][200:230] = rng2.normal(0, 0.02, 30)              # SC100 goes its own way from week 200
    for a in ("N100", "MC100", "Micro250"):
        panelx[a][203:230] = rng2.normal(0, 0.02, 27)                # the others 3 weeks later
    rx = pd.DataFrame(panelx, index=wk); rx["GoldMCX"] = rng2.normal(0, 0.01, 300); rx["VIXchg"] = 0.0
    Sx = corrbreak.series(rx)
    check("B corrbreak: G is the mean of the six pairwise correlations", np.isclose(Sx["G"].iloc[-1], np.mean([rx[a].iloc[-13:].corr(rx[b].iloc[-13:]) for a, b in [("N100","MC100"),("N100","SC100"),("N100","Micro250"),("MC100","SC100"),("MC100","Micro250"),("SC100","Micro250")]])))
    check("B corrbreak: c_N100 is the mean of its three pairwise correlations", np.isclose(Sx["c_N100"].iloc[-1], np.mean([rx["N100"].iloc[-13:].corr(rx[b].iloc[-13:]) for b in ("MC100", "SC100", "Micro250")])))
    # ordering logic on hand-built z-series: SC100 crosses the line 3 weeks before the group and the others
    zk = pd.Series(0.0, index=wk)
    Sz = pd.DataFrame(index=wk)
    for a in ("N100", "MC100", "Micro250"):
        Sz[f"c_{a}"] = 0.85; Sz[f"z_c_{a}"] = 0.0
        Sz.loc[wk[203]:wk[215], f"z_c_{a}"] = -2.0; Sz.loc[wk[203]:wk[215], f"c_{a}"] = 0.60
    Sz["c_SC100"] = 0.85; Sz["z_c_SC100"] = 0.0
    Sz.loc[wk[200]:wk[215], "z_c_SC100"] = -2.5; Sz.loc[wk[200]:wk[215], "c_SC100"] = 0.40
    Sz["G"] = 0.85; Sz["z_G"] = 0.0
    Sz.loc[wk[203]:wk[215], "z_G"] = -2.2; Sz.loc[wk[203]:wk[215], "G"] = 0.55
    rz = pd.DataFrame(0.0, index=wk, columns=["N100", "MC100", "SC100", "Micro250", "GoldMCX", "VIXchg"])
    epz = corrbreak.episodes(Sz)
    check("B corrbreak: one episode, starting the week z_G crosses the line", len(epz) == 1 and epz.iloc[0]["start"] == wk[203] and epz.iloc[0]["end"] == wk[216])
    abz = corrbreak.asset_breaks(Sz, epz, rz)
    scz = abz[abz["asset"] == "SC100"].iloc[0]
    check("B corrbreak: the early decoupler has lead -3 and rank 1 by date", scz["lead_vs_group_weeks"] == -3 and scz["rank"] == 1 and (abz[abz["asset"] != "SC100"]["rank"] > 1).all())
    check("B corrbreak: it is also deepest at the group break and has the biggest drop", scz["rank_deepest"] == 1 and scz["rank_drop"] == 1)
    check("B corrbreak: drop shares sum to 1 within an episode", np.isclose(abz["drop_share"].sum(), 1.0))
    smz = corrbreak.summary(abz)
    check("B corrbreak: summary names it first on every measure", "FIRST-BREAKER" in smz.loc["SC100", "verdict"] and smz.loc["SC100", "share_first_alone"] == 1.0)

    print("C. signal")
    wts = np.arange(1, C.LOOKBACK + 1, dtype=float) / np.arange(1, C.LOOKBACK + 1).sum()
    check("C LW ATR weights are 1/15 .. 5/15", np.allclose(wts, [1 / 15, 2 / 15, 3 / 15, 4 / 15, 5 / 15]))
    check("C states in {+1, -1}", set(np.unique(st.dropna().values.astype(int))) <= {1, -1})

    print("E. pairs, volatility, pair trade")
    px_w_ = px_w[[c for c in C.ASSETS if c in px_w.columns]]
    Z, Bd = PR.ratio_series(px_w_)
    lr = np.log(px_w_["N100"]) - np.log(px_w_["SC100"])
    i = 300
    zman = (lr.iloc[i] - lr.iloc[i - C.PAIR_WINDOW + 1:i + 1].mean()) / lr.iloc[i - C.PAIR_WINDOW + 1:i + 1].std()
    check("E ratio z = (log ratio - 52w mean) / 52w sd", np.isclose(Z["N100|SC100"].iloc[i], zman))
    check("E z_of(b, a) = -z_of(a, b)", np.allclose(PR.z_of(Z, "SC100", "N100").dropna(), -Z["N100|SC100"].dropna()))
    idx2 = pd.date_range("2016-01-08", periods=80, freq="W-FRI")
    r2 = pd.Series(np.arange(80, dtype=float), index=idx2)
    check("E pairs.forward(h) at t = sum of r[t+1..t+h]", np.isclose(PR.forward(r2, 13).iloc[10], sum(range(11, 24))) and PR.forward(r2, 13).iloc[-13:].isna().all() and np.isclose(PR.forward(r2, 4).iloc[10], sum(range(11, 15))))
    # planted: a mean-reverting ratio (AR(1), half-life ~8 weeks) vs a random-walk ratio
    rng3 = np.random.default_rng(11)
    n = 520; wk3 = pd.date_range("2016-01-08", periods=n, freq="W-FRI")
    phi = 0.5 ** (1 / 8)
    u = np.zeros(n)
    for k in range(1, n):
        u[k] = phi * u[k - 1] + rng3.normal(0, 0.02)
    common = np.cumsum(rng3.normal(0.001, 0.02, n))
    pA = 100 * np.exp(common + u / 2); pB = 100 * np.exp(common - u / 2)            # ln(A/B) = u, mean-reverting
    pC = 100 * np.exp(common + np.cumsum(rng3.normal(0, 0.015, n)))                  # ln(C/B) = random walk
    pxp = pd.DataFrame({"N100": pA, "MC100": pB, "SC100": pC}, index=wk3)
    ct = PR.coint_table(pxp)
    check("E planted mean-reverting ratio is called 'pulls back' with a half-life near 8 weeks", ct.loc["N100|MC100", "verdict"].startswith("pulls back") and 4 <= ct.loc["N100|MC100", "halflife_weeks_full"] <= 16, f"{ct.loc['N100|MC100', 'verdict']} hl {ct.loc['N100|MC100', 'halflife_weeks_full']:.1f}")
    check("E planted random-walk ratio is called 'no long-run level'", ct.loc["MC100|SC100", "verdict"].startswith("no long-run") and ct.loc["MC100|SC100", "adf_p_log_ratio"] > 0.05, ct.loc["MC100|SC100", "verdict"])
    Zp, _ = PR.ratio_series(pxp)
    rp = np.log(pxp).diff().iloc[1:]
    zfp = PR.z_forward(Zp, rp)
    g = zfp[(zfp["pair"] == "N100|MC100") & (zfp["horizon_weeks"] == 13)].set_index("bucket")
    lo, hi = PR.BUCKET_ORDER[1], PR.BUCKET_ORDER[3]
    check("E z_forward: the planted reverting ratio RISES after a lag and FALLS after a lead", g.loc[lo, "mean_ratio_change"] > 0 and g.loc[hi, "mean_ratio_change"] < 0 and g.loc[lo, "share_reverted"] > 0.5, f"{g.loc[lo, 'mean_ratio_change']:.3f} / {g.loc[hi, 'mean_ratio_change']:.3f}")
    g2 = zfp[(zfp["pair"] == "MC100|SC100") & (zfp["horizon_weeks"] == 13)].set_index("bucket")
    check("E z_forward: the planted random walk shows no reliable pull-back", abs(g2.loc[lo, "hac_t_ratio_change"]) < 2.5 or np.isnan(g2.loc[lo, "hac_t_ratio_change"]), f"t {g2.loc[lo, 'hac_t_ratio_change']:.2f}")
    # the pair-trade rule on the planted panel
    stp = pd.DataFrame(1, index=wk3, columns=["N100", "MC100", "SC100"])
    trp = pairtrade.trades(Zp, rp, "N100", "MC100", st=stp)
    ok_entry = (trp["z_entry"].abs() >= C.PT_ENTRY).all()
    ok_exit = ((trp["exit_reason"] == "z back inside") & (trp["z_exit"].abs() <= C.PT_EXIT) | (trp["exit_reason"] == "max hold") & (trp["weeks"] == C.PT_MAX_HOLD) | (trp["exit_reason"] == "end of data")).all()
    ok_overlap = (trp["entry_date"].iloc[1:].values > trp["exit_date"].iloc[:-1].values).all() if len(trp) > 1 else True
    t0 = trp.iloc[0]
    i0, i1 = rp.index.get_loc(t0["entry_date"]), rp.index.get_loc(t0["exit_date"])
    gross_manual = t0["side"] * (rp["N100"] - rp["MC100"]).iloc[i0 + 1:i1 + 1].sum()
    check("E pair trade: enters at |z| >= 2, exits at |z| <= 0.5 or 26 weeks, never overlaps", len(trp) >= 5 and ok_entry and ok_exit and ok_overlap, f"n {len(trp)} entry {ok_entry} exit {ok_exit} overlap {ok_overlap}")
    check("E pair trade: gross = side x spread over t+1..exit; cost = 4 x bp", np.isclose(t0["gross"], gross_manual) and np.isclose(t0["cost"], 4 * C.PT_COST_BPS / 1e4) and np.isclose(t0["net"], t0["gross"] - t0["cost"]))
    check("E pair trade: long the cheap leg - it buys the lagging leg", ((trp["side"] == 1) == (trp["z_entry"] < 0)).all() and ((trp["cheap_leg"] == "N100") == (trp["side"] == 1)).all())
    check("E pair trade: makes money on the planted mean-reverting ratio", trp["net"].mean() > 0 and (trp["net"] > 0).mean() > 0.6, f"mean {trp['net'].mean():.3f} win {(trp['net'] > 0).mean():.2f}")
    smp = pairtrade.summary(trp, rp)
    check("E pair trade: summary verdict says 'worked' on the planted ratio", smp.loc["N100|MC100", "verdict"] == "worked on this history", smp.loc["N100|MC100", "verdict"])
    # volatility
    vr = vol.rolling_vol(rw)
    check("E rolling vol = sd of the last 13 weekly returns x sqrt 52", np.isclose(vr["N100"].iloc[200], rw["N100"].iloc[188:201].std() * np.sqrt(52)))
    vbs = vol.by_state(st, rw)
    check("E vol by state: BUY + SELL weeks = the asset's weeks with a state", int(vbs.loc["N100", "weeks_BUY"] + vbs.loc["N100", "weeks_SELL"]) == int(st["N100"].reindex(rw.index).notna().sum()))
    # every asset's flips, not only N100's; the flip-week value is the series at the flip
    afl = sigcorr.around_flip(st, rw)
    check("E around_flip covers every asset as the reference", set(afl["ref"]) == set(a for a in C.ASSETS if a in st.columns))
    s_sc = st["SC100"].reindex(rw.index); flips_sc = rw.index[((s_sc == -1) & (s_sc.shift(1) == 1)).fillna(False)]
    mc_ = sigcorr.rolling_mean_corr(rw, C.EQUITY)
    manual = float(np.nanmean([mc_.loc[d] for d in flips_sc]))
    got = afl[(afl["ref"] == "SC100") & (afl["event"] == "flip to SELL") & (afl["weeks_from_flip"] == 0)]["mean_corr"].iloc[0]
    check("E around_flip at k = 0 for SC100 = mean of G on SC100's SELL-flip weeks", np.isclose(got, manual))
    vfl = vol.around_flip(st, rw, lvl)
    check("E vol.around_flip covers every asset and carries the VIX level when the file is present", set(vfl["ref"]) == set(a for a in C.ASSETS if a in st.columns) and (("vix_level" in vfl.columns) == (lvl is not None and len(lvl.dropna()) > 0)))
    regs, regl_ = vol.corr_by_regime(rw, lvl)
    check("E corr_by_regime: three regime definitions, quintiles cover every pair's weeks once", set(regs) >= {"N100 vol", "pair's own vol"} and (regl_.groupby(["regime", "pair"])["weeks"].sum() > 0).all())
    ok_ = regl_[(regl_["regime"] == "N100 vol") & (regl_["pair"] == "N100|SC100")].set_index("bucket")
    check("E corr_by_regime: Q1 < Q5 for N100|SC100 under N100's vol (the stated finding)", ok_.loc["Q1", "mean_corr"] < ok_.loc["Q5", "mean_corr"])
    rtb = PR.returns_table(rw, px_w_, st)
    xy_ = rw[["N100", "SC100"]].dropna()
    check("E pair returns: total = exp(sum of log returns) - 1 over the common weeks; spread = a - b", np.isclose(rtb.loc["N100|SC100", "total_a"], np.exp(xy_["N100"].sum()) - 1) and np.isclose(rtb.loc["N100|SC100", "spread_total"], rtb.loc["N100|SC100", "total_a"] - rtb.loc["N100|SC100", "total_b"]))
    check("E pair returns: both-BUY + both-SELL + disagree weeks = weeks with both states", int(rtb.loc["N100|SC100", "both_BUY_weeks"] + rtb.loc["N100|SC100", "both_SELL_weeks"] + rtb.loc["N100|SC100", "disagree_weeks"]) == int((st["N100"].reindex(xy_.index).notna() & st["SC100"].reindex(xy_.index).notna()).sum()))

    print("G. every pair beside the group")
    PC = sigcorr.pair_corrs(rw)
    ten = {f"{a}|{b}" for a, b in __import__("itertools").combinations([a for a in C.ASSETS if a in rw.columns], 2)}
    check("G pair_corrs carries all ten pairs, gold included", set(PC.columns) == ten)
    G_ = corrbreak.series(rw)["G"]; Gm = PC[[p for p in PC.columns if "Gold" not in p]].mean(axis=1, skipna=True)
    check("G the group correlation G = the mean of the six index-pair correlations, week by week", np.allclose(G_.dropna(), Gm.reindex(G_.dropna().index), atol=1e-9))
    afp_ = sigcorr.around_flip_pairs(st, rw, refs=["N100"])
    one = afp_[(afp_["event"] == "flip to SELL") & (afp_["weeks_from_flip"] == 0)].set_index("pair")["mean_corr"]
    flips_ = st["N100"].reindex(PC.index); flips_ = PC.index[((flips_ == -1) & (flips_.shift(1) == 1)).fillna(False)]
    check("G around_flip_pairs at k = 0 = the pair's correlation averaged over the flip weeks", np.isclose(one["N100|SC100"], PC.loc[flips_, "N100|SC100"].dropna().mean()))
    chk = afp_[(afp_["event"] == "flip to SELL")].pivot(index="weeks_from_flip", columns="pair", values="mean_corr")
    afg = sigcorr.around_flip(st, rw, refs=["N100"]); afg = afg[afg["event"] == "flip to SELL"].set_index("weeks_from_flip")["mean_corr"]
    check("G the G event study is close to the mean of the six index-pair event studies (differs only where a pair is missing a flip)", (chk[[p for p in chk.columns if "Gold" not in p]].mean(axis=1) - afg).abs().max() < 0.05)
    bpd_ = sigcorr.by_breadth_pairs(st, rw); bb_ = sigcorr.by_breadth(st, rw)
    check("G by_breadth_pairs: the bands' weeks sum to the weeks with a correlation; the index pairs' mean tracks G's", int(bpd_["weeks"].sum()) == int(bb_["weeks"].sum()) and (bpd_[[p for p in bpd_.columns if "|" in p and "Gold" not in p]].mean(axis=1) - bb_["mean"]).abs().max() < 0.03)
    qp_ = histcorr.same_quarter_pairs(rw)
    g_ = qp_[(qp_["pair"] == "N100|SC100") & (qp_["leg"] == "N100")]
    check("G same_quarter_pairs: five buckets per leg, weeks sum to the pair's weeks, every bucket about a fifth", len(g_) == 5 and int(g_["weeks"].sum()) == int(PC["N100|SC100"].notna().sum()) and (g_["weeks"].max() - g_["weeks"].min()) <= 1)
    hpl_, hps_ = histcorr.episodes_pairs(rw)
    check("G episodes_pairs: a stretch is 8+ consecutive weeks of that pair above 0.90, and no gold pair has one", (hpl_["weeks"] >= 8).all() and all(PC[r["pair"]].loc[str(r["start"]):str(r["end"])].gt(0.90).all() for _, r in hpl_.head(10).iterrows()) and (hps_.loc[[p for p in hps_.index if "Gold" in p], "stretches"] == 0).all())
    tp_ = vol.together_pairs(rw, lvl)
    own_paths_, own_per_ = vol.corr_rise_paths_own(tp_)
    zc_ = tp_["z_N100|SC100"]; ev_own = vol.corr_rise_events(pd.DataFrame({"z_corr": zc_}))
    check("G corr_rise_paths_own: a pair's jumps are on its OWN z (first >= +1 after < +0.5, 8 weeks apart) and the path at k = 0 is that z", int(own_per_.loc["N100|SC100", "jumps"]) == len(ev_own) and all(zc_.loc[d] >= 1 for d in ev_own) and np.isclose(own_paths_[(own_paths_["pair"] == "N100|SC100") & (own_paths_["series"] == "z_corr") & (own_paths_["k"] == 0)]["mean_z"].iloc[0], zc_.loc[ev_own].mean()))
    check("G together_pairs: the pair's own vol = the mean of its two legs' 13-week vol", np.allclose(tp_["v_N100|SC100"].dropna(), ((vol.rolling_vol(rw)["N100"] + vol.rolling_vol(rw)["SC100"]) / 2).dropna(), atol=1e-9))
    check("G together_pairs: level and z for all ten pairs; z is the pair's own 104-week z-score", set(c for c in tp_.columns if "|" in c and not c.startswith(("z_", "v_", "zv_"))) == ten and np.allclose(tp_["z_N100|SC100"].dropna(), ((tp_["N100|SC100"] - tp_["N100|SC100"].rolling(C.CB_NORM, min_periods=int(0.5 * C.CB_NORM)).mean()) / tp_["N100|SC100"].rolling(C.CB_NORM, min_periods=int(0.5 * C.CB_NORM)).std()).dropna(), atol=1e-9))

    atr_ = vol.atr_pct(panel)
    import supertrend as ST_
    bars_ = ST_.resample_ohlc(panel["N100"], C.SIGNAL_BAR)
    check("G atr_pct = the desk's 5-bar LW ATR on weekly bars divided by the close, from START", np.isclose(atr_["N100"].dropna().iloc[-1], (ST_.atr(bars_) / bars_["close"]).dropna().iloc[-1]) and atr_.index[0] >= pd.Timestamp(C.START))
    regs_, regl2 = vol.corr_by_regime(rw, lvl, atrp=atr_)
    check("G corr_by_regime carries each leg's vol and the ATR regimes, with weeks per fifth summing to the pair's weeks", {"first leg's vol", "second leg's vol", "N100 ATR%", "pair's own ATR%"} <= set(regs_) and int(regl2[(regl2["regime"] == "pair's own vol") & (regl2["pair"] == "N100|SC100")]["weeks"].sum()) == int(sigcorr.pair_corrs(rw)["N100|SC100"].notna().sum()))
    gwk_, gsum_ = histcorr.gold_corr_rank(rw)
    PCg = sigcorr.pair_corrs(rw)
    both_ = PCg[["SC100|GoldMCX", "N100|GoldMCX"]].dropna()
    check("G gold_corr_rank: weeks SC100xGold > N100xGold counted on the weeks both exist", int(gsum_.loc["SC100", "weeks_above_N100"]) == int((both_["SC100|GoldMCX"] > both_["N100|GoldMCX"]).sum()) and int(gsum_.loc["SC100", "weeks_compared"]) == len(both_))
    gq_, gpath_, gper_ = histcorr.gold_when_decoupled(rw, corrbreak.episodes(corrbreak.series(rw)))
    check("G gold_when_decoupled: five G-quintiles per gold pair with the weeks summing to the pair's weeks; |corr| >= |mean corr| in every cell", all(int(gq_[gq_["pair"] == p]["weeks"].sum()) == int(PCg[p].notna().sum()) for p in set(gq_["pair"])) and (gq_["mean_abs_corr"] >= gq_["mean_corr"].abs() - 1e-12).all())

    print("H. the review of 2026-10-19")
    # H1. the around-flip robustness columns: the median is the median over the same flips; ex-year drops exactly that year's flips
    afr = sigcorr.around_flip(st, rw, refs=["N100"]); sellf = afr[afr["event"] == "flip to SELL"].set_index("weeks_from_flip")
    sN = st["N100"].reindex(rw.index); fl_ = rw.index[((sN == -1) & (sN.shift(1) == 1)).fillna(False)]
    Gm_ = sigcorr.rolling_mean_corr(rw, C.EQUITY)
    at0 = Gm_.loc[fl_].dropna(); at0x = Gm_.loc[[d for d in fl_ if d.year != C.FLIP_EXCLUDE_YEAR]].dropna()
    check("H around_flip: median_corr at k = 0 is the median of G over the flip weeks; mean_corr_exYEAR drops that year's flips", np.isclose(sellf.loc[0, "median_corr"], at0.median()) and np.isclose(sellf.loc[0, f"mean_corr_ex{C.FLIP_EXCLUDE_YEAR}"], at0x.mean()) and int(sellf.loc[0, f"n_events_ex{C.FLIP_EXCLUDE_YEAR}"]) == len(at0x) and len(at0x) < len(at0))
    # H2. the quintile hit rate counts only weeks with a forward return; the next-QUINT_FWD-week return is t+1..t+F
    lgq, _ = histcorr.same_quarter(rw)
    q5 = lgq[(lgq["asset"] == "N100") & (lgq["bucket"] == "Q5")].iloc[0]
    mcq = sigcorr.rolling_mean_corr(rw, C.EQUITY, C.HIST_WINDOW); bkq = pd.qcut(mcq, C.N_BUCKETS, labels=[f"Q{i + 1}" for i in range(C.N_BUCKETS)])
    fwdq = rw["N100"].shift(-1).rolling(C.QUINT_FWD).sum().shift(-(C.QUINT_FWD - 1)); inq5 = fwdq[bkq == "Q5"].dropna()
    i_ = rw.index.get_loc(inq5.index[10])
    check("H same_quarter: the next-F-week return = t+1..t+F; hit = share positive over the weeks that HAVE a forward return", np.isclose(inq5.iloc[10], rw["N100"].iloc[i_ + 1:i_ + 1 + C.QUINT_FWD].sum()) and np.isclose(q5[f"next_{C.QUINT_FWD}w_hit"], (inq5 > 0).mean()) and int(q5[f"n_next_{C.QUINT_FWD}w"]) == len(inq5) and len(inq5) <= int(q5["weeks"]) and int(q5["n_next"]) < int(q5["weeks"]))
    # H3. the gold-vs-group fit: the slope is OLS of y on G and the deciles hold the weeks
    gvw_, gvf_, gvb_ = histcorr.gold_vs_group(rw)
    d_ = gvw_[["G", "N100|GoldMCX"]].dropna(); b_ = np.polyfit(d_["G"], d_["N100|GoldMCX"], 1)[0]
    check("H gold_vs_group: slope = OLS of the gold-pair correlation on G; Pearson matches; decile weeks sum to n", np.isclose(gvf_.loc["N100|GoldMCX", "slope"], b_) and np.isclose(gvf_.loc["N100|GoldMCX", "pearson"], d_["G"].corr(d_["N100|GoldMCX"])) and int(gvb_[gvb_["y"] == "N100|GoldMCX"]["weeks"].sum()) == int(gvf_.loc["N100|GoldMCX", "n"]))
    # H4. the gap: the AR(1) half-life follows from phi; an extreme's forward return uses t+1..t+h; events are first weeks of runs
    ggs_, gga_, ggw_ = histcorr.gap_reversion(rw)
    s0_ = ggs_.iloc[0]; gap_ = ggw_["gap"].dropna()
    up_ = (gap_ > s0_["mean"] + C.GAP_SD * s0_["sd"]); starts_ = gap_.index[up_ & ~up_.shift(1, fill_value=False)]
    a4_ = gga_[(gga_["horizon_weeks"] == C.GAP_FWD[0]) & gga_["extreme"].str.contains("above")].iloc[0]
    relf = (rw["SC100"] - rw["N100"]).shift(-1).rolling(C.GAP_FWD[0]).sum().shift(-(C.GAP_FWD[0] - 1))
    ok_ = [d for d in starts_ if pd.notna((ggw_["gap"].shift(-C.GAP_FWD[0]) - ggw_["gap"]).get(d))]
    check("H gap_reversion: half-life = -ln2/ln(phi); events = first weeks of runs beyond +1 sd; forward relative return = t+1..t+h", np.isclose(s0_["half_life_weeks"], -np.log(2) / np.log(s0_["ar1_phi"])) and int(a4_["events"]) == len(ok_) and np.isclose(a4_["SC100_minus_N100_mean"], relf.loc[ok_].mean()))
    # H5. gold after a break: k = 0 is the episode's first week and the positive-week count is what it says
    gbp_, gba_, gbe_ = histcorr.gold_after_break(rw, corrbreak.episodes(corrbreak.series(rw)))
    e0 = gbe_.iloc[0]; seg_ = PCg[e0["pair"]].loc[pd.Timestamp(e0["start"]):].iloc[:C.GOLD_AFTER_BREAK + 1]
    check("H gold_after_break: corr_at_break = the pair's correlation in the episode's first week; positive weeks counted over the window", np.isclose(e0["corr_at_break"], seg_.iloc[0]) and int(e0[f"weeks_positive_of_first_{C.GOLD_AFTER_BREAK}"]) == int((seg_ > 0).sum()))
    # H6. the non-overlapping lead/lag: X on t-L+1..t and Y on t+1..t+L share no week, and the level rows are those block values
    L_ = C.LL_BLOCKS[0]
    cL = rw["N100"].rolling(L_).corr(rw["SC100"]); vL = (rw["N100"].rolling(L_).std() + rw["SC100"].rolling(L_).std()) / 2 * np.sqrt(C.ANNUALISE)
    llt = vol.lead_lag_blocks(rw[["N100", "SC100"]], None, blocks=(L_,))
    lev = llt[(llt["direction"] == "correlation → own vol") & (llt["kind"] == "level")].iloc[0]
    man = pd.DataFrame({"x": cL, "y": vL.shift(-L_)}).dropna()
    j_ = rw.index.get_loc(man.index[50])
    y_manual = (rw["N100"].iloc[j_ + 1:j_ + 1 + L_].std() + rw["SC100"].iloc[j_ + 1:j_ + 1 + L_].std()) / 2 * np.sqrt(C.ANNUALISE)
    check("H lead_lag_blocks: Y is the block t+1..t+L (no week shared with X's block t-L+1..t); Pearson matches; the non-overlapping sample steps every L weeks", np.isclose(man["y"].iloc[50], y_manual) and np.isclose(lev["pearson"], man["x"].corr(man["y"])) and int(lev["n_nonoverlap"]) == len(man.iloc[::L_]))
    # H7. the VIX surge and the flip lead/lag
    if lvl is not None and len(lvl.dropna()):
        sg_ = vol.vix_surges(lvl, rw.index); x_ = lvl.reindex(rw.index); ch_ = x_ / x_.shift(C.VIX_SURGE_WEEKS) - 1
        check("H vix_surges: each surge is the FIRST week of a run with the VIX up >= VIX_SURGE over VIX_SURGE_WEEKS", all(ch_.loc[d] >= C.VIX_SURGE and not (ch_.shift(1).loc[d] >= C.VIX_SURGE) for d in sg_["date"]) and len(sg_) > 0)
        vfe_, vse_, vfs_, vfp_ = vol.vix_vs_flips(st, rw, lvl)
        fN = vfe_[(vfe_["asset"] == "N100") & vfe_["vix_data"]]
        check("H vix_vs_flips: the share of flips with a surge before = the share with a surge within VIX_FLIP_WINDOW weeks before; the path at k = 0 is 1", np.isclose(vfs_.loc["N100", "share_flips_with_surge_before"], fN["weeks_since_last_surge"].notna().mean()) and (fN["weeks_since_last_surge"].dropna().between(1, C.VIX_FLIP_WINDOW)).all() and np.isclose(vfp_.set_index("k").loc[0, "vix_ratio_to_flip_week_mean"], 1.0))
    # H8. the state distribution and the regime stretches
    pts_ = vol.state_points(st, rw, lvl); sd_ = vol.state_dist(pts_)
    rowN = sd_[(sd_["asset"] == "N100") & (sd_["state"] == "SELL") & (sd_["measure"] == "abs_weekly_return")].iloc[0]
    xs_ = rw["N100"].abs()[st["N100"].reindex(rw.index) == -1].dropna()
    check("H state_dist: n / median / max of |weekly return| in N100's SELL weeks match the raw weeks", int(rowN["n"]) == len(xs_) and np.isclose(rowN["median"], xs_.median()) and np.isclose(rowN["max"], xs_.max()))
    rwk_ = vol.regime_weekly(rw); stx_ = vol.regime_stretches(rwk_)
    r0 = stx_[stx_["pair"] == "N100|SC100"].iloc[0]
    check("H regime_stretches: every week of a stretch is in Q5 of the pair's own vol; weeks and corr_inside match", (rwk_["q_N100|SC100"].loc[pd.Timestamp(r0["start"]):pd.Timestamp(r0["end"])] == "Q5").all() and int(r0["weeks"]) == len(rwk_.loc[pd.Timestamp(r0["start"]):pd.Timestamp(r0["end"])]) and np.isclose(r0["corr_inside"], rwk_["N100|SC100"].loc[pd.Timestamp(r0["start"]):pd.Timestamp(r0["end"])].mean()))
    # H9. the ratio study: the forward change is the first leg's return minus the second's over t+1..t+F
    rs_, rsw_ = histcorr.ratio_study(rw, st)
    w0 = rsw_[(rsw_["pair"] == "N100|MC100")].dropna(subset=["next"]).iloc[20]; k_ = rw.index.get_loc(pd.Timestamp(w0["date"]))
    check("H ratio_study: next = (r_a - r_b) summed over t+1..t+F; past over t-B+1..t; quintiles equal within a pair", np.isclose(w0["next"], (rw["N100"] - rw["MC100"]).iloc[k_ + 1:k_ + 1 + C.RATIO_FWD].sum()) and np.isclose(w0["past"], (rw["N100"] - rw["MC100"]).iloc[k_ - C.RATIO_BACK + 1:k_ + 1].sum()) and (rs_[(rs_["pair"] == "N100|MC100") & (rs_["cut"] == "correlation quintile")]["weeks"].max() - rs_[(rs_["pair"] == "N100|MC100") & (rs_["cut"] == "correlation quintile")]["weeks"].min()) <= 1)

    print("F. replacement")
    stf = pd.DataFrame(1, index=wk3, columns=["N100", "MC100", "SC100"], dtype=float)
    stf.loc[wk3[300]:wk3[320], "N100"] = -1                   # N100 flips to SELL at week 300, back to BUY at 321
    stf.loc[wk3[306]:wk3[330], "MC100"] = -1                  # MC100's own SELL starts 6 weeks later
    evf = replace.events(stf, rp.index)
    check("F events are exactly the BUY -> SELL flips", len(evf) == 2 and set(evf["date"]) == {wk3[300], wk3[306]} and set(evf["seller"]) == {"N100", "MC100"})
    rp2 = rp.copy(); rp2["Liquid"] = 0.0005
    crf = replace.candidate_rows(evf, stf, rp2, Zp)
    row = crf[(crf["seller"] == "N100") & (crf["candidate"] == "MC100")].iloc[0]
    i = rp2.index.get_loc(wk3[300])
    check("F candidate forward return = t+1..t+4 / t+1..t+13", np.isclose(row["fwd_cand_4"], rp2["MC100"].iloc[i + 1:i + 5].sum()) and np.isclose(row["fwd_cand_13"], rp2["MC100"].iloc[i + 1:i + 14].sum()) and np.isclose(row["fwd_seller_13"], rp2["N100"].iloc[i + 1:i + 14].sum()))
    check("F hold until the candidate's own SELL stops at its first SELL week (6 weeks later)", row["hold_weeks"] == 6 and row["hold_end"] == "own SELL" and np.isclose(row["hold_ret"], rp2["MC100"].iloc[i + 1:i + 7].sum()))
    check("F z of the candidate vs the seller = -z of the pair", np.isclose(row["z_cand_vs_seller"], -Zp["N100|MC100"].loc[wk3[300]]))
    Rf, Pf = replace.rules(crf)
    pk = Pf[(Pf["rule"] == "correlated & lagging, in BUY") & (Pf["seller"] == "N100")].iloc[0]
    check("F a rule with no candidate falls back to cash", (not pk["had_pick"]) and pk["pick"] == C.CASH and np.isclose(pk["ret_13"], row["fwd_cash_13"]) or pk["had_pick"])
    st_pick = Pf[(Pf["rule"] == "stay in the seller") & (Pf["seller"] == "N100")].iloc[0]
    check("F 'stay in the seller' = the seller's own forward return", np.isclose(st_pick["ret_13"], row["fwd_seller_13"]))

    print("D. hygiene")
    env = C.ROOT / ".env"
    pw = next((l.split("=", 1)[1].strip() for l in env.read_text().splitlines() if l.startswith("WM_DB_PASSWORD=")), None) if env.exists() else None
    if pw:
        hits = [str(p.relative_to(C.ROOT)) for p in C.ROOT.rglob("*") if p.is_file() and p.suffix in (".py", ".md", ".txt", ".json", ".csv", ".html", ".example") and ".git" not in p.parts and pw in p.read_text(encoding="utf-8", errors="ignore")]
        check("D password appears in no other file", not hits, ", ".join(hits))
    check("D .gitignore ignores .env", ".env" in (C.ROOT / ".gitignore").read_text().split())
    names = next((l.split("=", 1)[1].strip() for l in env.read_text().splitlines() if l.startswith("WM_REDACT_NAMES=")), "") if env.exists() else ""
    names = [n.strip() for n in names.split(",") if n.strip()]
    if names:                                                  # no person's name anywhere in the code, docs, tables or reports (the list itself lives only in .env)
        pat = re.compile(r"\b(" + "|".join(re.escape(n) for n in names) + r")\b", re.I)
        hits = sorted({str(p.relative_to(C.ROOT)) for p in C.ROOT.rglob("*") if p.is_file() and p.suffix in (".py", ".md", ".txt", ".json", ".csv", ".html", ".example") and ".git" not in p.parts and p.name != ".env" and pat.search(p.read_text(encoding="utf-8", errors="ignore"))})
        check("D no person's name in any file (list in .env: WM_REDACT_NAMES)", not hits, ", ".join(hits))
    if (C.ROOT / ".git").exists():
        check("D .env not tracked", ".env" not in subprocess.run(["git", "ls-files"], cwd=C.ROOT, capture_output=True, text=True).stdout.split())
    rep = C.REPORT_HTML
    check("D report exists", rep.exists())
    if rep.exists():
        txt = rep.read_text(encoding="utf-8")
        meta = json.loads((C.TABLES / "meta.json").read_text())
        check("D report names no missing or failed chart", "class='missing'" not in txt)
        check("D report embeds the plotly library once and at least 35 interactive figures", txt.count("plotly.min.js") <= 1 and "Plotly" in txt and txt.count("class=\"plotly-graph-div\"") >= 35)
        check("D synthetic stamp matches the data", ("SYNTHETIC DEMO DATA" in txt) == bool(meta.get("synthetic")))
    gd = C.REPORT_GUIDE_HTML
    check("D reading guide exists with its three parts and no failed chart", gd.exists() and all(s in gd.read_text(encoding="utf-8") for s in ("Part A", "Part B", "Part C", "Reading it for each combination")) and "failed:" not in gd.read_text(encoding="utf-8"))
    gtxt = gd.read_text(encoding="utf-8") if gd.exists() else ""
    check("D reading guide reads every G study pair by pair (around flips, breadth, lead, jumps, quintiles, >0.90, breaks, replacement)", gtxt.count("Reading it for each pair") >= 6 and all(f"fig_{n}" in gtxt for n in ("1_corr_around_flip_pairs", "1_corr_by_breadth", "2_corr_quintiles_pairs", "5_vol_corr_lead", "5_vol_corr_lead_pairs", "5_together", "5_together_pairs", "5_corr_rise_paths", "5_corr_rise_paths_own", "0_gold_when_decoupled", "5_corr_by_regime_more")))
    check("D reading guide carries the 2026-10-19 additions (gold scatter / gap / after-break, state distribution, VIX vs flip, regime components, non-overlapping lead/lag, ratio) and checks the review's inferences", all(f"fig_{n}" in gtxt for n in ("0_gold_vs_group", "0_gold_gap_reversion", "0_gold_after_break", "5_vol_state_box", "5_vix_flip_leadlag", "5_regime_scatter", "5_regime_timeline", "5_lead_lag_blocks", "2_ratio_by_corr", "2_ratio_by_state")) and "Inference → what the non-overlapping test says" in gtxt and "B10b" in gtxt)

    print(f"\n{len(P)} passed, {len(F)} failed")
    if data.is_synthetic():
        print("NOTE: SYNTHETIC data run.")
    if F:
        print("FAILED:", *F, sep="\n  ")
    return 1 if F else 0


if __name__ == "__main__":
    sys.exit(main())
