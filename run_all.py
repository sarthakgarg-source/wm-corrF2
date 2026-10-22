"""
Run the study and write every table to outputs/tables/.

    python run_all.py            # ~10 s
    python run_all.py --no-charts
"""
from __future__ import annotations

import json
import sys
import time
import warnings

import numpy as np
import pandas as pd

import breaks
import config as C
import corrbreak
import data
import histcorr
import pairs
import pairtrade
import replace
import sigcorr
import vol

warnings.filterwarnings("ignore")


def _w(df, name, **kw):
    C.TABLES.mkdir(parents=True, exist_ok=True)
    kw.setdefault("float_format", "%.6f")
    df.to_csv(C.TABLES / f"{name}.csv", **kw)


def run(charts: bool = True) -> None:
    t0 = time.time()
    C.TABLES.mkdir(parents=True, exist_ok=True)
    for f in C.TABLES.glob("*.csv"):
        f.unlink()
    panel = data.load_panel()
    missing = [a for a in C.ASSETS if a not in panel]
    if missing:
        raise SystemExit(f"STOP: required series missing: {missing}. See the NOT FOUND line above.")
    px = data.close_panel(panel)
    px_w = data.resample_close(px, "W-FRI")
    rw = data.returns(px, "weekly")
    lvl = data.vix_level(px, "weekly") if C.VIX in panel else None
    st = sigcorr.states(panel)
    _w(px_w, "px_weekly"); _w(rw, "ret_weekly"); _w(st, "sig_states_weekly"); _w(data.coverage(rw), "coverage_weekly")
    meta = {"synthetic": data.is_synthetic(), "synthetic_files": sorted(data.SYNTHETIC_SEEN), "start": C.START,
            "last_session": str(px.index[-1].date()), "assets": C.ASSETS, "run_at": pd.Timestamp.now().isoformat(timespec="seconds")}

    # 0. correlation between the asset classes: full sample, last 52 weeks, last 13 weeks (pairwise-complete)
    cols0 = [c for c in C.ASSETS if c in rw.columns] + [c for c in (C.VIXCHG, C.CASH) if c in rw.columns]
    def _cm(df, min_obs):                      # a series with too few weeks in the window gets no cell (the VIX file ends early)
        m = df.corr(min_periods=min_obs)
        thin = [c for c in df.columns if df[c].notna().sum() < min_obs]
        m.loc[thin, :] = np.nan; m.loc[:, thin] = np.nan
        return m
    _w(_cm(rw[cols0], 20), "0_corr_matrix_full")
    _w(_cm(rw[cols0].iloc[-52:], 26), "0_corr_matrix_52w")
    _w(_cm(rw[cols0].iloc[-13:], 10), "0_corr_matrix_13w")
    _w(pd.DataFrame({c: [rw[c].dropna().index[0].date(), rw[c].dropna().index[-1].date(), int(rw[c].notna().sum())] for c in cols0}, index=["first", "last", "weeks"]).T, "0_corr_matrix_coverage")
    print("0. correlation between the asset classes: full / 52w / 13w matrices")

    # 1. correlation vs signal
    bs, dec = sigcorr.by_state(st, rw)
    _w(bs, "1_corr_by_signal_state", index=False)
    _w(dec, "1_cov_decomposition", index=False)
    _w(sigcorr.around_flip(st, rw), "1_corr_around_flip", index=False)
    _w(sigcorr.around_flip_pairs(st, rw), "1_corr_around_flip_pairs", index=False)
    _w(sigcorr.by_breadth(st, rw), "1_corr_by_breadth")
    _w(sigcorr.by_breadth_pairs(st, rw), "1_corr_by_breadth_pairs")
    _w(sigcorr.corr_timeseries(st, rw), "1_corr_timeseries")
    print("1. correlation vs signal: by state, covariance decomposition, around flips (G and every pair), by breadth (G and every pair)")

    # 2. correlation vs returns, history
    _w(histcorr.by_year(rw), "2_by_year")
    _w(histcorr.by_year_pairs(rw), "2_by_year_pairs")
    lg, piv = histcorr.same_quarter(rw)
    _w(lg, "2_corr_quintile_returns_long", index=False)
    _w(piv, "2_corr_quintile_returns")
    _w(histcorr.same_quarter_pairs(rw), "2_corr_quintile_returns_pairs", index=False)
    _w(histcorr.timeline(rw, lvl), "2_timeline")
    _w(histcorr.episodes(rw), "2_high_corr_episodes", index=False)
    gwk, gsum = histcorr.gold_corr_rank(rw)
    _w(gwk, "0_gold_corr_weekly"); _w(gsum, "0_gold_corr_rank")
    hp, hps = histcorr.episodes_pairs(rw)
    _w(hp, "2_high_corr_episodes_pairs", index=False); _w(hps, "2_high_corr_pairs_summary")
    gvw, gvf, gvb = histcorr.gold_vs_group(rw)
    _w(gvw, "0_gold_vs_group_weekly"); _w(gvf, "0_gold_vs_group_fit"); _w(gvb, "0_gold_vs_group_bins", index=False)
    ggs, gga, ggw = histcorr.gap_reversion(rw)
    _w(ggs, "0_gold_gap_stats", index=False); _w(gga, "0_gold_gap_after_extreme", index=False); _w(ggw, "0_gold_gap_weekly")
    rs, rsw = histcorr.ratio_study(rw, st)
    _w(rs, "2_ratio_study", index=False); _w(rsw, "2_ratio_weekly", index=False)
    print("2. history: by year, quintiles (+ next 4 weeks), timeline, high-correlation episodes — G and every pair; gold vs the group, the gold gap, the ratio study")

    # 3. who breaks the CORRELATION first
    S = corrbreak.series(rw)
    cep = corrbreak.episodes(S)
    ab = corrbreak.asset_breaks(S, cep, rw)
    _w(S, "3_corr_series")
    _w(cep, "3_corr_episodes", index=False)
    _w(ab, "3_asset_breaks", index=False)
    _w(corrbreak.summary(ab), "3_summary")
    for col, name in (("rank", "date"), ("rank_deepest", "deepest"), ("rank_drop", "drop")):
        _w(ab.pivot(index="episode", columns="asset", values=col), f"3_rank_{name}")
    _w(ab.pivot(index="episode", columns="asset", values="lead_vs_group_weeks"), "3_lead_weeks")
    pb, psm = corrbreak.pair_breaks(rw, cep)
    _w(pb, "3_pair_breaks", index=False); _w(psm, "3_pair_summary")
    pep = breaks.episodes(px_w)
    _w(corrbreak.before_selloff(S, pep), "3_before_selloff", index=False)
    gq, gpath, gper = histcorr.gold_when_decoupled(rw, cep)
    _w(gq, "0_gold_by_group_corr", index=False); _w(gpath, "0_gold_around_decoupling", index=False); _w(gper, "0_gold_in_episodes", index=False)
    ipg, ipgw = histcorr.index_pair_vs_gold_pair(rw)
    _w(ipg, "0_index_vs_gold_pairs", index=False); _w(ipgw, "0_index_vs_gold_pairs_weekly")
    gbp, gba, gbe = histcorr.gold_after_break(rw, cep)
    _w(gbp, "0_gold_after_break_paths", index=False); _w(gba, "0_gold_after_break_avg", index=False); _w(gbe, "0_gold_after_break_episodes", index=False)
    afb, afs = corrbreak.after_break(ab, rw)
    _w(afb, "3_after_break", index=False); _w(afs, "3_after_break_summary", index=False)
    sens, sens_long = corrbreak.sensitivity(rw)
    _w(sens, "3_sensitivity"); _w(sens_long, "3_sensitivity_long", index=False)
    meta_sens = {f"w{k[0]} z{k[1]}": int(v) for k, v in sens.attrs["episodes"].items()}
    print(f"3. who breaks the correlation first: {len(cep)} decorrelation episodes")

    # 4. who breaks in price / signal
    ep = pep
    dp = breaks.dates_per_asset(ep, px_w, st)
    ld = breaks.leads(dp)
    _w(ep, "4_price_episodes", index=False)
    _w(ld, "4_dates_and_leads", index=False)
    for col in ("peak", "signal_sell", "cross_10pct", "recover_buy"):
        _w(ld.pivot(index="episode", columns="asset", values=f"lead_{col}"), f"4_lead_{col}")
        _w(breaks.order_table(ld, col), f"4_rank_{col}")
    _w(breaks.summary(ld), "4_summary")
    print(f"4. who breaks in price / signal: {len(ep)} sell-offs")

    # 5. volatility
    _w(vol.rolling_vol(rw), "5_vol_rolling")
    _w(vol.by_state(st, rw, lvl), "5_vol_by_state")
    _w(vol.around_flip(st, rw, lvl), "5_vol_around_flip", index=False)
    cv, cv_long = vol.corr_by_vol_regime(rw)
    _w(cv, "5_corr_by_vol_regime"); _w(cv_long, "5_corr_by_vol_regime_long", index=False)
    _w(cv.attrs["edges"], "5_vol_regime_edges")
    atrp = vol.atr_pct(panel)
    _w(atrp, "5_atr_pct_weekly")
    regs, regs_long = vol.corr_by_regime(rw, lvl, atrp=atrp)
    _w(regs_long, "5_corr_by_regime_long", index=False)
    for name, w in regs.items():
        _w(w, "5_corr_by_regime_" + name.replace("'", "").replace(" ", "_").replace("%", "pct"))
    _w(regs_long.pivot_table(index=["regime", "pair"], columns="bucket", values="weeks", aggfunc="sum"), "5_regime_weeks")
    vl = vol.vol_corr_lead(rw, lvl)
    _w(vl, "5_vol_corr_lead")
    _w(vol.vol_corr_lead_pairs(rw, lvl), "5_vol_corr_lead_pairs", index=False)
    meta["vol_corr_level"] = {k: float(v) for k, v in vl.attrs.items()}
    ll = vol.lead_lag_blocks(rw, lvl)
    _w(ll, "5_lead_lag_blocks", index=False); _w(vol.lead_lag_verdicts(ll), "5_lead_lag_verdicts", index=False)
    rwk = vol.regime_weekly(rw)
    _w(rwk, "5_regime_weekly"); _w(vol.regime_stretches(rwk), "5_regime_stretches", index=False)
    pts = vol.state_points(st, rw, lvl)
    _w(pts, "5_state_points", index=False); _w(vol.state_dist(pts), "5_state_dist", index=False)
    if lvl is not None and len(lvl.dropna()):
        vfe, vse, vfs, vfp = vol.vix_vs_flips(st, rw, lvl)
        _w(vol.vix_surges(lvl, rw.index), "5_vix_surges", index=False)
        _w(vfe, "5_vix_flip_events", index=False); _w(vse, "5_vix_surge_events", index=False); _w(vfs, "5_vix_flip_summary"); _w(vfp, "5_vix_flip_paths", index=False)
        meta["n_vix_surges"] = int(len(vol.vix_surges(lvl, rw.index)))
    _w(vol.timeseries(st, rw, lvl), "5_vol_timeseries")
    tg = vol.together(rw, lvl)
    _w(tg, "5_together")
    tp = vol.together_pairs(rw, lvl)
    _w(tp, "5_together_pairs")
    own_paths, own_per = vol.corr_rise_paths_own(tp)
    _w(own_paths, "5_corr_rise_paths_own", index=False); _w(own_per, "5_corr_rise_own_summary"); _w(vol.corr_rise_events_own(tp), "5_corr_rise_own_events", index=False)
    paths, per = vol.corr_rise_paths(tg)
    _w(paths, "5_corr_rise_paths", index=False); _w(per, "5_corr_rise_events", index=False)
    pp, pps = vol.corr_rise_paths_pairs(tg, tp)
    _w(pp, "5_corr_rise_paths_pairs", index=False); _w(pps, "5_corr_rise_pairs_summary")
    meta["n_corr_rises"] = int(len(per))
    meta["vix_last"] = str(panel[C.VIX].index[-1].date()) if C.VIX in panel else None
    print("5. volatility: by state (+VIX), around every asset's flips, correlation by three regimes, vol-vs-correlation lead")

    # 6. each pair separately
    Z, bands = pairs.ratio_series(px_w)
    _w(Z, "6_ratio_z"); _w(bands, "6_ratio_bands")
    _w(pairs.stats_table(rw, px_w, st, Z), "6_pair_stats")
    _w(pairs.coint_table(px_w), "6_pair_coint")
    _w(pairs.z_forward(Z, rw), "6_z_forward", index=False)
    _w(pairs.path_after(Z), "6_path_after_2sd", index=False)
    _w(pairs.current(Z), "6_current")
    _w(pairs.pair_corr_series(rw), "6_pair_corr13")
    _w(pairs.returns_table(rw, px_w, st), "6_pair_returns")
    print(f"6. each pair: {Z.shape[1]} pairs, deviation z-score, pull-back tests")

    # 7. the pair-trading rule
    tr = pairtrade.all_trades(Z, rw, st=st)
    _w(tr, "7_trades", index=False)
    _w(pairtrade.summary(tr, rw), "7_summary")
    _w(pairtrade.equity(tr, rw), "7_equity")
    _w(pairtrade.sensitivity(Z, rw, st), "7_sensitivity", index=False)
    print(f"7. pair-trading rule: {len(tr)} trades")

    # 8. who replaces a SELL
    ev = replace.events(st, rw.index)
    cr = replace.candidate_rows(ev, st, rw, Z)
    _w(ev, "8_events", index=False); _w(cr, "8_candidates", index=False)
    _w(replace.cells(cr), "8_cells", index=False)
    _w(replace.by_pair(cr), "8_by_pair", index=False)
    R, P = replace.rules(cr)
    _w(R, "8_rules"); _w(P, "8_rule_picks", index=False); _w(replace.rules_by_seller(P), "8_rules_by_seller", index=False)
    ep = replace.event_paths(ev, Z)
    _w(ep, "8_event_paths", index=False); _w(replace.z_path(ep), "8_z_path", index=False)
    _w(replace.current(st, rw, Z), "8_current", index=False)
    print(f"8. who replaces a SELL: {len(ev)} SELL flips, {len(cr)} candidate rows")

    meta["sensitivity_episodes"] = meta_sens
    meta["n_pairs"] = int(Z.shape[1]); meta["n_trades"] = int(len(tr)); meta["n_sell_flips"] = int(len(ev))
    meta["cash_present"] = C.CASH in rw.columns
    meta["elapsed_s"] = round(time.time() - t0, 1)
    (C.TABLES / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    if charts:
        import plots
        plots.all_charts()
        print("[charts] written")
    print(f"done in {time.time() - t0:.0f}s -> {C.TABLES.relative_to(C.ROOT)}/")


if __name__ == "__main__":
    run(charts="--no-charts" not in sys.argv)
