# wm-BREAK — Study I: correlation — between the asset classes, with respect to the signal, with volatility, and with returns

Built around four things (Part I of the report), on N100, MC100, SC100, Micro250 and gold, weekly, 2016 → :

- **§2 Correlation between the asset classes** — the correlation matrix (whole sample / last 52 weeks / last 13 weeks, with the VIX change and cash for reference) and every pair's rolling correlation.
- **§3 Correlation with respect to the signal** — each pair in both-BUY / disagree / both-SELL weeks, side by side; correlation around every asset's flips; by how many signals are in SELL.
- **§4 Correlation and volatility** — vol by signal state (realised and VIX), the covariance split, correlation by volatility regime (three definitions), which moves first.
- **§5 The historical pattern between correlation and returns** — by quintile (same quarter / next quarter), by year, the timeline, the stretches above 0.90.

Part II (folded away at the end of the report, each with its one-line finding visible):

1. **How do correlation and covariance behave with respect to the supertrend signal?** (report §3) Pair correlation / covariance / volatility in both-BUY vs both-SELL weeks, the covariance rise split into "more correlated" vs "more volatile", correlation in the 8 weeks around **every** asset's flips, correlation vs how many signals are in SELL.
2. **How does volatility behave with respect to the signal, and how does it move correlation?** (§4) Each asset's realised vol and the VIX level in its own BUY vs SELL weeks; vol and the VIX around **every** asset's flips; every pair's correlation by volatility regime, the regime defined three ways (N100's vol, the pair's own vol, the VIX level); whether vol — realised or implied — moves before correlation or with it.
3. **What has the historical pattern between correlation and returns been?** (§5) By year; by correlation quintile → return over the same quarter and the next; the stretches above 0.90.
4. **Which asset breaks the correlation first?** (Part II §6) Each index's 13-week correlation with the others as a z-score against its own two-year norm; decorrelation episodes; who crossed first / was deepest / collapsed most; which way it broke; pair-level; early-warning; what followed; sensitivity. **And in price / signal?** (§7) Every 10%+ fall in N100: whose supertrend fired first, who crossed −10% first, who recovered first.
5. **Each pair separately.** (§8) For all ten pairs: each leg's cumulative return and a returns table (total / annualised / last 13w / last 52w / by joint signal state), the price ratio and its **deviation z-score** against the trailing 52 weeks, the pair's correlation and each leg's volatility with the VIX, whether the ratio pulls back (ADF / Engle-Granger / half-life, and the ratio's actual change after a deviation), plus the pair's own pair trades and SELL-flip replacements. One card per pair, five-panel interactive chart each.
6. **The pair-trading rule.** (§9) Enter at |z| ≥ 2, long the cheap leg / short the rich leg, exit at |z| ≤ 0.5 or 26 weeks, 10 bp per leg per side; every trade listed; long-only reading (the cheap leg's return vs the rich leg's); sensitivity grid.
7. **When an asset's signal flips to SELL, which asset should replace it?** (§10) Every SELL flip; each candidate's correlation with the seller, its deviation from the seller (lagging / neutral / leading) and its own signal at the flip; its return over the next 4 / 13 weeks and until its own SELL, against staying put, cash and gold; twelve replacement rules on the same flips; the deviation z-path around every flip.

    pip install -r requirements.txt
    python run_all.py          # ~50 s (add --no-charts to skip the static PNGs)
    python make_report.py      # outputs/report.html — the CLEAN report (~10 MB; interactive, works offline)
    python make_report_full.py # outputs/report_full.html — everything expanded, five-part scaffold per section
    python make_guide.py       # outputs/report_guide.html — the READING GUIDE: every question with its answer, every chart with what is on it and a reading per pair / per asset, every column name
    python checks.py           # 71 checks; must say "0 failed"

Read `GLOSSARY.md` first — every term and column, in plain words. `LOGIC.md` has the exact definitions. The clean report keeps one shape per chapter — *the question → the answer in one line → how we know (the chart, and where on it to look) → the key numbers as tiles → what it means for the book* — with the detail (every episode, every trade, the extra tables and charts) folded into collapsible **More…** boxes; the full report expands all of it with a five-part scaffold (question, method, how to read, what it says, what it does not say). §0 of the clean report is the seven answers; §1 walks one sell-off (the March-2020 N100 flip) through every measure the report uses — correlation, volatility, the VIX, the signal, gold — on one table, before the measures are used across the whole history; every technical word is dotted-underlined and shows its meaning on hover. Wherever a study uses the group correlation G, it is repeated for every pair and the pair lines are on the same chart (gold pairs dashed, off until clicked; one colour per pair throughout), with the `_pairs` tables beside the group tables. **Every chart's bold title is its finding and the grey line under it is the method**; boxed notes on the chart point at where the finding can be seen. **Charts are interactive:** hover for values, click a legend entry to add or remove a trace, double-click to isolate it, drag to zoom.

## Layout
    config.py     every parameter                    data.py / supertrend.py   loaders and the desk's supertrend (copies of wm-VIXcorr)
    sigcorr.py    §2 (tables 1_)                      vol.py        §3 (5_)
    histcorr.py   §4 (2_)                             corrbreak.py  §5 (3_)        breaks.py  §6 (4_)
    pairs.py      §7 (6_)                             pairtrade.py  §8 (7_)        replace.py §9 (8_)
    figs.py       the interactive charts (Plotly) used by the report      plots.py  static PNG versions (optional)
    run_all.py / make_report.py (clean) / make_report_full.py (everything) / make_guide.py (the reading guide) / checks.py
    db.py, db_probe.py, fetch_*.py        Postgres access (credentials in .env only; nothing here needs the DB)

## Data
`data/index_ohlcv_all.csv`, `data/MAUc1_daily.csv` (from a sibling repo). Optional: `india_vix.csv` (one timeline chart), `liquidEtfFinal.csv` (the cash comparator `Liquid` in §9; without it the cash columns are blank).

## Tables (`outputs/tables/`, prefixed by module)
`0_corr_matrix_{full,52w,13w}`, `0_corr_matrix_coverage` ·
`1_corr_by_signal_state`, `1_cov_decomposition`, `1_corr_around_flip` (every asset as the reference), `1_corr_by_breadth`, `1_corr_timeseries` ·
`2_by_year`, `2_corr_quintile_returns` (+ `_long`), `2_timeline`, `2_high_corr_episodes` ·
`3_corr_series`, `3_corr_episodes`, `3_asset_breaks`, `3_summary`, `3_rank_{date,deepest,drop}`, `3_lead_weeks`, `3_pair_breaks`, `3_pair_summary`, `3_before_selloff`, `3_after_break(_summary)`, `3_sensitivity` ·
`4_price_episodes`, `4_dates_and_leads`, `4_lead_{…}`, `4_rank_{…}`, `4_summary` ·
`5_vol_rolling`, `5_vol_by_state`, `5_vol_around_flip`, `5_corr_by_regime_long` (+ one wide table per regime), `5_corr_by_vol_regime` (+ `_long`, `5_vol_regime_edges`), `5_vol_corr_lead`, `5_vol_timeseries` ·
`6_ratio_z`, `6_ratio_bands`, `6_pair_stats`, `6_pair_returns`, `6_pair_coint`, `6_z_forward`, `6_path_after_2sd`, `6_current`, `6_pair_corr13` ·
`7_trades`, `7_summary`, `7_equity`, `7_sensitivity` ·
`8_events`, `8_candidates`, `8_cells`, `8_by_pair`, `8_rules`, `8_rule_picks`, `8_rules_by_seller`, `8_event_paths`, `8_z_path`, `8_current`

## Charts
The report's charts are built by `figs.py` (Plotly, embedded in `report.html`, no internet needed). `python plots.py` writes static PNG copies to `outputs/charts/` (N100-only versions of the flip studies) for decks.
