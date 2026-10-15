# LOGIC — exact definitions

Weekly bars (W-FRI, last close). r_i,t = ln P_t − ln P_{t−1}. States s_i,t ∈ {+1, −1} from the desk's supertrend (5-bar LW ATR, ×2, (H+L)/2, ratcheting) on weekly OHLC. Annualisation: mean ×52, cov ×52, vol ×√52.

## §2 — `sigcorr.py` (tables 1_)
* `by_state`: for each asset pair and each cell {both BUY, both SELL, disagree, all}: corr, cov, vol_a, vol_b, mean return, on the weeks in the cell (≥ `MIN_CELL` = 20). Decomposition: cov_corr_only = corr_SELL × vol_a,BUY × vol_b,BUY; share_from_corr = (cov_corr_only − cov_BUY) / (cov_SELL − cov_BUY); share_from_vol = 1 − that.
* `around_flip`: mc_t = mean of the six pairwise 13-week rolling correlations among the indices present. For EVERY asset ref: for each flip of ref to SELL (s = −1 and s_{t−1} = +1) and to BUY, mc at offsets k = −8..+8, averaged over flips; also ref's own mean 13-week correlation with the other indices, and its trailing 13-week return, at each offset.
* `by_breadth`: mc_t grouped by (number of index signals in SELL) / (number existing), in five bands.
* **Every pair beside G** (`pair_corrs`: ρ_ab,t = 13-week rolling correlation of weekly log returns, min 10 of 13 weeks, all ten pairs incl. gold; G_t = mean over the six index pairs present): `around_flip_pairs` = around_flip with ρ_ab,t in place of mc_t, for every pair; `by_breadth_pairs` = by_breadth with ρ_ab,t (weeks = weeks in the band with any index-pair ρ); `corr_timeseries` carries the ten ρ_ab,t columns. In `histcorr`: `by_year_pairs` (mean ρ_ab,t by calendar year), `same_quarter_pairs` (ρ_ab,t cut into its own quintiles over the pair's weeks; same / next as in same_quarter, for each leg), `timeline` carries ρ_ab,t, `episodes_pairs` (stretches of ≥ 8 weeks with ρ_ab,t > 0.90, an open stretch at the end counting; each leg's return during and over t+1..t+13 after; per-pair summary). In `corrbreak.series`: ρ_ab,t and its z (same 104-week norm) for every pair. In `vol`: `vol_corr_lead_pairs` (corr( Δ13 (v_a + v_b)/2 at t , Δ13 ρ_ab at t+k ), and against Δ13 ln VIX; needs > 30 weeks), `together_pairs` (ρ_ab,t and z_ab,t), `corr_rise_paths_pairs` (around the same G jumps: z_ab at k = −8..+8 averaged; per pair the share of jumps in which z_ab ≥ 1 somewhere in k = −8..+4, and z_ab at k = 0). `together_pairs` also carries v_ab,t = (v_a,t + v_b,t)/2 and its 104-week z, and z of ln VIX. `corr_rise_paths_own`: per pair, jumps = corr_rise_events on z_ab (first ≥ 1 after < 0.5, ≥ 8 weeks apart); around them z_ab, z(v_ab) and z(ln VIX) at k = −8..+8 averaged; per pair the mean at k = −4 and 0 of each, and the share of jumps with z(v_ab) ≥ 1 (resp. z VIX ≥ 1) somewhere in k = −8..−1.

## §4 — `histcorr.py` (tables 2_)
* `by_year`: mc_t (13-week) grouped by calendar year: mean/min/max, weeks > 0.90; each asset's return = exp(Σ r) − 1 over the year's weeks.
* `same_quarter`: mc_t cut into quintiles over the full sample. same = Σ r over t−12..t; next = Σ r over t+1..t+13. Per asset and bucket: mean, hit rate (share > 0), n.
* `episodes`: runs of mc_t > 0.90 lasting ≥ 8 weeks; returns during and over the 13 weeks after.

## §5 — `corrbreak.py` — who breaks the correlation first (tables 3_)
* pairwise ρ_ab,t = 13-week rolling correlation of weekly log returns. c_i,t = mean over b≠i of ρ_ib,t (the four indices). G_t = mean over all six pairs. c_gold,t = mean over the four of ρ_gold,b,t.
* z_x,t = (x_t − mean_104(x)) / sd_104(x), rolling, min 52 obs.
* break(x) = first t with z_x,t ≤ −1 after z_x > −0.5 ("armed"); the stretch ends at the first t with z_x,t > −0.5. Episodes of G closer than 8 weeks are merged. G_before = mean G over the 8 weeks before the start.
* Per episode and index: break date searched in [start − 8 weeks, end]; lead = (date − start)/7; rank by date (min method); z_at_group_break = z_c,i at the start week, rank_deepest; corr_drop = min c_i in window − mean c_i over the 8 weeks before, drop_share = corr_drop / Σ_i corr_drop, rank_drop; rel_13w = Σ_{t−12..t} r_i − mean over others of the same sum, at the break week; direction by its sign.
* summary: shares, averages, medians as named; combined_rank = mean(avg_rank, avg_rank_deepest, avg_rank_drop); verdicts by argmin.
* pair_breaks: the same break rule on each ρ_ab,t with its own 104-week z.
* before_selloff: for each price episode (§4), the 13 weeks ending at N100's peak: min z_G, whether ≤ −1, and each index's first crossing (weeks before the peak).
* after_break: for each asset break, Σ r over t+1..t+4 and t+1..t+13 for the breaker, the mean of the other indices, and gold; grouped by direction.
* sensitivity: windows {13, 26} × thresholds {−1, −1.5}; episodes and by-date ranks recomputed.

## §6 — `breaks.py` — who breaks in price / signal (tables 4_)
* `episodes`: on weekly N100 closes, running high H_t; drawdown d_t = P_t/H_t − 1. An episode opens the first week d ≤ −10% and closes the week d ≥ 0 (or the last bar). peak = the last week at the running high before the open; trough = argmin d inside; merged if peaks are < 8 weeks apart.
* `dates_per_asset`: for each episode and asset, the window runs from 8 weeks before N100's peak to 4 weeks after the trough. peak = argmax of the asset's close in [window start, trough]; signal_sell = first flip to −1 after the peak, within the window; cross_10pct = first week the asset's close ≤ 0.9 × its running high (running high from the window start) after the peak, within the window; recover_buy = first flip to +1 after the trough within 52 weeks; own_fall = P_trough / P_peak − 1.
* `leads`: lead_x = (asset's date − N100's date) in weeks. `order_table`: rank within episode, min method (ties share). `summary`: medians, first shares, average ranks, n per measure; verdict = argmin average rank among the four indices on signal_sell, cross_10pct, recover_buy.

## §3 — `vol.py` (tables 5_)
* v_i,t = sd of r_i over t−12..t × √52 (`VOL_WINDOW` = 13, min 11 obs).
* `by_state`: per asset, on the weeks its own state is +1 and on the weeks it is −1 (≥ `MIN_CELL`): vol, mean r, share r < 0, min r, mean and median VIX level; ratios vol_SELL / vol_BUY and VIX_SELL / VIX_BUY.
* `around_flip`: for every asset ref, v_i (all assets) and the VIX level at offsets k = −8..+8 around each ref flip (to −1 and to +1), averaged over flips.
* `corr_by_regime`: three regime series — v_N100,t; (v_a,t + v_b,t)/2 per pair; VIX_t — each cut into quintiles over its sample (per pair for the own-vol regime); for each of the ten pairs ρ_ab,t (13-week) averaged inside each quintile; Q5 − Q1; Spearman(ρ_ab,t, regime_t). `corr_by_vol_regime` = the N100-vol case with its edges.
* `vol_corr_lead`: corr( Δ13 v_N100,t , Δ13 G_t+k ) and corr( Δ13 ln VIX_t , Δ13 G_t+k ), k = −4..+4; also corr and Spearman of the levels (v_N100,t, G_t), (VIX_t, G_t), and corr(VIX_t, v_N100,t).

## §7 — `pairs.py` (tables 6_)
* Pairs (a, b) = combinations of ASSETS in config order. lr_t = ln P_a,t − ln P_b,t on weekly closes.
* z_t = (lr_t − mean(lr over t−51..t)) / sd(lr over t−51..t) (`PAIR_WINDOW` = 52, min 42). z_of(x, y) = z of ln(P_x/P_y) = −z_of(y, x).
* buckets: z < −2, [−2, −1), [−1, 1), [1, 2), ≥ 2 (`PAIR_Z_EDGES`).
* `returns_table`: per pair on the common weeks: total = exp(Σ r) − 1, annualised = (1 + total)^(1/years) − 1, last 13w / 52w, vol, share up, worst / best week; mean r × 52 on both-BUY / both-SELL / disagree weeks (≥ `MIN_CELL`); spread = a − b of each; share of weeks r_a > r_b.
* `stats_table`: corr full / 13w / 52w; vol per leg full / 13w; beta_a|b = cov/var over 52w and full; shares |z| > 1, > 2; n of first crossings of |z| ≥ 2; ratio total change = exp(lr_T − lr_0) − 1.
* `coint_table`: ADF(lr, AIC) p; EG(a on b) p, EG(b on a) p (`statsmodels.coint`, trend c); half-life from Δlr_t = α + β lr_t−1: −ln2 / ln(1+β) if −1 < β < 0 else ∞; the same on lr − its 52-week mean; drift = mean(Δlr) × 52. Verdict tiers: ADF < 0.05 & HL ≤ 52 → pulls back; min EG < 0.05 & HL ≤ 52 → pulls back with a hedge ratio; ADF < 0.10 & HL ≤ 52 → some pull-back, not established; else no long-run level.
* `z_forward`: F_h,t = Σ_{t+1..t+h} (r_a − r_b), h ∈ {4, 13}; per bucket mean, median, share(sign F = −sign z), mean(−sign z · F), HAC t (4 lags, ≥ 20 obs); pooled over pairs with the t on the per-week mean across pairs.
* `path_after`: events = first t with |z_t| ≥ 2 after |z_t−1| < 2; z_t+k · sign(z_t), k = 0..13, averaged; share ≤ 1; share having crossed 0 by k.

## §8 — `pairtrade.py` (tables 7_)
* State machine per pair. At t with no open trade: if |z_t| ≥ `PT_ENTRY` (2): side = +1 if z_t < 0 (long a short b) else −1; optional: the cheap leg's state must be +1 (`require_buy`). Earns side × (r_a − r_b) over u = t+1 .. until the first u with |z_u| ≤ `PT_EXIT` (0.5) or u − t ≥ `PT_MAX_HOLD` (26) or the last bar. gross = Σ side × spread over t+1..u; cost = 4 × `PT_COST_BPS` / 1e4; net = gross − cost; worst_point = min running gross. Next entry considered from u + 1.
* `summary`: per pair and pooled; t = mean / (sd/√n) on trade nets (n ≥ 5); net_per_year = Σ net / years of the pair's data; drawdown on the trade-ordered cumulative net; verdict as in GLOSSARY.
* `equity`: weekly series, side × spread while in a trade, cost on the exit bar, cumulative.
* `sensitivity`: entry {1.5, 2, 2.5} × exit {0, 0.5} × require_buy {no, yes}, pooled.

## §9 — `replace.py` (tables 8_)
* Events: (A, t) with s_A,t = −1 and s_A,t−1 = +1, t in the weekly return index.
* Per event and candidate B ≠ A with P_B,t present: corr = 52-week rolling corr(r_A, r_B) at t; correlated = corr ≥ `RP_CORR_MIN` (0.5); z = z_of(B, A)_t; deviation = lagging if z ≤ −`RP_Z_DEV` (−1), leading if z ≥ +1, else neutral (no z if NaN); state = s_B,t; rel_13w = Σ_{t−12..t} r_B − Σ r_A. Forward: Σ_{t+1..t+h} r for B, A, cash, gold, h ∈ `RP_FWD` (4, 13). Hold: from t+1 to the first u > t with s_B,u = −1 (inclusive), capped at u − t = `RP_MAX_HOLD` (26); the same weeks for A and cash.
* `cells`: group means by deviation × correlated × state, with `any` pools; rows = candidate rows, events = distinct (A, t).
* `rules`: per event one pick per rule (definitions in GLOSSARY); fallback = cash when the rule's set is empty; `had_pick` false then. Per rule: means, medians, hit rates, mean minus cash, HAC t of (ret − cash) over events sorted by date (≥ 20 events), the same on picked events only, hold variant on picked events.
* `event_paths` / `z_path`: z_of(B, A) at t+k, k = −13..+13, per event; averaged per (A, B) and per A over all B.
* `current`: at the last week every index has a close: corr, z, deviation, state for every ordered pair.

## Not here
No t-stats on the ten decorrelation episodes or six sell-offs. The pair-trade and replacement rules are evaluations of rules on history (one unit per leg, flat costs, no sizing, no portfolio overlap), not backtests of strategies; a pre-registered test inside the rotation framework is the next step.
