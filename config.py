"""
wm-BREAK  ·  Study I  ·  Correlation, volatility and the signal — and what to do with a pair

  1. How do correlation and covariance behave with respect to the supertrend signal?
  2. How does volatility behave with respect to the signal, and how does it move correlation?
  3. What has the historical pattern between correlation and returns been?
  4. Which asset breaks the correlation first (and which breaks first in price / signal)?
  5. Each pair on its own: deviation (ratio z-score), correlation, volatility, pull-back.
  6. A pair-trading rule built from the deviation, evaluated pair by pair.
  7. When an asset's signal flips to SELL, which asset should replace it, and how did that go?

Weekly only. Every parameter lives here.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
CHARTS = OUT / "charts"

# ------------------------------------------------------------- data files --
INDEX_CSV = DATA / "index_ohlcv_all.csv"
GOLD_SIGNAL_CSV = DATA / "MAUc1_daily.csv"
VIX_CSV = DATA / "india_vix.csv"              # optional here: used only as a regime line on the timeline chart
LIQUID_CSV = DATA / "liquidEtfFinal.csv"      # optional: the cash leg ("Liquid") used as a comparator in the replacement study
CASH = "Liquid"

# --------------------------------------------------------------- universe --
INDICES = {
    "nifty100":     ("N100",     ["Nifty100", "N100", "Nifty 100"]),
    "midcap100":    ("MC100",    ["MC100", "NiftyMidcap100", "Midcap100", "Nifty Midcap 100"]),
    "smallcap100":  ("SC100",    ["SC100", "NiftySmallcap100", "Smallcap100", "Nifty Smallcap 100"]),
    "microcap250":  ("Micro250", ["NiftyMicrocap250", "Micro250", "Microcap250", "Nifty Microcap 250"]),
    "nifty50":      ("N50",      ["NSEI", "N50", "Nifty50", "Nifty 50"]),      # calendar + reference only
}
GOLD = "GoldMCX"
VIX = "VIX"
VIXCHG = "VIXchg"
ASSETS = ["N100", "MC100", "SC100", "Micro250", "GoldMCX"]
EQUITY = ["N100", "MC100", "SC100", "Micro250"]
UNIVERSE = ASSETS + [VIXCHG]
ORDER = UNIVERSE
BENCHMARK = "N100"

# ------------------------------------------------------------ frequencies --
FREQS = {"daily": None, "weekly": "W-FRI"}
HEADLINE_FREQ = "weekly"
START = "2016-01-01"
MIN_OBS = {"daily": 120, "weekly": 20}
RETURN_KIND = "log"
ANNUALISE = 52

# ---------------------------------------------------------------- signal --
LOOKBACK, MULTIPLIER, ATR_METHOD = 5, 2.0, "lw"      # the desk's supertrend, recovered not tuned
SIGNAL_BAR = "W-FRI"

# --------------------------------------------- 1. correlation vs signal --
CORR_WINDOW = 13                 # rolling window (weeks) for the "current" correlation - one quarter
FLIP_WINDOW = 8                  # weeks before / after a flip in the event-study
MIN_CELL = 20                    # weeks needed before a cell's correlation is reported
FLIP_EXCLUDE_YEAR = 2020         # the around-flip study is repeated without the flips of this year (the COVID crash, when correlation went to 1) and with the median

# ------------------------------------------ 2. correlation vs returns --
HIST_WINDOW = 13                 # trailing-quarter correlation and trailing-quarter return
FWD_WINDOW = 13                  # the following quarter (kept in the tables)
QUINT_FWD = 4                    # weeks: the forward return the quintile chart shows (t+1 … t+4) — Manu, 2026-10-19: "4 weeks forward rather than 13"
N_BUCKETS = 5
GAP_SD = 1.0                     # the SC100×Gold − N100×Gold gap counts as "extreme" beyond this many of its own sd
GAP_FWD = (4, 13)                # weeks ahead for what followed an extreme gap
GOLD_AFTER_BREAK = 26            # weeks after a decorrelation episode starts over which the gold pairs' correlation is followed
RATIO_BACK = 4                   # weeks: the ratio's change over the last 4 weeks (t−3 … t)
RATIO_FWD = 4                    # weeks: the ratio's change over the next 4 weeks (t+1 … t+4)

# ------------------------------------- 3. who breaks the CORRELATION first --
CB_WINDOW = 13                   # weeks: each asset's "current" correlation with the others
CB_NORM = 104                    # weeks: the history its current correlation is judged against (2 years)
CB_Z = -1.0                      # a BREAK = current correlation this many sd below its own 2-year norm
CB_RESET = -0.5                  # a break has ended when the z-score is back above this
CB_LOOKBACK = 8                  # weeks before the group break in which an asset's own break is looked for
CB_MIN_GAP = 8                   # weeks between episodes
CB_FWD = (4, 13)                 # weeks ahead for the after-break returns
CB_SENSITIVITY = {"window": (13, 26), "z": (-1.0, -1.5)}   # re-run the ranking at these settings

# ---------------------------------------- 4. who breaks in price / signal --
DD_THRESHOLD = 0.10              # a sell-off = N100 falls this much from its running high (weekly closes)
DD_CROSS = 0.10                  # the "break" line for each asset: this far below its own running high
DD_PEAK_LOOKBACK = 8             # weeks before N100's peak from which each asset's own peak is looked for
DD_FALL_GRACE = 4                # weeks past the trough within which a SELL flip / -10% cross still counts
DD_MIN_GAP = 8                   # weeks of separation between episodes
RECOVERY_WINDOW = 52             # weeks after the trough in which the first BUY flip is looked for

# --------------------------------------------------------- 5. volatility --
VOL_WINDOW = 13                  # weeks: realised volatility (sd of weekly returns x sqrt 52), recomputed every week
VOL_BUCKETS = 5                  # quintiles of N100's rolling vol = the "vol regimes"
VOL_LEAD_LAGS = 4                # weeks: lead / lag between changes in vol and changes in correlation (the first version of B9 — its windows overlap)
LL_BLOCKS = (4, 13)              # weeks: the block lengths of the NON-overlapping lead/lag study (B9 redone 2026-10-19) — X on t−L+1 … t, Y on t+1 … t+L
LL_MIN_N = 20                    # blocks needed before a lead/lag number is reported
VIX_SURGE = 0.25                 # a VIX "surge" = the VIX level up at least this much (25%) over VIX_SURGE_WEEKS weeks, counted once per run
VIX_SURGE_WEEKS = 4
VIX_FLIP_WINDOW = 8              # weeks: a surge and a SELL flip within this many weeks of each other count as "related"; also the path length around each
REGIME_STRETCH_MIN = 2           # weeks: a run of the pair's own vol in its wildest fifth (Q5) must last this long to be listed as a stretch

# ----------------------------------------------------------- 6. each pair --
PAIR_WINDOW = 52                 # weeks: the deviation window - the ratio z-score is judged against its own trailing year
PAIR_MIN_OBS = 42                # weeks of history needed before a z-score is reported (80% of the window)
PAIR_Z_LINES = (1.0, 2.0)        # the z lines drawn on every pair chart
PAIR_Z_EDGES = (-2.0, -1.0, 1.0, 2.0)   # z buckets: <= -2, -2..-1, -1..1, 1..2, > 2
PAIR_FWD = (4, 13)               # weeks ahead for "what the ratio did next"
PAIR_PATH_WEEKS = 13             # weeks: the average z path after a 2-sd deviation
HAC_LAGS = 4                     # Newey-West lags for overlapping forward returns
COINT_MAX_HALFLIFE = 52          # weeks: a half-life longer than this counts as "does not pull back in practice"

# ------------------------------------------------- 7. the pair-trading rule --
PT_ENTRY = 2.0                   # enter when |z| reaches this (long the cheap leg, short the rich leg)
PT_EXIT = 0.5                    # exit when |z| is back inside this
PT_MAX_HOLD = 26                 # weeks: exit anyway
PT_COST_BPS = 10                 # one-way cost per leg in basis points; a round trip of a two-leg trade costs 4x this
PT_MIN_TRADES = 5                # a pair with fewer trades gets no verdict
PT_SENSITIVITY = {"entry": (1.5, 2.0, 2.5), "exit": (0.0, 0.5), "require_buy": (False, True)}

# -------------------------------------------------- 8. who replaces a SELL --
RP_CORR_WINDOW = 52              # weeks: the correlation that decides whether a candidate is "correlated" with the seller
RP_CORR_MIN = 0.5                # a candidate counts as correlated if that correlation is at least this
RP_Z_DEV = 1.0                   # a candidate counts as deviated if its z-score against the seller is at least this far from 0
RP_FWD = (4, 13)                 # weeks held after the SELL flip
RP_MAX_HOLD = 26                 # weeks: cap for the "hold until the candidate's own SELL" variant
RP_PATH_WEEKS = 13               # weeks before / after the flip in the z-path chart
RP_MIN_EVENTS = 8                # a cell with fewer events gets no verdict

# --------------------------------------------------------------- reporting --
REPORT_HTML = OUT / "report.html"             # the clean report (make_report.py)
REPORT_FULL_HTML = OUT / "report_full.html"   # every table and chart (make_report_full.py)
REPORT_GUIDE_HTML = OUT / "report_guide.html" # the reading guide: questions, chart by chart, column dictionary (make_guide.py)
CHART_DPI = 150

# ---- 9. the reading guide
GUIDE_INCLUDE_TRADES = False     # B16–B20 (deviations & pair cards, the pair-trading rule, who replaces a SELL) — switched off 2026-10-16; the tables and the full report keep them
