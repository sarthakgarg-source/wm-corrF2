# SETUP
    python -m venv .venv && .venv\Scripts\activate
    pip install -r requirements.txt     # includes plotly (the interactive charts)
If `run_all.py` stops at `import statsmodels` with "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x" / "numpy.core.multiarray failed to import", the SciPy on that interpreter predates NumPy 2: `python -m pip install -U numpy scipy statsmodels pandas matplotlib scikit-learn` (or pin `numpy<2`), then re-run.
Copy `index_ohlcv_all.csv` and `MAUc1_daily.csv` (and optionally `india_vix.csv` for the timeline chart and `liquidEtfFinal.csv` for the cash comparator in §9) into `data/` from a sibling repo, then:
    python data.py        # every series resolves, no SYNTHETIC banner
    python run_all.py     # tables + charts
    python make_report.py          # clean report; python make_report_full.py for everything expanded; python make_guide.py for the reading guide
    python checks.py      # "0 failed"
Change the correlation window (`CORR_WINDOW`, `HIST_WINDOW`), the sell-off threshold (`DD_THRESHOLD`, `DD_CROSS`), the deviation window (`PAIR_WINDOW`), the pair-trade lines (`PT_ENTRY`, `PT_EXIT`, `PT_MAX_HOLD`, `PT_COST_BPS`) or the replacement thresholds (`RP_CORR_MIN`, `RP_Z_DEV`, `RP_FWD`, `RP_MAX_HOLD`) in `config.py` only; re-run `checks.py` after any code edit. `run_all.py` takes ~50 s (ten pair cards and five event grids are the slow part); `--no-charts` skips them.
