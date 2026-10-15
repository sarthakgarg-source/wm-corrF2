# Links and access
No secret is written in this file. `.env` (git-ignored) carries the Windmill Postgres login for `db_probe.py` / `fetch_*.py`; nothing in this study needs the database. Host `wm-prod-internal-postgres.cae4omyvwslq.ap-south-1.rds.amazonaws.com`, port 5432, db `windmill`, `sslmode=require`, user and password in `.env`. Delete `.env` before forwarding a zip.

Sibling studies: `wm-CORR` (G, structure and cointegration), `wm-VIXcorr` (H, VIX, forward returns, divergence, signal disagreement), `wm-IDXrot` / `wm-EQrot` / `wm-EQtilt` (D/E/F, the rotation strategies), `wm-STstrat` (A/B/C, the supertrend engine).
