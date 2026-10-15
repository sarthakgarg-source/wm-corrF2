"""
Connection check. Confirms the credentials work, the source table exists, and
every ticker this repo can fetch is present, with its date span. Also lists any
index-like tickers, so the open question "is there an index OHLC source in
Postgres?" can be settled by inspection.

    python db_probe.py
"""
import sys

import pandas as pd
from sqlalchemy import text

import db

TICKERS = ["GOLDBEES", "LIQUIDBEES", "LIQUIDCASE", "LIQUIDADD", "LIQUIDSHRI", "LIQUIDPLUS", "CASHIETF"]


def main() -> int:
    eng = db.engine()
    with eng.connect() as conn:
        n = conn.execute(text(f"SELECT COUNT(*) FROM {db.TABLE}")).scalar()
        print(f"connected. {db.TABLE}: {n:,} rows")
        q = text(f"""SELECT ticker, MIN(date) AS first, MAX(date) AS last, COUNT(*) AS rows,
                            COUNT(DISTINCT close) AS distinct_closes
                     FROM {db.TABLE} WHERE ticker = ANY(:t) AND close IS NOT NULL GROUP BY ticker ORDER BY ticker""")
        print(pd.read_sql(q, conn, params={"t": TICKERS}).to_string(index=False))
        n_mau = conn.execute(text(f"SELECT COUNT(*) FROM {db.TABLE} WHERE ric = 'MAUc1' OR ticker = 'MAUc1'")).scalar()
        print(f"\nMAUc1 (MCX gold futures, the gold signal) rows: {n_mau}  "
              f"({'a fetch script could be written' if n_mau else 'not here - data/MAUc1_daily.csv must be refreshed by hand'})")
        print("\nindex-like tickers (is there an index OHLC source here? UNDERSTANDING.md open question 1):")
        q = text(f"""SELECT DISTINCT ticker FROM {db.TABLE}
                     WHERE ticker ILIKE '%NIFTY%' OR ticker ILIKE '%SMALLCAP%' OR ticker ILIKE '%MICROCAP%'
                        OR ticker ILIKE '%MIDCAP%' OR ric ILIKE '.NIF%' OR ric ILIKE '.NSE%' ORDER BY 1 LIMIT 80""")
        print(", ".join(pd.read_sql(q, conn)["ticker"].tolist()) or "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
