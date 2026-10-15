"""
Refresh data/india_vix.csv (date, open, high, low, close) from the Windmill
Postgres, if the table carries India VIX under any of the usual names.

    python fetch_vix.py

If nothing is found, download the history from NSE (Reports > India VIX
historical data) and save it as data/india_vix.csv; the loader accepts NSE's
column layout as is. Credentials come from db.py / .env.
"""
import sys

import pandas as pd
from sqlalchemy import text

import config as C
import db

CANDIDATES = ["INDIAVIX", "INDIA VIX", "INDIA_VIX", "VIX", "NIFVIX", ".NIFVIX", "INDIAVIX.NS"]


def main() -> int:
    eng = db.engine()
    with eng.connect() as conn:
        q = text(f"""SELECT DISTINCT ticker, ric FROM {db.TABLE}
                     WHERE UPPER(ticker) = ANY(:t) OR UPPER(ric) = ANY(:t)
                        OR ticker ILIKE '%VIX%' OR ric ILIKE '%VIX%' LIMIT 20""")
        hits = pd.read_sql(q, conn, params={"t": CANDIDATES})
        if hits.empty:
            print("no VIX-like ticker in the table. Download India VIX history from NSE and save it as data/india_vix.csv")
            return 1
        print("candidates:\n" + hits.to_string(index=False))
        tk = hits.iloc[0]["ticker"]
        q = text(f"SELECT date, close FROM {db.TABLE} WHERE ticker = :tk AND close IS NOT NULL ORDER BY date")
        df = pd.read_sql(q, conn, params={"tk": tk})
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").set_index("date")
    C.DATA.mkdir(exist_ok=True)
    df.to_csv(C.VIX_CSV, float_format="%.4f")
    print(f"wrote {C.VIX_CSV.relative_to(C.ROOT)} from ticker {tk}: {len(df):,} rows, {df.index[0].date()} .. {df.index[-1].date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
