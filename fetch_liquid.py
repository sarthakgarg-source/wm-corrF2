"""
Refresh data/liquid_prices.csv (LIQUIDCASE + 4 peer liquid ETFs, growth option)
from the Windmill Postgres. Used only to cross-check liquidEtfFinal.csv.

    python fetch_liquid.py
"""
import sys

import pandas as pd
from sqlalchemy import text

import config as C
import db

TICKERS = ["LIQUIDCASE", "LIQUIDADD", "LIQUIDSHRI", "LIQUIDPLUS", "CASHIETF"]


def main() -> int:
    eng = db.engine()
    q = text(f"""SELECT date, ticker, close FROM {db.TABLE}
                 WHERE ticker = ANY(:t) AND close IS NOT NULL ORDER BY date""")
    with eng.connect() as conn:
        df = pd.read_sql(q, conn, params={"t": TICKERS})
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    out = C.DATA / "liquid_prices.csv"
    wide.to_csv(out, float_format="%.4f")
    print(f"wrote {out.relative_to(C.ROOT)}: {len(wide):,} rows, {wide.index[0].date()} .. {wide.index[-1].date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
