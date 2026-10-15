"""
Refresh data/sleeve_prices.csv (date, gold, liquid) from the Windmill Postgres.

    python fetch_sleeve.py

gold   = GOLDBEES close (the traded gold leg; no dividend, so close ~ total return)
liquid = LIQUIDBEES close (dividend-option fund pinned near 1000; ~0% return.
         Kept for parity with the rotation repos; the cash leg of this study is
         data/liquidEtfFinal.csv when present - see data.load_liquid).

Ported from wm-IDXrot/fetch_sleeve.py. Credentials come from db.py / .env.
"""
import sys

import pandas as pd
from sqlalchemy import text

import config as C
import db


def main() -> int:
    eng = db.engine()
    q = text(f"""SELECT date, ticker, close FROM {db.TABLE}
                 WHERE ticker IN ('GOLDBEES', 'LIQUIDBEES') AND close IS NOT NULL ORDER BY date""")
    with eng.connect() as conn:
        df = pd.read_sql(q, conn)
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    wide = wide.rename(columns={"GOLDBEES": "gold", "LIQUIDBEES": "liquid"})[["gold", "liquid"]]
    C.DATA.mkdir(exist_ok=True)
    wide.to_csv(C.SLEEVE_CSV, float_format="%.4f")
    print(f"wrote {C.SLEEVE_CSV.relative_to(C.ROOT)}: {len(wide):,} rows, {wide.index[0].date()} .. {wide.index[-1].date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
