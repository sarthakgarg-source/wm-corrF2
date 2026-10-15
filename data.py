"""
Load the price series into one panel, then build the return matrices.

    panel = load_panel()           # dict label -> DataFrame[open, high, low, close] on its own calendar
    px    = close_panel(panel)     # closes on the NSE calendar (MCX gold forward-filled; VIX level as is)
    rets  = returns(px, "weekly")  # log returns; the VIX column is the log change of its LEVEL
    lvl   = vix_level(px, "weekly")

The index CSV has no fetch script (UNDERSTANDING.md open question 1). The
loader accepts long or wide layouts and prints the mapping it resolved.

VIX: `data/india_vix.csv` with a date column and a close column (OHLC accepted;
NSE's download layout 'Date, Open, High, Low, Close, Prev. Close, Change, %Change'
works). `fetch_vix.py` tries the Postgres table first.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config as C

SYNTHETIC_SEEN: set[str] = set()


def _norm(s: str) -> str:
    return re.sub(r"[\s_\-\.]+", "", str(s)).lower()


def _read_raw(path) -> pd.DataFrame:
    with open(path, encoding="utf-8") as f:
        head = f.readline()
    skip = 0
    if head.startswith("#"):
        skip = 1
        if "SYNTHETIC" in head.upper():
            SYNTHETIC_SEEN.add(Path(path).name)
    return pd.read_csv(path, skiprows=skip)


DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d-%b-%y", "%m/%d/%Y")


def _parse_dates(col: pd.Series) -> pd.Series:
    """Try each explicit format on the whole column; the first that parses every
    non-empty value wins. DD-MM-YYYY (liquidEtfFinal.csv) and DD-Mon-YYYY (NSE
    downloads) are therefore never silently read month-first."""
    raw = col.astype(str).str.strip()
    nonempty = raw[(raw != "") & (raw.str.lower() != "nan")]
    for fmt in DATE_FORMATS:
        try:
            out = pd.to_datetime(nonempty, format=fmt, errors="coerce")
        except (ValueError, TypeError):
            continue
        if out.notna().all():
            return pd.to_datetime(raw, format=fmt, errors="coerce")
    return pd.to_datetime(raw, errors="coerce", dayfirst=True)


def _read_csv_dates(path, date_col: str | None = None) -> pd.DataFrame:
    df = _read_raw(path)
    df.columns = [str(c).strip() for c in df.columns]
    if date_col is None:
        cands = [c for c in df.columns if _norm(c) in ("date", "datetime", "timestamp", "time", "day")]
        date_col = cands[0] if cands else df.columns[0]
    df[date_col] = _parse_dates(df[date_col])
    df = df.dropna(subset=[date_col]).rename(columns={date_col: "date"})
    return df.sort_values("date")


def _ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {_norm(c): c for c in df.columns}
    out = pd.DataFrame(index=pd.DatetimeIndex(df["date"]))
    for k in ("open", "high", "low", "close"):
        src = cols.get(k) or cols.get(f"adj{k}") or cols.get(f"{k}price")
        if src is None and k == "close":
            src = cols.get("adjclose") or cols.get("price") or cols.get("nav") or cols.get("level") or cols.get("vix")
        if src is None and k == "close":                       # a two-column file (date + one value): that value is the close
            num = [c for c in df.columns if c != "date" and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
            src = num[0] if len(num) == 1 else None
        if src is not None:
            out[k] = pd.to_numeric(pd.Series(df[src].values).astype(str).str.replace(",", ""), errors="coerce").values
    if "close" not in out:
        raise ValueError(f"no close column among {list(df.columns)}")
    for k in ("open", "high", "low"):
        if k not in out:
            out[k] = out["close"]
    out = out[~out.index.duplicated(keep="last")].dropna(subset=["close"])
    return out[out["close"] > 0]


def load_indices(path=C.INDEX_CSV, verbose: bool = True) -> dict[str, pd.DataFrame]:
    df = _read_csv_dates(path)
    cols_norm = {_norm(c): c for c in df.columns}
    panel, found = {}, {}
    key_col = cols_norm.get("indexkey")
    sheet_col = next((cols_norm[k] for k in ("sourcesheet", "sheet", "index", "ticker", "symbol", "name") if k in cols_norm), None)
    if key_col is not None:                                       # long layout with an explicit key column
        keys = {_norm(k): k for k in df[key_col].dropna().unique()}
        for key, (label, aliases) in C.INDICES.items():
            hit = keys.get(_norm(key))
            if hit is None:
                hit = next((keys[_norm(a)] for a in aliases if _norm(a) in keys), None)
            if hit is None:
                continue
            panel[label] = _ohlc(df[df[key_col] == hit])
            found[label] = f"{key_col}={hit}"
        if sheet_col is not None:                                 # fall back to the sheet name for anything left
            sheets = {_norm(s): s for s in df[sheet_col].dropna().unique()}
            for key, (label, aliases) in C.INDICES.items():
                if label in panel:
                    continue
                hit = next((sheets[_norm(a)] for a in aliases if _norm(a) in sheets), None)
                if hit is not None:
                    panel[label] = _ohlc(df[df[sheet_col] == hit])
                    found[label] = hit
    elif sheet_col is not None:
        sheets = {_norm(s): s for s in df[sheet_col].dropna().unique()}
        for key, (label, aliases) in C.INDICES.items():
            hit = next((sheets[_norm(a)] for a in aliases if _norm(a) in sheets), None)
            if hit is None:
                continue
            panel[label] = _ohlc(df[df[sheet_col] == hit])
            found[label] = hit
    else:
        for key, (label, aliases) in C.INDICES.items():
            for a in aliases:
                na = _norm(a)
                close_c = cols_norm.get(f"{na}close") or cols_norm.get(na)
                if close_c is None:
                    continue
                sub = pd.DataFrame({"date": df["date"], "close": df[close_c]})
                for k in ("open", "high", "low"):
                    c = cols_norm.get(f"{na}{k}")
                    if c is not None:
                        sub[k] = df[c]
                panel[label] = _ohlc(sub)
                found[label] = close_c
                break
    missing = [lab for _, (lab, _) in C.INDICES.items() if lab not in panel]
    if verbose:
        print(f"index file: {path.name}")
        for lab, src in found.items():
            p = panel[lab]
            print(f"  {lab:9s} <- {src:24s} {p.index[0].date()} .. {p.index[-1].date()}  ({len(p):,} sessions)")
        if missing:
            present = sorted(df[key_col].dropna().unique()) if key_col else (sorted(df[sheet_col].dropna().unique()) if sheet_col else list(df.columns))
            print(f"  NOT FOUND: {missing}  -- names present in the file: {present}\n  add the right one to config.INDICES aliases")
    if not panel:
        raise SystemExit("no index matched - fix config.INDICES aliases")
    return panel


def load_gold(path=C.GOLD_SIGNAL_CSV) -> pd.DataFrame | None:
    return _ohlc(_read_csv_dates(path)) if path.exists() else None


def load_vix(path=C.VIX_CSV) -> pd.DataFrame | None:
    return _ohlc(_read_csv_dates(path)) if path.exists() else None


def load_liquid(path=C.LIQUID_CSV) -> pd.DataFrame | None:
    """The cash leg (liquid ETF NAV, growth). Dates in that file are DD-MM-YYYY; _parse_dates handles it."""
    return _ohlc(_read_csv_dates(path)) if path.exists() else None


def is_synthetic() -> bool:
    return bool(SYNTHETIC_SEEN)


def load_panel(verbose: bool = True) -> dict[str, pd.DataFrame]:
    SYNTHETIC_SEEN.clear()
    panel = load_indices(verbose=verbose)
    g = load_gold()
    if g is not None:
        panel[C.GOLD] = g
    v = load_vix()
    if v is not None:
        panel[C.VIX] = v
    q = load_liquid()
    if q is not None:
        panel[C.CASH] = q
    if verbose:
        notes = {C.VIX: "optional here (timeline chart only)", C.CASH: "optional (cash comparator in the replacement study)", C.GOLD: "add data/MAUc1_daily.csv"}
        for lab in (C.GOLD, C.VIX, C.CASH):
            if lab in panel:
                p = panel[lab]
                print(f"  {lab:9s}    {p.index[0].date()} .. {p.index[-1].date()}  ({len(p):,} sessions)")
            else:
                print(f"  {lab:9s}    file not present - {notes[lab]}")
        if SYNTHETIC_SEEN:
            print("\n" + "!" * 78 + f"\n!!  SYNTHETIC DEMO DATA in {sorted(SYNTHETIC_SEEN)} - results are NOT the desk's\n" + "!" * 78)
    return panel


def nse_calendar(panel: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    idx = None
    for lab in ("N50", "N100", "MC100", "SC100", "Micro250"):
        if lab in panel:
            idx = panel[lab].index if idx is None else idx.union(panel[lab].index)
    return idx[idx >= pd.Timestamp(C.START)]


def close_panel(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Closes on the NSE calendar. GoldMCX forward-filled onto it (MCX-only days
    dropped); the VIX level kept as is. NaN before a series' first session."""
    cal = nse_calendar(panel)
    out = {}
    for lab in C.ASSETS + [C.VIX, C.CASH, "N50"]:
        if lab in panel and len(panel[lab]):
            s = panel[lab]["close"]
            r = s.reindex(cal.union(s.index)).ffill().reindex(cal)
            r[cal < s.index[0]] = np.nan
            r[cal > s.index[-1]] = np.nan          # never carry a series past its last real print
            out[lab] = r
    return pd.DataFrame(out, index=cal)


def resample_close(px: pd.DataFrame, freq: str | None) -> pd.DataFrame:
    return px if freq is None else px.resample(freq).last().dropna(how="all")


def returns(px: pd.DataFrame, freq_name: str) -> pd.DataFrame:
    """Log returns of the assets; the VIX column becomes VIXchg = log change of the level."""
    p = resample_close(px, C.FREQS[freq_name])
    r = np.log(p).diff().iloc[1:]
    if C.VIX in r.columns:
        r = r.rename(columns={C.VIX: C.VIXCHG})
    cols = [c for c in C.UNIVERSE if c in r.columns] + [c for c in r.columns if c not in C.UNIVERSE]
    return r[cols]


def vix_level(px: pd.DataFrame, freq_name: str) -> pd.Series:
    p = resample_close(px, C.FREQS[freq_name])
    return p[C.VIX].rename(C.VIX) if C.VIX in p.columns else pd.Series(dtype=float, name=C.VIX)


def coverage(r: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in r.columns:
        s = r[c].dropna()
        rows.append({"series": c, "first": s.index[0].date() if len(s) else None,
                     "last": s.index[-1].date() if len(s) else None, "obs": len(s)})
    return pd.DataFrame(rows).set_index("series")


if __name__ == "__main__":
    panel = load_panel()
    px = close_panel(panel)
    for f in C.FREQS:
        r = returns(px, f)
        print(f"\n{f}: {r.shape[0]} bars x {r.shape[1]} series, {r.index[0].date()} .. {r.index[-1].date()}")
        print(coverage(r).to_string())
    sys.exit(0)
