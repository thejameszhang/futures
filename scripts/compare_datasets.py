#!/usr/bin/env python3
"""Grade fresh-vs-current datasets over the OVERLAP window (dates present in the
current dataset). For each dataset file, per-instrument Pearson correlation of
fresh vs current returns; PASS iff median >= 0.99. Fresh data past the current
end date is expected and excluded from grading."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
FRESH = Path.home() / "futures_verify" / "datasets"
CURRENT = REPO / "datasets"
THRESHOLD = 0.99

DATASETS = [
    "tier1/sync/sync_daily.csv", "tier1/async/async_daily.csv",
    "tier1/async/async_monthly.csv",
    "tier2/sync/sync_daily.csv", "tier2/async/async_daily.csv",
    "tier2/async/async_monthly.csv",
]


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    out = df.set_index("date").sort_index().apply(pd.to_numeric, errors="coerce")
    # .apply() is typed DataFrame | Series (its return shape isn't statically known);
    # applying pd.to_numeric column-by-column over a DataFrame always yields a
    # DataFrame back, so assert it rather than let the ambiguity leak into every caller.
    assert isinstance(out, pd.DataFrame)
    return out


def grade_one(rel: str):
    cur = _load(CURRENT / rel)
    fresh = _load(FRESH / rel)
    overlap_end = cur.index.max()                        # e.g. 2025-12-31
    fresh = fresh[fresh.index <= overlap_end]            # grade only the overlap
    syms = [c for c in cur.columns if c in fresh.columns]
    corrs = []
    for s in syms:
        # df[label] is typed Series | DataFrame -- pandas returns a DataFrame instead of
        # a Series if the source CSV has a duplicate column name for `s`. That would
        # silently corrupt the grading below, so assert the expected shape rather than
        # let it through.
        cur_s, fresh_s = cur[s], fresh[s]
        assert isinstance(cur_s, pd.Series) and isinstance(fresh_s, pd.Series), (
            f"duplicate column {s!r} in source data"
        )
        j = pd.DataFrame({"cur": cur_s, "fresh": fresh_s}).dropna()
        if len(j) < 24:
            continue
        j_cur, j_fresh = j["cur"], j["fresh"]
        assert isinstance(j_cur, pd.Series) and isinstance(j_fresh, pd.Series)
        r = j_cur.corr(j_fresh)
        if pd.notna(r):
            corrs.append((s, r))
    med = float(np.median([r for _, r in corrs])) if corrs else float("nan")
    worst = sorted(corrs, key=lambda x: x[1])[:5]
    return med, len(corrs), worst


def main() -> int:
    fail = 0
    for rel in DATASETS:
        if not (FRESH / rel).exists() or not (CURRENT / rel).exists():
            print(f"SKIP {rel} (missing fresh or current)")
            continue
        med, n, worst = grade_one(rel)
        status = "PASS" if med >= THRESHOLD else "FAIL"
        if med < THRESHOLD:
            fail += 1
        print(f"[{status}] {rel:34s} median={med:.4f} n={n}  worst={[(s, round(r,3)) for s,r in worst]}")
    print(f"\n{'ALL PASS' if fail == 0 else str(fail)+' dataset(s) below '+str(THRESHOLD)}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
