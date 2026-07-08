#!/usr/bin/env python3
"""Emit + validate the Tier-A path list for the Dropbox bundle. Does NOT copy
the ~1.5 TB; prints the tar command. Tier-A only, so interims (B), moved files
(C) and the quarantined confidential series (outside data/) are excluded by
construction. tar recurses the listed directories (trades/quotes)."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_data import REPO, resolve

OUT = REPO / "scripts" / "dropbox_include.txt"
FORBIDDEN = ["passive_returns", ".parquet", "spot_equity_returns",
             "synthetic_fx_returns", "3month_libor_rates", "tousd_panel",
             "instrumentlists", "old-data/", "reference/"]
MUST = ["data/datastream/equities/ds2indexdata.csv",
        "data/datastream/futures/datastream_continuous_series.csv",
        "data/misc/VIX_History.csv",
        "data/tickhistory/trades", "data/tickhistory/quotes"]


def main() -> int:
    paths = sorted({unit for unit, rule, err in resolve()
                    if not err and rule and rule["tier"] == "A"})
    OUT.write_text("\n".join(paths) + "\n")

    fail = []
    for p in paths:
        for bad in FORBIDDEN:
            if bad in p:
                fail.append(f"forbidden ({bad}): {p}")
    for m in MUST:
        if m not in paths:
            fail.append(f"missing expected tier-A path: {m}")
    if fail:
        for x in fail:
            print("FAIL:", x)
        print("VALIDATION FAILED")
        return 1

    du = subprocess.run(["du", "-sch", *[str(REPO / p) for p in paths]],
                        capture_output=True, text=True)
    total = du.stdout.strip().splitlines()[-1].split("\t")[0] if du.stdout else "?"
    print(f"include list: {OUT}  ({len(paths)} paths, {total} total)")
    print("\nTo build the bundle (run from repo root):")
    print("  cp scripts/DROPBOX_README.md README_DATA_BUNDLE.md")
    print("  tar --zstd -cf globalmacro_data_bundle.tar.zst README_DATA_BUNDLE.md -T scripts/dropbox_include.txt")
    print("  # then scp globalmacro_data_bundle.tar.zst to Dropbox")
    return 0


if __name__ == "__main__":
    sys.exit(main())
