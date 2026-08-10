#!/usr/bin/env python3
"""Single source of truth for data/ tier assignment + a manifest audit.

resolve() applies literal-beats-glob precedence and is imported by
reorg_data.py, check_data_integrity.py and package_dropbox.py so every tool
agrees which rule owns each live file. Never greps repo root '.' (would scan
the 2.5 TB data/ tree); greps code dirs only, with fixed strings.
"""
from __future__ import annotations

import csv
import fnmatch
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
MANIFEST = REPO / "scripts" / "data_manifest.tsv"
SRC = REPO / "src" / "globalmacro"
ANALYSIS = REPO / "analysis"

# Dirs whose DIRECT children are each their own decision-unit (mixed tiers).
PER_CHILD = {
    "data/datastream/futures", "data/datastream/equities",
    "data/datastream/fx", "data/datastream/economics",
    "data/comp", "data/jkp", "data/misc", "data/tickhistory",
}
# Dirs we recurse THROUGH to reach PER_CHILD dirs / leaf units.
CONTAINER = {"data", "data/datastream"}

# Mirror of download.py's PULL_SPECS (consumed-only tables). Keep in sync: the
# broad prefix pulls were tightened, so the ex-download extras (fx ds2mktval/
# primqt*/scdqt*, eco*/dsf* extras, ds2equityindex) are now orphans, not kept.
DOWNLOAD = {
    "equities": (None, {"ds2indexdata"}),
    "fx": (None, {"ds2fxcode", "ds2fxrate"}),
    "futures": (None, {"dsfutclass", "dsfutcontr", "dsfuttrdcycle",
                       "dsfutcontrinfo", "dsfutcontrval"}),
    "economics": (None, {"ecodata"}),
    "comp": (None, {"exrt_dly"}),
}


def load_rules() -> list[dict]:
    with open(MANIFEST, newline="") as fh:
        return [{k: v.strip() for k, v in row.items()}
                for row in csv.DictReader(fh, delimiter="\t")]


def decision_units() -> list[str]:
    units: list[str] = []

    def walk(d: Path):
        rel = d.relative_to(REPO).as_posix()
        if rel in PER_CHILD:
            for c in sorted(d.iterdir()):
                units.append(c.relative_to(REPO).as_posix())
        elif rel in CONTAINER:
            for c in sorted(d.iterdir()):
                walk(c) if c.is_dir() else units.append(c.relative_to(REPO).as_posix())
        else:
            units.append(rel)

    if DATA.exists():
        walk(DATA)
    return sorted(set(units))


def resolve() -> list[tuple]:
    """[(unit, rule|None, error|None)] with literal-beats-glob precedence."""
    rules = load_rules()
    out = []
    for unit in decision_units():
        exact = [r for r in rules if r["pattern"] == unit]
        globs = [r for r in rules if r["pattern"] != unit and fnmatch.fnmatch(unit, r["pattern"])]
        if len(exact) == 1:
            out.append((unit, exact[0], None))
        elif len(exact) > 1:
            out.append((unit, None, "AMBIGUOUS-LITERAL: " + ", ".join(r["pattern"] for r in exact)))
        elif len(globs) == 1:
            out.append((unit, globs[0], None))
        elif len(globs) > 1:
            out.append((unit, None, "AMBIGUOUS-GLOB: " + ", ".join(r["pattern"] for r in globs)))
        else:
            out.append((unit, None, "UNCLASSIFIED"))
    return out


def _grep(root: Path, *needles: str) -> bool:
    if not root.exists():
        return False
    args = ["grep", "-rIlqF"]
    for n in needles:
        args += ["-e", n]
    return subprocess.run(args + [str(root)]).returncode == 0


def _stem(pattern: str) -> str:
    """Longest literal prefix of the pattern's basename (producer may build the
    rest with an f-string, e.g. datastream_futures_{...}.parquet)."""
    base = pattern.split("/")[-1]
    pre = base.split("*")[0].rstrip("_.-")
    return pre if len(pre) >= 3 else base


def main() -> int:
    errors, warnings = [], []
    rules = load_rules()

    for unit, rule, err in resolve():
        if err:
            if unit.endswith("_passive_returns.csv") and err == "UNCLASSIFIED":
                warnings.append(f"quarantine-pending (classification not yet run): {unit}")
            else:
                errors.append(f"{err}: {unit}")
            continue
        tier, cons = rule["tier"], rule["consumer"]
        base = unit.split("/")[-1]
        # HARD read-safety: a moved-out file must not be read by the producer.
        if tier in ("C1", "C2") and not cons.startswith("producer-output"):
            # Quoted-token match (the producer references inputs as quoted string
            # literals). A bare substring match would false-positive when a dead
            # file's name is a substring of a live input's, e.g. daily_ind_gics.csv
            # inside "updated_daily_ind_gics.csv".
            if _grep(SRC, f'"{base}"', f"'{base}'"):
                errors.append(f"tier-{tier} unit READ BY PRODUCER (would break pipeline): {unit}")
        # Soft evidence checks (warn only — grep is heuristic).
        if tier == "A" and cons.split(":")[0] in ("producer", "validation") and not _grep(SRC, _stem(rule["pattern"])):
            warnings.append(f"tier-A '{unit}' claims {cons} but producer stem not found")
        if tier == "B" and not _grep(SRC, _stem(rule["pattern"])):
            warnings.append(f"tier-B '{unit}' claims {cons} but producer stem not found")
        if tier == "C1" and cons.startswith("analysis") and not _grep(ANALYSIS, base):
            warnings.append(f"C1 '{unit}' consumer {cons} unverified (analysis grep miss) -> maybe C2")

    # HARD download-spec check (no grep).
    for r in rules:
        if r["consumer"].startswith("download:"):
            db = r["consumer"].split(":", 1)[1]
            prefix, tables = DOWNLOAD[db]
            fn = r["pattern"].split("/")[-1].replace(".csv", "")       # 'ds2*' or 'ds2indexdata'
            ok = (prefix and fn.startswith(prefix)) or \
                 (tables and any(fnmatch.fnmatch(t, fn) for t in tables))
            if not ok:
                errors.append(f"DOWNLOAD SPEC MISMATCH: {r['pattern']} not produced by download --database {db}")

    for w in warnings:
        print("WARN:", w)
    for e in errors:
        print("FAIL:", e)
    print(f"\n{len(decision_units())} live units, {len(rules)} rules, {len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
