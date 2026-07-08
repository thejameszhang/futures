#!/usr/bin/env python3
"""Move tier-C1/C2 units out of data/ per the resolved manifest. Per-file,
structure-preserving, logged, reversible. Default dry-run; --apply to execute.
Precedence is honored via resolve(): a tier-A file that also matches a C2 glob
is assigned to its literal rule and never appears here."""
from __future__ import annotations
import shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_data import REPO, resolve


def _shq(p: Path) -> str:
    return "'" + str(p).replace("'", "'\\''") + "'"


def _human(n: float) -> str:
    for u in "BKMGT":
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.0f}P"


def move_pairs() -> list[tuple[Path, Path]]:
    pairs = []
    for unit, rule, err in resolve():
        if err or rule is None or rule["tier"] not in ("C1", "C2") or rule["destination"] == "-":
            continue
        src, kind, dest = REPO / unit, rule["kind"], rule["destination"]
        if kind == "dir":
            for f in sorted(src.rglob("*")):
                if f.is_file():
                    pairs.append((f, REPO / dest / f.relative_to(src)))
        elif kind == "glob":
            pairs.append((src, REPO / dest / src.name))
        else:  # file
            pairs.append((src, REPO / dest))
    return pairs


def main() -> int:
    apply = "--apply" in sys.argv[1:]
    pairs = [(s, d) for (s, d) in move_pairs() if s.exists()]

    # Pre-flight: refuse to clobber anything.
    clobber = [d for _, d in pairs if d.exists()]
    if clobber:
        for d in clobber:
            print(f"REFUSE (dest exists): {d.relative_to(REPO)}")
        return 1

    total = sum(s.stat().st_size for s, _ in pairs)
    if not apply:
        for s, d in pairs:
            print(f"WOULD  {s.relative_to(REPO)} -> {d.relative_to(REPO)}")
        print(f"\n{len(pairs)} file-moves, {_human(total)} total  [DRY-RUN]")
        return 0

    (REPO / "old-data").mkdir(parents=True, exist_ok=True)
    rev = REPO / "old-data" / "REVERSAL.sh"
    # Append so successive reorgs preserve the full reversal (header written once).
    rlines = [] if rev.exists() else ["#!/usr/bin/env bash", "set -euo pipefail",
              "# Undo scripts/reorg_data.py --apply. Restores every moved file."]
    src_dirs = set()
    for s, d in pairs:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
        rlines.append(f"mkdir -p {_shq(s.parent)}; mv {_shq(d)} {_shq(s)}")
        src_dirs.add(s.parent)
        print(f"MOVED  {s.relative_to(REPO)} -> {d.relative_to(REPO)}")

    # Prune emptied source directories (deepest first); reversal recreates them.
    for d in sorted({p for sd in src_dirs for p in [sd, *sd.parents]},
                    key=lambda x: len(x.parts), reverse=True):
        try:
            if REPO in d.parents and d.is_dir() and not any(d.iterdir()) and d != REPO / "data":
                d.rmdir()
        except OSError:
            pass

    with rev.open("a") as fh:
        fh.write("\n".join(rlines) + "\n")
    rev.chmod(0o755)
    print(f"\n{len(pairs)} file-moves, {_human(total)} total  [APPLY]\nreversal: {rev}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
