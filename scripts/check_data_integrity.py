#!/usr/bin/env python3
"""Post-reorg gate. (1) No tier-C1/C2 unit lingers in data/. (2) Every tier-A/B
rule still has a live file (nothing over-moved). (3) No producer source
references a moved-out dir. Uses resolve() for precedence-correct assignment."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_data import REPO, SRC, load_rules, resolve


def main() -> int:
    errors = []
    res = resolve()

    for unit, rule, err in res:
        if err:
            errors.append(f"UNRESOLVED post-reorg: {unit} ({err})")
        elif rule and rule["tier"] in ("C1", "C2"):
            errors.append(f"tier-{rule['tier']} STILL IN data/: {unit}")

    live_patterns = {rule["pattern"] for _, rule, _ in res if rule}
    for rule in load_rules():
        if rule["tier"] in ("A", "B") and rule["pattern"] not in live_patterns:
            errors.append(f"tier-{rule['tier']} rule has NO live file (over-moved?): {rule['pattern']}")

    for rule in load_rules():
        # producer-output dirs (debug/) are WRITTEN by the producer and recreated
        # via os.makedirs(exist_ok=True) on the next run, so a reference to them
        # is expected and safe. Every OTHER moved-out dir must be unreferenced by
        # the producer (a reference there would mean a needed input was moved).
        if (rule["tier"] in ("C1", "C2") and rule["kind"] == "dir"
                and not rule["consumer"].startswith("producer-output")):
            name = rule["pattern"].split("/")[-1]
            if subprocess.run(["grep", "-rIlqF", "-e", f'"{name}"', "-e", f"'{name}'", str(SRC)]).returncode == 0:
                errors.append(f"producer references moved-out dir '{name}' ({rule['pattern']})")

    for e in errors:
        print("FAIL:", e)
    print(f"{len(errors)} integrity errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
