"""globalmacro.pipeline.download_public -- fetch the public, license-free raw inputs.

Three files the pipeline needs that are NOT behind WRDS and that researchers used
to place by hand:
  * Fama-French daily factors  -> data/misc/F-F_Research_Data_Factors_daily.csv   (rf, via load_rf)
  * FRED interbank/eurodollar   -> data/datastream/economics/ded3_wrds.csv        (ded3, via rates.py)
  * OECD 3-month interbank      -> data/datastream/economics/oecd.csv             (via rates.py)

Each is a plain HTTPS GET through the system `curl` (HTTP/2). FRED and OECD hang or
reject Python's urllib regardless of User-Agent, so curl -- proven reliable on the
login node -- is the transport. WRDS pulls stay in `download.py`; this module needs
no credentials. Fetchers skip an already-present destination unless `force=True`.
"""
import argparse
import io
import os
import re
import subprocess
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import polars as pl

from globalmacro.utils.paths import DATA_ROOT, ECONOMICS_PATH

FAMA_FRENCH_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip"
)
# 6 daily FRED series -> columns date,ded1,ded3,ded6,effr,dff,sofr (only ded3 is consumed;
# the rest keep ded3_wrds.csv drop-in shaped). Keyless CSV endpoint, no API key needed.
FRED_SERIES = ["DED1", "DED3", "DED6", "EFFR", "DFF", "SOFR"]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
# OECD SDMX: DF_KEI key = REF_AREA(all) . FREQ(M) . MEASURE(IR3TIB=3-month interbank) . ...(all).
# csvfilewithlabels carries the `Reference area` label rates.py pivots on.
OECD_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.STES,DSD_KEI@DF_KEI,4.0/.M.IR3TIB.....?format=csvfilewithlabels"
)
OECD_ACCEPT = "application/vnd.sdmx.data+csv"


def _curl(url: str, dest, accept: str | None = None) -> None:
    """GET `url` into `dest` atomically via the system curl (HTTP/2; -f fails on HTTP error)."""
    dest = Path(dest)
    tmp = dest.with_name(dest.name + ".part")
    cmd = ["curl", "-sSfL", "--remove-on-error", "--max-time", "180", "-o", str(tmp)]
    if accept:
        cmd += ["-H", f"Accept: {accept}"]
    cmd.append(url)
    subprocess.run(cmd, check=True)
    os.replace(tmp, dest)


def _trim_fama_french(text: str) -> str:
    """Drop the trailing blank line + 'Copyright ...' footer, keeping preamble+header+data.

    load_rf reads with skip_rows=4, so the 3 preamble lines + blank + `,Mkt-RF,SMB,HML,RF`
    header must stay; only the non-data tail is removed (else the date parse chokes on it).
    """
    lines = text.splitlines()
    last = max(i for i, line in enumerate(lines) if re.match(r"^\d{8},", line))
    return "\n".join(lines[: last + 1]) + "\n"


def _merge_fred_series(series_text: dict[str, str]) -> pl.DataFrame:
    """Outer-join the 6 FRED series on date into date,ded1,ded3,ded6,effr,dff,sofr.

    Each series is `observation_date,<ID>`; a missing observation is the literal `.`
    which cast(Float64, strict=False) turns into null. Column names are the lowercased IDs.
    NOTE: strict=False nulls *any* unparseable token, not just `.` -- in FRED's fredgraph
    CSV only `.` occurs, so this is fine, but a malformed value would be absorbed as null
    (then dropped by rates.py's drop_nulls) rather than failing loudly.
    """
    out: pl.DataFrame | None = None
    for sid in FRED_SERIES:
        col = sid.lower()
        df = (
            pl.read_csv(io.StringIO(series_text[sid]), schema_overrides={"observation_date": pl.Date})
            .rename({"observation_date": "date", sid: col})
            .with_columns(pl.col(col).cast(pl.Float64, strict=False))
        )
        out = df if out is None else out.join(df, on="date", how="full", coalesce=True)
    assert out is not None
    return out.sort("date")


def _skip(dest: Path, force: bool) -> bool:
    if dest.exists() and not force:
        print(f"[public] {dest} exists, skipping (use --force to refresh)")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    return False


def fetch_fama_french(dest=None, force: bool = False) -> Path:
    dest = Path(dest) if dest else DATA_ROOT / "misc" / "F-F_Research_Data_Factors_daily.csv"
    if _skip(dest, force):
        return dest
    with tempfile.TemporaryDirectory() as td:
        zip_path = Path(td) / "ff.zip"
        _curl(FAMA_FRENCH_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            text = z.read(z.namelist()[0]).decode("latin-1")
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_text(_trim_fama_french(text))
    os.replace(tmp, dest)  # atomic, mirroring download.py's write discipline
    print(f"[public] wrote {dest}")
    return dest


def fetch_fred_rates(dest=None, force: bool = False) -> Path:
    dest = Path(dest) if dest else ECONOMICS_PATH / "ded3_wrds.csv"
    if _skip(dest, force):
        return dest
    with tempfile.TemporaryDirectory() as td:
        series_text = {}
        for sid in FRED_SERIES:
            p = Path(td) / f"{sid}.csv"
            _curl(FRED_URL.format(series=sid), p)
            series_text[sid] = p.read_text()
    tmp = dest.with_name(dest.name + ".part")
    _merge_fred_series(series_text).write_csv(tmp)
    os.replace(tmp, dest)  # atomic, mirroring download.py's write discipline
    print(f"[public] wrote {dest}")
    return dest


def fetch_oecd_stir(dest=None, force: bool = False) -> Path:
    dest = Path(dest) if dest else ECONOMICS_PATH / "oecd.csv"
    if _skip(dest, force):
        return dest
    _curl(OECD_URL, dest, accept=OECD_ACCEPT)
    print(f"[public] wrote {dest}")
    return dest


FETCHERS: dict[str, Callable[..., Path]] = {
    "fama_french": fetch_fama_french,
    "fred": fetch_fred_rates,
    "oecd": fetch_oecd_stir,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="globalmacro download-public",
        description="Fetch the public (no-credentials) raw inputs: Fama-French rf, FRED ded3, OECD stir.",
    )
    ap.add_argument(
        "--only", choices=sorted(FETCHERS), action="append",
        help="fetch only these sources (repeatable); default = all three",
    )
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    ns = ap.parse_args(argv)
    for name in ns.only or list(FETCHERS):
        FETCHERS[name](force=ns.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
