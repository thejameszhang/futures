import argparse
import csv
import io
import os
import sys
from contextlib import closing
from typing import Any, cast

import wrds
from tqdm import tqdm

from globalmacro.utils.paths import COMPUSTAT_PATH, DATASTREAM_PATH, FUTURES_PATH

# database -> (WRDS library, explicit table list). Tightened to ONLY the tables
# the pipeline actually consumes (see plan). No prefix pulls: the old dsf*/eco*
# prefix pulls dragged in ~180 GB of unused tables.
PULL_SPECS: dict[str, tuple[str, list[str]]] = {
    "equities":  ("tr_ds_equities", ["ds2indexdata"]),
    "fx":        ("tr_ds_equities", ["ds2fxcode", "ds2fxrate"]),
    "futures":   ("tr_ds_fut",      ["dsfutclass", "dsfutcontr", "dsfuttrdcycle",
                                     "dsfutcontrinfo", "dsfutcontrval"]),
    "economics": ("tr_ds_econ",     ["ecodata"]),
    "comp":      ("comp",           ["exrt_dly"]),
}

# datastream_continuous is a JOIN, not a single-table COPY, so it can't be a PULL_SPECS
# entry. It reproduces ONLY the 6-column slice validation/datastream_comparison.py consumes
# (RollMethodCode==0, PositionFwdCode==0, Settlement not null). ClsCode is cast ::integer so
# the CSV emits a bare int (169, not 169.00) -- else scan_csv infers Float64 and the
# comparison's ClsCode.is_in(<ints>) silently matches nothing.
DATASTREAM_CONTINUOUS_SQL = (
    'SELECT i.calcseriesname AS "CalcSeriesName", i.clscode::integer AS "ClsCode", '
    'v.date_ AS "Date_", v.settlement AS "Settlement", '
    'i.rollmethodcode AS "RollMethodCode", i.positionfwdcode AS "PositionFwdCode" '
    "FROM tr_ds_fut.dsfutcalcserval v "
    "JOIN tr_ds_fut.dsfutcalcserinfo i ON v.calcseriescode = i.calcseriescode "
    "WHERE i.rollmethodcode = 0 AND i.positionfwdcode = 0 AND v.settlement IS NOT NULL"
)


def quote_ident(ident: str) -> str:
    return '"' + ident.replace('"', '""') + '"'

def has_rows(cursor, fq_table: str) -> bool:
    cursor.execute(f"SELECT 1 FROM {fq_table} LIMIT 1;")
    return cursor.fetchone() is not None

def get_columns(cursor, fq_table: str) -> list[str]:
    cursor.execute(f"SELECT * FROM {fq_table} LIMIT 0;")
    return [desc[0] for desc in cursor.description]

def write_csv_header(file_obj, columns: list[str]) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    file_obj.write(buffer.getvalue().encode("utf-8"))

def _connect():
    """Open a WRDS connection + its raw DBAPI connection (copy_expert-capable)."""
    from globalmacro.wrds_credentials import get_wrds_credentials
    creds = get_wrds_credentials()
    if creds.password is not None:
        os.environ["PGPASSWORD"] = creds.password
    conn = wrds.Connection(wrds_username=creds.username)
    dbapi_conn = cast(Any, conn.engine).raw_connection()
    return conn, dbapi_conn

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Connect to WRDS
    conn, dbapi_conn = _connect()

    # List all tables in the library
    tables = cast(list[str], conn.list_tables(library=LIBRARY))
    if TABLES is not None:
        tables = list(filter(lambda x: x in TABLES, tables))
    elif TABLE_PREFIX is not None:
        tables = list(filter(lambda x: x.startswith(cast(str, TABLE_PREFIX)), tables))
    else:
        raise ValueError("Either TABLES or TABLE_PREFIX must be provided")

    if not tables:
        print(f"No tables found in library '{LIBRARY}'.")
        sys.exit(1)

    print(f"Found {len(tables)} tables in '{LIBRARY}'. Downloading {tables} to CSV...")

    try:
        with closing(dbapi_conn.cursor()) as cursor:
            for t in tqdm(tables):
                fq = f"{quote_ident(LIBRARY)}.{quote_ident(t)}"
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                out_csv = os.path.join(OUTPUT_DIR, t + ".csv")
                tmp_csv = out_csv + ".tmp"

                try:
                    print(f"Querying {fq} ...")
                    if not has_rows(cursor, fq):
                        print(f"{fq} is empty; skipping.")
                        continue
                    columns = get_columns(cursor, fq)

                    with open(tmp_csv, "wb") as handle:
                        # Preserve pandas' default index column (blank header) for compatibility.
                        write_csv_header(handle, [""] + columns)
                        copy_sql = (
                            f"COPY (SELECT row_number() OVER () - 1 AS idx, t.* "
                            f"FROM {fq} t) TO STDOUT WITH (FORMAT CSV)"
                        )
                        cursor.copy_expert(copy_sql, handle)

                    os.replace(tmp_csv, out_csv)
                    print(f"Wrote {out_csv}")
                except Exception as e:
                    print(f"Failed {fq}: {e}")
                    dbapi_conn.rollback()
                    if os.path.exists(tmp_csv):
                        os.remove(tmp_csv)
                    continue
    finally:
        dbapi_conn.close()
        conn.close()

    print("Done.")

def pull_datastream_continuous(output_dir) -> str:
    """Pull the 6-column datastream_continuous validation slice via a JOIN + copy_expert."""
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, "datastream_continuous_series.csv")
    tmp_csv = out_csv + ".tmp"
    conn, dbapi_conn = _connect()
    try:
        with closing(dbapi_conn.cursor()) as cursor, open(tmp_csv, "wb") as handle:
            # No pandas index column: the on-disk file's header starts at the data cols.
            cursor.copy_expert(
                f"COPY ({DATASTREAM_CONTINUOUS_SQL}) TO STDOUT WITH (FORMAT CSV, HEADER)",
                handle,
            )
        os.replace(tmp_csv, out_csv)
        print(f"Wrote {out_csv}")
    except Exception:
        dbapi_conn.rollback()
        if os.path.exists(tmp_csv):
            os.remove(tmp_csv)
        raise
    finally:
        dbapi_conn.close()
        conn.close()
    return out_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--database', type=str, default='equities', help='Datastream library to download')
    args = parser.parse_args()
    DATABASE = args.database
    if DATABASE == "datastream_continuous":
        pull_datastream_continuous(FUTURES_PATH)
    else:
        if DATABASE == "comp":
            OUTPUT_DIR = COMPUSTAT_PATH
        else:
            OUTPUT_DIR = DATASTREAM_PATH / DATABASE
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if DATABASE not in PULL_SPECS:
            raise ValueError(f"Invalid database: {DATABASE} (known: {sorted(PULL_SPECS)} + datastream_continuous)")
        LIBRARY, TABLES = PULL_SPECS[DATABASE]
        TABLE_PREFIX = None
        main()
