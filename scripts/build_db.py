"""Build data/trucks.duckdb from the CSV exports of the EU-Truck-Electrification project.

The CSVs in data/csv are copied unchanged from
https://github.com/YiYang-09/EU-Truck-Electrification (folder exports/), which builds them
from Eurostat tables road_eqr_lormot and road_eqs_lormot.
"""
import pathlib
import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "csv"
DB_PATH = ROOT / "data" / "trucks.duckdb"

# Explicit types so the database does not depend on CSV type sniffing.
TYPES = {
    "flag": "VARCHAR", "new_total_flag": "VARCHAR", "new_electric_flag": "VARCHAR",
    "stock_total_flag": "VARCHAR", "stock_electric_flag": "VARCHAR",
}


def build(db_path: pathlib.Path = DB_PATH) -> pathlib.Path:
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    for csv in sorted(CSV_DIR.glob("*.csv")):
        table = csv.stem
        header = csv.read_text(encoding="utf-8").splitlines()[0].split(",")
        types = {c: t for c, t in TYPES.items() if c in header}
        types_sql = ", types={" + ", ".join(f"'{c}': '{t}'" for c, t in types.items()) + "}" if types else ""
        con.execute(f"create table {table} as select * from read_csv('{csv.as_posix()}', header=true{types_sql})")
    con.close()
    return db_path


if __name__ == "__main__":
    path = build()
    con = duckdb.connect(str(path), read_only=True)
    for (t,) in con.execute("select table_name from information_schema.tables order by 1").fetchall():
        print(f"{t:24s} {con.execute(f'select count(*) from {t}').fetchone()[0]:>6d} rows")
