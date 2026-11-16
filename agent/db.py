"""Read-only access to the DuckDB database, with a SQL guard, a row limit and a timeout."""
from __future__ import annotations

import pathlib
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import duckdb

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "trucks.duckdb"

# Table functions and keywords that could reach the file system or change state. The connection is
# already read-only with external access disabled; this list is a second, explicit layer that also
# gives the model a clear error message.
_BLOCKED = re.compile(
    r"\b(read_csv\w*|read_parquet|read_json\w*|read_text|read_blob|parquet_\w+|glob|sniff_csv|"
    r"attach|detach|install|load|copy|export|import|pragma|checkpoint|query_table)\b",
    re.IGNORECASE,
)


class SQLGuardError(ValueError):
    """The SQL was rejected before execution."""


def check_sql(sql: str) -> str:
    """Return the cleaned SQL if it is a single read-only SELECT statement, else raise SQLGuardError."""
    cleaned = sql.strip().rstrip(";").strip()
    if not cleaned:
        raise SQLGuardError("Empty SQL.")
    try:
        statements = duckdb.extract_statements(cleaned)
    except duckdb.Error as e:  # syntax errors surface here
        raise SQLGuardError(f"SQL could not be parsed: {e}") from e
    if len(statements) != 1:
        raise SQLGuardError(f"Exactly one statement is allowed, got {len(statements)}.")
    if statements[0].type != duckdb.StatementType.SELECT:
        raise SQLGuardError(f"Only SELECT queries are allowed, got {statements[0].type.name}.")
    # Ignore string literals when looking for blocked words.
    no_strings = re.sub(r"'(?:[^']|'')*'", "''", cleaned)
    m = _BLOCKED.search(no_strings)
    if m:
        raise SQLGuardError(f"'{m.group(0)}' is not allowed. Query the tables in the database only.")
    return cleaned


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool = False
    elapsed_s: float = 0.0
    error: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    def to_markdown(self, max_rows: int = 30) -> str:
        if not self.ok:
            return f"ERROR: {self.error}"
        head = "| " + " | ".join(self.columns) + " |"
        sep = "|" + "---|" * len(self.columns)
        body = ["| " + " | ".join(_fmt(v) for v in r) + " |" for r in self.rows[:max_rows]]
        lines = [head, sep, *body]
        shown = min(len(self.rows), max_rows)
        note = f"{len(self.rows)} row(s)"
        if shown < len(self.rows):
            note += f", first {shown} shown"
        if self.truncated:
            note += f" (result truncated at the row limit)"
        return "\n".join(lines) + f"\n\n{note}"

    def to_dict(self) -> dict:
        return {"sql": self.sql, "columns": self.columns, "rows": [list(r) for r in self.rows],
                "truncated": self.truncated, "elapsed_s": round(self.elapsed_s, 4), "error": self.error}


def _fmt(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


class Database:
    def __init__(self, path: str | pathlib.Path = DEFAULT_DB, max_rows: int = 500, timeout_s: float = 15.0):
        path = pathlib.Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run: python scripts/build_db.py")
        self.path = path
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self.con = duckdb.connect(str(path), read_only=True, config={"enable_external_access": False})
        self._lock = threading.Lock()

    def tables(self) -> list[str]:
        return [r[0] for r in self.con.execute(
            "select table_name from information_schema.tables where table_schema = 'main' order by 1").fetchall()]

    def columns(self, table: str) -> list[tuple[str, str]]:
        return [(r[0], r[1]) for r in self.con.execute(
            "select column_name, data_type from information_schema.columns "
            "where table_schema = 'main' and table_name = ? order by ordinal_position", [table]).fetchall()]

    def ddl(self) -> str:
        """A compact CREATE TABLE listing of the whole schema (no descriptions)."""
        out = []
        for t in self.tables():
            cols = ", ".join(f"{c} {ty}" for c, ty in self.columns(t))
            out.append(f"{t}({cols})")
        return "\n".join(out)

    def run(self, sql: str) -> QueryResult:
        try:
            cleaned = check_sql(sql)
        except SQLGuardError as e:
            return QueryResult(sql=sql, columns=[], rows=[], error=str(e))
        start = time.perf_counter()
        with self._lock:
            timer = threading.Timer(self.timeout_s, self.con.interrupt)
            timer.start()
            try:
                cur = self.con.execute(cleaned)
                columns = [d[0] for d in cur.description]
                rows = cur.fetchmany(self.max_rows + 1)
            except duckdb.InterruptException:
                return QueryResult(sql=cleaned, columns=[], rows=[], elapsed_s=time.perf_counter() - start,
                                   error=f"Query timed out after {self.timeout_s:.0f}s.")
            except duckdb.Error as e:
                return QueryResult(sql=cleaned, columns=[], rows=[], elapsed_s=time.perf_counter() - start,
                                   error=f"{type(e).__name__}: {e}")
            finally:
                timer.cancel()
        truncated = len(rows) > self.max_rows
        return QueryResult(sql=cleaned, columns=columns, rows=[tuple(r) for r in rows[: self.max_rows]],
                           truncated=truncated, elapsed_s=time.perf_counter() - start)
