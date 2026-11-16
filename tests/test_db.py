import pytest

from agent.db import SQLGuardError, check_sql


@pytest.mark.parametrize("sql", [
    "drop table dim_year", "delete from dim_year", "create table x as select 1", "select 1; select 2",
    "copy dim_year to 'x.csv'", "attach 'other.db'", "select * from read_csv('/etc/passwd')",
    "install httpfs", "pragma version", "",
])
def test_guard_rejects(sql):
    with pytest.raises(SQLGuardError):
        check_sql(sql)


@pytest.mark.parametrize("sql", [
    "select 1", "with a as (select 1 as x) select * from a", "select 'copy' as word",
    "select * from dim_year order by year;",
])
def test_guard_accepts(sql):
    assert check_sql(sql)


def test_row_limit_and_errors(db):
    r = db.run("select * from fct_new_lorries_bi")
    assert r.ok and r.truncated and len(r.rows) == db.max_rows
    r = db.run("select nope from dim_year")
    assert not r.ok and "nope" in r.error


def test_connection_is_read_only(db):
    # Even if the guard were bypassed, the connection refuses writes.
    with pytest.raises(Exception):
        db.con.execute("create table x as select 1")
