"""PostgreSQL only (VERITAS_TEST_PG_URL): reads must not leave the connection idle in a transaction."""
import pytest
from psycopg.pq import TransactionStatus

from app.models import database as db

pg = pytest.mark.skipif(db.connection().dialect != "postgres", reason="PostgreSQL only")


@pg
def test_a_read_does_not_leave_a_transaction_open():
    conn = db.connection()
    assert conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"] >= 0
    assert conn._conn.info.transaction_status == TransactionStatus.IDLE


@pg
def test_a_read_inside_a_write_transaction_keeps_it_open():
    conn = db.connection()
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS t_idle (x int)")
    conn.commit()
    conn.execute("INSERT INTO t_idle VALUES (1)")
    conn.execute("SELECT COUNT(*) AS n FROM t_idle").fetchone()
    assert conn._conn.info.transaction_status == TransactionStatus.INTRANS
    conn.rollback()
    assert conn.execute("SELECT COUNT(*) AS n FROM t_idle").fetchone()["n"] == 0
