# -*- coding: utf-8 -*-
"""Lightweight SQLite usage tracker — no external dependencies beyond stdlib."""

import sqlite3
import os
from datetime import date

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "usage_tracker.db")


def _get_connection():
    return sqlite3.connect(DB_PATH)


def _ensure_table(cursor):
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            email        TEXT PRIMARY KEY,
            last_login_date TEXT NOT NULL,
            usage_count  INTEGER NOT NULL DEFAULT 0
        )
    """)


def track_login(email: str) -> dict:
    """Record a login event. Creates DB / table on first run.

    Returns dict with last_login_date and usage_count for display.
    """
    conn = _get_connection()
    cur = conn.cursor()
    _ensure_table(cur)

    today = date.today().isoformat()

    cur.execute("SELECT usage_count FROM usage_log WHERE email = ?", (email,))
    row = cur.fetchone()

    if row is None:
        cur.execute(
            "INSERT INTO usage_log (email, last_login_date, usage_count) "
            "VALUES (?, ?, 1)",
            (email, today),
        )
        result = {"last_login_date": today, "usage_count": 1}
    else:
        cur.execute(
            "UPDATE usage_log SET last_login_date = ?, usage_count = usage_count + 1 "
            "WHERE email = ?",
            (today, email),
        )
        result = {"last_login_date": today, "usage_count": row[0] + 1}

    conn.commit()
    conn.close()
    return result


def get_user_stats(email: str) -> dict | None:
    """Return stats dict for an email, or None."""
    conn = _get_connection()
    cur = conn.cursor()
    _ensure_table(cur)
    cur.execute(
        "SELECT last_login_date, usage_count FROM usage_log WHERE email = ?",
        (email,),
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {"last_login_date": row[0], "usage_count": row[1]}
    return None
