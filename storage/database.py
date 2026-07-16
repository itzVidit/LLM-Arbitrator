"""
storage/database.py
-------------------
Async SQLite persistence layer for the LLM Output Arbitration System.

Schema
------
  arbitrations   — one row per FinalVerdict, verdict JSON stored as TEXT
  analytics_rows — one row per critic per arbitration run (for fast aggregation)

All public functions are async and use aiosqlite.  The module exposes a
single Database singleton accessed via `get_db()`.

Usage
-----
    from storage.database import get_db

    db = await get_db()
    await db.init()

    arb_id = await db.save_verdict(verdict)
    row     = await db.get_verdict(arb_id)
    rows    = await db.list_verdicts(limit=20, offset=0)
    report  = await db.get_analytics()
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from config import settings

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
#  Schema SQL
# ─────────────────────────────────────────────────────────────────────

_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS arbitrations (
    id              TEXT PRIMARY KEY,
    session_id      TEXT,
    created_at      TEXT NOT NULL,
    recommendation  TEXT NOT NULL,
    overall_score   INTEGER NOT NULL,
    quality_score   INTEGER NOT NULL DEFAULT 0,
    overall_confidence REAL NOT NULL,
    critic_agreement   REAL NOT NULL,
    adjudication_used  INTEGER NOT NULL DEFAULT 0,
    fast_path          INTEGER NOT NULL DEFAULT 0,
    total_latency_ms   REAL,
    verdict_json    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_arbitrations_created
    ON arbitrations (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_arbitrations_recommendation
    ON arbitrations (recommendation);
CREATE INDEX IF NOT EXISTS idx_arbitrations_session
    ON arbitrations (session_id);

CREATE TABLE IF NOT EXISTS analytics_rows (
    id              TEXT PRIMARY KEY,
    arbitration_id  TEXT NOT NULL REFERENCES arbitrations(id) ON DELETE CASCADE,
    created_at      TEXT NOT NULL,
    dimension       TEXT NOT NULL,
    model_used      TEXT NOT NULL,
    score           INTEGER NOT NULL,
    confidence      REAL NOT NULL,
    passed          INTEGER NOT NULL,
    latency_ms      REAL,
    used_fallback   INTEGER NOT NULL DEFAULT 0,
    issue_count     INTEGER NOT NULL DEFAULT 0,
    critical_count  INTEGER NOT NULL DEFAULT 0,
    major_count     INTEGER NOT NULL DEFAULT 0,
    was_overruled   INTEGER NOT NULL DEFAULT 0,
    overrule_direction TEXT,
    issue_categories TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS idx_analytics_dimension
    ON analytics_rows (dimension);
CREATE INDEX IF NOT EXISTS idx_analytics_model
    ON analytics_rows (model_used);
CREATE INDEX IF NOT EXISTS idx_analytics_arbitration
    ON analytics_rows (arbitration_id);
"""


# ─────────────────────────────────────────────────────────────────────
#  Database class
# ─────────────────────────────────────────────────────────────────────

class Database:
    """
    Wraps an aiosqlite connection to the arbitrations SQLite file.

    Call `await db.init()` once at application startup (e.g. in
    FastAPI's lifespan handler) before making any other calls.
    """

    def __init__(self, db_path: str) -> None:
        self._path = db_path
        self._conn: Optional[aiosqlite.Connection] = None

    # ── Lifecycle ─────────────────────────────────────────────────────

    async def init(self) -> None:
        """Open the connection and create tables if they don't exist."""
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self._path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(_DDL)
        await self._conn.commit()
        log.info("Database ready at %s", self._path)

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def conn(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database.init() has not been called")
        return self._conn

    # ── Verdict storage ───────────────────────────────────────────────

    async def save_verdict(self, verdict) -> str:
        """
        Persist a FinalVerdict and its per-critic analytics rows.

        Parameters
        ----------
        verdict : FinalVerdict

        Returns
        -------
        str
            The UUID assigned to this arbitration record.
        """
        arb_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()

        verdict_json = verdict.model_dump_json()

        await self.conn.execute(
            """
            INSERT INTO arbitrations
                (id, session_id, created_at, recommendation, overall_score,
                 quality_score, overall_confidence, critic_agreement,
                 adjudication_used, fast_path, total_latency_ms, verdict_json)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                arb_id,
                verdict.session_id,
                now,
                verdict.recommendation,
                verdict.overall_score,
                verdict.quality_score,
                verdict.overall_confidence,
                verdict.critic_agreement,
                int(verdict.adjudication_used),
                int(verdict.fast_path),
                verdict.total_latency_ms,
                verdict_json,
            ),
        )

        # Determine which critic dimensions were overruled by the adjudicator.
        # "Overruled" = adjudicator changed a dimension's score by >10 pts or
        # flipped its pass/fail.  We detect this by comparing stored
        # dimension_scores against each critique's original score.
        overruled: dict[str, str] = {}
        if verdict.adjudication_used:
            for critique in verdict.critiques:
                dim = critique.dimension.value
                final_score = verdict.dimension_scores.get(dim, critique.score)
                delta = final_score - critique.score
                final_passed = verdict.dimension_passed.get(dim, critique.passed)
                if abs(delta) > 10 or final_passed != critique.passed:
                    direction = "raised" if delta > 0 else "lowered"
                    overruled[dim] = direction

        fallback_dims = {fb.dimension.value for fb in verdict.fallbacks_used}

        for critique in verdict.critiques:
            dim = critique.dimension.value
            issue_cats = json.dumps(
                list({i.severity for i in critique.issues})
            )
            await self.conn.execute(
                """
                INSERT INTO analytics_rows
                    (id, arbitration_id, created_at, dimension, model_used,
                     score, confidence, passed, latency_ms, used_fallback,
                     issue_count, critical_count, major_count,
                     was_overruled, overrule_direction, issue_categories)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    str(uuid.uuid4()),
                    arb_id,
                    now,
                    dim,
                    critique.model_used,
                    critique.score,
                    critique.confidence,
                    int(critique.passed),
                    critique.latency_ms,
                    int(dim in fallback_dims),
                    len(critique.issues),
                    critique.critical_count,
                    critique.major_count,
                    int(dim in overruled),
                    overruled.get(dim),
                    issue_cats,
                ),
            )

        await self.conn.commit()
        log.debug("Saved arbitration %s  rec=%s", arb_id, verdict.recommendation)
        return arb_id

    # ── Retrieval ──────────────────────────────────────────────────────

    async def get_verdict(self, arb_id: str) -> Optional[dict[str, Any]]:
        """
        Retrieve one arbitration row by its UUID.

        Returns the full row dict (including verdict_json) or None.
        """
        async with self.conn.execute(
            "SELECT * FROM arbitrations WHERE id = ?", (arb_id,)
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return None
        return dict(row)

    async def list_verdicts(
        self,
        limit: int = 50,
        offset: int = 0,
        recommendation: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        List arbitration summary rows (no verdict_json) newest-first.

        Parameters
        ----------
        limit          : max rows to return (capped at 200)
        offset         : pagination offset
        recommendation : optional filter — "APPROVE", "REVISE", or "REJECT"
        """
        limit = min(limit, 200)
        if recommendation:
            sql = """
                SELECT id, session_id, created_at, recommendation,
                       overall_score, quality_score, overall_confidence,
                       critic_agreement, adjudication_used, fast_path,
                       total_latency_ms
                FROM arbitrations
                WHERE recommendation = ?
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            params = (recommendation.upper(), limit, offset)
        else:
            sql = """
                SELECT id, session_id, created_at, recommendation,
                       overall_score, quality_score, overall_confidence,
                       critic_agreement, adjudication_used, fast_path,
                       total_latency_ms
                FROM arbitrations
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            params = (limit, offset)

        async with self.conn.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def count_verdicts(self) -> int:
        async with self.conn.execute("SELECT COUNT(*) FROM arbitrations") as cur:
            row = await cur.fetchone()
        return row[0] if row else 0

    # ── Analytics queries ──────────────────────────────────────────────

    async def get_analytics_rows(self) -> list[dict[str, Any]]:
        """Return all analytics_rows joined with their arbitration metadata."""
        sql = """
            SELECT
                ar.*,
                a.recommendation,
                a.overall_score,
                a.adjudication_used,
                a.created_at AS arb_created_at
            FROM analytics_rows ar
            JOIN arbitrations a ON a.id = ar.arbitration_id
            ORDER BY ar.created_at DESC
        """
        async with self.conn.execute(sql) as cur:
            rows = await cur.fetchall()
        return [dict(r) for r in rows]

    async def get_recommendation_counts(self) -> dict[str, int]:
        """Count verdicts grouped by recommendation."""
        async with self.conn.execute(
            "SELECT recommendation, COUNT(*) as n FROM arbitrations GROUP BY recommendation"
        ) as cur:
            rows = await cur.fetchall()
        return {r["recommendation"]: r["n"] for r in rows}


# ─────────────────────────────────────────────────────────────────────
#  Module-level singleton
# ─────────────────────────────────────────────────────────────────────

_DB: Optional[Database] = None


async def get_db() -> Database:
    """
    Return the module-level Database singleton.
    Caller is responsible for calling init() before use.
    """
    global _DB
    if _DB is None:
        _DB = Database(settings.storage_path)
    return _DB
