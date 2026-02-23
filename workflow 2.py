"""
approval/workflow.py — Human-in-the-Loop Trade Approval System
==============================================================

DESIGN PHILOSOPHY:
  The agents produce recommendations; humans make decisions. This module
  is the formal boundary between those two worlds. Nothing in this codebase
  touches a wallet or places an order without a record in this module
  showing that a human explicitly authorised it.

HOW IT WORKS — THE LIFECYCLE OF A RECOMMENDATION:
  1. PENDING:   The agent system finishes its analysis and calls
                ApprovalWorkflow.submit(). The recommendation is written
                to the database with status='pending' and a unique ID.

  2. REVIEW:    A human reviews the recommendation via the dashboard UI
                or the /api/v1/approvals endpoint. They see the full
                context: price, technicals, risk parameters, agent rationale.

  3. DECISION:  The human calls .approve() or .reject(). Both actions:
                  a) Update the database record with the decision and a timestamp
                  b) Record the human's reasoning (required field — you must
                     articulate WHY you approved or rejected, building a log
                     that lets you audit your own decision-making over time)
                  c) If approved, trigger the configured execution callback

  4. EXECUTED / REJECTED:  Terminal states. The audit trail is permanent.

  5. EXPIRED:   Any pending recommendation older than APPROVAL_TIMEOUT_HOURS
                is automatically expired and cannot be acted upon. This
                prevents stale analysis from being executed hours later when
                market conditions have completely changed.

AUDIT TRAIL:
  Every action — submission, approval, rejection, expiry — is written to
  the audit_log table. This gives you a complete, tamper-evident record
  of every decision the system and its human operators made, which is
  essential for reviewing performance and understanding losses.

DATABASE SCHEMA:
  Two tables:
    - trade_recommendations: one row per agent recommendation
    - audit_log:             one row per action taken on any recommendation
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

APPROVAL_TIMEOUT_HOURS = 4   # recommendations expire after this many hours
DB_PATH = "./memory/approvals.db"


# ---------------------------------------------------------------------------
# Status enum — the recommendation lifecycle in code form
# ---------------------------------------------------------------------------

class RecommendationStatus(str, Enum):
    PENDING  = "pending"    # awaiting human review
    APPROVED = "approved"   # human said yes; ready to execute
    REJECTED = "rejected"   # human said no; archived
    EXECUTED = "executed"   # approved AND execution confirmed
    EXPIRED  = "expired"    # timeout reached before human reviewed


class RiskLevel(str, Enum):
    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"
    EXTREME = "extreme"


# ---------------------------------------------------------------------------
# Core workflow engine
# ---------------------------------------------------------------------------

class ApprovalWorkflow:
    """
    Manages the full lifecycle of trade recommendations from agent
    submission through human approval and optional execution.

    The two most important properties of this class are:

    IMMUTABILITY: Once a recommendation is submitted, its agent-generated
    content (symbol, action, prices, rationale) is never modified. Only
    the status, decision fields, and audit log change. This ensures you
    always have an accurate record of what the agents actually recommended,
    even if you edited your own notes.

    REQUIRED REASONING: Both .approve() and .reject() require the human
    to provide a reason. This isn't bureaucratic friction — it's what
    separates a thoughtful trading operation from one that just clicks
    "yes" on everything the AI says. Articulating your reasoning even
    briefly ("strong volume confirmation, agree with entry") builds a
    feedback loop that helps you evaluate agent quality over time.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row   # rows behave like dicts
        conn.execute("PRAGMA journal_mode=WAL")  # safer concurrent access
        return conn

    def _init_db(self):
        """Create tables on first run. Safe to call repeatedly — uses IF NOT EXISTS."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trade_recommendations (
                    id                   TEXT PRIMARY KEY,
                    session_id           TEXT NOT NULL,
                    symbol               TEXT NOT NULL,
                    action               TEXT NOT NULL,
                    entry_price_low      REAL,
                    entry_price_high     REAL,
                    target_price         REAL,
                    stop_loss            REAL,
                    position_size_usd    REAL,
                    position_size_pct    REAL,
                    risk_level           TEXT,
                    risk_reward_ratio    REAL,
                    confidence           REAL,
                    strategy_used        TEXT,
                    timeframe            TEXT,
                    agent_rationale      TEXT,
                    key_risks            TEXT,     -- JSON array
                    full_analysis        TEXT,     -- full JSON blob from agents
                    status               TEXT NOT NULL DEFAULT 'pending',
                    submitted_at         TEXT NOT NULL,
                    expires_at           TEXT NOT NULL,
                    reviewed_at          TEXT,
                    reviewed_by          TEXT,
                    human_reason         TEXT,
                    execution_tx         TEXT,     -- tx hash if executed on-chain
                    execution_price      REAL,
                    executed_at          TEXT
                );

                CREATE TABLE IF NOT EXISTS audit_log (
                    id               INTEGER PRIMARY KEY AUTOINCREMENT,
                    recommendation_id TEXT NOT NULL,
                    action           TEXT NOT NULL,   -- 'submitted','approved','rejected','executed','expired'
                    actor            TEXT,            -- 'agent_system' | human identifier
                    details          TEXT,            -- free-form JSON
                    timestamp        TEXT NOT NULL,
                    FOREIGN KEY (recommendation_id) REFERENCES trade_recommendations(id)
                );

                CREATE INDEX IF NOT EXISTS idx_status ON trade_recommendations(status);
                CREATE INDEX IF NOT EXISTS idx_symbol ON trade_recommendations(symbol);
                CREATE INDEX IF NOT EXISTS idx_submitted ON trade_recommendations(submitted_at);
            """)

    # ── Submission ───────────────────────────────────────────────────────────

    def submit(
        self,
        session_id:        str,
        symbol:            str,
        agent_outputs:     dict,          # the full state from AgentState
        execution_callback: Optional[Callable] = None,
    ) -> str:
        """
        Submit a new recommendation from the agent system for human review.

        The agent_outputs dict is the complete final state returned by the
        LangGraph — it contains market_data, technical_analysis, risk_assessment,
        and trade_recommendation. We flatten the key fields for easy querying
        while preserving the full blob for the review UI.

        Returns the unique recommendation ID (a UUID) which you can use
        to retrieve or act on the recommendation later.
        """
        rec_id    = str(uuid.uuid4())
        now_utc   = datetime.now(timezone.utc)
        expires   = now_utc + timedelta(hours=APPROVAL_TIMEOUT_HOURS)

        # Extract the structured fields from agent outputs
        trade_rec = agent_outputs.get("trade_recommendation", {})
        risk      = agent_outputs.get("risk_assessment", {})
        entry     = trade_rec.get("entry_zone", {})

        with self._conn() as conn:
            conn.execute("""
                INSERT INTO trade_recommendations (
                    id, session_id, symbol, action,
                    entry_price_low, entry_price_high, target_price, stop_loss,
                    position_size_usd, position_size_pct,
                    risk_level, risk_reward_ratio, confidence,
                    strategy_used, timeframe,
                    agent_rationale, key_risks, full_analysis,
                    status, submitted_at, expires_at
                ) VALUES (
                    ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    ?, ?,
                    ?, ?, ?,
                    'pending', ?, ?
                )
            """, (
                rec_id, session_id, symbol.upper(),
                trade_rec.get("action", "hold").upper(),
                entry.get("low"),  entry.get("high"),
                trade_rec.get("target_price"),
                trade_rec.get("stop_loss"),
                risk.get("max_position_size_usd"),
                risk.get("max_position_size_pct"),
                risk.get("risk_level"),
                risk.get("risk_reward_ratio"),
                trade_rec.get("confidence"),
                trade_rec.get("strategy_used"),
                trade_rec.get("timeframe"),
                trade_rec.get("rationale"),
                json.dumps(trade_rec.get("key_risks", [])),
                json.dumps(agent_outputs, default=str),
                now_utc.isoformat(),
                expires.isoformat(),
            ))

            # Write the first audit entry
            self._log(conn, rec_id, "submitted", actor="agent_system", details={
                "symbol": symbol, "action": trade_rec.get("action"),
                "confidence": trade_rec.get("confidence"),
            })

        # Store callback for later if provided
        if execution_callback:
            self._callbacks[rec_id] = execution_callback

        return rec_id

    # ── Human decisions ──────────────────────────────────────────────────────

    def approve(
        self,
        rec_id:    str,
        reason:    str,
        reviewer:  str = "human_operator",
    ) -> dict:
        """
        Approve a pending recommendation.

        `reason` is a required string. Even a short phrase — "RSI confirms,
        volume looks solid" — creates a feedback trail that lets you review
        which types of reasoning led to winning vs losing trades.

        Returns a result dict with the updated recommendation and next steps.
        """
        if not reason or len(reason.strip()) < 5:
            return {
                "success": False,
                "error": "Approval requires a reason (at least 5 characters). "
                         "Articulate why you agree with this recommendation."
            }

        rec = self._get_or_raise(rec_id)
        self._check_not_expired(rec)

        if rec["status"] != RecommendationStatus.PENDING:
            return {
                "success": False,
                "error": f"Cannot approve — recommendation is already '{rec['status']}'"
            }

        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE trade_recommendations
                SET status='approved', reviewed_at=?, reviewed_by=?, human_reason=?
                WHERE id=?
            """, (now, reviewer, reason.strip(), rec_id))
            self._log(conn, rec_id, "approved", actor=reviewer, details={"reason": reason})

        # Fire the execution callback if one was registered
        callback = self._callbacks.pop(rec_id, None)
        execution_result = None
        if callback:
            try:
                execution_result = callback(rec)
                self._mark_executed(rec_id, execution_result)
            except Exception as e:
                execution_result = {"error": str(e)}

        return {
            "success":          True,
            "recommendation_id": rec_id,
            "status":           "approved",
            "execution_result": execution_result,
            "message":          (
                "✅ Recommendation approved and queued for execution."
                if execution_result is None else
                f"✅ Approved and executed. Result: {execution_result}"
            )
        }

    def reject(
        self,
        rec_id:   str,
        reason:   str,
        reviewer: str = "human_operator",
    ) -> dict:
        """
        Reject a pending recommendation.

        Rejected recommendations are preserved in the database — they're
        not deleted. This is intentional: your rejections are as valuable
        as your approvals for understanding agent quality over time.
        """
        if not reason or len(reason.strip()) < 5:
            return {
                "success": False,
                "error": "Rejection requires a reason. Note why you disagree — "
                         "this feedback is how you evaluate agent quality."
            }

        rec = self._get_or_raise(rec_id)

        if rec["status"] != RecommendationStatus.PENDING:
            return {
                "success": False,
                "error": f"Cannot reject — recommendation is already '{rec['status']}'"
            }

        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE trade_recommendations
                SET status='rejected', reviewed_at=?, reviewed_by=?, human_reason=?
                WHERE id=?
            """, (now, reviewer, reason.strip(), rec_id))
            self._log(conn, rec_id, "rejected", actor=reviewer, details={"reason": reason})

        # Remove any execution callback
        self._callbacks.pop(rec_id, None)

        return {
            "success":           True,
            "recommendation_id": rec_id,
            "status":            "rejected",
            "message":           "❌ Recommendation rejected and archived."
        }

    # ── Queries ──────────────────────────────────────────────────────────────

    def get_pending(self) -> list[dict]:
        """
        Return all pending recommendations that haven't expired yet.
        This is what the review UI calls to populate its queue.
        """
        self._expire_stale()   # expire any timed-out items first
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT * FROM trade_recommendations
                WHERE status = 'pending'
                ORDER BY submitted_at DESC
            """).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_recommendation(self, rec_id: str) -> Optional[dict]:
        """Fetch a single recommendation by ID, including full agent analysis."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trade_recommendations WHERE id=?", (rec_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_history(
        self,
        limit:  int = 50,
        status: Optional[str] = None,
        symbol: Optional[str] = None,
    ) -> list[dict]:
        """
        Return completed recommendations (approved, rejected, executed, expired).
        Useful for performance review and agent quality auditing.
        """
        where_clauses = []
        params = []

        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if symbol:
            where_clauses.append("symbol = ?")
            params.append(symbol.upper())

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM trade_recommendations {where_sql} "
                f"ORDER BY submitted_at DESC LIMIT ?",
                params + [limit]
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_audit_log(self, rec_id: str) -> list[dict]:
        """Return the full audit trail for a specific recommendation."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE recommendation_id=? ORDER BY timestamp",
                (rec_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """
        Return aggregate statistics useful for evaluating agent performance.
        The approval_rate tells you how much you trust the agents.
        The win_rate_of_approved tells you whether your trust is calibrated.
        """
        with self._conn() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM trade_recommendations"
            ).fetchone()[0]
            by_status = dict(conn.execute(
                "SELECT status, COUNT(*) FROM trade_recommendations GROUP BY status"
            ).fetchall())
            executed_wins = conn.execute("""
                SELECT COUNT(*) FROM trade_recommendations
                WHERE status='executed' AND execution_price > entry_price_low
            """).fetchone()[0]
            total_executed = by_status.get("executed", 0)

        return {
            "total_recommendations": total,
            "by_status":             by_status,
            "pending":               by_status.get("pending", 0),
            "approval_rate_pct":     round(
                (by_status.get("approved", 0) + by_status.get("executed", 0))
                / max(total, 1) * 100, 1
            ),
            "execution_count":       total_executed,
            "win_rate_of_executed":  round(executed_wins / max(total_executed, 1) * 100, 1),
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    _callbacks: dict[str, Callable] = {}   # class-level callback registry

    def _get_or_raise(self, rec_id: str) -> sqlite3.Row:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM trade_recommendations WHERE id=?", (rec_id,)
            ).fetchone()
        if not row:
            raise ValueError(f"Recommendation {rec_id} not found")
        return row

    def _check_not_expired(self, rec: sqlite3.Row):
        expires_at = datetime.fromisoformat(rec["expires_at"])
        if datetime.now(timezone.utc) > expires_at:
            with self._conn() as conn:
                conn.execute(
                    "UPDATE trade_recommendations SET status='expired' WHERE id=?",
                    (rec["id"],)
                )
                self._log(conn, rec["id"], "expired", actor="system", details={
                    "expired_at": datetime.now(timezone.utc).isoformat()
                })
            raise ValueError(
                f"Recommendation {rec['id']} expired at {rec['expires_at']}. "
                "Market conditions may have changed — request a fresh analysis."
            )

    def _expire_stale(self):
        """Mark all pending recommendations past their expiry time as expired."""
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            stale_ids = conn.execute("""
                SELECT id FROM trade_recommendations
                WHERE status='pending' AND expires_at < ?
            """, (now,)).fetchall()
            for row in stale_ids:
                conn.execute(
                    "UPDATE trade_recommendations SET status='expired' WHERE id=?",
                    (row[0],)
                )
                self._log(conn, row[0], "expired", actor="system", details={})

    def _mark_executed(self, rec_id: str, execution_result: dict):
        now = datetime.now(timezone.utc).isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE trade_recommendations
                SET status='executed', executed_at=?, execution_tx=?, execution_price=?
                WHERE id=?
            """, (
                now,
                execution_result.get("transaction_hash"),
                execution_result.get("price"),
                rec_id,
            ))
            self._log(conn, rec_id, "executed", actor="system", details=execution_result)

    def _log(self, conn: sqlite3.Connection, rec_id: str, action: str,
             actor: str, details: dict):
        conn.execute("""
            INSERT INTO audit_log (recommendation_id, action, actor, details, timestamp)
            VALUES (?, ?, ?, ?, ?)
        """, (rec_id, action, actor, json.dumps(details),
              datetime.now(timezone.utc).isoformat()))

    def mark_executed(
        self,
        rec_id:  str,
        tx_hash: str,
        details: dict | None = None,
    ) -> dict:
        """
        Mark an approved recommendation as executed after an on-chain
        transaction has been confirmed. Records the transaction hash
        permanently in the audit trail.

        This method is called by the /execute endpoint AFTER the CDP SDK
        swap has returned a transaction hash. If the swap reverted on-chain
        (slippage exceeded, etc.) this method is never called — the rec
        remains 'approved' and can be retried or rejected.

        Args:
            rec_id:  The recommendation UUID.
            tx_hash: The on-chain transaction hash (0x...).
            details: Additional execution metadata (slippage_bps, position_usd, etc.)
        """
        with self._conn() as conn:
            row = conn.execute(
                "SELECT status FROM trade_recommendations WHERE id=?", (rec_id,)
            ).fetchone()

            if not row:
                return {"success": False, "error": f"Recommendation {rec_id} not found"}
            if row["status"] != "approved":
                return {"success": False, "error": f"Status is '{row['status']}', expected 'approved'"}

            now = datetime.now(timezone.utc).isoformat()
            conn.execute("""
                UPDATE trade_recommendations
                SET status     = 'executed',
                    decided_at = ?
                WHERE id = ?
            """, (now, rec_id))

            self._log(conn, rec_id, "executed", actor="system", details={
                "transaction_hash": tx_hash,
                **(details or {}),
            })

        return {"success": True, "rec_id": rec_id, "transaction_hash": tx_hash, "status": "executed"}

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        # Parse JSON fields back to native types for API responses
        for field in ("key_risks", "full_analysis"):
            if d.get(field) and isinstance(d[field], str):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d
