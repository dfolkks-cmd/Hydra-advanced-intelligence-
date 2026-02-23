"""
memory/memory_manager.py — Persistent memory using SQLite + LangGraph checkpointing.

Stores:
  - Conversation summaries (compressed long-term memory)
  - Trade history & outcomes (for learning)
  - User preferences
  - Notable market events
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional


class CryptoMemoryManager:
    """SQLite-backed persistent memory for the crypto agent system."""

    def __init__(self, db_path: str = "./memory/agent_memory.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT NOT NULL,
                    summary     TEXT NOT NULL,
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS trade_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT NOT NULL,
                    action          TEXT NOT NULL,
                    entry_price     REAL,
                    exit_price      REAL,
                    position_size   REAL,
                    pnl_usd         REAL,
                    pnl_pct         REAL,
                    outcome         TEXT,   -- 'win', 'loss', 'open'
                    strategy        TEXT,
                    notes           TEXT,
                    created_at      TEXT NOT NULL,
                    closed_at       TEXT
                );

                CREATE TABLE IF NOT EXISTS user_preferences (
                    key         TEXT PRIMARY KEY,
                    value       TEXT NOT NULL,
                    updated_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS market_notes (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol      TEXT,
                    note        TEXT NOT NULL,
                    tags        TEXT,
                    created_at  TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    snapshot_json   TEXT NOT NULL,
                    total_value_usd REAL,
                    created_at      TEXT NOT NULL
                );
            """)

    # ── Conversation Memory ──────────────────────────────────────────────────

    def save_summary(self, session_id: str, summary: str):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO conversation_summaries (session_id, summary, created_at) VALUES (?, ?, ?)",
                (session_id, summary, datetime.utcnow().isoformat()),
            )

    def get_recent_summaries(self, limit: int = 5) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM conversation_summaries ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Trade History ────────────────────────────────────────────────────────

    def record_trade(
        self,
        symbol: str,
        action: str,
        entry_price: float,
        position_size: float,
        strategy: str,
        notes: str = "",
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO trade_history
                   (symbol, action, entry_price, position_size, outcome, strategy, notes, created_at)
                   VALUES (?, ?, ?, ?, 'open', ?, ?, ?)""",
                (symbol, action, entry_price, position_size, strategy, notes,
                 datetime.utcnow().isoformat()),
            )
            return cur.lastrowid

    def close_trade(self, trade_id: int, exit_price: float, notes: str = ""):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT entry_price, position_size FROM trade_history WHERE id=?", (trade_id,)
            ).fetchone()
            if not row:
                return
            pnl_usd = (exit_price - row["entry_price"]) * row["position_size"]
            pnl_pct = (exit_price - row["entry_price"]) / row["entry_price"] * 100
            outcome = "win" if pnl_usd >= 0 else "loss"
            conn.execute(
                """UPDATE trade_history SET
                   exit_price=?, pnl_usd=?, pnl_pct=?, outcome=?, closed_at=?, notes=?
                   WHERE id=?""",
                (exit_price, round(pnl_usd, 2), round(pnl_pct, 2), outcome,
                 datetime.utcnow().isoformat(), notes, trade_id),
            )

    def get_trade_stats(self) -> dict:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_history WHERE outcome != 'open'"
            ).fetchall()
        if not rows:
            return {"total_trades": 0}
        trades = [dict(r) for r in rows]
        wins = [t for t in trades if t["outcome"] == "win"]
        total_pnl = sum(t["pnl_usd"] or 0 for t in trades)
        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(trades) - len(wins),
            "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
            "total_pnl_usd": round(total_pnl, 2),
            "avg_win_usd": round(
                sum(t["pnl_usd"] for t in wins) / len(wins), 2
            ) if wins else 0,
        }

    def get_recent_trades(self, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM trade_history ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    # ── User Preferences ─────────────────────────────────────────────────────

    def set_preference(self, key: str, value):
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO user_preferences (key, value, updated_at) VALUES (?, ?, ?)",
                (key, json.dumps(value), datetime.utcnow().isoformat()),
            )

    def get_preference(self, key: str, default=None):
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM user_preferences WHERE key=?", (key,)
            ).fetchone()
        return json.loads(row["value"]) if row else default

    # ── Market Notes ─────────────────────────────────────────────────────────

    def add_market_note(self, note: str, symbol: Optional[str] = None, tags: list[str] = None):
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO market_notes (symbol, note, tags, created_at) VALUES (?, ?, ?, ?)",
                (symbol, note, json.dumps(tags or []), datetime.utcnow().isoformat()),
            )

    def get_market_notes(self, symbol: Optional[str] = None, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM market_notes WHERE symbol=? ORDER BY created_at DESC LIMIT ?",
                    (symbol.upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM market_notes ORDER BY created_at DESC LIMIT ?", (limit,)
                ).fetchall()
        return [dict(r) for r in rows]

    # ── Context Builder ───────────────────────────────────────────────────────

    def build_context(self, symbol: Optional[str] = None) -> str:
        """Build a compact memory context string to inject at session start."""
        parts = []

        # Recent conversation summaries
        summaries = self.get_recent_summaries(3)
        if summaries:
            parts.append("📝 RECENT SESSION HISTORY:")
            for s in summaries:
                parts.append(f"  [{s['created_at'][:10]}] {s['summary']}")

        # Trade stats
        stats = self.get_trade_stats()
        if stats.get("total_trades", 0) > 0:
            parts.append(
                f"\n📊 TRADE HISTORY: {stats['total_trades']} trades | "
                f"Win rate: {stats['win_rate_pct']}% | "
                f"Total P&L: ${stats['total_pnl_usd']:,.2f}"
            )

        # Recent trades
        recent = self.get_recent_trades(3)
        if recent:
            parts.append("Recent trades:")
            for t in recent:
                status = f"P&L: ${t['pnl_usd']:.2f}" if t['pnl_usd'] else "OPEN"
                parts.append(f"  {t['symbol']} {t['action']} @ ${t['entry_price']} | {status}")

        # Market notes for this symbol
        if symbol:
            notes = self.get_market_notes(symbol, 5)
            if notes:
                parts.append(f"\n🔖 NOTES ON {symbol.upper()}:")
                for n in notes:
                    parts.append(f"  [{n['created_at'][:10]}] {n['note']}")

        # User preferences
        pref_keys = ["risk_tolerance", "preferred_timeframe", "favorite_coins"]
        prefs = {k: self.get_preference(k) for k in pref_keys}
        prefs = {k: v for k, v in prefs.items() if v is not None}
        if prefs:
            parts.append(f"\n⚙️  USER PREFERENCES: {json.dumps(prefs)}")

        return "\n".join(parts) if parts else "No prior memory found."
