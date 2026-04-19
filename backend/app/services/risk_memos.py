"""Asyncpg-backed reads for risk memos.

Replaces the SQLAlchemy path for the narrator / quick-read flows, which only
need a few columns and benefit from the lower overhead of raw asyncpg.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from uuid import UUID

from app.db.asyncpg_pool import get_pool


@dataclass
class RiskMemoSummary:
    contract_id: UUID
    filename: str
    sector: Optional[str]
    overall_risk: Optional[float]
    high_risk_clauses: int
    uploaded_at: str


async def list_recent_risk_memos(limit: int = 20) -> List[RiskMemoSummary]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT c.id, c.filename, c.sector, c.uploaded_at,
                   rs.overall_risk,
                   (SELECT COUNT(*) FROM clauses cl
                     WHERE cl.contract_id = c.id AND cl.risk_level = 'high') AS high_risk_clauses
            FROM contracts c
            LEFT JOIN risk_scores rs ON rs.contract_id = c.id
            ORDER BY c.uploaded_at DESC
            LIMIT $1
            """,
            limit,
        )
    return [
        RiskMemoSummary(
            contract_id=r["id"],
            filename=r["filename"],
            sector=r["sector"],
            overall_risk=r["overall_risk"],
            high_risk_clauses=r["high_risk_clauses"] or 0,
            uploaded_at=r["uploaded_at"].isoformat() if r["uploaded_at"] else "",
        )
        for r in rows
    ]


async def get_risk_memo(contract_id: UUID) -> Optional[dict]:
    pool = await get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT c.id, c.filename, c.sector, c.uploaded_at, c.word_count,
                   rs.overall_risk, rs.confidence, rs.summary_text
            FROM contracts c
            LEFT JOIN risk_scores rs ON rs.contract_id = c.id
            WHERE c.id = $1
            """,
            contract_id,
        )
        if not row:
            return None
        clauses = await conn.fetch(
            """
            SELECT id, clause_type, text, risk_level, confidence, explanation
            FROM clauses
            WHERE contract_id = $1
            ORDER BY start_offset ASC
            """,
            contract_id,
        )
    return {
        "contract_id": str(row["id"]),
        "filename": row["filename"],
        "sector": row["sector"],
        "uploaded_at": row["uploaded_at"].isoformat() if row["uploaded_at"] else None,
        "word_count": row["word_count"],
        "overall_risk": row["overall_risk"],
        "confidence": row["confidence"],
        "summary_text": row["summary_text"],
        "clauses": [dict(c) for c in clauses],
    }
