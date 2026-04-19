"""Thin asyncpg pool for the shared portfolio Cloud SQL `slc` database.

Provides a single `get_pool()` coroutine that returns a lazily-initialized
connection pool. Prefer this for new code paths; the legacy SQLAlchemy layer
in `app/db/` stays in place for existing models until fully migrated.

Usage:
    from app.db.asyncpg_pool import get_pool

    async with (await get_pool()).acquire() as conn:
        row = await conn.fetchrow("SELECT 1 AS ok")
"""
from __future__ import annotations

import os
from typing import Optional

import asyncpg


_pool: Optional[asyncpg.Pool] = None


def _build_dsn() -> str:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy backend/.env.example and run "
            "scripts/db-proxy.sh first."
        )
    # asyncpg does not accept +driver qualifiers like SQLAlchemy does.
    # Strip them so the same URL works for both layers.
    if dsn.startswith("postgresql+asyncpg://"):
        dsn = "postgresql://" + dsn[len("postgresql+asyncpg://") :]
    elif dsn.startswith("postgresql+psycopg://"):
        dsn = "postgresql://" + dsn[len("postgresql+psycopg://") :]
    return dsn


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            dsn=_build_dsn(),
            min_size=1,
            max_size=int(os.environ.get("ASYNCPG_MAX_SIZE", "8")),
            command_timeout=30,
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
