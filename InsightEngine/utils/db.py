"""
通用数据库工具（异步）

此模块提供基于 SQLAlchemy 2.x 异步引擎的数据库访问封装，支持 MySQL 与 PostgreSQL。
数据模型定义位置：
- 无（本模块仅提供连接与查询工具，不定义数据模型）
"""

from __future__ import annotations
from urllib.parse import quote_plus
import asyncio
import os
from typing import Any, Dict, Iterable, List, Optional, Union

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy import text
from InsightEngine.utils.config import settings

import re

__all__ = [
    "get_async_engine",
    "fetch_all",
]


_engine: Optional[AsyncEngine] = None
_is_postgres: Optional[bool] = None


def _build_database_url() -> str:
    """根据配置拼出 SQLAlchemy 异步连接串。

    优先用环境变量 DATABASE_URL；否则按 DB_DIALECT 选驱动：postgresql -> asyncpg，
    其余默认 mysql -> aiomysql。密码经 quote_plus 转义以兼容特殊字符。
    """
    dialect: str = (settings.DB_DIALECT or "mysql").lower()
    host: str = settings.DB_HOST or ""
    port: str = str(settings.DB_PORT or "")
    user: str = settings.DB_USER or ""
    password: str = settings.DB_PASSWORD or ""
    db_name: str = settings.DB_NAME or ""

    if os.getenv("DATABASE_URL"):
        return os.getenv("DATABASE_URL")  # 直接使用外部提供的完整URL

    password = quote_plus(password)

    if dialect in ("postgresql", "postgres"):
        # PostgreSQL 使用 asyncpg 驱动
        return f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{db_name}"

    # 默认 MySQL 使用 aiomysql 驱动
    return f"mysql+aiomysql://{user}:{password}@{host}:{port}/{db_name}"


def get_async_engine() -> AsyncEngine:
    """返回进程级单例异步引擎（首次调用时创建）。

    pool_pre_ping 防止用到失效连接、pool_recycle=1800 定期回收，规避数据库空闲断连；
    同时记录是否为 PostgreSQL，供 _normalize_sql 做方言转换。
    """
    global _engine, _is_postgres
    if _engine is None:
        database_url: str = _build_database_url()
        _is_postgres = "postgresql" in database_url or "postgres" in database_url
        _engine = create_async_engine(
            database_url,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    return _engine


def _normalize_sql(query: str) -> str:
    """将 MySQL 风格 SQL 转换为 PostgreSQL 兼容格式。

    - 去掉反引号（PostgreSQL 不支持）
    - CAST(x AS UNSIGNED) → CAST(x AS BIGINT)
    """
    if not _is_postgres:
        return query
    query = query.replace('`', '')
    query = re.sub(r'CAST\((.+?) AS UNSIGNED\)', r'CAST(\1 AS BIGINT)', query)
    return query


async def fetch_all(query: str, params: Optional[Union[Iterable[Any], Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    执行只读查询并返回字典列表。

    params 支持两种形式：
    - dict: 命名参数，SQL 中使用 :name 占位符
    - tuple/list: 位置参数，SQL 中使用 %s 占位符（自动转换为命名参数）
    """
    engine: AsyncEngine = get_async_engine()

    if params is not None and not isinstance(params, dict):
        params = tuple(params)
        named = {}
        counter = 0

        def _replace_placeholder(m):
            nonlocal counter
            key = f"p{counter}"
            counter += 1
            return f":{key}"

        query = re.sub(r'%s', _replace_placeholder, query)
        for i in range(counter):
            named[f"p{i}"] = params[i] if i < len(params) else None
        params = named

    query = _normalize_sql(query)

    async with engine.connect() as conn:
        result = await conn.execute(text(query), params or {})
        rows = result.mappings().all()
        return [dict(row) for row in rows]


