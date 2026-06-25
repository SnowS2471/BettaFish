"""
LLM 模块

导出 QueryEngine 统一的 OpenAI 兼容客户端 LLMClient（实现见 base.py）。
"""

from .base import LLMClient

__all__ = ["LLMClient"]
