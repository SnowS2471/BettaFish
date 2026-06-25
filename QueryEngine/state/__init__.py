"""
状态管理模块

导出贯穿全流程的状态数据结构 State / Paragraph / Research / Search（定义见 state.py）。
"""

from .state import State, Paragraph, Research, Search

__all__ = ["State", "Paragraph", "Research", "Search"]
