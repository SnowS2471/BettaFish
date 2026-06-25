"""
ForumEngine —— 监控三个引擎日志、汇集 SummaryNode 总结并由 LLM 主持人引导讨论。

对外导出 LogMonitor（后台日志监控器）；主持人生成在 llm_host.py，读取端在 utils/forum_reader.py。
"""

from .monitor import LogMonitor

__all__ = ['LogMonitor']
