"""
Deep Search Agent（InsightEngine 包入口）

一个无框架（hand-rolled）的深度搜索 AI 代理：对外导出 DeepSearchAgent / create_agent
以及配置 Settings / settings，供 SingleEngineApp 与编排器导入使用。
"""

from .agent import DeepSearchAgent, create_agent
from .utils.config import settings, Settings

__version__ = "1.0.0"
__author__ = "Deep Search Agent Team"

__all__ = ["DeepSearchAgent", "create_agent", "settings", "Settings"]
