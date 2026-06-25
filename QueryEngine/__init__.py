"""
Deep Search Agent（QueryEngine 包入口）

一个无框架（hand-rolled）的深度搜索 AI 代理：基于 Tavily 网络搜索做新闻分析报告。
对外导出 DeepSearchAgent / create_agent 与配置 Settings，供 SingleEngineApp 与编排器使用。
"""

from .agent import DeepSearchAgent, create_agent
from .utils.config import Settings

__version__ = "1.0.0"
__author__ = "Deep Search Agent Team"

__all__ = ["DeepSearchAgent", "create_agent", "Settings"]
