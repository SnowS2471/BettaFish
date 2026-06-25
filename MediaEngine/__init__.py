"""
Deep Search Agent（MediaEngine 包入口）

一个无框架（hand-rolled）的深度搜索 AI 代理：默认 Bocha 多模态搜索、可切换 Anspire。
对外导出 DeepSearchAgent / AnspireSearchAgent / create_agent 与配置 Settings。
"""

from .agent import DeepSearchAgent, AnspireSearchAgent, create_agent
from .utils.config import Settings

__version__ = "1.0.0"
__author__ = "Deep Search Agent Team"

__all__ = ["DeepSearchAgent", "AnspireSearchAgent", "create_agent", "Settings"]
