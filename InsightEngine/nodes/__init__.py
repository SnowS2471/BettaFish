"""
节点处理模块

汇集 Deep Search Agent 的各处理步骤并统一导出。典型执行顺序：
ReportStructureNode（拆段落） -> FirstSearchNode（首次检索词） -> FirstSummaryNode（段落初稿）
-> ReflectionNode + ReflectionSummaryNode（多轮补充） -> ReportFormattingNode（汇总成报告）。
另含两个旁路分析节点：CrossPlatformAnalysisNode（跨平台对比）、XPropagationAnalysisNode（X 传播）。
"""

from .base_node import BaseNode
from .report_structure_node import ReportStructureNode
from .search_node import FirstSearchNode, ReflectionNode
from .summary_node import FirstSummaryNode, ReflectionSummaryNode
from .formatting_node import ReportFormattingNode
from .cross_platform_node import CrossPlatformAnalysisNode
from .x_propagation_node import XPropagationAnalysisNode

__all__ = [
    "BaseNode",
    "ReportStructureNode",
    "FirstSearchNode",
    "ReflectionNode",
    "FirstSummaryNode",
    "ReflectionSummaryNode",
    "ReportFormattingNode",
    "CrossPlatformAnalysisNode",
    "XPropagationAnalysisNode",
]
