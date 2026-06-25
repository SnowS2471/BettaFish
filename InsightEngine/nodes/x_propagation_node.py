"""
X平台传播特征分析节点

协调 XPropagationAnalyzer 执行全量分析，
并通过 LLM 生成中文传播特征总结。
"""

from typing import Any, Optional

from loguru import logger

from .base_node import BaseNode
from ..tools.x_propagation_analyzer import XPropagationAnalyzer, XPropagationReport


X_PROPAGATION_SUMMARY_PROMPT = """你是一位专业的社交媒体传播分析师。请根据以下X平台(Twitter)传播特征分析数据，生成一份简洁的中文分析总结，适合用于学术论文的实验分析章节。

要求：
1. 分析推文的传播规模和类型分布特征
2. 分析时间分布规律（发布高峰时段、日期趋势）
3. 分析高频关键词和话题标签反映的讨论焦点
4. 分析活跃账号特征和影响力分布
5. 分析互动数据（点赞、转发、评论）的分布特征
6. 识别高热度内容的共同特征
7. 总结不超过800字

{analysis_data}

请直接输出分析总结："""


class XPropagationAnalysisNode(BaseNode):
    """X 平台传播特征分析节点：调 XPropagationAnalyzer 全量分析 + LLM 总结，产出 XPropagationReport。

    由 agent.x_propagation_analysis 调用（毕设扩展，独立于 research() 主流程）。
    """

    def __init__(
        self,
        llm_client,
        analyzer: Optional[XPropagationAnalyzer] = None,
    ):
        super().__init__(llm_client, "XPropagationAnalysisNode")
        self._analyzer = analyzer or XPropagationAnalyzer()

    def run(self, input_data: Any, **kwargs) -> XPropagationReport:
        """统计 X 平台传播特征（规模/时间分布/关键词/账号/互动等）并让 LLM 总结。

        kwargs: start_date / end_date / top_n（排行榜数量，默认 20）。topic 为空则分析全部推文。
        """
        topic = input_data if isinstance(input_data, str) else str(input_data)
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        top_n = kwargs.get("top_n", 20)

        self.log_info(f"开始X平台传播分析: {topic}")

        report = self._analyzer.analyze_all(
            topic=topic if topic else None,
            start_date=start_date,
            end_date=end_date,
            top_n=top_n,
        )

        self.log_info("生成LLM传播分析总结...")
        try:
            analysis_text = self._analyzer.format_report_for_llm(report)
            prompt = X_PROPAGATION_SUMMARY_PROMPT.format(analysis_data=analysis_text)
            summary = self.llm_client.invoke(
                system_prompt="你是一位专业的社交媒体传播分析师，擅长X平台数据分析。",
                user_prompt=prompt,
            )
            report.llm_summary = summary
        except Exception as e:
            self.log_warning(f"LLM总结生成失败: {e}")
            report.llm_summary = ""

        self.log_info(
            f"传播分析完成: 共分析 {report.total_tweets_analyzed} 条推文"
        )
        return report
