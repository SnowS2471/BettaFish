"""
跨平台对比分析节点

协调搜索、过滤、分析、LLM总结的完整流程，
输出结构化的跨平台舆情对比报告。
"""

from typing import Any, Optional

from loguru import logger

from .base_node import BaseNode
from ..tools.search import MediaCrawlerDB
from ..tools.relevance_filter import TweetRelevanceFilter
from ..tools.cross_platform_analyzer import CrossPlatformAnalyzer, CrossPlatformReport


CROSS_PLATFORM_SUMMARY_PROMPT = """你是一位专业的舆情分析师。请根据以下跨平台对比分析数据，生成一份简洁的中文分析总结。

要求：
1. 分析新闻媒体与社交平台在该主题上的关注点差异
2. 对比两个平台的情感倾向差异
3. 分析时间趋势上的异同
4. 识别值得关注的叙事差异
5. 总结不超过500字

{analysis_data}

请直接输出分析总结："""


class CrossPlatformAnalysisNode(BaseNode):
    """跨平台对比分析节点：编排「搜索 -> 过滤 -> 对比分析 -> LLM 总结」，产出 CrossPlatformReport。

    由 agent.cross_platform_research 调用（毕设扩展，独立于 research() 主流程）。
    """

    def __init__(
        self,
        llm_client,
        search_db: Optional[MediaCrawlerDB] = None,
        sentiment_analyzer=None,
        embedding_model=None,
    ):
        super().__init__(llm_client, "CrossPlatformAnalysisNode")
        self._search_db = search_db or MediaCrawlerDB()
        self._relevance_filter = TweetRelevanceFilter(embedding_model=embedding_model)
        self._analyzer = CrossPlatformAnalyzer(sentiment_analyzer=sentiment_analyzer)

    def run(self, input_data: Any, **kwargs) -> CrossPlatformReport:
        """执行「南非新闻 vs X 推文」对比的完整流程并返回结构化报告。

        步骤：①查 sa_news 新闻 -> ②查 X 推文 -> ③相关性过滤推文 ->
        ④CrossPlatformAnalyzer 做关键词/情感/时间/叙事对比 -> ⑤LLM 生成中文总结。
        kwargs: news_limit / tweet_limit / relevance_threshold。
        """
        topic = input_data if isinstance(input_data, str) else str(input_data)
        self.log_info(f"开始跨平台分析: {topic}")

        # 1. 搜索南非新闻
        self.log_info("搜索南非新闻...")
        news_response = self._search_db.search_topic_on_platform(
            platform="sa_news", topic=topic, limit=kwargs.get("news_limit", 100)
        )
        news_results = news_response.results
        self.log_info(f"获取到 {len(news_results)} 条新闻")

        # 2. 搜索 X 推文
        self.log_info("搜索 X 推文...")
        tweet_response = self._search_db.search_topic_on_platform(
            platform="x", topic=topic, limit=kwargs.get("tweet_limit", 200)
        )
        raw_tweets = tweet_response.results
        self.log_info(f"获取到 {len(raw_tweets)} 条原始推文")

        # 3. 过滤低相关推文
        filtered_tweets = self._relevance_filter.filter_tweets(
            raw_tweets,
            topic,
            threshold=kwargs.get("relevance_threshold", 0.35),
        )

        # 4. 运行跨平台分析
        self.log_info("运行跨平台对比分析...")
        report = self._analyzer.analyze(topic, news_results, filtered_tweets)

        # 5. LLM 生成总结
        self.log_info("生成 LLM 总结...")
        try:
            analysis_text = self._analyzer.format_report_for_llm(report)
            prompt = CROSS_PLATFORM_SUMMARY_PROMPT.format(analysis_data=analysis_text)
            summary = self.llm_client.invoke(
                system_prompt="你是一位专业的跨平台舆情分析师。",
                user_prompt=prompt,
            )
            report.llm_summary = summary
        except Exception as e:
            self.log_warning(f"LLM 总结生成失败: {e}")
            report.llm_summary = ""

        self.log_info(
            f"分析完成: 新闻 {len(news_results)} 条, "
            f"推文 {len(filtered_tweets)}/{len(raw_tweets)} 条"
        )
        return report
