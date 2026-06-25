"""
工具调用模块（InsightEngine 的工具层统一出口）

对外导出四类工具，供 agent.py 与各节点使用：
- 数据库查询：MediaCrawlerDB / QueryResult / DBResponse / print_response_summary
- 关键词优化：KeywordOptimizer / keyword_optimizer（Qwen，把查询转成网民口语）
- 情感分析：WeiboMultilingualSentimentAnalyzer / multilingual_sentiment_analyzer 等
- 毕设扩展：相关性过滤(TweetRelevanceFilter)、跨平台对比(CrossPlatform*)、
           X 传播分析(XPropagation*)、南非报告取数(SAReportDataProvider)
"""

from .search import (
    MediaCrawlerDB,
    QueryResult,
    DBResponse,
    print_response_summary
)
from .keyword_optimizer import (
    KeywordOptimizer,
    KeywordOptimizationResponse,
    keyword_optimizer
)
from .sentiment_analyzer import (
    WeiboMultilingualSentimentAnalyzer,
    SentimentResult,
    BatchSentimentResult,
    multilingual_sentiment_analyzer,
    analyze_sentiment
)
from .relevance_filter import TweetRelevanceFilter
from .cross_platform_analyzer import (
    CrossPlatformAnalyzer,
    CrossPlatformReport,
    KeywordStats,
    SentimentComparison,
    TimeTrend,
    NarrativeDifference,
)
from .x_propagation_analyzer import (
    XPropagationAnalyzer,
    XPropagationReport,
    TweetCountStats,
    TimeDistribution,
    KeywordFrequency,
    HashtagFrequency,
    ActiveAccountStats,
    HotTweetRanking,
    EngagementStats,
    InteractionHotnessRanking,
)
from .sa_report_data_provider import SAReportDataProvider

__all__ = [
    "MediaCrawlerDB",
    "QueryResult",
    "DBResponse",
    "print_response_summary",
    "KeywordOptimizer",
    "KeywordOptimizationResponse",
    "keyword_optimizer",
    "WeiboMultilingualSentimentAnalyzer",
    "SentimentResult",
    "BatchSentimentResult",
    "multilingual_sentiment_analyzer",
    "analyze_sentiment",
    "TweetRelevanceFilter",
    "CrossPlatformAnalyzer",
    "CrossPlatformReport",
    "KeywordStats",
    "SentimentComparison",
    "TimeTrend",
    "NarrativeDifference",
    "XPropagationAnalyzer",
    "XPropagationReport",
    "TweetCountStats",
    "TimeDistribution",
    "KeywordFrequency",
    "HashtagFrequency",
    "ActiveAccountStats",
    "HotTweetRanking",
    "EngagementStats",
    "InteractionHotnessRanking",
    "SAReportDataProvider",
]
