"""
跨平台舆情对比分析器

将南非新闻（英文原文 + 中文译文）与 X 推文纳入统一分析框架，
输出关键词统计、情感分布、时间趋势、叙事差异等结构化对比数据。
"""

import re
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ── 停用词 ──────────────────────────────────────────────────────────

ENGLISH_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "not", "no", "nor",
    "so", "if", "then", "than", "that", "this", "these", "those", "it",
    "its", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "him", "her", "his", "they", "them", "their", "what", "which", "who",
    "whom", "how", "when", "where", "why", "all", "each", "every", "both",
    "few", "more", "most", "other", "some", "such", "only", "own", "same",
    "just", "also", "very", "about", "up", "out", "as", "into", "over",
    "after", "before", "between", "under", "again", "further", "once",
    "here", "there", "any", "because", "during", "through", "above",
    "below", "while", "too", "s", "t", "re", "ve", "ll", "d", "m",
    "rt", "amp", "https", "http", "co", "www",
})

# ── 数据结构 ────────────────────────────────────────────────────────

@dataclass
class KeywordStats:
    platform: str
    top_keywords: List[Tuple[str, int]] = field(default_factory=list)
    total_documents: int = 0

@dataclass
class SentimentComparison:
    platform: str
    distribution: Dict[str, int] = field(default_factory=dict)
    total_analyzed: int = 0
    dominant_sentiment: str = ""

@dataclass
class TimeTrend:
    platform: str
    daily_counts: Dict[str, int] = field(default_factory=dict)
    peak_date: str = ""
    peak_count: int = 0

@dataclass
class NarrativeDifference:
    news_only_keywords: List[Tuple[str, int]] = field(default_factory=list)
    tweet_only_keywords: List[Tuple[str, int]] = field(default_factory=list)
    common_keywords: List[Tuple[str, int]] = field(default_factory=list)

@dataclass
class CrossPlatformReport:
    topic: str = ""
    news_keyword_stats_en: Optional[KeywordStats] = None
    news_keyword_stats_zh: Optional[KeywordStats] = None
    tweet_keyword_stats: Optional[KeywordStats] = None
    sentiment_comparison: List[SentimentComparison] = field(default_factory=list)
    time_trends: List[TimeTrend] = field(default_factory=list)
    narrative_differences: Optional[NarrativeDifference] = None
    common_focus_points: List[str] = field(default_factory=list)
    llm_summary: str = ""


# ── 关键词提取 ──────────────────────────────────────────────────────

def _extract_english_keywords(texts: List[str], top_n: int = 30) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for text in texts:
        words = re.findall(r"[a-zA-Z'\-]{2,}", text.lower())
        counter.update(w for w in words if w not in ENGLISH_STOPWORDS and len(w) > 2)
    return counter.most_common(top_n)


def _extract_chinese_keywords(texts: List[str], top_n: int = 30) -> List[Tuple[str, int]]:
    try:
        import jieba
    except ImportError:
        logger.warning("jieba 未安装，中文关键词提取将使用简单字符分割")
        return _extract_chinese_keywords_fallback(texts, top_n)

    counter: Counter = Counter()
    for text in texts:
        words = jieba.cut(text)
        counter.update(w for w in words if len(w) >= 2 and not w.isspace())
    return counter.most_common(top_n)


def _extract_chinese_keywords_fallback(texts: List[str], top_n: int = 30) -> List[Tuple[str, int]]:
    counter: Counter = Counter()
    for text in texts:
        chars = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        counter.update(chars)
    return counter.most_common(top_n)


# ── 时间趋势 ────────────────────────────────────────────────────────

def _compute_time_trend(items: list, platform_name: str) -> TimeTrend:
    daily: Counter = Counter()
    for item in items:
        pt = item.publish_time
        if pt:
            day_str = pt.strftime("%Y-%m-%d")
            daily[day_str] += 1

    sorted_days = dict(sorted(daily.items()))
    peak_date, peak_count = "", 0
    if sorted_days:
        peak_date = max(sorted_days, key=sorted_days.get)
        peak_count = sorted_days[peak_date]

    return TimeTrend(
        platform=platform_name,
        daily_counts=sorted_days,
        peak_date=peak_date,
        peak_count=peak_count,
    )


# ── 叙事差异 ────────────────────────────────────────────────────────

def _compute_narrative_diff(
    news_kw: List[Tuple[str, int]],
    tweet_kw: List[Tuple[str, int]],
    top_n: int = 15,
) -> NarrativeDifference:
    news_set = {w for w, _ in news_kw}
    tweet_set = {w for w, _ in tweet_kw}

    news_only = [(w, c) for w, c in news_kw if w not in tweet_set][:top_n]
    tweet_only = [(w, c) for w, c in tweet_kw if w not in news_set][:top_n]
    common = [(w, c) for w, c in news_kw if w in tweet_set][:top_n]

    return NarrativeDifference(
        news_only_keywords=news_only,
        tweet_only_keywords=tweet_only,
        common_keywords=common,
    )


# ── 主分析器 ────────────────────────────────────────────────────────

class CrossPlatformAnalyzer:
    """跨平台对比分析器，接收 QueryResult 列表，输出结构化对比报告。"""

    def __init__(self, sentiment_analyzer=None):
        self._sentiment_analyzer = sentiment_analyzer

    def analyze(
        self,
        topic: str,
        news_results: list,
        tweet_results: list,
    ) -> CrossPlatformReport:
        """对新闻与推文做五步对比分析，返回结构化报告。

        步骤：①中英文关键词统计 -> ②情感分布（需注入 sentiment_analyzer）->
        ③每日时间趋势与峰值 -> ④叙事差异（新闻独有/推文独有/共同）-> ⑤共同关注点。
        """
        report = CrossPlatformReport(topic=topic)

        # ── 1. 关键词统计 ──
        news_en_texts = [r.title_or_content for r in news_results if r.title_or_content]
        # 说明：news_zh_texts 目前恒为空（未从 content_zh 提取译文），因此下方中文关键词
        # 实际是对英文原文做的提取，属占位实现，后续可改为读取 sa_news 的中文译文字段。
        news_zh_texts = []
        tweet_texts = [r.title_or_content for r in tweet_results if r.title_or_content]

        for r in news_results:
            if hasattr(r, 'source_table') and r.source_table == 'sa_news_article':
                pass

        news_en_kw = _extract_english_keywords(news_en_texts)
        report.news_keyword_stats_en = KeywordStats(
            platform="sa_news_en", top_keywords=news_en_kw, total_documents=len(news_en_texts)
        )

        if news_zh_texts:
            news_zh_kw = _extract_chinese_keywords(news_zh_texts)
        else:
            news_zh_kw = _extract_chinese_keywords(news_en_texts)
        report.news_keyword_stats_zh = KeywordStats(
            platform="sa_news_zh", top_keywords=news_zh_kw, total_documents=len(news_zh_texts) or len(news_en_texts)
        )

        tweet_kw = _extract_english_keywords(tweet_texts)
        report.tweet_keyword_stats = KeywordStats(
            platform="x_twitter", top_keywords=tweet_kw, total_documents=len(tweet_texts)
        )

        # ── 2. 情感分布 ──
        if self._sentiment_analyzer:
            for label, items, platform_name in [
                ("news", news_results, "sa_news"),
                ("tweet", tweet_results, "x_twitter"),
            ]:
                try:
                    texts = [r.title_or_content for r in items if r.title_or_content]
                    if texts:
                        batch_result = self._sentiment_analyzer.analyze_batch(texts)
                        dist: Counter = Counter()
                        for sr in batch_result.results:
                            dist[sr.sentiment_label] += 1
                        dominant = dist.most_common(1)[0][0] if dist else "中性"
                        report.sentiment_comparison.append(SentimentComparison(
                            platform=platform_name,
                            distribution=dict(dist),
                            total_analyzed=batch_result.total_processed,
                            dominant_sentiment=dominant,
                        ))
                except Exception as e:
                    logger.warning(f"情感分析失败 ({platform_name}): {e}")

        # ── 3. 时间趋势 ──
        report.time_trends.append(_compute_time_trend(news_results, "sa_news"))
        report.time_trends.append(_compute_time_trend(tweet_results, "x_twitter"))

        # ── 4. 叙事差异 ──
        report.narrative_differences = _compute_narrative_diff(news_en_kw, tweet_kw)

        # ── 5. 共同关注点 ──
        report.common_focus_points = [w for w, _ in report.narrative_differences.common_keywords]

        logger.info(
            f"[CrossPlatformAnalyzer] 分析完成: 新闻 {len(news_results)} 条, "
            f"推文 {len(tweet_results)} 条, 共同关键词 {len(report.common_focus_points)} 个"
        )
        return report

    def format_report_for_llm(self, report: CrossPlatformReport) -> str:
        """将结构化报告格式化为 LLM 可读的文本，用于生成自然语言总结。"""
        lines = [f"## 跨平台舆情对比分析数据 — 主题: {report.topic}\n"]

        lines.append("### 关键词统计")
        if report.news_keyword_stats_en:
            kw_str = ", ".join(f"{w}({c})" for w, c in report.news_keyword_stats_en.top_keywords[:15])
            lines.append(f"新闻原文(英文) [{report.news_keyword_stats_en.total_documents}篇]: {kw_str}")
        if report.news_keyword_stats_zh:
            kw_str = ", ".join(f"{w}({c})" for w, c in report.news_keyword_stats_zh.top_keywords[:15])
            lines.append(f"新闻译文(中文) [{report.news_keyword_stats_zh.total_documents}篇]: {kw_str}")
        if report.tweet_keyword_stats:
            kw_str = ", ".join(f"{w}({c})" for w, c in report.tweet_keyword_stats.top_keywords[:15])
            lines.append(f"X推文 [{report.tweet_keyword_stats.total_documents}条]: {kw_str}")

        if report.sentiment_comparison:
            lines.append("\n### 情感分布")
            for sc in report.sentiment_comparison:
                dist_str = ", ".join(f"{k}: {v}" for k, v in sc.distribution.items())
                lines.append(f"{sc.platform} (共{sc.total_analyzed}条, 主导情感: {sc.dominant_sentiment}): {dist_str}")

        if report.time_trends:
            lines.append("\n### 时间趋势")
            for tt in report.time_trends:
                if tt.daily_counts:
                    lines.append(f"{tt.platform}: 峰值 {tt.peak_date} ({tt.peak_count}条), 共{sum(tt.daily_counts.values())}条")

        if report.narrative_differences:
            nd = report.narrative_differences
            lines.append("\n### 叙事差异")
            if nd.news_only_keywords:
                lines.append(f"新闻独有: {', '.join(w for w, _ in nd.news_only_keywords[:10])}")
            if nd.tweet_only_keywords:
                lines.append(f"推文独有: {', '.join(w for w, _ in nd.tweet_only_keywords[:10])}")
            if nd.common_keywords:
                lines.append(f"共同关注: {', '.join(w for w, _ in nd.common_keywords[:10])}")

        return "\n".join(lines)

