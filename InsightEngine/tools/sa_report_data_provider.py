"""
南非专题报告数据预处理器

负责从 sa_news_article 和 twitter_tweet 表预拉取结构化数据，
为 ReportEngine 的章节生成提供 dataBundles。
"""

import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..utils.config import settings
from ..utils.db import fetch_all
from .search import QueryResult
from .cross_platform_analyzer import (
    CrossPlatformAnalyzer,
    CrossPlatformReport,
    _extract_english_keywords,
    _extract_chinese_keywords,
)
from .x_propagation_analyzer import XPropagationAnalyzer, XPropagationReport


class SAReportDataProvider:
    """南非专题报告数据预处理器

    从 sa_news_article 和 twitter_tweet 表预拉取并加工结构化数据，打包成 dataBundles 供
    ReportEngine 章节生成使用。各 get_* 方法对应报告不同章节（数据来源 / 新闻概览 / X 传播 /
    跨平台对比 / 传播趋势）；build_all_data_bundles 汇总它们。get_translation_quality_stats
    仅供内部质量监控、不进 bundles。内部复用 XPropagationAnalyzer 与 CrossPlatformAnalyzer。
    """

    def __init__(self, topic: str, start_date: Optional[str] = None, end_date: Optional[str] = None):
        self._topic = topic
        self._start_date = start_date
        self._end_date = end_date
        self._is_postgres = (settings.DB_DIALECT or "mysql").lower() in ("postgresql", "postgres")

    def _execute_query(self, query: str, params=None) -> List[Dict[str, Any]]:
        try:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(fetch_all(query, params))
        except Exception as e:
            logger.exception(f"SAReportDataProvider 数据库查询错误: {e}")
            return []

    def _build_news_where(self) -> Tuple[str, list]:
        clauses, params = [], []
        if self._topic:
            term = f"%{self._topic}%"
            clauses.append("(`title` LIKE %s OR `content` LIKE %s OR `source_keyword` LIKE %s OR `tags` LIKE %s)")
            params.extend([term, term, term, term])
        if self._start_date and self._end_date:
            clauses.append("`publish_time` >= %s AND `publish_time` <= %s")
            params.extend([self._start_date, self._end_date])
        where = " AND ".join(clauses) if clauses else "1=1"
        return where, params

    def _build_tweet_where(self) -> Tuple[str, list]:
        clauses, params = [], []
        if self._topic:
            term = f"%{self._topic}%"
            clauses.append("(`content` LIKE %s OR `hashtags` LIKE %s OR `source_keyword` LIKE %s)")
            params.extend([term, term, term])
        if self._start_date and self._end_date:
            start_ts = int(datetime.strptime(self._start_date, "%Y-%m-%d").timestamp())
            end_ts = int((datetime.strptime(self._end_date, "%Y-%m-%d") + timedelta(days=1)).timestamp())
            clauses.append("`create_time` >= %s AND `create_time` < %s")
            params.extend([start_ts, end_ts])
        where = " AND ".join(clauses) if clauses else "1=1"
        return where, params

    # ── 第 2 章：数据来源说明 ──

    def get_data_source_stats(self) -> Dict[str, Any]:
        """数据源统计：新闻总数、推文总数、来源站点分布、时间范围"""
        news_where, news_params = self._build_news_where()
        tweet_where, tweet_params = self._build_tweet_where()

        news_stats = self._execute_query(
            f"SELECT COUNT(*) as total, COUNT(DISTINCT `source_site`) as source_count, "
            f"MIN(`publish_time`) as earliest, MAX(`publish_time`) as latest "
            f"FROM `sa_news_article` WHERE {news_where}",
            news_params,
        )

        tweet_stats = self._execute_query(
            f"SELECT COUNT(*) as total, COUNT(DISTINCT `user_id`) as unique_users, "
            f"MIN(`create_date_time`) as earliest, MAX(`create_date_time`) as latest "
            f"FROM `twitter_tweet` WHERE {tweet_where}",
            tweet_params,
        )

        source_dist = self._execute_query(
            f"SELECT `source_site`, COUNT(*) as cnt "
            f"FROM `sa_news_article` WHERE {news_where} "
            f"GROUP BY `source_site` ORDER BY cnt DESC",
            news_params,
        )

        ns = news_stats[0] if news_stats else {}
        ts = tweet_stats[0] if tweet_stats else {}

        return {
            "news": {
                "total": ns.get("total", 0),
                "source_count": ns.get("source_count", 0),
                "earliest": ns.get("earliest", ""),
                "latest": ns.get("latest", ""),
            },
            "tweets": {
                "total": ts.get("total", 0),
                "unique_users": ts.get("unique_users", 0),
                "earliest": ts.get("earliest", ""),
                "latest": ts.get("latest", ""),
            },
            "source_distribution": [
                {"site": r["source_site"], "count": r["cnt"]} for r in source_dist
            ],
        }

    # ── 第 3 章：南非新闻概览 ──

    def get_news_overview(self) -> Dict[str, Any]:
        """新闻概览：来源分布、分类分布、时间趋势、关键词、摘要"""
        news_where, news_params = self._build_news_where()

        category_dist = self._execute_query(
            f"SELECT `category`, COUNT(*) as cnt "
            f"FROM `sa_news_article` WHERE {news_where} AND `category` IS NOT NULL "
            f"GROUP BY `category` ORDER BY cnt DESC LIMIT 15",
            news_params,
        )

        time_trend = self._execute_query(
            f"SELECT LEFT(`publish_time`, 10) as pub_date, COUNT(*) as cnt "
            f"FROM `sa_news_article` WHERE {news_where} AND `publish_time` IS NOT NULL "
            f"GROUP BY pub_date ORDER BY pub_date",
            news_params,
        )

        key_articles = self._execute_query(
            f"SELECT `title`, `source_site`, `publish_time`, `article_url` "
            f"FROM `sa_news_article` WHERE {news_where} "
            f"ORDER BY `publish_time` DESC LIMIT 10",
            news_params,
        )

        all_titles = self._execute_query(
            f"SELECT `title`, `content` FROM `sa_news_article` WHERE {news_where} "
            f"AND `title` IS NOT NULL LIMIT 200",
            news_params,
        )
        texts = [
            f"{r.get('title', '')} {(r.get('content', '') or '')[:300]}"
            for r in all_titles if r.get("title")
        ]
        en_keywords = _extract_english_keywords(texts, top_n=20) if texts else []

        zh_titles = self._execute_query(
            f"SELECT `title_zh` FROM `sa_news_article` WHERE {news_where} "
            f"AND `title_zh` IS NOT NULL LIMIT 200",
            news_params,
        )
        zh_texts = [r["title_zh"] for r in zh_titles if r.get("title_zh")]
        zh_keywords = _extract_chinese_keywords(zh_texts, top_n=20) if zh_texts else []

        return {
            "category_distribution": [
                {"category": r["category"], "count": r["cnt"]} for r in category_dist
            ],
            "time_trend": [
                {"date": r["pub_date"], "count": r["cnt"]} for r in time_trend
            ],
            "key_articles": [
                {
                    "title": r.get("title", ""),
                    "source": r.get("source_site", ""),
                    "time": r.get("publish_time", ""),
                    "url": r.get("article_url", ""),
                }
                for r in key_articles
            ],
            "en_keywords": [{"word": w, "count": c} for w, c in en_keywords],
            "zh_keywords": [{"word": w, "count": c} for w, c in zh_keywords],
        }

    # ── 内部监控：翻译质量统计（不注入报告，仅用于后台质量监控）──

    def get_translation_quality_stats(self) -> Dict[str, Any]:
        """（内部使用）翻译质量统计：覆盖率、五维评分、状态分布、典型样本
        此方法不参与 build_all_data_bundles()，仅供系统内部质量监控调用。"""
        news_where, news_params = self._build_news_where()

        total_row = self._execute_query(
            f"SELECT COUNT(*) as total FROM `sa_news_article` WHERE {news_where}",
            news_params,
        )
        total = total_row[0]["total"] if total_row else 0

        translated_row = self._execute_query(
            f"SELECT COUNT(*) as cnt FROM `sa_news_article` "
            f"WHERE {news_where} AND `translation_status` = 'done'",
            news_params,
        )
        translated_count = translated_row[0]["cnt"] if translated_row else 0

        eval_stats = self._execute_query(
            f"SELECT "
            f"AVG(`eval_accuracy`) as avg_accuracy, "
            f"AVG(`eval_fluency`) as avg_fluency, "
            f"AVG(`eval_terminology`) as avg_terminology, "
            f"AVG(`eval_completeness`) as avg_completeness, "
            f"AVG(`eval_overall`) as avg_overall, "
            f"MIN(`eval_overall`) as min_overall, "
            f"MAX(`eval_overall`) as max_overall, "
            f"COUNT(*) as evaluated_count "
            f"FROM `sa_news_article` WHERE {news_where} AND `eval_overall` IS NOT NULL",
            news_params,
        )

        quality_dist = self._execute_query(
            f"SELECT `quality_flag`, COUNT(*) as cnt "
            f"FROM `sa_news_article` WHERE {news_where} AND `quality_flag` IS NOT NULL "
            f"GROUP BY `quality_flag`",
            news_params,
        )

        good_samples = self._execute_query(
            f"SELECT `title`, `title_zh`, `eval_overall`, `eval_accuracy`, `eval_fluency` "
            f"FROM `sa_news_article` WHERE {news_where} AND `eval_overall` IS NOT NULL "
            f"ORDER BY `eval_overall` DESC LIMIT 3",
            news_params,
        )

        poor_samples = self._execute_query(
            f"SELECT `title`, `title_zh`, `eval_overall`, `eval_accuracy`, `eval_fluency` "
            f"FROM `sa_news_article` WHERE {news_where} AND `eval_overall` IS NOT NULL "
            f"ORDER BY `eval_overall` ASC LIMIT 3",
            news_params,
        )

        es = eval_stats[0] if eval_stats else {}
        coverage_rate = (translated_count / total * 100) if total > 0 else 0

        return {
            "total_articles": total,
            "translated_count": translated_count,
            "coverage_rate": round(coverage_rate, 1),
            "evaluated_count": es.get("evaluated_count", 0),
            "avg_scores": {
                "accuracy": round(float(es.get("avg_accuracy") or 0), 2),
                "fluency": round(float(es.get("avg_fluency") or 0), 2),
                "terminology": round(float(es.get("avg_terminology") or 0), 2),
                "completeness": round(float(es.get("avg_completeness") or 0), 2),
                "overall": round(float(es.get("avg_overall") or 0), 2),
            },
            "min_overall": es.get("min_overall", 0),
            "max_overall": es.get("max_overall", 0),
            "quality_distribution": [
                {"flag": r["quality_flag"], "count": r["cnt"]} for r in quality_dist
            ],
            "good_samples": [
                {
                    "title": r.get("title", ""),
                    "title_zh": r.get("title_zh", ""),
                    "overall": r.get("eval_overall", 0),
                    "accuracy": r.get("eval_accuracy", 0),
                    "fluency": r.get("eval_fluency", 0),
                }
                for r in good_samples
            ],
            "poor_samples": [
                {
                    "title": r.get("title", ""),
                    "title_zh": r.get("title_zh", ""),
                    "overall": r.get("eval_overall", 0),
                    "accuracy": r.get("eval_accuracy", 0),
                    "fluency": r.get("eval_fluency", 0),
                }
                for r in poor_samples
            ],
        }

    # ── 第 5 章：X 平台舆情概览 ──

    def get_x_propagation_data(self) -> Dict[str, Any]:
        """X 平台传播数据：复用 XPropagationAnalyzer"""
        analyzer = XPropagationAnalyzer()
        report: XPropagationReport = analyzer.analyze_all(
            topic=self._topic,
            start_date=self._start_date,
            end_date=self._end_date,
        )
        llm_text = analyzer.format_report_for_llm(report)
        chart_configs = analyzer.to_chart_configs(report)
        return {
            "report_text": llm_text,
            "chart_configs": chart_configs,
            "total_tweets": report.total_tweets_analyzed,
        }

    # ── 第 6-7 章：跨平台对比 ──

    def get_cross_platform_data(self, sentiment_analyzer=None) -> Dict[str, Any]:
        """跨平台对比数据：复用 CrossPlatformAnalyzer"""
        analyzer = CrossPlatformAnalyzer(sentiment_analyzer=sentiment_analyzer)

        news_where, news_params = self._build_news_where()
        news_rows = self._execute_query(
            f"SELECT `title`, `content`, `title_zh`, `content_zh`, `publish_time` "
            f"FROM `sa_news_article` WHERE {news_where} LIMIT 300",
            news_params,
        )

        tweet_where, tweet_params = self._build_tweet_where()
        tweet_rows = self._execute_query(
            f"SELECT `content`, `create_date_time`, `like_count`, `retweet_count` "
            f"FROM `twitter_tweet` WHERE {tweet_where} LIMIT 500",
            tweet_params,
        )

        news_results = [
            QueryResult(
                platform="sa_news",
                content_type="news",
                title_or_content=r.get("title", "") or r.get("content", "")[:300],
                source_table="sa_news_article",
            )
            for r in news_rows if r.get("title") or r.get("content")
        ]
        tweet_results = [
            QueryResult(
                platform="x",
                content_type="tweet",
                title_or_content=r.get("content", ""),
                source_table="twitter_tweet",
            )
            for r in tweet_rows if r.get("content")
        ]

        report: CrossPlatformReport = analyzer.analyze(
            topic=self._topic,
            news_results=news_results,
            tweet_results=tweet_results,
        )

        result = {
            "news_keywords_en": (
                [{"word": w, "count": c} for w, c in report.news_keyword_stats_en.top_keywords]
                if report.news_keyword_stats_en else []
            ),
            "news_keywords_zh": (
                [{"word": w, "count": c} for w, c in report.news_keyword_stats_zh.top_keywords]
                if report.news_keyword_stats_zh else []
            ),
            "tweet_keywords": (
                [{"word": w, "count": c} for w, c in report.tweet_keyword_stats.top_keywords]
                if report.tweet_keyword_stats else []
            ),
            "sentiment_comparison": [
                {
                    "platform": s.platform,
                    "distribution": s.distribution,
                    "dominant": s.dominant_sentiment,
                    "total": s.total_analyzed,
                }
                for s in report.sentiment_comparison
            ],
            "time_trends": [
                {
                    "platform": t.platform,
                    "daily_counts": t.daily_counts,
                    "peak_date": t.peak_date,
                    "peak_count": t.peak_count,
                }
                for t in report.time_trends
            ],
            "narrative_differences": None,
            "common_focus_points": report.common_focus_points,
        }

        if report.narrative_differences:
            nd = report.narrative_differences
            result["narrative_differences"] = {
                "news_only": [{"word": w, "count": c} for w, c in nd.news_only_keywords[:15]],
                "tweet_only": [{"word": w, "count": c} for w, c in nd.tweet_only_keywords[:15]],
                "common": [{"word": w, "count": c} for w, c in nd.common_keywords[:15]],
            }

        return result

    # ── 第 8 章：传播趋势与热点内容 ──

    def get_propagation_trends(self) -> Dict[str, Any]:
        """传播趋势与热点：时间对比 + 热门推文"""
        news_where, news_params = self._build_news_where()
        tweet_where, tweet_params = self._build_tweet_where()

        news_daily = self._execute_query(
            f"SELECT LEFT(`publish_time`, 10) as pub_date, COUNT(*) as cnt "
            f"FROM `sa_news_article` WHERE {news_where} AND `publish_time` IS NOT NULL "
            f"GROUP BY pub_date ORDER BY pub_date",
            news_params,
        )

        tweet_daily = self._execute_query(
            f"SELECT `create_date_time` FROM `twitter_tweet` WHERE {tweet_where} "
            f"AND `create_date_time` IS NOT NULL LIMIT 5000",
            tweet_params,
        )
        tweet_day_counter: Counter = Counter()
        for row in tweet_daily:
            dt_str = row.get("create_date_time", "")
            if dt_str and len(dt_str) >= 10:
                tweet_day_counter[dt_str[:10]] += 1

        hot_tweets = self._execute_query(
            f"SELECT `tweet_id`, `content`, `username`, `nickname`, "
            f"`like_count`, `retweet_count`, `reply_count`, `view_count`, `tweet_url` "
            f"FROM `twitter_tweet` WHERE {tweet_where} "
            f"ORDER BY (`like_count` + `retweet_count` * 10 + `reply_count` * 5) DESC LIMIT 10",
            tweet_params,
        )

        return {
            "news_daily_trend": [
                {"date": r["pub_date"], "count": r["cnt"]} for r in news_daily
            ],
            "tweet_daily_trend": [
                {"date": d, "count": c}
                for d, c in sorted(tweet_day_counter.items())
            ],
            "hot_tweets": [
                {
                    "content": (r.get("content", "") or "")[:120],
                    "username": r.get("username", ""),
                    "likes": r.get("like_count", 0),
                    "retweets": r.get("retweet_count", 0),
                    "replies": r.get("reply_count", 0),
                    "views": r.get("view_count", 0),
                    "url": r.get("tweet_url", ""),
                }
                for r in hot_tweets
            ],
        }

    # ── 汇总 ──

    def build_all_data_bundles(self) -> List[Dict[str, Any]]:
        """构建完整的 dataBundles 列表，供 ReportEngine 注入"""
        logger.info(f"SAReportDataProvider: 开始构建数据包, topic={self._topic}")

        bundles = []

        try:
            data_source = self.get_data_source_stats()
            bundles.append({"key": "data_source_stats", "data": data_source})
        except Exception as e:
            logger.error(f"获取数据源统计失败: {e}")

        try:
            news_overview = self.get_news_overview()
            bundles.append({"key": "news_overview", "data": news_overview})
        except Exception as e:
            logger.error(f"获取新闻概览失败: {e}")

        try:
            x_data = self.get_x_propagation_data()
            bundles.append({"key": "x_propagation", "data": x_data})
        except Exception as e:
            logger.error(f"获取X平台传播数据失败: {e}")

        try:
            cross_platform = self.get_cross_platform_data()
            bundles.append({"key": "cross_platform", "data": cross_platform})
        except Exception as e:
            logger.error(f"获取跨平台对比数据失败: {e}")

        try:
            trends = self.get_propagation_trends()
            bundles.append({"key": "propagation_trends", "data": trends})
        except Exception as e:
            logger.error(f"获取传播趋势数据失败: {e}")

        logger.info(f"SAReportDataProvider: 数据包构建完成, 共 {len(bundles)} 个")
        return bundles
