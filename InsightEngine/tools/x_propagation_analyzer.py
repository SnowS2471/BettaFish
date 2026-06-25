"""
X平台传播特征分析器

对 twitter_tweet 表中的推文数据进行传播特征统计分析，
输出推文数量、时间分布、关键词、话题标签、活跃账号、热门推文、互动指标等结构化结果。
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
from .cross_platform_analyzer import _extract_english_keywords, _extract_chinese_keywords


# ── 数据结构 ────────────────────────────────────────────────────────

@dataclass
class TweetCountStats:
    total_tweets: int = 0
    original_tweets: int = 0
    retweets: int = 0
    quotes: int = 0
    replies: int = 0
    type_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class TimeDistribution:
    hourly_counts: Dict[int, int] = field(default_factory=dict)
    daily_counts: Dict[str, int] = field(default_factory=dict)
    peak_hour: int = 0
    peak_date: str = ""
    peak_hour_count: int = 0
    peak_date_count: int = 0


@dataclass
class KeywordFrequency:
    top_keywords: List[Tuple[str, int]] = field(default_factory=list)
    total_documents: int = 0


@dataclass
class HashtagFrequency:
    top_hashtags: List[Tuple[str, int]] = field(default_factory=list)
    total_tweets_with_hashtags: int = 0


@dataclass
class ActiveAccount:
    username: str = ""
    nickname: str = ""
    user_id: str = ""
    tweet_count: int = 0
    total_likes: int = 0
    total_retweets: int = 0
    total_views: int = 0
    is_verified: bool = False
    verified_type: str = ""
    followers_count: int = 0

@dataclass
class ActiveAccountStats:
    top_accounts: List[ActiveAccount] = field(default_factory=list)
    total_unique_users: int = 0
    verified_user_count: int = 0


@dataclass
class HotTweet:
    tweet_id: str = ""
    content: str = ""
    username: str = ""
    nickname: str = ""
    like_count: int = 0
    retweet_count: int = 0
    reply_count: int = 0
    quote_count: int = 0
    view_count: int = 0
    bookmark_count: int = 0
    hotness_score: float = 0.0
    create_time: Optional[datetime] = None
    tweet_url: str = ""
    lang: str = ""


@dataclass
class HotTweetRanking:
    top_tweets: List[HotTweet] = field(default_factory=list)


@dataclass
class EngagementStats:
    total_likes: int = 0
    total_retweets: int = 0
    total_replies: int = 0
    total_quotes: int = 0
    total_bookmarks: int = 0
    total_views: int = 0
    avg_likes: float = 0.0
    avg_retweets: float = 0.0
    avg_replies: float = 0.0
    avg_views: float = 0.0
    max_likes: int = 0
    max_retweets: int = 0
    max_views: int = 0


@dataclass
class InteractionHotnessRanking:
    top_by_likes: List[HotTweet] = field(default_factory=list)
    top_by_retweets: List[HotTweet] = field(default_factory=list)
    top_by_replies: List[HotTweet] = field(default_factory=list)
    top_by_hotness: List[HotTweet] = field(default_factory=list)


@dataclass
class XPropagationReport:
    topic: str = ""
    analysis_time_range: str = ""
    total_tweets_analyzed: int = 0
    tweet_count_stats: Optional[TweetCountStats] = None
    time_distribution: Optional[TimeDistribution] = None
    keyword_frequency: Optional[KeywordFrequency] = None
    hashtag_frequency: Optional[HashtagFrequency] = None
    active_account_stats: Optional[ActiveAccountStats] = None
    hot_tweet_ranking: Optional[HotTweetRanking] = None
    engagement_stats: Optional[EngagementStats] = None
    interaction_hotness_ranking: Optional[InteractionHotnessRanking] = None
    llm_summary: str = ""


# ── 分析器 ──────────────────────────────────────────────────────────

class XPropagationAnalyzer:
    """X 平台传播特征分析器：直接对 twitter_tweet 表做多维统计，产出 XPropagationReport。

    每个 get_* 方法负责一个维度（数量/时间/关键词/标签/账号/热门/互动），analyze_all 串起全部；
    SQL 通过 _is_postgres 及若干方言辅助函数（_from_unixtime_expr 等）兼容 MySQL/PostgreSQL。
    """

    # 互动热度权重（沿用与 MediaCrawlerDB 一致的思路：转发 > 评论 > 点赞 > 浏览）
    W_LIKE = 1.0
    W_REPLY = 5.0
    W_RETWEET = 10.0
    W_VIEW = 0.1

    def __init__(self):
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
            logger.exception(f"数据库查询时发生错误: {e}")
            return []

    def _build_where(
        self,
        topic: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Tuple[str, list]:
        clauses, params = [], []
        if topic:
            term = f"%{topic}%"
            clauses.append("(`content` LIKE %s OR `hashtags` LIKE %s OR `source_keyword` LIKE %s)")
            params.extend([term, term, term])
        if start_date and end_date:
            start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_ts = int((datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)).timestamp())
            clauses.append("`create_time` >= %s AND `create_time` < %s")
            params.extend([start_ts, end_ts])
        where = " AND ".join(clauses) if clauses else "1=1"
        return where, params

    def _ts_to_datetime(self, ts: Any) -> Optional[datetime]:
        if not ts:
            return None
        try:
            val = float(ts)
            return datetime.fromtimestamp(val / 1000 if val > 1_000_000_000_000 else val)
        except (ValueError, TypeError):
            return None

    def _from_unixtime_expr(self, col: str) -> str:
        if self._is_postgres:
            return f"TO_TIMESTAMP({col})"
        return f"FROM_UNIXTIME({col})"

    def _hour_expr(self, ts_expr: str) -> str:
        if self._is_postgres:
            return f"EXTRACT(HOUR FROM {ts_expr})::INT"
        return f"HOUR({ts_expr})"

    def _date_expr(self, ts_expr: str) -> str:
        if self._is_postgres:
            return f"({ts_expr})::DATE"
        return f"DATE({ts_expr})"

    def _cast_unsigned(self, col: str) -> str:
        return f"COALESCE(CAST(`{col}` AS UNSIGNED), 0)"

    # ── 1. 推文数量统计 ──

    def get_tweet_count_stats(
        self, topic: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None,
    ) -> TweetCountStats:
        where, params = self._build_where(topic, start_date, end_date)
        query = (
            f"SELECT COUNT(*) as total,"
            f" SUM(CASE WHEN `tweet_type`='tweet' THEN 1 ELSE 0 END) as originals,"
            f" SUM(CASE WHEN `tweet_type`='retweet' THEN 1 ELSE 0 END) as rts,"
            f" SUM(CASE WHEN `tweet_type`='quote' THEN 1 ELSE 0 END) as quotes,"
            f" SUM(CASE WHEN `tweet_type`='reply' THEN 1 ELSE 0 END) as replies"
            f" FROM `twitter_tweet` WHERE {where}"
        )
        rows = self._execute_query(query, params)
        if not rows:
            return TweetCountStats()
        r = rows[0]
        stats = TweetCountStats(
            total_tweets=int(r.get("total") or 0),
            original_tweets=int(r.get("originals") or 0),
            retweets=int(r.get("rts") or 0),
            quotes=int(r.get("quotes") or 0),
            replies=int(r.get("replies") or 0),
        )
        stats.type_distribution = {
            "tweet": stats.original_tweets,
            "retweet": stats.retweets,
            "quote": stats.quotes,
            "reply": stats.replies,
        }
        return stats

    # ── 2. 时间分布统计 ──

    def get_time_distribution(
        self, topic: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None,
    ) -> TimeDistribution:
        where, params = self._build_where(topic, start_date, end_date)
        ts_expr = self._from_unixtime_expr("`create_time`")

        hourly_q = (
            f"SELECT {self._hour_expr(ts_expr)} as h, COUNT(*) as cnt"
            f" FROM `twitter_tweet` WHERE `create_time` IS NOT NULL AND {where}"
            f" GROUP BY h ORDER BY h"
        )
        hourly_rows = self._execute_query(hourly_q, list(params))

        daily_q = (
            f"SELECT {self._date_expr(ts_expr)} as d, COUNT(*) as cnt"
            f" FROM `twitter_tweet` WHERE `create_time` IS NOT NULL AND {where}"
            f" GROUP BY d ORDER BY d"
        )
        daily_rows = self._execute_query(daily_q, list(params))

        dist = TimeDistribution()
        for r in hourly_rows:
            h = int(r["h"])
            dist.hourly_counts[h] = int(r["cnt"])
        for r in daily_rows:
            d = str(r["d"])
            dist.daily_counts[d] = int(r["cnt"])

        if dist.hourly_counts:
            dist.peak_hour = max(dist.hourly_counts, key=dist.hourly_counts.get)
            dist.peak_hour_count = dist.hourly_counts[dist.peak_hour]
        if dist.daily_counts:
            dist.peak_date = max(dist.daily_counts, key=dist.daily_counts.get)
            dist.peak_date_count = dist.daily_counts[dist.peak_date]
        return dist

    # ── 3. 高频关键词统计 ──

    def get_keyword_frequency(
        self, topic: Optional[str] = None, start_date: Optional[str] = None,
        end_date: Optional[str] = None, top_n: int = 30,
    ) -> KeywordFrequency:
        where, params = self._build_where(topic, start_date, end_date)
        query = (
            f"SELECT `content` FROM `twitter_tweet`"
            f" WHERE `content` IS NOT NULL AND `content` != '' AND {where}"
            f" LIMIT 10000"
        )
        rows = self._execute_query(query, params)
        texts = [r["content"] for r in rows if r.get("content")]
        top_kw = _extract_english_keywords(texts, top_n=top_n)
        return KeywordFrequency(top_keywords=top_kw, total_documents=len(texts))

    # ── 4. 高频话题标签统计 ──

    def get_hashtag_frequency(
        self, topic: Optional[str] = None, start_date: Optional[str] = None,
        end_date: Optional[str] = None, top_n: int = 30,
    ) -> HashtagFrequency:
        where, params = self._build_where(topic, start_date, end_date)
        query = (
            f"SELECT `hashtags` FROM `twitter_tweet`"
            f" WHERE `hashtags` IS NOT NULL AND `hashtags` != '' AND `hashtags` != '[]' AND {where}"
        )
        rows = self._execute_query(query, params)
        counter: Counter = Counter()
        tweets_with_tags = 0
        for r in rows:
            raw = r.get("hashtags", "")
            try:
                tags = json.loads(raw) if isinstance(raw, str) else raw
                if isinstance(tags, list) and tags:
                    tweets_with_tags += 1
                    counter.update(t.lower().strip() for t in tags if isinstance(t, str) and t.strip())
            except (json.JSONDecodeError, TypeError):
                continue
        return HashtagFrequency(top_hashtags=counter.most_common(top_n), total_tweets_with_hashtags=tweets_with_tags)

    # ── 5. 活跃账号统计 ──

    def get_active_accounts(
        self, topic: Optional[str] = None, start_date: Optional[str] = None,
        end_date: Optional[str] = None, top_n: int = 20,
    ) -> ActiveAccountStats:
        where, params = self._build_where(topic, start_date, end_date)
        like_c = self._cast_unsigned("like_count")
        rt_c = self._cast_unsigned("retweet_count")
        view_c = self._cast_unsigned("view_count")

        agg_q = (
            f"SELECT `user_id`, `username`, MAX(`nickname`) as nickname,"
            f" MAX(`user_verified`) as verified, MAX(`user_verified_type`) as vtype,"
            f" COUNT(*) as cnt, SUM({like_c}) as tlikes,"
            f" SUM({rt_c}) as trts, SUM({view_c}) as tviews"
            f" FROM `twitter_tweet` WHERE `user_id` IS NOT NULL AND {where}"
            f" GROUP BY `user_id`, `username` ORDER BY cnt DESC LIMIT %s"
        )
        rows = self._execute_query(agg_q, params + [top_n])

        accounts = []
        user_ids = []
        for r in rows:
            acc = ActiveAccount(
                user_id=str(r.get("user_id", "")),
                username=str(r.get("username", "")),
                nickname=str(r.get("nickname", "")),
                tweet_count=int(r.get("cnt") or 0),
                total_likes=int(r.get("tlikes") or 0),
                total_retweets=int(r.get("trts") or 0),
                total_views=int(r.get("tviews") or 0),
                is_verified=bool(r.get("verified")),
                verified_type=str(r.get("vtype") or ""),
            )
            accounts.append(acc)
            user_ids.append(acc.user_id)

        if user_ids:
            placeholders = ", ".join(["%s"] * len(user_ids))
            creator_q = (
                f"SELECT `user_id`, COALESCE(CAST(`followers_count` AS UNSIGNED), 0) as followers"
                f" FROM `twitter_creator` WHERE `user_id` IN ({placeholders})"
            )
            creator_rows = self._execute_query(creator_q, user_ids)
            followers_map = {str(r["user_id"]): int(r["followers"]) for r in creator_rows}
            for acc in accounts:
                acc.followers_count = followers_map.get(acc.user_id, 0)

        unique_q = (
            f"SELECT COUNT(DISTINCT `user_id`) as uniq,"
            f" COUNT(DISTINCT CASE WHEN `user_verified`=1 THEN `user_id` END) as verified_cnt"
            f" FROM `twitter_tweet` WHERE {where}"
        )
        uniq_rows = self._execute_query(unique_q, params)
        total_unique = int(uniq_rows[0]["uniq"]) if uniq_rows else 0
        verified_cnt = int(uniq_rows[0]["verified_cnt"]) if uniq_rows else 0

        return ActiveAccountStats(top_accounts=accounts, total_unique_users=total_unique, verified_user_count=verified_cnt)

    # ── 6. 热门推文排行 ──

    def _hotness_formula(self) -> str:
        like_c = self._cast_unsigned("like_count")
        reply_c = self._cast_unsigned("reply_count")
        rt_c = self._cast_unsigned("retweet_count")
        view_c = f"COALESCE(CAST(`view_count` AS DECIMAL(20,2)), 0)"
        return (
            f"({like_c} * {self.W_LIKE} + {reply_c} * {self.W_REPLY}"
            f" + {rt_c} * {self.W_RETWEET} + {view_c} * {self.W_VIEW})"
        )

    def _row_to_hot_tweet(self, r: Dict[str, Any]) -> HotTweet:
        return HotTweet(
            tweet_id=str(r.get("tweet_id", "")),
            content=str(r.get("content", "")),
            username=str(r.get("username", "")),
            nickname=str(r.get("nickname", "")),
            like_count=int(r.get("likes") or 0),
            retweet_count=int(r.get("retweets") or 0),
            reply_count=int(r.get("replies") or 0),
            quote_count=int(r.get("quotes") or 0),
            view_count=int(r.get("views") or 0),
            bookmark_count=int(r.get("bookmarks") or 0),
            hotness_score=float(r.get("hotness_score") or 0),
            create_time=self._ts_to_datetime(r.get("create_time")),
            tweet_url=str(r.get("tweet_url") or ""),
            lang=str(r.get("lang") or ""),
        )

    def _select_tweet_fields(self) -> str:
        like_c = self._cast_unsigned("like_count")
        rt_c = self._cast_unsigned("retweet_count")
        reply_c = self._cast_unsigned("reply_count")
        quote_c = self._cast_unsigned("quote_count")
        view_c = self._cast_unsigned("view_count")
        bm_c = self._cast_unsigned("bookmark_count")
        return (
            f"`tweet_id`, `content`, `username`, `nickname`, `tweet_url`, `lang`, `create_time`,"
            f" {like_c} as likes, {rt_c} as retweets, {reply_c} as replies,"
            f" {quote_c} as quotes, {view_c} as views, {bm_c} as bookmarks"
        )

    def get_hot_tweet_ranking(
        self, topic: Optional[str] = None, start_date: Optional[str] = None,
        end_date: Optional[str] = None, top_n: int = 20,
    ) -> HotTweetRanking:
        where, params = self._build_where(topic, start_date, end_date)
        query = (
            f"SELECT {self._select_tweet_fields()},"
            f" {self._hotness_formula()} as hotness_score"
            f" FROM `twitter_tweet` WHERE {where}"
            f" ORDER BY hotness_score DESC LIMIT %s"
        )
        rows = self._execute_query(query, params + [top_n])
        return HotTweetRanking(top_tweets=[self._row_to_hot_tweet(r) for r in rows])

    # ── 7. 互动指标统计 ──

    def get_engagement_stats(
        self, topic: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None,
    ) -> EngagementStats:
        where, params = self._build_where(topic, start_date, end_date)
        cols = ["like_count", "retweet_count", "reply_count", "quote_count", "bookmark_count", "view_count"]
        agg_parts = []
        for c in cols:
            cast = self._cast_unsigned(c)
            short = c.replace("_count", "")
            agg_parts.append(f"SUM({cast}) as sum_{short}")
            agg_parts.append(f"AVG({cast}) as avg_{short}")
            agg_parts.append(f"MAX({cast}) as max_{short}")
        agg_parts.append("COUNT(*) as total")
        query = f"SELECT {', '.join(agg_parts)} FROM `twitter_tweet` WHERE {where}"
        rows = self._execute_query(query, params)
        if not rows:
            return EngagementStats()
        r = rows[0]
        return EngagementStats(
            total_likes=int(r.get("sum_like") or 0),
            total_retweets=int(r.get("sum_retweet") or 0),
            total_replies=int(r.get("sum_reply") or 0),
            total_quotes=int(r.get("sum_quote") or 0),
            total_bookmarks=int(r.get("sum_bookmark") or 0),
            total_views=int(r.get("sum_view") or 0),
            avg_likes=float(r.get("avg_like") or 0),
            avg_retweets=float(r.get("avg_retweet") or 0),
            avg_replies=float(r.get("avg_reply") or 0),
            avg_views=float(r.get("avg_view") or 0),
            max_likes=int(r.get("max_like") or 0),
            max_retweets=int(r.get("max_retweet") or 0),
            max_views=int(r.get("max_view") or 0),
        )

    # ── 8. 互动热度排行 ──

    def get_interaction_hotness_ranking(
        self, topic: Optional[str] = None, start_date: Optional[str] = None,
        end_date: Optional[str] = None, top_n: int = 10,
    ) -> InteractionHotnessRanking:
        where, params = self._build_where(topic, start_date, end_date)
        fields = self._select_tweet_fields()
        formula = self._hotness_formula()

        ranking = InteractionHotnessRanking()
        for attr, order_col in [
            ("top_by_likes", self._cast_unsigned("like_count")),
            ("top_by_retweets", self._cast_unsigned("retweet_count")),
            ("top_by_replies", self._cast_unsigned("reply_count")),
            ("top_by_hotness", formula),
        ]:
            query = (
                f"SELECT {fields}, {formula} as hotness_score"
                f" FROM `twitter_tweet` WHERE {where}"
                f" ORDER BY {order_col} DESC LIMIT %s"
            )
            rows = self._execute_query(query, list(params) + [top_n])
            setattr(ranking, attr, [self._row_to_hot_tweet(r) for r in rows])
        return ranking

    # ── 全量分析 ──

    def analyze_all(
        self,
        topic: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        top_n: int = 20,
    ) -> XPropagationReport:
        """依次跑完 8 个维度的统计，汇总成一份 XPropagationReport（topic 为空则统计全部推文）。"""
        logger.info(f"[XPropagationAnalyzer] 开始全量分析: topic={topic}, range={start_date}~{end_date}")

        report = XPropagationReport(topic=topic or "")
        if start_date and end_date:
            report.analysis_time_range = f"{start_date} ~ {end_date}"

        report.tweet_count_stats = self.get_tweet_count_stats(topic, start_date, end_date)
        report.total_tweets_analyzed = report.tweet_count_stats.total_tweets
        logger.info(f"  推文数量: {report.total_tweets_analyzed}")

        report.time_distribution = self.get_time_distribution(topic, start_date, end_date)
        report.keyword_frequency = self.get_keyword_frequency(topic, start_date, end_date, top_n=top_n)
        report.hashtag_frequency = self.get_hashtag_frequency(topic, start_date, end_date, top_n=top_n)
        report.active_account_stats = self.get_active_accounts(topic, start_date, end_date, top_n=top_n)
        report.hot_tweet_ranking = self.get_hot_tweet_ranking(topic, start_date, end_date, top_n=top_n)
        report.engagement_stats = self.get_engagement_stats(topic, start_date, end_date)
        report.interaction_hotness_ranking = self.get_interaction_hotness_ranking(topic, start_date, end_date, top_n=min(top_n, 10))

        logger.info(f"[XPropagationAnalyzer] 全量分析完成: {report.total_tweets_analyzed} 条推文")
        return report

    # ── LLM 格式化 ──

    def format_report_for_llm(self, report: XPropagationReport) -> str:
        """把结构化报告拍平成 LLM 可读文本，供 XPropagationAnalysisNode 生成自然语言总结。"""
        lines = [f"## X平台传播特征分析数据 — 主题: {report.topic or '全部'}\n"]

        if report.tweet_count_stats:
            s = report.tweet_count_stats
            lines.append("### 推文数量统计")
            lines.append(f"总计: {s.total_tweets} 条 (原创 {s.original_tweets}, 转发 {s.retweets}, 引用 {s.quotes}, 回复 {s.replies})")

        if report.time_distribution:
            td = report.time_distribution
            lines.append("\n### 时间分布")
            if td.peak_hour_count:
                lines.append(f"发布高峰时段: {td.peak_hour}时 ({td.peak_hour_count}条)")
            if td.peak_date_count:
                lines.append(f"发布高峰日期: {td.peak_date} ({td.peak_date_count}条)")

        if report.keyword_frequency and report.keyword_frequency.top_keywords:
            kw = report.keyword_frequency
            kw_str = ", ".join(f"{w}({c})" for w, c in kw.top_keywords[:15])
            lines.append(f"\n### 高频关键词 [{kw.total_documents}篇]")
            lines.append(kw_str)

        if report.hashtag_frequency and report.hashtag_frequency.top_hashtags:
            ht = report.hashtag_frequency
            ht_str = ", ".join(f"#{t}({c})" for t, c in ht.top_hashtags[:15])
            lines.append(f"\n### 高频话题标签 [{ht.total_tweets_with_hashtags}条含标签]")
            lines.append(ht_str)

        if report.active_account_stats:
            aa = report.active_account_stats
            lines.append(f"\n### 活跃账号 (共{aa.total_unique_users}个用户, {aa.verified_user_count}个认证)")
            for a in aa.top_accounts[:10]:
                v = " [V]" if a.is_verified else ""
                lines.append(f"  @{a.username}{v}: {a.tweet_count}条, {a.total_likes}赞, {a.followers_count}粉丝")

        if report.engagement_stats:
            es = report.engagement_stats
            lines.append("\n### 互动指标汇总")
            lines.append(f"总点赞: {es.total_likes}, 总转发: {es.total_retweets}, 总评论: {es.total_replies}, 总浏览: {es.total_views}")
            lines.append(f"均点赞: {es.avg_likes:.1f}, 均转发: {es.avg_retweets:.1f}, 均评论: {es.avg_replies:.1f}")

        if report.hot_tweet_ranking and report.hot_tweet_ranking.top_tweets:
            lines.append("\n### 热门推文 TOP5")
            for i, t in enumerate(report.hot_tweet_ranking.top_tweets[:5], 1):
                preview = t.content[:80].replace("\n", " ")
                lines.append(f"  {i}. @{t.username}: {preview}... (赞{t.like_count} 转{t.retweet_count} 评{t.reply_count} 看{t.view_count})")

        return "\n".join(lines)

    # ── 图表配置输出 ──

    def to_chart_configs(self, report: XPropagationReport) -> List[Dict[str, Any]]:
        """把报告里的时间/互动数据转成 Chart.js widget 配置（折线/柱状/雷达），供 ReportEngine 渲染图表。"""
        configs: List[Dict[str, Any]] = []

        if report.time_distribution and report.time_distribution.daily_counts:
            dc = report.time_distribution.daily_counts
            configs.append({
                "type": "widget", "widgetType": "chart.js/line",
                "widgetId": "x-prop-daily-trend",
                "props": {"title": "每日发布趋势"},
                "data": {
                    "labels": list(dc.keys()),
                    "datasets": [{"label": "推文数量", "data": list(dc.values())}],
                },
            })

        if report.time_distribution and report.time_distribution.hourly_counts:
            hc = report.time_distribution.hourly_counts
            labels = [f"{h}时" for h in range(24)]
            data = [hc.get(h, 0) for h in range(24)]
            configs.append({
                "type": "widget", "widgetType": "chart.js/bar",
                "widgetId": "x-prop-hourly-dist",
                "props": {"title": "发布时段分布"},
                "data": {
                    "labels": labels,
                    "datasets": [{"label": "推文数量", "data": data}],
                },
            })

        if report.engagement_stats:
            es = report.engagement_stats
            configs.append({
                "type": "widget", "widgetType": "chart.js/radar",
                "widgetId": "x-prop-engagement-radar",
                "props": {"title": "互动指标概览 (均值)"},
                "data": {
                    "labels": ["点赞", "转发", "评论", "引用", "书签"],
                    "datasets": [{
                        "label": "平均互动",
                        "data": [es.avg_likes, es.avg_retweets, es.avg_replies,
                                 float(es.total_quotes / max(report.total_tweets_analyzed, 1)),
                                 float(es.total_bookmarks / max(report.total_tweets_analyzed, 1))],
                    }],
                },
            })

        return configs
